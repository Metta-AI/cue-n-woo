#!/usr/bin/env python3
"""Fleet load monitor + graph endpoint for the cue-n-woo SkyServe worker fleet.

WHAT IT DOES
  - Discovers the fleet's replicas (via SkyServe's local state DB) and polls each
    replica's /health every --interval seconds.
  - Derives queries/min from the per-replica `requests_served` counter (delta /
    interval), plus replica count, queue depth, latency-ish signals, and RAM/VRAM.
  - Appends every sample to a JSONL log (durable history) and keeps an in-memory
    ring for the live graph.
  - Serves a self-contained HTML dashboard at  /          (auto-refreshing chart)
    and the raw series as JSON at             /data       (?minutes=N to window).

WHERE IT RUNS
  On the SkyServe CONTROLLER. The controller is the one always-on box that can
  reach every replica by its (internal) URL and already has the serve state DB.
  Replicas are NOT reachable from a laptop, so this can't run locally.

  # on the controller:
  python3 fleet_metrics.py --service cue-n-woo-workers --port 8000 --interval 15
  # then browse http://<controller-EIP>:8000/   (controller SG already allows 30001-30020;
  # pick a port in that range, e.g. --port 30010, or open the chosen port in the SG)

WHY POLL /health (vs scraping SkyServe)
  SkyServe's /autoscaler/info exposes only {target,min,max} replicas -- no QPS
  history. The request-rate data it uses internally for autoscaling is a rolling
  60s window that is never persisted as a time series. The worker's /health, by
  contrast, already reports a monotonic `requests_served` counter (added for this
  purpose), so sampling it and differencing gives a clean, durable QPS series with
  zero changes to the workers.

This is a single stdlib-only file (no deps beyond what SkyServe already installs:
it uses `requests` if present, else urllib). Safe to scp to the controller and run.
"""
import argparse
import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Replica discovery
# ---------------------------------------------------------------------------


def discover_replica_urls(service_name):
    """Return {replica_id: base_url} for the READY replicas of the service.

    Reads SkyServe's local state DB (we're on the controller). Falls back to an
    empty dict if SkyServe isn't importable so the monitor degrades gracefully
    instead of crashing.
    """
    try:
        from sky.serve import serve_state
    except Exception as e:  # pragma: no cover - only on misconfigured host
        print(f"[discover] cannot import sky.serve.serve_state: {e}")
        return {}
    urls = {}
    try:
        for info in serve_state.get_replica_infos(service_name):
            # is_ready: the replica passed its readiness probe and is in rotation.
            if not getattr(info, "is_ready", False):
                continue
            url = info.url  # internal http://IP:port, reachable from controller
            if url:
                urls[info.replica_id] = url
    except Exception as e:
        print(f"[discover] error reading replica infos: {e}")
    return urls


# ---------------------------------------------------------------------------
# HTTP fetch (requests if available, else urllib)
# ---------------------------------------------------------------------------

try:
    import requests  # noqa: F401

    def _get_json(url, timeout):
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
except Exception:

    def _get_json(url, timeout):
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Poller
# ---------------------------------------------------------------------------


class FleetMonitor:
    def __init__(self, service_name, interval, logfile, history_minutes):
        self.service_name = service_name
        self.interval = interval
        self.logfile = logfile
        # in-memory ring large enough for `history_minutes` of samples
        maxlen = max(60, int(history_minutes * 60 / max(interval, 1)) + 10)
        self.samples = deque(maxlen=maxlen)
        self._prev = {}  # replica_id -> (timestamp, requests_served)
        self._lock = threading.Lock()
        self._load_existing()

    def _load_existing(self):
        """Re-hydrate the in-memory ring from the JSONL log on restart."""
        if not self.logfile or not os.path.exists(self.logfile):
            return
        try:
            with open(self.logfile) as f:
                lines = f.readlines()[-self.samples.maxlen:]
            for line in lines:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))
        except Exception as e:
            print(f"[monitor] could not reload history: {e}")

    def poll_once(self):
        now = time.time()
        urls = discover_replica_urls(self.service_name)
        replicas = []
        agg_qpm = 0.0
        agg_queue = 0
        ready = 0
        for rid, base in urls.items():
            entry = {"replica_id": rid, "url": base, "ok": False}
            try:
                h = _get_json(base.rstrip("/") + "/health", timeout=10)
                entry["ok"] = True
                entry["load_state"] = h.get("load_state")
                served = h.get("requests_served")
                entry["requests_served"] = served
                entry["queue_depth"] = h.get("queue_depth")
                entry["allocated_vram_mb"] = h.get("allocated_vram_mb")
                entry["reserved_vram_mb"] = h.get("reserved_vram_mb")
                entry["ram_used_pct"] = h.get("ram_used_pct")
                entry["uptime_seconds"] = h.get("uptime_seconds")
                # queries/min for this replica from the served-counter delta
                qpm = None
                prev = self._prev.get(rid)
                if served is not None and prev is not None:
                    pt, pserved = prev
                    dt = now - pt
                    dserved = served - pserved
                    # guard against counter reset (worker restart) -> skip negative
                    if dt > 0 and dserved >= 0:
                        qpm = dserved / dt * 60.0
                if served is not None:
                    self._prev[rid] = (now, served)
                entry["qpm"] = round(qpm, 2) if qpm is not None else None
                if h.get("load_state") == "loaded":
                    ready += 1
                if qpm is not None:
                    agg_qpm += qpm
                if h.get("queue_depth"):
                    agg_queue += h.get("queue_depth") or 0
            except Exception as e:
                entry["error"] = str(e)
            replicas.append(entry)

        # drop stale replicas from the prev map so a torn-down replica's counter
        # doesn't linger and skew a future same-id replica's first delta
        for rid in list(self._prev.keys()):
            if rid not in urls:
                self._prev.pop(rid, None)

        sample = {
            "ts": round(now, 1),
            "replica_count": len(urls),
            "ready_count": ready,
            "total_qpm": round(agg_qpm, 2),
            "total_queue_depth": agg_queue,
            "replicas": replicas,
        }
        with self._lock:
            self.samples.append(sample)
            if self.logfile:
                try:
                    with open(self.logfile, "a") as f:
                        f.write(json.dumps(sample) + "\n")
                except Exception as e:
                    print(f"[monitor] log write failed: {e}")
        return sample

    def run_forever(self):
        print(f"[monitor] polling '{self.service_name}' every {self.interval}s; "
              f"log -> {self.logfile}")
        while True:
            try:
                s = self.poll_once()
                print(f"[monitor] {time.strftime('%H:%M:%S')} "
                      f"replicas={s['replica_count']} ready={s['ready_count']} "
                      f"qpm={s['total_qpm']} queue={s['total_queue_depth']}")
            except Exception as e:
                print(f"[monitor] poll error: {e}")
            time.sleep(self.interval)

    def window(self, minutes):
        cutoff = time.time() - minutes * 60
        with self._lock:
            return [s for s in self.samples if s["ts"] >= cutoff]


# ---------------------------------------------------------------------------
# HTTP server: / (HTML dashboard) and /data (JSON)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>cue-n-woo fleet load</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body{font-family:system-ui,sans-serif;margin:0;background:#0d1117;color:#c9d1d9}
  header{padding:14px 20px;background:#161b22;border-bottom:1px solid #30363d;
         display:flex;align-items:center;gap:24px;flex-wrap:wrap}
  h1{font-size:16px;margin:0;font-weight:600}
  .stat{font-size:13px}.stat b{color:#58a6ff;font-size:18px}
  .controls{margin-left:auto;font-size:13px}
  select{background:#0d1117;color:#c9d1d9;border:1px solid #30363d;border-radius:6px;padding:4px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:16px}
  .card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}
  .card h2{font-size:13px;margin:0 0 8px;color:#8b949e;font-weight:600}
  @media(max-width:900px){.grid{grid-template-columns:1fr}}
</style></head>
<body>
<header>
  <h1>cue-n-woo fleet load</h1>
  <div class="stat">replicas <b id="s-rep">-</b></div>
  <div class="stat">ready <b id="s-ready">-</b></div>
  <div class="stat">queries/min <b id="s-qpm">-</b></div>
  <div class="stat">queue <b id="s-q">-</b></div>
  <div class="controls">
    window <select id="win">
      <option value="15">15 min</option>
      <option value="60" selected>1 hr</option>
      <option value="180">3 hr</option>
      <option value="720">12 hr</option>
      <option value="1440">24 hr</option>
    </select>
    refresh <select id="ref">
      <option value="5">5s</option>
      <option value="15" selected>15s</option>
      <option value="60">60s</option>
    </select>
  </div>
</header>
<div class="grid">
  <div class="card"><h2>Queries / min (aggregate + per replica)</h2><canvas id="qpm"></canvas></div>
  <div class="card"><h2>Replica count (total vs ready)</h2><canvas id="rep"></canvas></div>
  <div class="card"><h2>Queue depth (aggregate)</h2><canvas id="queue"></canvas></div>
  <div class="card"><h2>VRAM allocated / replica (MB)</h2><canvas id="vram"></canvas></div>
</div>
<script>
const COLORS=['#58a6ff','#3fb950','#f778ba','#d29922','#a371f7','#ff7b72','#79c0ff','#56d364','#e3b341','#ffa657'];
let charts={};
function mk(id,labelfmt){const c=document.getElementById(id);
  return new Chart(c,{type:'line',data:{datasets:[]},options:{animation:false,responsive:true,
    interaction:{mode:'index',intersect:false},
    scales:{x:{type:'linear',ticks:{color:'#8b949e',callback:v=>new Date(v*1000).toLocaleTimeString()}},
            y:{beginAtZero:true,ticks:{color:'#8b949e'},grid:{color:'#21262d'}}},
    plugins:{legend:{labels:{color:'#c9d1d9',boxWidth:12}}}}});}
function ds(label,color,fill){return{label,borderColor:color,backgroundColor:color+'33',
  fill:fill||false,borderWidth:2,pointRadius:0,tension:0.25};}
async function refresh(){
  const win=document.getElementById('win').value;
  const r=await fetch('/data?minutes='+win); const data=await r.json();
  const S=data.samples||[];
  if(S.length){const last=S[S.length-1];
    document.getElementById('s-rep').textContent=last.replica_count;
    document.getElementById('s-ready').textContent=last.ready_count;
    document.getElementById('s-qpm').textContent=last.total_qpm;
    document.getElementById('s-q').textContent=last.total_queue_depth;}
  // collect replica ids seen in window
  const ids=[...new Set(S.flatMap(s=>s.replicas.map(p=>p.replica_id)))].sort((a,b)=>a-b);
  // qpm: aggregate + per replica
  let qd=[ds('total','#ffffff',true)]; qd[0].borderWidth=3;
  qd[0].data=S.map(s=>({x:s.ts,y:s.total_qpm}));
  ids.forEach((id,i)=>{const d=ds('r'+id,COLORS[i%COLORS.length]);
    d.data=S.map(s=>{const p=s.replicas.find(x=>x.replica_id===id);return{x:s.ts,y:p?p.qpm:null};});qd.push(d);});
  charts.qpm.data.datasets=qd; charts.qpm.update();
  // replica count
  charts.rep.data.datasets=[
    Object.assign(ds('total','#58a6ff',true),{data:S.map(s=>({x:s.ts,y:s.replica_count})),stepped:true}),
    Object.assign(ds('ready','#3fb950',true),{data:S.map(s=>({x:s.ts,y:s.ready_count})),stepped:true})];
  charts.rep.update();
  // queue
  charts.queue.data.datasets=[Object.assign(ds('queue','#d29922',true),
    {data:S.map(s=>({x:s.ts,y:s.total_queue_depth}))})];
  charts.queue.update();
  // vram per replica
  charts.vram.data.datasets=ids.map((id,i)=>{const d=ds('r'+id,COLORS[i%COLORS.length]);
    d.data=S.map(s=>{const p=s.replicas.find(x=>x.replica_id===id);return{x:s.ts,y:p?p.allocated_vram_mb:null};});return d;});
  charts.vram.update();
}
charts.qpm=mk('qpm'); charts.rep=mk('rep'); charts.queue=mk('queue'); charts.vram=mk('vram');
let timer=null;
function reschedule(){if(timer)clearInterval(timer);
  timer=setInterval(refresh,document.getElementById('ref').value*1000);}
document.getElementById('ref').onchange=reschedule;
document.getElementById('win').onchange=refresh;
refresh(); reschedule();
</script>
</body></html>"""


def make_handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, DASHBOARD_HTML.encode("utf-8"), "text/html")
            elif path == "/data":
                minutes = 60
                if "?" in self.path:
                    from urllib.parse import parse_qs

                    q = parse_qs(self.path.split("?", 1)[1])
                    try:
                        minutes = float(q.get("minutes", [60])[0])
                    except ValueError:
                        pass
                body = json.dumps({"samples": monitor.window(minutes)}).encode("utf-8")
                self._send(200, body, "application/json")
            elif path == "/healthz":
                self._send(200, b"ok", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--service", default="cue-n-woo-workers",
                    help="SkyServe service name to monitor")
    ap.add_argument("--port", type=int, default=8000,
                    help="port to serve the dashboard on (use one open in the SG)")
    ap.add_argument("--interval", type=int, default=15,
                    help="seconds between /health polls")
    ap.add_argument("--logfile",
                    default=os.path.expanduser("~/fleet_metrics.jsonl"),
                    help="JSONL sample log (durable history)")
    ap.add_argument("--history-minutes", type=int, default=1440,
                    help="how much history to hold in memory for the graph")
    args = ap.parse_args()

    monitor = FleetMonitor(args.service, args.interval, args.logfile,
                           args.history_minutes)
    t = threading.Thread(target=monitor.run_forever, daemon=True)
    t.start()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(monitor))
    host = socket.gethostname()
    print(f"[server] dashboard on http://0.0.0.0:{args.port}/  (host {host})")
    print(f"[server] JSON series at /data?minutes=N")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] shutting down")


if __name__ == "__main__":
    main()
