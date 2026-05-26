import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FLAS 9B Steering Chat</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #fff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9dee7;
      --accent: #1f766b;
      --accent-dark: #155b52;
      --error: #b42318;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    main { min-height: 100vh; display: grid; grid-template-columns: 340px minmax(0, 1fr); }
    aside { padding: 18px; border-right: 1px solid var(--line); background: #fbfcfd; }
    section { padding: 22px; }
    h1 { margin: 0 0 16px; font-size: 20px; }
    h2 { margin: 20px 0 10px; color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }
    label { display: block; margin: 12px 0 6px; font-size: 13px; font-weight: 650; }
    input, textarea, button {
      width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff;
      color: var(--ink); font: inherit; padding: 9px 10px;
    }
    textarea { min-height: 112px; resize: vertical; line-height: 1.45; }
    button { cursor: pointer; background: var(--accent); color: #fff; border-color: var(--accent); font-weight: 700; }
    button:hover { background: var(--accent-dark); }
    button.secondary { background: #fff; color: var(--ink); border-color: var(--line); }
    button:disabled { cursor: not-allowed; opacity: .55; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .chat { max-width: 1040px; margin: 0 auto; display: grid; gap: 14px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .status { min-height: 22px; margin-top: 10px; font-size: 13px; color: var(--muted); white-space: pre-wrap; }
    .error { color: var(--error); }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; font: 14px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .small { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .tokens { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 8px; }
    .token { border: 1px solid var(--line); border-radius: 6px; padding: 8px; font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: #fff; }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
<main>
  <aside>
    <h1>FLAS 9B</h1>
    <button id="loadBtn" class="secondary">Load model</button>
    <div id="loadStatus" class="status">Loads automatically on first generation.</div>

    <h2>Steering</h2>
    <label for="concept">Concept</label>
    <textarea id="concept">Talk like a pirate</textarea>
    <div class="row">
      <div><label for="flowtime">Flowtime</label><input id="flowtime" type="number" step="0.1" value="2.0"></div>
      <div><label for="steps">Steps</label><input id="steps" type="number" min="1" max="8" value="3"></div>
    </div>
    <div class="small">Flowtime is steering strength. Start around 1.0-2.0.</div>

    <h2>Generation</h2>
    <div class="row">
      <div><label for="maxTokens">New tokens</label><input id="maxTokens" type="number" min="1" max="512" value="128"></div>
      <div><label for="temperature">Temperature</label><input id="temperature" type="number" min="0" step="0.05" value="0.7"></div>
    </div>
  </aside>

  <section>
    <div class="chat">
      <div class="panel">
        <label for="prompt">Prompt</label>
        <textarea id="prompt">Write a short paragraph about the ocean.</textarea>
        <div class="row" style="margin-top:10px">
          <button id="generateBtn">Generate</button>
          <button id="compareBtn" class="secondary">Compare</button>
        </div>
        <div id="status" class="status"></div>
      </div>
      <div class="panel">
        <h2>Steered</h2>
        <pre id="steered"></pre>
        <h2>First Token Probabilities</h2>
        <div id="steeredTokens" class="tokens"></div>
      </div>
      <div class="panel">
        <h2>Baseline</h2>
        <pre id="baseline"></pre>
        <h2>First Token Probabilities</h2>
        <div id="baselineTokens" class="tokens"></div>
      </div>
    </div>
  </section>
</main>
<script>
const $ = (id) => document.getElementById(id);

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {})
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function request(compare=false) {
  return {
    prompt: $("prompt").value,
    concept: $("concept").value,
    flowtime: Number($("flowtime").value),
    n_steps: Number($("steps").value),
    max_tokens: Number($("maxTokens").value),
    temperature: Number($("temperature").value),
    compare
  };
}

function setStatus(text, isError=false) {
  $("status").textContent = text;
  $("status").className = isError ? "status error" : "status";
}

function renderTokens(id, tokens) {
  const root = $(id);
  root.innerHTML = "";
  for (const [token, prob] of tokens || []) {
    const div = document.createElement("div");
    div.className = "token";
    div.textContent = `${JSON.stringify(token)}  ${prob.toExponential(3)}`;
    root.appendChild(div);
  }
}

$("loadBtn").addEventListener("click", async () => {
  $("loadBtn").disabled = true;
  $("loadStatus").textContent = "Loading Gemma 2 9B IT + FLAS...";
  try {
    const data = await api("/api/load", {});
    $("loadStatus").textContent = data.message;
  } catch (err) {
    $("loadStatus").textContent = err.message;
    $("loadStatus").className = "status error";
  } finally {
    $("loadBtn").disabled = false;
  }
});

async function run(compare) {
  $("generateBtn").disabled = true;
  $("compareBtn").disabled = true;
  setStatus(compare ? "Generating steered and baseline outputs..." : "Generating...");
  try {
    const data = await api("/api/generate", request(compare));
    $("steered").textContent = data.steered;
    $("baseline").textContent = data.baseline || "";
    renderTokens("steeredTokens", data.steered_first_tokens);
    renderTokens("baselineTokens", data.baseline_first_tokens);
    setStatus(data.message);
    $("loadStatus").textContent = "Model loaded.";
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    $("generateBtn").disabled = false;
    $("compareBtn").disabled = false;
  }
}

$("generateBtn").addEventListener("click", () => run(false));
$("compareBtn").addEventListener("click", () => run(true));
</script>
</body>
</html>
"""


class FlasBackend:
    def __init__(self, args):
        self.args = args
        self.gen = None
        self.lock = threading.Lock()

    def load(self):
        with self.lock:
            if self.gen is not None:
                return "Model already loaded."
            from flas.generate import load_generator

            self.gen = load_generator(
                self.args.flow_ckpt,
                model_id=self.args.model_id,
                layer=self.args.layer,
                num_blocks=self.args.num_blocks,
            )
            return f"Loaded {self.args.model_id} with FLAS checkpoint {self.args.flow_ckpt}."

    def generate(self, payload):
        if not payload.get("prompt", "").strip():
            raise ValueError("Prompt is required.")
        if not payload.get("concept", "").strip():
            raise ValueError("Concept is required.")
        if self.gen is None:
            self.load()

        flowtime = float(payload.get("flowtime", 2.0))
        n_steps = int(payload.get("n_steps", self.args.n_steps))
        max_tokens = int(payload.get("max_tokens", 128))
        temperature = float(payload.get("temperature", 0.7))
        compare = bool(payload.get("compare", False))

        with self.lock:
            steered_first_tokens = self.first_token_probs(
                prompt=payload["prompt"],
                concept_text=payload["concept"],
                flowtime=flowtime,
                n_steps=n_steps,
            )
            steered = self.gen.generate_batch(
                prompts=[payload["prompt"]],
                concept_text=payload["concept"],
                flowtimes=[flowtime],
                n_steps=n_steps,
                max_tokens=max_tokens,
                temperature=temperature,
                max_batch=1,
            )[0]["generation"]

            baseline = ""
            baseline_first_tokens = []
            if compare:
                baseline_first_tokens = self.first_token_probs(
                    prompt=payload["prompt"],
                    concept_text=payload["concept"],
                    flowtime=0.0,
                    n_steps=n_steps,
                )
                baseline = self.gen.generate_batch(
                    prompts=[payload["prompt"]],
                    concept_text=payload["concept"],
                    flowtimes=[0.0],
                    n_steps=n_steps,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_batch=1,
                )[0]["generation"]

        return {
            "steered": steered,
            "baseline": baseline,
            "steered_first_tokens": steered_first_tokens,
            "baseline_first_tokens": baseline_first_tokens,
            "message": f"flowtime={flowtime}, steps={n_steps}, tokens={max_tokens}",
        }

    def first_token_probs(self, prompt, concept_text, flowtime, n_steps, top_k=10):
        import torch

        gen = self.gen
        gen._n_steps = n_steps
        concept_hidden, concept_mask = gen.encode_concept(concept_text)
        formatted = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = gen.tokenizer(
            [formatted],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        ).to("cuda")

        attention_mask = enc.attention_mask
        gen._concept_hidden = concept_hidden
        gen._concept_mask = concept_mask
        gen._flowtimes = torch.tensor([flowtime], device="cuda", dtype=torch.float32)
        gen._padding_mask = attention_mask.float()
        gen._sa_caches = [None] * n_steps
        gen._is_prefill = True
        gen._past_len = 0
        gen._position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)

        gen._install_hook()
        gen._active = True
        try:
            with torch.no_grad():
                out = gen.llm(
                    enc.input_ids,
                    attention_mask=attention_mask,
                    position_ids=gen._position_ids,
                    use_cache=True,
                )
                logits = out.logits[0, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)
                values, indices = torch.topk(probs, k=top_k)
                return [
                    (gen.tokenizer.decode([idx.item()]), float(value.item()))
                    for value, idx in zip(values, indices)
                ]
        finally:
            gen._active = False
            gen._sa_caches = None
            torch.cuda.empty_cache()


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}")

        def do_GET(self):
            if self.path == "/":
                data = INDEX_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(body)
                if self.path == "/api/load":
                    self.write_json({"message": backend.load()})
                    return
                if self.path == "/api/generate":
                    self.write_json(backend.generate(payload))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

        def write_json(self, payload, status=HTTPStatus.OK):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Run a browser UI for FLAS Gemma 2 9B IT steering.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--flow-ckpt", default="checkpoints/flas-gemma-2-9b-it/flas-gemma-2-9b-it.safetensors")
    parser.add_argument("--model-id", default="unsloth/gemma-2-9b-it")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    backend = FlasBackend(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(backend))
    print(f"Serving FLAS UI at http://{args.host}:{args.port}")
    print("Use Ctrl-C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
