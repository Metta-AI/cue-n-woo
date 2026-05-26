#!/usr/bin/env python3
import argparse
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from app import INDEX_HTML


class ProxyHandler(BaseHTTPRequestHandler):
    backend = "http://127.0.0.1:7860"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/alice", "/bob"}:
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.proxy()

    def do_POST(self):
        self.proxy()

    def proxy(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        req = urllib.request.Request(
            self.backend + self.path,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() not in {"connection", "transfer-encoding"}:
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(data)
        except urllib.error.HTTPError as err:
            data = err.read()
            self.send_response(err.code)
            for key, value in err.headers.items():
                if key.lower() not in {"connection", "transfer-encoding"}:
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--backend", default="http://127.0.0.1:7860")
    args = parser.parse_args()
    ProxyHandler.backend = args.backend.rstrip("/")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
    print(f"Serving patched UI at http://127.0.0.1:{args.port}")
    print(f"Proxying API requests to {ProxyHandler.backend}")
    server.serve_forever()


if __name__ == "__main__":
    main()
