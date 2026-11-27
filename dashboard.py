#!/usr/bin/env python3
"""Browser dashboard for the qwen-vllm server: load, throughput, GPU and per-client traffic.

Runs on the server next to vLLM (standard library only) and serves http://<server>:8080/.
Every 2 s it samples vLLM's /metrics and nvidia-smi; every 10 s it reads the service journal
and the ufw log for the client table. The last 30 minutes are kept in memory.

  python3 dashboard.py [--port 8080] [--vllm http://localhost:8000]
"""
import argparse
import json
import re
import subprocess
import threading
import time
import urllib.request
from collections import defaultdict, deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GPU_QUERY = ("power.draw,power.limit,temperature.gpu,fan.speed,utilization.gpu,"
             "memory.used,memory.total,clocks.sm")
HTML = Path(__file__).with_name("dashboard.html")

METRIC_RE = re.compile(r'^(vllm:\w+)(?:\{[^}]*\})?\s+(\S+)$')
ACCESS_RE = re.compile(r'^(\S+) .*INFO:\s+(\d+\.\d+\.\d+\.\d+):\d+ - "(\w+) (\S+) [^"]*" (\d{3})')
USAGE_RE = re.compile(r'^(\S+) .*USAGE ip=(\S+) path=\S+ status=\d+ prompt=(\d+) completion=(\d+)')
UFW_RE = re.compile(r'^(\S+) .*\[UFW BLOCK\].* SRC=(\S+) ')

LOGS_SCRIPT = """\
journalctl -u qwen-vllm --since '-{hours}h' -o short-iso --no-pager -q | grep -F -e ' - "' -e 'USAGE ip='
echo @@UFW
grep -h ' DPT=8000 ' /var/log/ufw.log 2>/dev/null | tail -n 5000
"""


def fetch_metrics(base):
    """Sum each vllm:* sample in /metrics over its labels."""
    with urllib.request.urlopen(base + "/metrics", timeout=5) as r:
        text = r.read().decode()
    m = defaultdict(float)
    for line in text.splitlines():
        hit = METRIC_RE.match(line)
        if hit:
            try:
                m[hit.group(1)] += float(hit.group(2))
            except ValueError:
                pass
    return m


def fetch_logs(hours):
    """Access-log lines from the service journal and port-8000 drops from the ufw log."""
    out = subprocess.run(["sh", "-c", LOGS_SCRIPT.format(hours=hours)],
                         capture_output=True, text=True, timeout=20)
    if "@@UFW" not in out.stdout:
        return {"error": (out.stderr.strip() or "no output").splitlines()[-1]}
    access, ufw = out.stdout.split("@@UFW", 1)
    return {"access": access.splitlines(), "ufw": ufw.splitlines()}


def parse_time(s):
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def summarize_clients(logs, hours):
    clients = defaultdict(lambda: {"ok": 0, "err": 0, "last": None, "blocked": 0, "prompt": 0, "gen": 0})
    recent = []
    for line in logs["access"]:
        hit = USAGE_RE.match(line)
        if hit:
            c = clients[hit.group(2)]
            c["prompt"] += int(hit.group(3))
            c["gen"] += int(hit.group(4))
            continue
        hit = ACCESS_RE.match(line)
        if not hit:
            continue
        ts, ip, method, path, status = hit.groups()
        if not path.startswith("/v1/"):          # browser visits, /metrics and /health polls
            continue
        c = clients[ip]
        c["ok" if status.startswith("2") else "err"] += 1
        c["last"] = parse_time(ts)
        recent.append((c["last"], ip, method, path, status))
    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    for line in logs["ufw"]:
        hit = UFW_RE.match(line)
        if not hit:
            continue
        t = parse_time(hit.group(1))
        if t and t.timestamp() >= cutoff:
            c = clients[hit.group(2)]
            c["blocked"] += 1
            if c["last"] is None or t > c["last"]:
                c["last"] = t
    return clients, recent


class Collector:
    def __init__(self, vllm, interval, history_s, hours):
        self.vllm, self.interval, self.hours = vllm, interval, hours
        self.history = deque(maxlen=int(history_s / interval))
        self.lock = threading.Lock()
        self.state = {"ok": False, "error": "starting"}
        self.clients, self.recent, self.logs_error = [], [], None

    def gpu(self):
        try:
            out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout
            v = [float(x) for x in out.strip().splitlines()[0].split(",")]
            keys = ["power", "power_limit", "temp", "fan", "util", "mem_used", "mem_total", "sm_clock"]
            return dict(zip(keys, v))
        except Exception:
            return None

    def logs(self):
        logs = fetch_logs(self.hours)
        if "error" in logs:
            self.logs_error = logs["error"]
            return
        self.logs_error = None
        clients, recent = summarize_clients(logs, self.hours)
        iso = lambda t: t.astimezone(timezone.utc).isoformat() if t else None
        self.clients = sorted(
            ({"ip": ip, "ok": c["ok"], "err": c["err"], "blocked": c["blocked"], "last": iso(c["last"]),
              "prompt": c["prompt"], "gen": c["gen"]}
             for ip, c in clients.items()),
            key=lambda c: c["last"] or "", reverse=True)
        self.recent = [{"t": iso(t), "ip": ip, "method": m, "path": p, "status": int(s)}
                       for t, ip, m, p, s in recent[-15:]][::-1]

    def run(self):
        prev, prev_t, logs_t = None, None, 0.0
        while True:
            t0 = time.monotonic()
            if t0 - logs_t >= 10:
                try:
                    self.logs()
                except Exception as e:
                    self.logs_error = str(e)
                logs_t = t0
            gpu = self.gpu()
            try:
                m = fetch_metrics(self.vllm)
                err = None
            except Exception as e:
                m, err = None, f"vLLM unreachable: {e}"
            now = time.monotonic()
            point = {"t": time.time(), "gpu": gpu}
            if m is not None:
                point.update(running=m["vllm:num_requests_running"], waiting=m["vllm:num_requests_waiting"],
                             kv=m["vllm:kv_cache_usage_perc"] * 100)
                if prev is not None and now > prev_t:
                    dt = now - prev_t
                    rate = lambda k: max(m[k] - prev[k], 0) / dt
                    point.update(gen_tps=rate("vllm:generation_tokens_total"),
                                 prompt_tps=rate("vllm:prompt_tokens_total"))
                prev, prev_t = m, now
            totals = None
            if m is not None:
                cnt = lambda k: m[k + "_count"]
                avg = lambda k: m[k + "_sum"] / cnt(k) if cnt(k) else None
                drafts = m["vllm:spec_decode_num_drafts_total"]
                q = m["vllm:prefix_cache_queries_total"]
                totals = {
                    "requests": m["vllm:request_success_total"],
                    "prompt_tokens": m["vllm:prompt_tokens_total"],
                    "gen_tokens": m["vllm:generation_tokens_total"],
                    "prefix_hit": m["vllm:prefix_cache_hits_total"] / q * 100 if q else None,
                    "ttft": avg("vllm:time_to_first_token_seconds"),
                    "e2e": avg("vllm:e2e_request_latency_seconds"),
                    "mtp": 1 + m["vllm:spec_decode_num_accepted_tokens_total"] / drafts if drafts else None,
                    "preemptions": m["vllm:num_preemptions_total"],
                }
            with self.lock:
                self.history.append(point)
                self.state = {"ok": m is not None, "error": err, "now": point, "totals": totals,
                              "clients": self.clients, "recent": self.recent, "logs_error": self.logs_error,
                              "hours": self.hours, "interval": self.interval,
                              "server_time": datetime.now(timezone.utc).isoformat()}
            time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))

    def snapshot(self, with_history):
        with self.lock:
            s = dict(self.state)
            if with_history:
                s["history"] = list(self.history)
            return s


def make_handler(collector):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self.send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                full = "history" in self.path
                self.send(200, json.dumps(collector.snapshot(full)).encode(), "application/json")
            else:
                self.send(404, b"not found", "text/plain")

        def log_message(self, *a):
            pass

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--vllm", default="http://localhost:8000")
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--history-minutes", type=float, default=30)
    p.add_argument("--hours", type=float, default=24, help="client table window (default 24 h)")
    a = p.parse_args()
    c = Collector(a.vllm, a.interval, a.history_minutes * 60, a.hours)
    threading.Thread(target=c.run, daemon=True).start()
    print(f"dashboard on http://{a.bind}:{a.port}/", flush=True)
    ThreadingHTTPServer((a.bind, a.port), make_handler(c)).serve_forever()


if __name__ == "__main__":
    main()
