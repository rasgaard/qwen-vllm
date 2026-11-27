"""ASGI middleware for vLLM that logs token usage per client IP.

Loaded with `vllm serve --middleware usage_log.UsageLogMiddleware`. For every completion request
it prints one line to stdout (the service journal), which dashboard.py reads:

  USAGE ip=10.0.0.5 path=/v1/chat/completions status=200 prompt=1234 completion=56 ms=789

Token counts come from the `usage` field vLLM puts in each response. Streaming responses only
carry it when the client sets stream_options.include_usage, so the middleware sets it and, if the
client didn't ask for it, removes the extra usage chunk again: clients see the same stream as
before. Any failure here only loses the log line, never the response.
"""
import json
import time

PATHS = ("/v1/chat/completions", "/v1/completions")
MAX_BODY = 32 << 20     # don't buffer non-streaming responses beyond this


class UsageLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in PATHS:
            return await self.app(scope, receive, send)

        # Read the request body so stream_options can be set.
        chunks, more = [], True
        while more:
            msg = await receive()
            if msg["type"] != "http.request":
                return await self.app(scope, _replay([msg], receive), send)
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        body = b"".join(chunks)

        injected = False
        try:
            req = json.loads(body)
            if isinstance(req, dict) and req.get("stream"):
                opts = req.get("stream_options") or {}
                if not opts.get("include_usage"):
                    req["stream_options"] = {**opts, "include_usage": True}
                    body = json.dumps(req).encode()
                    injected = True
        except Exception:
            pass
        if injected:
            headers = [(k, v) for k, v in scope["headers"] if k.lower() != b"content-length"]
            scope = {**scope, "headers": headers + [(b"content-length", str(len(body)).encode())]}

        state = {"status": None, "usage": None, "buf": [], "size": 0, "stream": False}
        start = time.monotonic()

        async def send_wrapper(msg):
            try:
                if msg["type"] == "http.response.start":
                    state["status"] = msg["status"]
                    ctype = dict(msg.get("headers") or []).get(b"content-type", b"")
                    state["stream"] = b"text/event-stream" in ctype
                elif msg["type"] == "http.response.body":
                    data = msg.get("body", b"")
                    if state["stream"]:
                        if b'"usage"' in data:
                            data = _scan_sse(data, state, injected)
                            msg = {**msg, "body": data}
                    elif state["size"] < MAX_BODY:
                        state["buf"].append(data)
                        state["size"] += len(data)
                    if not msg.get("more_body", False):
                        _finish(scope, state, start)
            except Exception:
                pass
            await send(msg)

        await self.app(scope, _replay([{"type": "http.request", "body": body, "more_body": False}], receive),
                       send_wrapper)


def _replay(first, receive):
    pending = list(first)

    async def wrapped():
        if pending:
            return pending.pop(0)
        return await receive()
    return wrapped


def _scan_sse(data, state, injected):
    """Record usage from SSE events; drop the usage-only event if we asked for it ourselves."""
    out = []
    for event in data.split(b"\n\n"):
        line = event.strip()
        if line.startswith(b"data:") and b'"usage"' in line:
            try:
                obj = json.loads(line[5:])
                if obj.get("usage"):
                    state["usage"] = obj["usage"]
                    if injected and not obj.get("choices"):
                        continue
            except Exception:
                pass
        out.append(event)
    return b"\n\n".join(out)


def _finish(scope, state, start):
    usage = state["usage"]
    if usage is None and state["buf"]:
        try:
            usage = json.loads(b"".join(state["buf"])).get("usage")
        except Exception:
            usage = None
    usage = usage or {}
    ip = (scope.get("client") or ("?",))[0]
    print(f"USAGE ip={ip} path={scope['path']} status={state['status']} "
          f"prompt={usage.get('prompt_tokens', 0)} completion={usage.get('completion_tokens', 0)} "
          f"ms={(time.monotonic() - start) * 1000:.0f}", flush=True)
