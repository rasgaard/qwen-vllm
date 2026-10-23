"""Single-stream (and optional concurrent) decode benchmark against the vLLM OpenAI server.

    uv run python bench.py                 # 1 stream, 8 prompts
    uv run python bench.py --concurrency 8 # aggregate throughput
"""

import argparse
import asyncio
import re
import statistics
import time

import httpx
from openai import AsyncOpenAI


def default_key():
    try:
        return open(".api-key").read().strip()
    except OSError:
        return "x"


PROMPTS = {
    "chat": [
        "Explain why the sky is blue to a curious ten-year-old.",
        "Write a short, warm email declining a meeting invitation.",
    ],
    "code": [
        "Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
        "Implement an LRU cache class in TypeScript with get/put and O(1) operations.",
    ],
    "math": [
        "Solve step by step: how many positive integers below 1000 are divisible by 3 or 7 but not 5?",
        "Derive the closed form of the sum of the first n squares and prove it by induction.",
    ],
    "json": [
        "Return a JSON array of 8 fictional employees with fields id, name, department, salary, start_date.",
        "Convert this to JSON with keys city, country, population: Paris France 2.1M; Tokyo Japan 14M; Lima Peru 10M; Oslo Norway 0.7M.",
    ],
}


async def one(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    first = last = None
    n = 0
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": False},
    )
    async for chunk in stream:
        if chunk.choices and (chunk.choices[0].delta.content or getattr(chunk.choices[0].delta, "reasoning_content", None)):
            now = time.perf_counter()
            first = first or now
            last = now
        if chunk.usage:
            n = chunk.usage.completion_tokens
    return {"ttft": first - t0, "tokens": n, "decode_tps": (n - 1) / (last - first) if last > first else 0.0, "wall": last - t0}


async def spec_counters(base):
    try:
        text = (await httpx.AsyncClient().get(base.replace("/v1", "/metrics"))).text
    except Exception:
        return None
    vals = {}
    for name in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total"):
        m = re.search(rf"^{re.escape(name)}\{{[^}}]*\}} ([0-9.e+]+)", text, re.M)
        if m:
            vals[name] = float(m.group(1))
    return vals


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=default_key())
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=1)
    args = ap.parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=600)

    await one(client, args.model, "Say hi.", 16)  # warm-up
    before = await spec_counters(args.base_url)

    if args.concurrency == 1:
        rows = []
        for domain, prompts in PROMPTS.items():
            for p in prompts:
                r = await one(client, args.model, p, args.max_tokens)
                rows.append((domain, r))
                print(f"{domain:5s} {r['tokens']:4d} tok  ttft {r['ttft']*1000:6.0f} ms  decode {r['decode_tps']:6.1f} tok/s")
        tot_tok = sum(r["tokens"] for _, r in rows)
        tot_time = sum(r["wall"] - r["ttft"] for _, r in rows)
        print(f"\nmean decode {statistics.mean(r['decode_tps'] for _, r in rows):.1f} tok/s | token-weighted {tot_tok / tot_time:.1f} tok/s")
    else:
        prompts = [p for ps in PROMPTS.values() for p in ps]
        jobs = [prompts[i % len(prompts)] + f" (variant {i})" for i in range(args.concurrency * 2)]
        sem = asyncio.Semaphore(args.concurrency)

        async def run(p):
            async with sem:
                return await one(client, args.model, p, args.max_tokens)

        t0 = time.perf_counter()
        res = await asyncio.gather(*(run(p) for p in jobs))
        dt = time.perf_counter() - t0
        print(f"concurrency {args.concurrency}: aggregate {sum(r['tokens'] for r in res) / dt:.1f} tok/s, "
              f"per-stream decode {statistics.mean(r['decode_tps'] for r in res):.1f} tok/s")

    after = await spec_counters(args.base_url)
    if before and after and after.get("vllm:spec_decode_num_drafts_total"):
        drafts = after["vllm:spec_decode_num_drafts_total"] - before.get("vllm:spec_decode_num_drafts_total", 0)
        acc = after["vllm:spec_decode_num_accepted_tokens_total"] - before.get("vllm:spec_decode_num_accepted_tokens_total", 0)
        if drafts:
            print(f"spec decode: mean acceptance length {1 + acc / drafts:.2f} tokens/step")


asyncio.run(main())
