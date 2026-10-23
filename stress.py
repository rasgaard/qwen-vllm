"""Worst-case memory stress: long-prompt prefills running alongside decodes.

Watches nvidia-smi while it runs and reports peak VRAM, so a pinned
--kv-cache-memory can be checked for OOM headroom before real users hit it.

    uv run python stress.py --prompt-tokens 30000 --long 2 --short 4
"""

import argparse
import asyncio
import random
import subprocess
import time

from openai import AsyncOpenAI


def default_key():
    try:
        return open(".api-key").read().strip()
    except OSError:
        return "x"


WORDS = "alpha beta gamma delta kernel tensor cache prefix decode token scheduler block".split()


def long_prompt(n_tokens, seed):
    rnd = random.Random(seed)
    body = " ".join(rnd.choice(WORDS) for _ in range(n_tokens))
    return f"Here is a log:\n{body}\n\nSummarize the log in three sentences."


async def request(client, model, prompt, max_tokens):
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return r.usage.prompt_tokens, r.usage.completion_tokens, time.perf_counter() - t0


async def watch_vram(stop, peak):
    while not stop.is_set():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.split(",")
        peak[0] = max(peak[0], int(out[0]))
        peak[1] = int(out[1])
        await asyncio.sleep(0.2)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=default_key())
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--prompt-tokens", type=int, default=30000)
    ap.add_argument("--long", type=int, default=2, help="concurrent long-prompt requests")
    ap.add_argument("--short", type=int, default=4, help="concurrent short decode-heavy requests")
    args = ap.parse_args()
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800)

    stop, peak = asyncio.Event(), [0, 0]
    watcher = asyncio.create_task(watch_vram(stop, peak))
    jobs = [request(client, args.model, long_prompt(args.prompt_tokens, i), 256) for i in range(args.long)]
    jobs += [request(client, args.model, f"Write a detailed essay about topic #{i}: caching.", 1024) for i in range(args.short)]
    t0 = time.perf_counter()
    results = await asyncio.gather(*jobs, return_exceptions=True)
    stop.set()
    await watcher

    for r in results:
        print(f"FAILED: {r!r}"[:300] if isinstance(r, Exception) else f"ok  prompt {r[0]:6d}  completion {r[1]:5d}  {r[2]:6.1f}s")
    ok = sum(not isinstance(r, Exception) for r in results)
    print(f"\n{ok}/{len(results)} succeeded in {time.perf_counter() - t0:.1f}s | peak VRAM {peak[0]} / {peak[1]} MiB "
          f"(headroom {peak[1] - peak[0]} MiB)")


asyncio.run(main())
