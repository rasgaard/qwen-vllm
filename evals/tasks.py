"""Task accuracy (GSM8K, MMLU-Pro) against the running server.

    uv run python evals/tasks.py gsm8k --n 200 --out evals/results/gsm8k_mtp.jsonl
    uv run python evals/tasks.py mmlupro --n 280 --out evals/results/mmlupro_mtp.jsonl
    uv run python evals/tasks.py compare a.jsonl b.jsonl   # paired comparison of two runs

Items are a fixed seeded subset, so runs against different server configs are
paired item-for-item. Sampling defaults come from the model's generation_config.
"""

import argparse
import asyncio
import json
import math
import random
import re
from pathlib import Path

import pyarrow.parquet as pq
from openai import AsyncOpenAI

DATA = Path(__file__).parent / "data"


def default_key():
    try:
        return (Path(__file__).parent.parent / ".api-key").read_text().strip()
    except OSError:
        return "x"


def load_gsm8k(n, seed):
    rows = pq.read_table(DATA / "gsm8k/main/test-00000-of-00001.parquet").to_pylist()
    rows = random.Random(seed).sample(rows, n)
    items = []
    for i, r in enumerate(rows):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        prompt = r["question"] + "\n\nSolve the problem. End with a final line of the form 'ANSWER: <number>'."
        items.append({"id": f"gsm8k-{i}", "prompt": prompt, "gold": gold})
    return items


def load_mmlupro(n, seed):
    rows = pq.read_table(DATA / "MMLU-Pro/data/test-00000-of-00001.parquet").to_pylist()
    by_cat = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    rnd = random.Random(seed)
    per = max(1, n // len(by_cat))
    picked = [r for cat in sorted(by_cat) for r in rnd.sample(by_cat[cat], per)]
    items = []
    for r in picked:
        letters = "ABCDEFGHIJ"[: len(r["options"])]
        opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, r["options"]))
        prompt = (f"{r['question']}\n\n{opts}\n\nAnswer the multiple-choice question. "
                  "End with a final line of the form 'ANSWER: <letter>'.")
        items.append({"id": f"mmlupro-{r['question_id']}", "prompt": prompt, "gold": r["answer"], "category": r["category"]})
    return items


def extract(task, text):
    m = re.findall(r"ANSWER:\s*\**\s*([^\s*]+)", text or "")
    if not m:
        return None
    a = m[-1].strip().rstrip(".")
    if task == "gsm8k":
        a = a.replace(",", "").replace("$", "")
        try:
            return str(int(float(a))) if float(a) == int(float(a)) else str(float(a))
        except ValueError:
            return a
    return a[:1].upper()


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


async def run(args):
    items = (load_gsm8k if args.task == "gsm8k" else load_mmlupro)(args.n, args.seed)
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=3600)
    sem = asyncio.Semaphore(args.concurrency)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    kwargs = {"thinking": args.thinking, "effort": args.effort}

    async def one(it):
        async with sem:
            r = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": it["prompt"]}],
                max_tokens=args.max_tokens,
                seed=args.seed,
                **({"temperature": 0} if args.greedy else {}),
                extra_body={"chat_template_kwargs": {"enable_thinking": args.thinking, "reasoning_effort": args.effort}},
            )
            text = r.choices[0].message.content
            pred = extract(args.task, text)
            rec = {**it, "pred": pred, "correct": pred == it["gold"], "finish": r.choices[0].finish_reason,
                   "completion_tokens": r.usage.completion_tokens, "content": text, "config": kwargs}
            with out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            return rec

    todo = [it for it in items if it["id"] not in done]
    print(f"{args.task}: {len(items)} items, {len(todo)} to run")
    await asyncio.gather(*(one(it) for it in todo))
    summarize(out)


def summarize(path):
    recs = [json.loads(l) for l in Path(path).read_text().splitlines()]
    k, n = sum(r["correct"] for r in recs), len(recs)
    lo, hi = wilson(k, n)
    trunc = sum(r["finish"] == "length" for r in recs)
    toks = sum(r["completion_tokens"] for r in recs) / max(n, 1)
    print(f"{path}: {k}/{n} = {k / n:.1%}  (95% CI {lo:.1%}-{hi:.1%})  truncated {trunc}  mean tokens {toks:.0f}")


def compare(a, b):
    ra = {json.loads(l)["id"]: json.loads(l) for l in Path(a).read_text().splitlines()}
    rb = {json.loads(l)["id"]: json.loads(l) for l in Path(b).read_text().splitlines()}
    ids = sorted(ra.keys() & rb.keys())
    only_a = sum(ra[i]["correct"] and not rb[i]["correct"] for i in ids)
    only_b = sum(rb[i]["correct"] and not ra[i]["correct"] for i in ids)
    same_text = sum(ra[i]["content"] == rb[i]["content"] for i in ids)
    # exact McNemar (two-sided binomial on discordant pairs)
    d = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(d, x) for x in range(0, min(only_a, only_b) + 1)) / 2**d) if d else 1.0
    print(f"paired items {len(ids)}: A {sum(ra[i]['correct'] for i in ids)}  B {sum(rb[i]['correct'] for i in ids)}  "
          f"A-only {only_a}  B-only {only_b}  McNemar p={p:.2f}  identical outputs {same_text}/{len(ids)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["gsm8k", "mmlupro", "compare", "summary"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=default_key())
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "xhigh"])
    ap.add_argument("--greedy", action="store_true")
    args = ap.parse_args()
    if args.task == "compare":
        compare(*args.files)
    elif args.task == "summary":
        for f in args.files:
            summarize(f)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
