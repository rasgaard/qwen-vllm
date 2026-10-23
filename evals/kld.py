"""Quantization damage vs. the BF16 model: perplexity, top-1 agreement, KL divergence.

Scores identical token sequences (WikiText-2 test) with vLLM prompt_logprobs and
saves per-position top-K logprobs, so any two configs can be compared offline.
Run with the server stopped (it needs the whole GPU).

    uv run python evals/kld.py score --model models/Qwen3.8-27B-BF16 --cpu-offload-gb 40 --out evals/results/bf16.npz
    uv run python evals/kld.py score --model models/Qwen3.8-27B-NVFP4-MTP --quantization modelopt --kv-cache-dtype fp8 --out evals/results/nvfp4_fp8kv.npz
    uv run python evals/kld.py compare evals/results/bf16.npz evals/results/nvfp4_fp8kv.npz
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
TOKENIZER = Path(__file__).parent.parent / "models/Qwen3.8-27B-NVFP4-MTP"


def sequences(n_seqs, seq_len):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    text = "".join(pq.read_table(DATA).column("text").to_pylist())
    ids = AutoTokenizer.from_pretrained(TOKENIZER).encode(text)
    assert len(ids) >= n_seqs * seq_len, f"only {len(ids)} tokens available"
    return [ids[i * seq_len:(i + 1) * seq_len] for i in range(n_seqs)]


def score(args):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    seqs = sequences(args.n_seqs, args.seq_len)
    llm = LLM(model=args.model, quantization=args.quantization, kv_cache_dtype=args.kv_cache_dtype,
              language_model_only=True, max_model_len=args.seq_len + 16, max_num_seqs=1,
              max_num_batched_tokens=args.chunk, enable_prefix_caching=False,
              cpu_offload_gb=args.cpu_offload_gb, gpu_memory_utilization=args.gpu_util,
              max_logprobs=args.topk, enforce_eager=args.cpu_offload_gb > 0)
    sp = SamplingParams(max_tokens=1, prompt_logprobs=args.topk, temperature=0)
    tok_lp, top_ids, top_lps = [], [], []
    for s in seqs:
        out = llm.generate([TokensPrompt(prompt_token_ids=s)], sp, use_tqdm=False)[0]
        for pos, d in enumerate(out.prompt_logprobs):
            if d is None:
                continue
            tok_lp.append(d[s[pos]].logprob)
            ranked = sorted(((lp.rank, tid, lp.logprob) for tid, lp in d.items() if lp.rank <= args.topk))[: args.topk]
            top_ids.append([t for _, t, _ in ranked])
            top_lps.append([l for _, _, l in ranked])
        print(f"scored {len(tok_lp)} tokens", flush=True)
    np.savez_compressed(args.out, tok_lp=np.array(tok_lp, np.float32), top_ids=np.array(top_ids, np.int32),
                        top_lps=np.array(top_lps, np.float32))
    print("saved", args.out)


def compare(ref_path, test_path):
    ref, tst = np.load(ref_path), np.load(test_path)
    n = len(ref["tok_lp"])
    assert n == len(tst["tok_lp"]), "different token counts"
    ppl_r, ppl_t = math.exp(-ref["tok_lp"].mean()), math.exp(-tst["tok_lp"].mean())
    top1 = (ref["top_ids"][:, 0] == tst["top_ids"][:, 0]).mean()
    # KL(ref || test) over ref's top-K. A ref token missing from test's top-K gets
    # test's K-th logprob (an upper bound), so this slightly under-counts KL.
    p_lp, t_lp = ref["top_lps"], tst["top_lps"]
    match = ref["top_ids"][:, :, None] == tst["top_ids"][:, None, :]  # [n, K_ref, K_test]
    q_lp = np.where(match.any(-1), (match * t_lp[:, None, :]).sum(-1), t_lp[:, -1:])
    kls = np.maximum((np.exp(p_lp) * (p_lp - q_lp)).sum(-1), 0)
    print(f"{Path(test_path).stem} vs {Path(ref_path).stem} ({n} tokens): "
          f"PPL {ppl_t:.4f} vs {ppl_r:.4f} ({(ppl_t / ppl_r - 1):+.2%})  top-1 agreement {top1:.2%}  "
          f"KLD mean {kls.mean():.4f}  median {np.median(kls):.4f}  p99 {np.quantile(kls, 0.99):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "compare"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--model")
    ap.add_argument("--quantization")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--cpu-offload-gb", type=float, default=0)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--n-seqs", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.cmd == "score":
        score(args)
    else:
        ref = args.files[0]
        for f in args.files[1:]:
            compare(ref, f)


if __name__ == "__main__":
    main()
