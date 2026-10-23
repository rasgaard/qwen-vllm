#!/usr/bin/env bash
# Serve Qwen3.8-27B (NVFP4) with vLLM on a single 24 GB Blackwell card.
#
#   ./serve.sh                 # default: built-in MTP head speculative decoding
#   SPEC=dspark MAX_LEN=4096 KV_MEM= ./serve.sh  # DSpark drafter: ~7% faster, but
#                              # its VRAM cost leaves room for only ~4k context
#   SPEC=none ./serve.sh       # plain autoregressive decode
#   MAX_LEN=32768 ./serve.sh   # override max context per request
#   KV_MEM= ./serve.sh         # let vLLM profile memory instead of pinning it
#
# Listens on all interfaces; clients need the key in .api-key (Bearer auth).
set -euo pipefail
cd "$(dirname "$0")"

SPEC=${SPEC:-mtp}
MAX_LEN=${MAX_LEN:-65536}
MAX_SEQS=${MAX_SEQS:-8}
PORT=${PORT:-8000}
GPU_UTIL=${GPU_UTIL:-0.95}
# 3.75 GiB: largest KV budget that survived stress.py with ~600 MiB to spare.
KV_MEM=${KV_MEM-4026531840}

HOST=${HOST:-0.0.0.0}

# Shared-secret auth for LAN clients; generated on first run.
if [[ ! -s .api-key ]]; then
  (umask 077; uv run python -c "import secrets; print('sk-' + secrets.token_urlsafe(32))" > .api-key)
fi
export VLLM_API_KEY="$(cat .api-key)"

MODEL=models/Qwen3.8-27B-NVFP4-MTP
DRAFTER=models/Qwen3.8-27B-DSpark-NVFP4

# FlashInfer JIT-compiles its SM120 FP4 kernels on first boot; use the nvcc
# that ships in the venv (nvidia-cuda-* wheels) instead of a system toolkit.
# The wheels lack the lib64/ and unversioned libcudart.so a toolkit has, so
# assemble a toolkit-shaped CUDA_HOME out of symlinks.
uv sync -q
CU13="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME="$PWD/.cuda-home"
mkdir -p "$CUDA_HOME/lib64"
for d in bin include nvvm cccl; do ln -sfn "$CU13/$d" "$CUDA_HOME/$d"; done
ln -sf "$CU13/lib/libcudart.so.13" "$CUDA_HOME/lib64/libcudart.so"
ln -sf "$CU13/lib/libcudadevrt.a" "$CUDA_HOME/lib64/libcudadevrt.a"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

case "$SPEC" in
  dspark) SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"model\":\"$DRAFTER\",\"num_speculative_tokens\":${NUM_SPEC:-7}}") ;;
  mtp)    SPEC_ARGS=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":${NUM_SPEC:-5}}") ;;
  none)   SPEC_ARGS=() ;;
  *) echo "unknown SPEC=$SPEC" >&2; exit 1 ;;
esac

# vLLM's profiler over-reserves activation memory on this model; pinning the
# KV cache size reclaims it (validated with stress.py, see README).
if [[ -n "$KV_MEM" ]]; then MEM_ARGS=(--kv-cache-memory "$KV_MEM"); else MEM_ARGS=(--gpu-memory-utilization "$GPU_UTIL"); fi

exec uv run vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b \
  --host "$HOST" \
  --port "$PORT" \
  --quantization modelopt \
  --language-model-only \
  --kv-cache-dtype fp8 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  "${MEM_ARGS[@]}" \
  --enable-prefix-caching \
  --async-scheduling \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  "${SPEC_ARGS[@]}" \
  "$@"
