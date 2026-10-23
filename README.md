# Qwen3.8-27B on vLLM (RTX PRO 4000 Blackwell, 24 GB)

OpenAI-compatible server on `http://10.183.65.74:8000/v1`, model name `qwen3.8-27b`,
for opencode clients on the LAN.

## Setup (fresh machine)

```bash
sudo apt-get install -y build-essential   # Triton/FlashInfer JIT need gcc
uv sync                                    # vLLM 0.32 + pinned CUDA 13.0 toolchain
uvx --from "huggingface_hub[hf_xet]" hf download gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --revision 35fd99fb8434ef70ee61372df3face5174bed2c8 --local-dir models/Qwen3.8-27B-NVFP4-MTP
# optional, only for SPEC=dspark:
uvx --from "huggingface_hub[hf_xet]" hf download gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4 \
  --revision eba1ac5a66c74902eaa95a4000a7c5eda96d8e95 --local-dir models/Qwen3.8-27B-DSpark-NVFP4
uv run python patch_drafter.py
```

## Run

```bash
./serve.sh        # foreground; first boot JIT-compiles FlashInfer FP4 kernels (~2 min)
./stop.sh
```

Or as a service (survives logout/reboot):

```bash
sudo cp qwen-vllm.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now qwen-vllm
journalctl -u qwen-vllm -f
```

LAN access needs the firewall (ufw) opened once:

```bash
sudo ufw allow from 10.183.65.0/24 to any port 8000 proto tcp
```

## Connect with opencode

1. Get the API key from the server owner (`.api-key` in this directory).
2. `export QWEN_LAN_API_KEY=sk-...` (shell profile).
3. Copy `opencode.example.json` to `~/.config/opencode/opencode.json`
   (or merge its `provider` block into an existing config).

## What it runs and why

| Choice | Reason |
|---|---|
| `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` @ `35fd99f` (`pre-final`) | Everything NVFP4 incl. GDN projections and `lm_head` (15.9 GiB on GPU); this revision still has the MTP head. NVIDIA's build keeps attention at FP8 (21.9 GB) and doesn't leave room for KV. |
| MTP speculative decoding, 5 tokens | ~2.8x single-stream decode for 0.85 GB. |
| `--kv-cache-memory 3.75 GiB` | vLLM's profiler reserves ~3.3 GiB for activations; real peak is far lower. 4.5 GiB OOMed under `stress.py`, 3.75 GiB passed with ~600 MiB to spare. |
| FP8 KV cache, prefix caching (`align` mamba mode) | Shared opencode system prompts / multi-turn history get reused. |
| `--language-model-only` | Skips the 0.9 GB vision tower (text only). |
| `MAX_LEN=65536`, 8 seqs | Pool is ~75k tokens; long sessions queue/preempt rather than fail. |

## Measured (bench.py, 512-token outputs, thinking off, temperature 0)

| Config | Single stream | Notes |
|---|---|---|
| No speculation | 37.5 tok/s | |
| MTP, 3 tokens | 91.5 tok/s | acceptance 3.33 |
| **MTP, 5 tokens (default)** | **103.5 tok/s** | acceptance 4.28; 166 tok/s aggregate @2 users, 270 @4 |
| DSpark drafter, 7 tokens | 110.7 tok/s | only ~4k context fits: drafter weights + 8 GDN state snapshots per request |

## Gotchas fixed along the way

- No system CUDA toolkit: nvcc comes from the `nvidia-cuda-*` wheels, pinned to 13.0
  in `pyproject.toml` (vllm[cu13] otherwise pulls 13.4 nvcc/nvvm, which clash with the
  13.0 runtime). `serve.sh` assembles a toolkit-shaped `.cuda-home/` from symlinks.
- Triton/FlashInfer JIT need `gcc` (`build-essential`) and Python headers; the venv
  uses uv-managed CPython (system Python lacks `Python.h`).
- DSpark drafter only: `patch_drafter.py` fixes its NVFP4 exclusion patterns for
  vLLM's module names (otherwise a 128 vs 256 shape error on load).
