# Qwen3.8-27B on vLLM (RTX PRO 4000 Blackwell, 24 GB)

OpenAI-compatible server on `http://$QWEN_LAN_HOST:8000/v1`, model name `qwen3.8-27b`,
for opencode clients on the LAN.

`QWEN_LAN_HOST` is the server's IP or hostname. The IP comes from DHCP and can change when the
machine moves; check it on the server with `ip -4 -br addr`. Clients on the same subnet can use
`<hostname>.local` instead (mDNS via avahi), but mDNS does not cross subnets.

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

On this machine it is installed as a systemd service: it starts on boot and restarts on
failure. It is ready about a minute after start (model load and warmup). Install it once with:

```bash
./stop.sh   # free the GPU and port if serve.sh is running in a terminal
sudo cp qwen-vllm.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now qwen-vllm
```

Day to day:

```bash
systemctl status qwen-vllm
journalctl -u qwen-vllm -f            # logs; no sudo needed for users in the adm group
sudo systemctl restart qwen-vllm      # e.g. after editing serve.sh
sudo systemctl stop qwen-vllm         # or ./stop.sh; starts again on next boot
```

The service caps GPU power at 110 W (card default and max 145 W, min 100 W) before starting
vLLM, because the card's blower fan is loud at full power. Decode is mostly memory-bandwidth
bound, so the cap costs less speed than power; prefill slows more. The limit resets on reboot
and is re-applied on every service start. To change it by hand (takes effect immediately):

```bash
sudo nvidia-smi -pl 110                                   # 100-145 W
nvidia-smi --query-gpu=power.limit,power.draw,temperature.gpu,fan.speed --format=csv
```

After editing `qwen-vllm.service`, reinstall it with
`sudo cp qwen-vllm.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl restart qwen-vllm`.

LAN access needs the firewall (ufw) opened once per client subnet. Clients on a subnet that
isn't allowed time out (ETIMEDOUT):

```bash
sudo ufw allow from <client-subnet>/24 to any port 8000 proto tcp
grep 'UFW BLOCK' /var/log/ufw.log | grep DPT=8000   # who is being dropped
```

## Dashboard

`dashboard.py` serves a live web dashboard on port 8080 at `http://$QWEN_LAN_HOST:8080/`.
It runs on the server next to vLLM, needs only the Python standard library, and refreshes
every 2 s. It shows:

- requests running and waiting, token throughput, KV cache use, prefix cache hits, latency and
  MTP acceptance
- GPU power against the cap, temperature, fan, utilization, VRAM and clock (nvidia-smi)
- charts of the last 30 minutes, kept in memory, so they start empty after a restart
- per-client requests, errors, prompt and generated tokens and firewall drops over the last
  24 h, plus recent requests (from the service journal and `/var/log/ufw.log`)

Tokens per client come from `usage_log.py`, an ASGI middleware that `serve.sh` loads into vLLM
(`--middleware`). It logs one `USAGE ip=... prompt=... completion=...` line per completion.
For streamed responses it turns on `stream_options.include_usage` and strips the extra usage
chunk again if the client didn't ask for it. Prompt tokens include tokens served from the prefix
cache. Generated tokens include reasoning tokens.

Install it as a service, and open the port to the same subnets as 8000:

```bash
sudo cp qwen-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now qwen-dashboard
sudo ufw allow from <client-subnet>/24 to any port 8080 proto tcp
```

The dashboard has no login. Anyone who can reach port 8080 sees client IP addresses and
request paths, but never prompts or the API key. `serve.sh` keeps `/metrics` polls out of
vLLM's access log.

## Connect with opencode

1. Get the API key from the server owner (`.api-key` in this directory).
2. `export QWEN_LAN_API_KEY=sk-...` and `export QWEN_LAN_HOST=<server IP or hostname>`
   (shell profile).
3. Copy `opencode.example.json` to `~/.config/opencode/opencode.json`
   (or merge its `provider` block into an existing config).

Troubleshooting from the client:
`curl -H "Authorization: Bearer $QWEN_LAN_API_KEY" http://$QWEN_LAN_HOST:8000/v1/models`.
ECONNREFUSED means the wrong network or the server is down, and ETIMEDOUT means the firewall
(see above). "Unauthorized" means the key isn't reaching opencode: export the variables in the shell that
starts opencode, then restart opencode.

Reasoning effort defaults to `medium`. The `off` / `low` / `medium` / `xhigh` variants switch
it with opencode's `variant_cycle` keybind. Leaving `reasoning_effort` unset in the request means `xhigh`, which thinks the longest.
The chat template only accepts `low`, `medium` and `xhigh` (`none` turns thinking off). Any
other value, such as `high` or `minimal`, makes every request fail with HTTP 400.

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

Context: 65,536 tokens per request, ~75k-token KV pool shared by all requests
(model native: 262k; the limit is VRAM).

## Quality vs. BF16

Likelihood on identical inputs (`evals/kld.py`: WikiText-2 test, 16 x 8k tokens = 131k
scored positions; BF16 reference run with 40 GB of weights offloaded to CPU):

| Config | PPL | Top-1 agreement with BF16 | Mean KLD (top-20) |
|---|---|---|---|
| BF16 | 5.861 | — | — |
| NVFP4 weights, BF16 KV | 6.278 (+7.1%) | 87.0% | 0.071 |
| NVFP4 weights, FP8 KV (production) | 6.278 (+7.1%) | 87.0% | 0.071 |

FP8 KV vs BF16 KV on its own: PPL identical, 99.7% top-1 agreement, KLD 0.0008.

MTP speculative decoding does not change quality: greedy GSM8K (100 items, thinking off)
95% with MTP vs 93% without (McNemar p=0.62). Outputs are not token-identical, but neither
are two runs of the same no-speculation config (15/100 identical) - batched GPU math is not
deterministic.

Task accuracy, production config (`evals/tasks.py`, thinking on, `reasoning_effort: medium`,
model-default sampling):

| Benchmark | Score | 95% CI | Published BF16* |
|---|---|---|---|
| GSM8K (200 items) | **97.5%** (195/200) | 94.3-98.9% | 95.5% |
| MMLU-Pro (280 items, 20 per subject) | **83.6%** (234/280) | 78.8-87.5% | 80.4% |

\*From the minima-ai model card, different harness: a rough reference, not a paired comparison.
The 4-bit perplexity cost does not show up as lost task accuracy at this sample size.

MMLU-Pro by subject: biology 20/20, psychology 19, chemistry 18, economics 18, math 18,
business 17, other 17, physics 17, philosophy 16, computer science 15, health 15, history 15,
law 15, engineering 14.

## Gotchas fixed along the way

- No system CUDA toolkit: nvcc comes from the `nvidia-cuda-*` wheels, pinned to 13.0
  in `pyproject.toml` (vllm[cu13] otherwise pulls 13.4 nvcc/nvvm, which clash with the
  13.0 runtime). `serve.sh` assembles a toolkit-shaped `.cuda-home/` from symlinks.
- Triton/FlashInfer JIT need `gcc` (`build-essential`) and Python headers; the venv
  uses uv-managed CPython (system Python lacks `Python.h`).
- DSpark drafter only: `patch_drafter.py` fixes its NVFP4 exclusion patterns for
  vLLM's module names (otherwise a 128 vs 256 shape error on load).
