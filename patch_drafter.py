"""Make the DSpark drafter's NVFP4 exclusion list match vLLM's module names.

The checkpoint excludes modules by their bare names ("fc", "markov_head*"), but
vLLM builds the drafter under a "model." prefix, so those patterns miss and the
BF16 Markov head / fc get allocated as packed FP4 params ("size of tensor a (128)
must match ... (256)" at load). Adding "*"-prefixed variants fixes the match.
Idempotent; keeps a .orig backup of each file.
"""

import json
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3.8-27B-DSpark-NVFP4")


def widen(patterns):
    out = list(patterns)
    for p in patterns:
        if not p.startswith("*") and "*" + p not in out:
            out.append("*" + p)
    return out


for name, path in (("hf_quant_config.json", ("quantization", "exclude_modules")),
                   ("config.json", ("quantization_config", "ignore"))):
    f = root / name
    if not (f.with_suffix(".json.orig")).exists():
        shutil.copy(f, f.with_suffix(".json.orig"))
    cfg = json.loads(f.read_text())
    section = cfg[path[0]]
    section[path[1]] = widen(section[path[1]])
    f.write_text(json.dumps(cfg, indent=4) + "\n")
    print(f"{f}: {len(section[path[1]])} exclusion patterns")
