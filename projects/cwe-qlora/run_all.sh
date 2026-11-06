#!/usr/bin/env bash
set -euo pipefail

python -c "import torch; assert torch.cuda.is_available(), 'A CUDA GPU is required'; print(torch.cuda.get_device_name(0))"
python -m pip install -q -r requirements.txt

python -m cwe_qlora prepare --out work_dry --run-mode dry-run --train-size 200 --val-size 50 --test-size 50
python -m cwe_qlora train --work work_dry --output outputs_dry/adapter --max-steps 20 --eval-steps 20 --save-steps 1000 --logging-steps 5
python -m cwe_qlora predict --work work_dry --out-dir results_dry --limit 16
python -m cwe_qlora predict --work work_dry --out-dir results_dry --limit 16 --adapter outputs_dry/adapter

python -m cwe_qlora prepare --out work
python -m cwe_qlora baselines --work work --out results
python -m cwe_qlora train --work work --output outputs/adapter
python -m cwe_qlora predict --work work --out-dir results
python -m cwe_qlora predict --work work --out-dir results --adapter outputs/adapter
python -m cwe_qlora report --work work --out results

python - <<'PY'
import zipfile
from pathlib import Path

members = sorted(Path("results").glob("*"))
members += [Path("work/prepare_meta.json"), Path("work/train_run.json"), Path("outputs/training_log.json")]
with zipfile.ZipFile("cwe_qlora_results.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
    for member in members:
        bundle.write(member, member.as_posix())
print("wrote cwe_qlora_results.zip")
PY
