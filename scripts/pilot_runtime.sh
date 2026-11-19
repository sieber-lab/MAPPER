#!/usr/bin/env bash
# Optional local-container bootstrap; model and experiment settings stay fixed.
set -euo pipefail
repo_root="${1:?Repository path required}"
pilot_root="${2:?Pilot output path required}"
mkdir -p "$pilot_root/logs"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export MPLBACKEND=Agg
export MPLCONFIGDIR="$pilot_root/runtime/matplotlib"
export HF_HOME="$pilot_root/runtime/huggingface"
export MLFLOW_TRACKING_URI="file:$pilot_root/runtime/mlruns"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=false
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1
printf 'Installing isolated GPU training requirements\n' | tee "$pilot_root/runtime_status.txt"
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124
python -m pip install torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r "$repo_root/requirements/training.txt" openpyxl==3.1.5 rdkit==2023.9.5
python -m pip freeze > "$pilot_root/runtime_packages.txt"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
printf 'Environment ready; waiting for fixed pilot plan\n' | tee "$pilot_root/runtime_status.txt"
while [ ! -f "$pilot_root/READY" ]; do sleep 2; done
cd "$repo_root"
printf 'Executing pilot queue\n' | tee "$pilot_root/runtime_status.txt"
exec python -u "$repo_root/scripts/run_repro_pilots.py" --root "$pilot_root"
