#!/bin/bash
#SBATCH --job-name=a100track-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%j.out
source "$HOME/venvs/refiner/bin/activate" || { echo "FATAL: venv missing"; exit 1; }
# Shared-node hygiene: keep every compile cache out of the node-local /tmp
export TMPDIR="$HOME/tmp"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
set -ex
cd "$SLURM_SUBMIT_DIR"
python - <<'PY'
import torch; print(torch.__version__, torch.cuda.get_device_name(0))
from torch.nn.attention.flex_attention import flex_attention  # smoke: import works on this build
PY
R=records/track_1_short/2024-10-10_Muon/train_gpt2.py
# 10-minute capped runs; success = per-step timing lines appear in the log
timeout 600 torchrun --standalone --nproc_per_node=1 $R || true
timeout 600 torchrun --standalone --nproc_per_node=2 $R || true
echo "SMOKE DONE — compare step times above for the 1v2 GPU scaling factor"
