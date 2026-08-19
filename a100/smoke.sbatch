#!/bin/bash
#SBATCH --job-name=a100track-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%j.out
# Prefer the self-contained conda env (own python + headers, node-independent);
# fall back to the venv only if it does not exist.
if [ -x "$HOME/envs/nanogpt/bin/python" ]; then
  export PATH="$HOME/envs/nanogpt/bin:$PATH"
elif [ -x "$HOME/venvs/refiner/bin/python" ]; then
  source "$HOME/venvs/refiner/bin/activate"
else
  echo "FATAL: no environment — build the conda env (a100/README, Environment)"; exit 1
fi
# Shared-node hygiene: keep every compile cache out of the node-local /tmp
export TMPDIR="$HOME/tmp"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
mkdir -p "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
# The GPU node lacks python3.11-devel (no Python.h), which breaks triton's
# cuda_utils build. Use a home-dir copy of the headers if one exists
# (made once on the login node: cp -r /usr/include/python3.11 ~/pyinclude311)
[ -d "$HOME/pyinclude311" ] && export CPATH="$HOME/pyinclude311:$CPATH"
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
