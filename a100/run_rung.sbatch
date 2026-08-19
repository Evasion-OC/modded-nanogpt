#!/bin/bash
#SBATCH --job-name=a100track-rung
#SBATCH --partition=gpu-standard
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
# Usage: RECORD=2024-10-10_Muon GPUS=2 sbatch a100/run_rung.sbatch
source "$HOME/venvs/refiner/bin/activate" || { echo "FATAL: venv missing"; exit 1; }
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
: "${RECORD:?set RECORD=<record dir name>}"
GPUS="${GPUS:-2}"
S=$(ls records/track_1_short/$RECORD/*.py | head -1)
mkdir -p a100/results/$RECORD
torchrun --standalone --nproc_per_node=$GPUS "$S" 2>&1 | tee a100/results/$RECORD/run_gpus$GPUS.log
