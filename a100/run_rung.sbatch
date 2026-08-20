#!/bin/bash
#SBATCH --job-name=a100track-rung
#SBATCH --partition=gpu-standard
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
# Usage: RECORD=2024-10-10_Muon GPUS=2 sbatch a100/run_rung.sbatch
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
: "${RECORD:?set RECORD=<record dir name>}"
GPUS="${GPUS:-2}"
DEVICE_BS="${DEVICE_BS:-32}"   # 40GB cards; global batch unchanged (accumulation absorbs it)
TAG="${TAG:-}"                 # names a variant run, e.g. TAG=eager
SUF="${TAG:+_$TAG}"
mkdir -p a100/results/$RECORD
P=a100/results/$RECORD/train_devbs${DEVICE_BS}${SUF}.py
# Most records ship no .py: the source is embedded in the record's log.
# get_script.py extracts it (preferring the README-linked log when a dir holds
# several) and applies anchor-verified adaptations. Flex-era records have no
# standard device_batch_size line; get_script says so and runs them unmodified.
# EXTRA_ARGS="--no-compile" builds the eager numerics-attribution variant (A0).
# LOGFILE=<name.txt> pins a specific log inside the record dir if ever needed.
python a100/get_script.py "records/track_1_short/$RECORD" "$P" --devbs "$DEVICE_BS" ${LOGFILE:+--log $LOGFILE} $EXTRA_ARGS
# TIMEOUT=600 turns any rung into a smoke test (flex-era memory/step_avg audit)
${TIMEOUT:+timeout $TIMEOUT} torchrun --standalone --nproc_per_node=$GPUS "$P" 2>&1 | tee "a100/results/$RECORD/run_gpus${GPUS}_bs${DEVICE_BS}${SUF}.log"
