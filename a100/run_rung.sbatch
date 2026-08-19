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
set -ex
cd "$SLURM_SUBMIT_DIR"
: "${RECORD:?set RECORD=<record dir name>}"
GPUS="${GPUS:-2}"
S=$(ls records/track_1_short/$RECORD/*.py | head -1)
mkdir -p a100/results/$RECORD
torchrun --standalone --nproc_per_node=$GPUS "$S" 2>&1 | tee a100/results/$RECORD/run_gpus$GPUS.log
