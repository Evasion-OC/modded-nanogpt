# The A100 track

The speedrun's records run on 8x H100. This directory reproduces the record
ladder on 1-2x A100-PCIE-40GB (a shared university node), measures what each
rung is worth there, and documents which rungs are H100-only (FP8, FA3) --
numbers the main repo does not have.

Method: same fixed target as the speedrun (val loss 3.28 on FineWeb), same
scripts. The early record scripts define the GLOBAL batch independently of
world size and compute gradient-accumulation steps at runtime, so a 1- or
2-GPU run reproduces the 8-GPU training trajectory exactly; only wall time
changes. Later rungs must be checked per script (classification.md).

MEASURED (smoke, 19 Aug 2026, Muon rung @ device_bs=32, global batch 512):
- step_avg 3,300 ms on 1x A100, 1,700 ms on 2x = **1.94x, 97% scaling efficiency**.
  The PCIe penalty predicted below did not materialise: with 16 accumulation
  micro-steps per optimizer step, gradient sync amortises almost completely.
- Full Muon rung projection: 6,200 steps x 1.70 s = **~2.9 h on 2 GPUs** (~5.7 h
  on 1). The 45-90 min prediction below was optimistic by ~2x; kept for the record.
- device_batch_size 64 OOMs on 40 GB (the 6.1 GB bf16 logits tensor); 32 fits.
  Global batch unchanged, so tokens-to-target stays comparable to the records.

Predictions, before measuring (kept for reconciliation, refiner-perf style):
- bf16-era rungs at ~3-8 min on 8xH100 land at roughly 45-90 min on 2xA100,
  2-4 h on 1xA100 (GPU count x per-GPU speed x PCIe; estimate, not measurement).
- DDP scaling on a PCIe pair will be visibly sub-linear; smoke.sbatch measures
  the 1-vs-2-GPU step-time ratio explicitly and it gets reported, not hidden.
- FlexAttention on sm80 / torch 2.13 should work; smoke-tested before use.

Workflow on the cluster (login node):
  1) Environment (ONE TIME, login node). python3.11-devel is missing on BOTH
     login and GPU nodes, so triton cannot compile its driver shim against the
     system python. Use a self-contained conda env instead:
       module load Anaconda3/2025.12-1
       conda create -y -p ~/envs/nanogpt -c conda-forge --override-channels python=3.11
       ~/envs/nanogpt/bin/pip install --upgrade pip
       ~/envs/nanogpt/bin/pip install torch scipy huggingface_hub tqdm
     Jobs pick it up automatically (PATH-first activation in the sbatch files).
  2) python data/cached_fineweb10B.py 24     # shards; ~few GB per unit, needs quota
  3) sbatch a100/smoke.sbatch                # 10-step sanity, 1 then 2 GPUs
  4) RECORD=2024-10-10_Muon GPUS=2 sbatch a100/run_rung.sbatch

Everything derives from KellerJordan/modded-nanogpt (MIT). Record scripts are
run as committed by their authors; per-rung changes, if any are ever needed,
live as diffs here and are declared in the results table.
