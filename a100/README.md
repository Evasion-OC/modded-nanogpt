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

Predictions, before measuring (kept for reconciliation, refiner-perf style):
- bf16-era rungs at ~3-8 min on 8xH100 land at roughly 45-90 min on 2xA100,
  2-4 h on 1xA100 (GPU count x per-GPU speed x PCIe; estimate, not measurement).
- DDP scaling on a PCIe pair will be visibly sub-linear; smoke.sbatch measures
  the 1-vs-2-GPU step-time ratio explicitly and it gets reported, not hidden.
- FlexAttention on sm80 / torch 2.13 should work; smoke-tested before use.

Workflow on the cluster (login node):
  1) source ~/venvs/refiner/bin/activate
  2) python data/cached_fineweb10B.py 24     # shards; ~few GB per unit, needs quota
  3) sbatch a100/smoke.sbatch                # 10-step sanity, 1 then 2 GPUs
  4) RECORD=2024-10-10_Muon GPUS=2 sbatch a100/run_rung.sbatch

Everything derives from KellerJordan/modded-nanogpt (MIT). Record scripts are
run as committed by their authors; per-rung changes, if any are ever needed,
live as diffs here and are declared in the results table.
