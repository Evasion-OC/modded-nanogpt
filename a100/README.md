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

MEASURED (first full rung, 20 Aug 2026, job 1055947): record #4, 2024-10-10_Muon
- 6,200 steps on 2x A100 in **2.92 h** train time, step_avg 1,696.63 ms — the
  600 s smoke's 1,700 ms projection was exact. Peak memory 19,711 MiB.
- **Conversion ratio kappa = 1696.63 / 216.33 = 7.84** (their 8xH100 step_avg
  from the record log). Dense-era rungs are projected at kappa x their record
  time; flex-era rungs get their own 600 s smoke first.
- Final val loss **3.3114 vs the record's 3.2785** at the same step: a drift of
  **Delta = +0.033**. It is not present at the start (their step-125 val 5.2017,
  ours 5.1995) and from step ~5600 the two curves run parallel (+0.031..0.033
  at every milestone), so the gap develops during training and stabilises.
- It is NOT gradient-accumulation rounding: this script keeps parameters — and
  therefore accumulated grads — in fp32 under bf16 autocast (model.cuda(), no
  bf16 cast), and 8-term fp32 sums cannot move a loss by 0.03. Prime suspect is
  kernel-level numerics of a different stack: torch 2.13 / inductor / sm80 here
  vs torch 2.4.1 / sm90 there.
- **Attribution run A0**: the same rung with compilation disabled
  (EXTRA_ARGS=--no-compile, which comments out torch.compile via get_script).
  If Delta moves, inductor codegen is implicated; if Delta stays, the drift
  lives in the bf16/SDPA kernels themselves. Either way the ladder metric is
  **Delta at the record's own step budget, every rung on our one fixed stack**,
  so rung-to-rung comparisons stay clean. No step-extension reruns.

## Run schedule (set 20 Aug; jobs run serially on the shared 2-GPU node)

| when | submit | projected wall |
|---|---|---|
| 20 Aug | A0 (eager Muon, sbatch --time=10:00:00) + rung #5 ModernArch | ~4-6 h + 2.0 h |
| 21 Aug | #6 DistributedMuon, #7 PyTorch25, #8 UntieEmbed | 1.7 + 1.6 + 1.4 h |
| 22 Aug | #9 ShortcutsTweaks, #10 CastBf16, #11 UNetDoubleLr | 1.1 + 1.0 + 0.9 h |
| 23 Aug | flex smokes #12-#18 (TIMEOUT=600 each): peak mem + step_avg on sm80 | ~1.5 h total |
| 24-25 Aug | flex full runs #12-#18, go/no-go per smoke | ~4-6 h total |
| 26-31 Aug | buffer: queue contention, redos; 1-vs-2-GPU scaling pair on #5 and #11; A0 follow-up if inductor is implicated | — |
| 1-12 Sep | no GPU needed: analysis + REPORT.md (Delta ladder, kappa by era, what-transferred taxonomy, numerics section) | — |

That lands all compute by ~1 Sep — a month inside the October access horizon,
with two spare weeks against surprises. After each rung: commit
a100/results/<record>/ and push, so the repo, not the cluster, is the record.

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
  4) RECORD=2024-10-14_ModernArch GPUS=2 sbatch a100/run_rung.sbatch
  5) Eager attribution variant (A0):
       RECORD=2024-10-10_Muon GPUS=2 TAG=eager EXTRA_ARGS=--no-compile \
         sbatch --time=10:00:00 a100/run_rung.sbatch
  6) Flex-era smoke (memory + step_avg audit before any full flex run):
       RECORD=2024-11-19_FlexAttention GPUS=2 TAG=smoke TIMEOUT=600 \
         sbatch --partition=gpu-short a100/run_rung.sbatch

Everything derives from KellerJordan/modded-nanogpt (MIT). Most records ship no
.py — the script that ran is embedded in the record's log; a100/get_script.py
extracts it (preferring the README-linked log when a dir holds several dozen
side logs) and applies only declared, anchor-verified adaptations. Data
assumption: the fineweb10B shards (kjj0/fineweb10B-gpt2) are the same frozen
set the records trained on. Per-rung changes are declared in classification.md
and the results table.
