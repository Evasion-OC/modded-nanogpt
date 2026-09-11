# Running the speedrun records on A100s

This branch runs records from the modded-nanogpt speedrun on two A100-PCIE-40GB GPUs, one node of a
university cluster, instead of the eight H100s the records were set on, and compares each run with
the published log. Everything outside `a100/` is the upstream repository.

## Which records can run

Records #19 onward use FP8, and records from mid-2025 also use FlashAttention 3. Both need H100s.
Records #4 to #18 use neither, so they run on an A100. Records #1 to #3 have no training script in
the repository. `classification.md` has the details for every record directory.

## Adaptations

- Device batch: 32 sequences per GPU instead of 64, since 64 runs out of memory on 40 GB cards
  (the bf16 logits tensor alone is 6.1 GB). The scripts derive gradient accumulation from the global
  batch of 512 sequences, which stays the same.
- Scripts: most record directories contain no `.py` file; the script that ran is
  printed at the top of the record's log. `get_script.py` extracts it, using the log the upstream
  README links when a directory holds several, and applies the device-batch change. Each change must
  apply exactly once or the script stops.
- Cluster environment: the nodes lack the Python 3.11 headers and have a GCC too old for
  `-std=c++20`, both of which break Triton or inductor builds. Jobs therefore use a conda environment
  with its own Python and GCC, and keep compiler caches in the home directory.

## Results so far

### Step time

Ten-minute test runs of the Muon record took 3,300 ms per step on one GPU and
1,700 ms on two, a 1.94× speedup from the second GPU.

### Muon record (#4)

| | published run, 8× H100 | this run, 2× A100 |
|---|---|---|
| steps | 6,200 | 6,200 |
| training time | 22.3 min | 2.92 h |
| time per step | 216 ms | 1,697 ms |
| final validation loss | 3.2785 | 3.3114 |

Peak memory on the first GPU was 19,711 MiB. The final loss is 0.033 higher than the published one. Compared
every 125 steps, the two validation curves differ by at most 0.008, in both directions, from step
250 to step 2,500. The gap then grows while the learning rate decays: +0.010 at step 3,750, +0.020 at 4,375,
+0.026 at 5,000 and +0.033 at 6,200. For scale, the 38 runs of a later record (ValueEmbed) logged in
this repository have a standard deviation of 0.004 in final loss.

### With and without torch.compile

The same record with `torch.compile` turned off finished at
3.3113, against 3.3114 compiled, and from step 3,750 on the two runs never differ by more than 0.003,
so the gap does not come from the compiler. Compiled steps were 1.94 times faster (1,697 against 3,284 ms) and
used less memory (19,711 against 32,822 MiB peak). The GPU architecture, the PyTorch version (2.13 here,
2.4.1 for the record), the attention backend and gradient accumulation have not been tested yet.

### Records #5 onward

The first ModernArch (#5) run stopped when inductor tried to build CPU kernels
with the node's GCC; jobs now use the conda environment's compiler. Results for records #5 to #8 are
not in this branch yet.

## Running a record

One-time setup on the login node:

```bash
module load Anaconda3/2025.12-1
conda create -y -p ~/envs/nanogpt -c conda-forge --override-channels python=3.11
conda install -y -p ~/envs/nanogpt -c conda-forge --override-channels gxx_linux-64 gcc_linux-64
~/envs/nanogpt/bin/pip install torch scipy huggingface_hub tqdm
python data/cached_fineweb10B.py 24
```

Then:

```bash
sbatch a100/smoke.sbatch                                    # ten-minute runs on 1 and 2 GPUs
RECORD=2024-10-14_ModernArch GPUS=2 sbatch a100/run_rung.sbatch
RECORD=2024-10-10_Muon GPUS=2 TAG=eager EXTRA_ARGS=--no-compile sbatch --time=10:00:00 a100/run_rung.sbatch
RECORD=2024-11-19_FlexAttention GPUS=2 TAG=smoke TIMEOUT=600 sbatch --partition=gpu-short a100/run_rung.sbatch
```

Records #12 to #18 run one 64K-token sequence per GPU and were set on 80 GB cards, so each gets a
ten-minute run (`TIMEOUT=600`) to check memory and step time before a full run. Each run writes the
adapted script and its log to `a100/results/<record>/`.

The rest of the repository is [KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
(MIT licence).
