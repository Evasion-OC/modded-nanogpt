# Which speedrun records run on an A100

Every entry below comes from the record's own training script, not its name. Most record directories
contain no `.py` file; the script that ran is printed at the top of the record's log, between two lines
of `=` characters. `get_script.py` extracts it, using the log that the upstream README links when a
directory holds several (some hold 40 to 75). Each extracted script was checked with `py_compile`, and
the features below (FP8, FlashAttention 3, FlexAttention, step count, batch shape) were read from it.
Published losses and times come from the same logs.

FP8 first appears in record #19 (2025-01-13_Fp8LmHead) and is used in every record after it.
FlashAttention 3 appears from mid-2025. Both need H100s, so records #4 to #18 are the ones that run here.

## Records #1 to #18

| # | record directory | status here | script | steps | published final loss / time (8× H100) |
|---|---|---|---|---|---|
| 1 | 2024-10-13_llmc | no training script in the repository | llm.c log, no code | | 45 min |
| 2 | 2024-06-06_AdamW | no training script in the repository | log predates the code-in-log convention | | 31.4 min |
| 3 | (first Muon record) | no training script in the repository | the README lists no log | | 24.9 min |
| 4 | 2024-10-10_Muon | run: 3.3114 in 2.92 h | `.py` | 6,200 | 3.2785 / 22.3 min |
| 5 | 2024-10-14_ModernArch | first attempt failed (compiler); rerun not in this branch yet | `.py` | 5,100 | 3.2741 / 15.2 min |
| 6 | 2024-10-17_DistributedMuon | not in this branch yet | log 22d24867 | 5,100 | 3.2747 / 13.1 min |
| 7 | 2024-10-18_PyTorch25 | not in this branch yet | log d4bfb25f | 5,100 | 3.2755 / 12.0 min |
| 8 | 2024-11-03_UntieEmbed | not in this branch yet | log d6b50d71 | 4,578 | 3.2762 / 10.8 min |
| 9 | 2024-11-06_ShortcutsTweaks | not run yet | log dd7304a6 (1 of 7) | 3,200 | 3.2791 / 8.2 min |
| 10 | 2024-11-08_CastBf16 | not run yet | log a833bed8 | 3,242 | 3.2781 / 7.8 min |
| 11 | 2024-11-10_UNetDoubleLr | not run yet | log c87bb826 | 3,000 | 3.2753 / 7.2 min |
| 12 | 2024-11-19_FlexAttention | not run yet | log 8384493d | 1,875 | 3.2783 / 5.03 min |
| 13 | 2024-11-24_WindowWarmup | not run yet | log cf9e4571 (1 of 7) | 1,750 | 3.2759 / 4.66 min |
| 14 | 2024-12-04_ValueEmbed | not run yet | `.py` (plus 38 seed logs) | 1,530 | 3.2791 (seed eecabe35) / 4.41 min |
| 15 | 2024-12-08_UNetValueEmbedsTweaks | not run yet | log e66b0dd9 (1 of 75) | 1,480 | 3.2782 / 3.95 min |
| 16 | 2024-12-10_MFUTweaks | not run yet | log 591496b6 (1 of 40) | 1,480 | 3.2764 / 3.80 min |
| 17 | 2024-12-17_SparsifyEmbeds | not run yet | log 165384d5 | 1,490 | 3.2794 / 3.57 min |
| 18 | 2025-01-04_SoftCap | not run yet | log 31d6c427 | 1,390 | 3.2785 / 3.4 min |

Batch shape, from the scripts:

- Records #4 to #11 set `device_batch_size : int = 64` and a global batch of 8 × 64 sequences of
  1,024 tokens. Here the device batch is 32 and gradient accumulation makes up the rest, so the global
  batch is unchanged; `get_script.py` makes and checks that change.
- Records #12 to #18 put one 64K-token sequence on each GPU, with a global batch of 8 sequences, so
  memory per GPU does not depend on the number of GPUs; with two GPUs each takes four accumulation
  steps. There is no device batch to change, and `get_script.py` runs these scripts as they are. They
  were set on 80 GB cards, so each gets a ten-minute run first to check that a 64K-token sequence fits
  in 40 GB and to measure the step time.

## Records #19 onward

None of these run here. Most use FP8 or FlashAttention 3, found by searching each extracted script
for `float8`, `e4m3`, `e5m2`, `scaled_mm` and `flash_attn`; the A100 has no FP8 support, and
FlashAttention 3 is written for Hopper GPUs. A few directories contain no training script.

| record directory | reason |
|---|---|
| 2025-01-13_Fp8LmHead | FP8 (the first record to use it) |
| 2025-01-16_Sub3Min | FP8 |
| 2025-01-26_BatchSize | FP8 |
| 2025-02-01_RuleTweak | FP8 (a re-timing of #21) |
| 2025-05-09_SkipMLPBlocks | FP8 and FlashAttention 3 |
| 2025-05-24_FasterReduce | FP8 |
| 2025-05-24_StableTorch | FP8 (a re-timing of #21) |
| 2025-05-25_EvenFasterReduce | FP8 |
| 2025-05-25_MuonWithAuxAdamExample | FP8 (an example, not a record) |
| 2025-05-30_noallreduce | FP8 |
| 2025-07-12_BosAlign | FP8 |
| 2025-07-13_UpgradeTorch190 | FP8 |
| 2025-07-18_TritonMuon | FP8 |
| 2025-08-23_SparseAttnGate | FP8 |
| 2025-09-03_FA3 | FP8 and FlashAttention 3 |
| 2025-09-05_SkipMLPBlocks | FP8 and FlashAttention 3 |
| 2025-09-10_Yarn | FP8 and FlashAttention 3 |
| 2025-09-11_VectSigmoidBFloat16 | FP8 and FlashAttention 3 |
| 2025-09-15_AsyncDataLoadAttnFinalWindow | FP8 and FlashAttention 3 |
| 2025-09-18_Smear | FP8 and FlashAttention 3 |
| 2025-09-21_DropAttn | FP8 and FlashAttention 3 |
| 2025-09-23_MuonCustomSizing | FP8 and FlashAttention 3 |
| 2025-09-27_BF16CE | FP8 and FlashAttention 3 |
| 2025-09-29_PolarExpress | FP8 and FlashAttention 3 |
| 2025-09-30_CustomBatching | FP8 and FlashAttention 3 |
| 2025-10-04_Backout | FP8 and FlashAttention 3 |
| 2025-10-24_NorMuon | FP8 and FlashAttention 3 |
| 2025-10-27_FixMuonLR | FP8 and FlashAttention 3 |
| 2025-10-31_AdamSyncGradientHook | FP8 and FlashAttention 3 |
| 2025-11-10_CautiousWD | FP8 and FlashAttention 3 |
| 2025-11-18_RefineSkip | FP8 and FlashAttention 3 |
| 2025-11-29_BatchSizeSchedule | FP8 and FlashAttention 3 |
| 2025-12-10_SALambdaOnWeights | FP8 and FlashAttention 3 |
| 2025-12-11_NorMuonOptimsAndFixes | FP8 and FlashAttention 3 |
| 2025-12-14_PartialKeyOffset | FP8 and FlashAttention 3 |
| 2025-12-18_CautiousWDAdam | FP8 and FlashAttention 3 |
| 2025-12-19_RetieLMHead | FP8 and FlashAttention 3 |
| 2025-12-21_SmoothedScalars | no training script in the directory |
| 2025-12-22_MultiTokenPrediction | FP8 and FlashAttention 3 |
| 2025-12-26_LogitRescale | FP8 and FlashAttention 3 |
| 2025-12-29_VeSkipGates | FP8 and FlashAttention 3 |
| 2025-12-31_GatesToCompiledAdam | no training script in the directory |
| 2026-01-04_MixedPrecisionInterweavedOptimizer | FP8 and FlashAttention 3 |
| 2026-01-07_PairedHeadAttention | FP8 and FlashAttention 3 |
| 2026-01-10_FusedLinearReLUSquare | FP8 and FlashAttention 3 |
| 2026-01-16_FusedSoftcappedEntropy | FP8 and FlashAttention 3 |
| 2026-01-18_UnifiedOptimizers | no training script in the directory |
| 2026-01-19 .. 2026-03-06 (11 directories with logs) | FP8 and FlashAttention 3. The scripts extracted from these logs also fail `py_compile` under Python 3.13 |
| 2026-01-30_MimeticValueOutput, 2026-02-16_FlattenForward, 2026-02-28_TransposeCopyBackward, 2026-03-22_VarlenMaxDocs, 2026-04-04_FuseCEFwdAndBwd, 2026-04-08_PairedHeadMuon, 2026-04-22_MuddFormer, 2026-04-29_XSAGatedLayers, 2026-05-19_FP8MLPUpProj, 2026-05-20_BigramsSignTrick, 2026-05-27-MuddGatedAndDC, 2026-06-11_RecursiveFromBest, 2026-07-13_PrefixTokenPrediction, 2026-07-17_FP8DownProjection | no training script in the directory |
| 2026-05-07_XSAGatedLayers | its `.py` is a helper, not the training script |

## Other record directories (not numbered records)

| record directory | what it is | here |
|---|---|---|
| 2024-10-09_SOAP | SOAP optimizer run, 6,000 steps | runs on an A100, not planned |
| 2024-10-20_ScaleUp1B | 1.5B-parameter run, 577 min on 8× H100 | too long here (about 75 h) |
| 2024-10-29_Optimizers | optimizer comparisons, 25.5 min | runs on an A100, not planned |
| 2024-11-04_50Bruns | 95,367-step run, 227 min | too long here |
| 2024-11-09_Replicateleloykun | replication run, 8.0 min | runs on an A100, not planned |
| 2024-11-10_ScaleShortcuts | 19,073-step run, 46 min | too long here |
| 2024-11-14_QuantizedFP4 | FP4 and FP8 quantization | needs an H100 |
