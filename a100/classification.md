# Record classification, track_1_short — VERIFIED 20 Aug 2026

Not name-based guessing: every verdict below comes from the record's actual
source. Most record dirs ship no `.py` — the training script is embedded in the
run log between two `====` bars (the log opens with a verbatim dump of the code
that ran). `a100/get_script.py` extracts it, preferring the log the README
table links when a dir holds several (some hold 40-75 side logs, and the
largest is not always the record). Each extract was syntax-checked with
py_compile; features (FP8, FlashAttention-3, FlexAttention, iteration count,
batch geometry) were grepped from the extracted source; "their" numbers are
harvested from the linked log's own output.

The gate is simple and it held everywhere: **FP8 enters at record #19
(2025-01-13_Fp8LmHead) and never leaves; FlashAttention-3 joins from mid-2025.
Both are H100-only. Everything before #19 is A100-clean.**

Projection basis: kappa = our measured step_avg / record's step_avg on the
completed Muon rung = 1696.63 / 216.33 = **7.84** (2x A100-PCIE-40GB,
device_bs 32, vs 8x H100). Valid for the dense-attention era (#4-#11, same
1024-token geometry). The flex era (#12-#18) changes the workload shape, so
each of those gets a 600 s smoke (TIMEOUT=600) to measure its own step_avg and
peak memory before the full run.

## The ladder (numbered records)

| # | dir | status | source | iters | their val / 8xH100 time | projected 2xA100 |
|---|---|---|---|---|---|---|
| 1 | 2024-10-13_llmc | NO-SOURCE | llm.c-format log, no embedded code | — | 45 min | not runnable from this repo |
| 2 | 2024-06-06_AdamW | NO-SOURCE | llm.c-format .log predates the code-dump convention | — | 31.4 min | not runnable |
| 3 | (Muon intro) | NO-SOURCE | README lists log as "none" | — | 24.9 min | not runnable |
| 4 | 2024-10-10_Muon | **DONE 20 Aug (job 1055947)** | py | 6200 | 3.2785 / 22.3 min | **measured: 3.3114 in 2.92 h, Delta +0.033** |
| 5 | 2024-10-14_ModernArch | READY | py | 5100 | 3.2741 / 15.2 min | ~2.0 h |
| 6 | 2024-10-17_DistributedMuon | READY | log 22d24867, compiles | 5100 | 3.2747 / 13.1 min | ~1.7 h |
| 7 | 2024-10-18_PyTorch25 | READY | log d4bfb25f, compiles | 5100 | 3.2755 / 12.0 min | ~1.6 h |
| 8 | 2024-11-03_UntieEmbed | READY | log d6b50d71, compiles | 4578 | 3.2762 / 10.8 min | ~1.4 h |
| 9 | 2024-11-06_ShortcutsTweaks | READY | log dd7304a6 (of 7 in dir), compiles | 3200 | 3.2791 / 8.2 min | ~1.1 h |
| 10 | 2024-11-08_CastBf16 | READY | log a833bed8, compiles | 3242 | 3.2781 / 7.8 min | ~1.0 h |
| 11 | 2024-11-10_UNetDoubleLr | READY | log c87bb826, compiles | 3000 | 3.2753 / 7.2 min | ~0.9 h |
| 12 | 2024-11-19_FlexAttention | READY, smoke first | log 8384493d, compiles | 1875 | 3.2783 / 5.03 min | ~40 min if dense kappa held |
| 13 | 2024-11-24_WindowWarmup | READY, smoke first | log cf9e4571 (of 7), compiles | 1750 | 3.2759 / 4.66 min | ~37 min " |
| 14 | 2024-12-04_ValueEmbed | READY, smoke first | py (+38 seed logs) | 1530 | 3.2791 (seed eecabe35) / 4.41 min | ~35 min " |
| 15 | 2024-12-08_UNetValueEmbedsTweaks | READY, smoke first | log e66b0dd9 (of 75!), compiles | 1480 | 3.2782 / 3.95 min | ~31 min " |
| 16 | 2024-12-10_MFUTweaks | READY, smoke first | log 591496b6 (of 40), compiles | 1480 | 3.2764 / 3.80 min | ~30 min " |
| 17 | 2024-12-17_SparsifyEmbeds | READY, smoke first | log 165384d5, compiles | 1490 | 3.2794 / 3.57 min | ~28 min " |
| 18 | 2025-01-04_SoftCap | READY, smoke first | log 31d6c427, compiles | 1390 | 3.2785 / 3.4 min | ~27 min " |

Batch geometry notes, verified in source:
- #4-#11: `device_batch_size : int = 64`, global batch `8*64` sequences of 1024
  tokens. Our standard adaptation (devbs 64 -> 32, accumulation absorbs it,
  global batch unchanged) applies; get_script performs and verifies it.
- #12-#18: one 64K-token sequence per device (`device_batch_size : int = 1` or
  hardcoded), global batch 8 sequences. Per-GPU memory is therefore
  **independent of GPU count** — 2 GPUs just means 4 accumulation micro-steps
  instead of none. No devbs adaptation exists or is needed (get_script prints
  DEVBS_PATTERN_MISSING and runs the source unmodified). Open question per rung
  is whether one 64K flex sequence fits 40 GB (records ran on 80 GB) and what
  step_avg it gets on sm80 — that is what the smoke answers.

## Excluded: the FP8 / FlashAttention-3 era (#19 onward)

All verdicts from source markers (`float8|e4m3|e5m2|scaled_mm`, `flash_attn`),
not names. A100 (sm80) has no FP8 hardware and FA3 targets Hopper.

| dir | why excluded |
|---|---|
| 2025-01-13_Fp8LmHead | FP8 (first record to use it) |
| 2025-01-16_Sub3Min | FP8 |
| 2025-01-26_BatchSize | FP8 |
| 2025-02-01_RuleTweak | FP8 (re-timing of #21) |
| 2025-05-09_SkipMLPBlocks | FP8 + FA3 |
| 2025-05-24_FasterReduce | FP8 |
| 2025-05-24_StableTorch | FP8 (re-timing of #21) |
| 2025-05-25_EvenFasterReduce | FP8 |
| 2025-05-25_MuonWithAuxAdamExample | FP8 (example, not a record) |
| 2025-05-30_noallreduce | FP8 |
| 2025-07-12_BosAlign | FP8 |
| 2025-07-13_UpgradeTorch190 | FP8 |
| 2025-07-18_TritonMuon | FP8 |
| 2025-08-23_SparseAttnGate | FP8 |
| 2025-09-03_FA3 | FP8 + FA3 |
| 2025-09-05_SkipMLPBlocks | FP8 + FA3 |
| 2025-09-10_Yarn | FP8 + FA3 |
| 2025-09-11_VectSigmoidBFloat16 | FP8 + FA3 |
| 2025-09-15_AsyncDataLoadAttnFinalWindow | FP8 + FA3 |
| 2025-09-18_Smear | FP8 + FA3 |
| 2025-09-21_DropAttn | FP8 + FA3 |
| 2025-09-23_MuonCustomSizing | FP8 + FA3 |
| 2025-09-27_BF16CE | FP8 + FA3 |
| 2025-09-29_PolarExpress | FP8 + FA3 |
| 2025-09-30_CustomBatching | FP8 + FA3 |
| 2025-10-04_Backout | FP8 + FA3 |
| 2025-10-24_NorMuon | FP8 + FA3 |
| 2025-10-27_FixMuonLR | FP8 + FA3 |
| 2025-10-31_AdamSyncGradientHook | FP8 + FA3 |
| 2025-11-10_CautiousWD | FP8 + FA3 |
| 2025-11-18_RefineSkip | FP8 + FA3 |
| 2025-11-29_BatchSizeSchedule | FP8 + FA3 |
| 2025-12-10_SALambdaOnWeights | FP8 + FA3 |
| 2025-12-11_NorMuonOptimsAndFixes | FP8 + FA3 |
| 2025-12-14_PartialKeyOffset | FP8 + FA3 |
| 2025-12-18_CautiousWDAdam | FP8 + FA3 |
| 2025-12-19_RetieLMHead | FP8 + FA3 |
| 2025-12-21_SmoothedScalars | no source in dir |
| 2025-12-22_MultiTokenPrediction | FP8 + FA3 |
| 2025-12-26_LogitRescale | FP8 + FA3 |
| 2025-12-29_VeSkipGates | FP8 + FA3 |
| 2025-12-31_GatesToCompiledAdam | no source in dir |
| 2026-01-04_MixedPrecisionInterweavedOptimizer | FP8 + FA3 |
| 2026-01-07_PairedHeadAttention | FP8 + FA3 |
| 2026-01-10_FusedLinearReLUSquare | FP8 + FA3 |
| 2026-01-16_FusedSoftcappedEntropy | FP8 + FA3 |
| 2026-01-18_UnifiedOptimizers | no source in dir |
| 2026-01-19 .. 2026-03-06 (11 dirs with logs) | FP8 + FA3; extracts also fail py_compile on py3.13 — extraction anomaly for the newest log format, moot given the era exclusion |
| 2026-01-30_MimeticValueOutput, 2026-02-16_FlattenForward, 2026-02-28_TransposeCopyBackward, 2026-03-22_VarlenMaxDocs, 2026-04-04_FuseCEFwdAndBwd, 2026-04-08_PairedHeadMuon, 2026-04-22_MuddFormer, 2026-04-29_XSAGatedLayers, 2026-05-19_FP8MLPUpProj, 2026-05-20_BigramsSignTrick, 2026-05-27-MuddGatedAndDC, 2026-06-11_RecursiveFromBest, 2026-07-13_PrefixTokenPrediction, 2026-07-17_FP8DownProjection | no source in dir |
| 2026-05-07_XSAGatedLayers | only a helper .py, not the training script; era-excluded regardless |

## Side experiments (not numbered records; not part of the ladder)

| dir | what it is | verdict |
|---|---|---|
| 2024-10-09_SOAP | SOAP-optimizer run, py, 6000 iters | optional curiosity, A100-clean |
| 2024-10-20_ScaleUp1B | 1.5B-param scale run, 577 min on 8xH100 | out of scope (~75 h here) |
| 2024-10-29_Optimizers | optimizer comparisons, 25.5 min | optional |
| 2024-11-04_50Bruns | 95,367-step long-horizon run, 227 min | out of scope |
| 2024-11-09_Replicateleloykun | replication side-run, 8.0 min | optional |
| 2024-11-10_ScaleShortcuts | 19,073-step scale run, 46 min | out of scope |
| 2024-11-14_QuantizedFP4 | FP4/FP8 quantization experiment | H100-only |
