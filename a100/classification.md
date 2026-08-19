# Record classification, track_1_short — DRAFT SKELETON

Status column is a name-based guess only. Step 1 of the A100 track is
reading each script and replacing the guess with a verified reason.

| record | script | draft status (VERIFY) | verified reason |
|---|---|---|---|
| 2024-06-06_AdamW | NO SCRIPT | expect A100-OK |  |
| 2024-10-09_SOAP | yes | expect A100-OK |  |
| 2024-10-10_Muon | yes | expect A100-OK |  |
| 2024-10-13_llmc | NO SCRIPT | expect A100-OK |  |
| 2024-10-14_ModernArch | yes | expect A100-OK |  |
| 2024-10-17_DistributedMuon | NO SCRIPT | expect A100-OK |  |
| 2024-10-18_PyTorch25 | NO SCRIPT | expect A100-OK |  |
| 2024-10-20_ScaleUp1B | NO SCRIPT | expect A100-OK |  |
| 2024-10-29_Optimizers | NO SCRIPT | expect A100-OK |  |
| 2024-11-03_UntieEmbed | NO SCRIPT | expect A100-OK |  |
| 2024-11-04_50Bruns | NO SCRIPT | expect A100-OK |  |
| 2024-11-06_ShortcutsTweaks | NO SCRIPT | expect A100-OK |  |
| 2024-11-08_CastBf16 | NO SCRIPT | expect A100-OK |  |
| 2024-11-09_Replicateleloykun | NO SCRIPT | expect A100-OK |  |
| 2024-11-10_ScaleShortcuts | NO SCRIPT | expect A100-OK |  |
| 2024-11-10_UNetDoubleLr | NO SCRIPT | expect A100-OK |  |
| 2024-11-14_QuantizedFP4 | NO SCRIPT | H100-only (fp8/fp4 in name) |  |
| 2024-11-19_FlexAttention | NO SCRIPT | expect A100-OK |  |
| 2024-11-24_WindowWarmup | NO SCRIPT | expect A100-OK |  |
| 2024-12-04_ValueEmbed | yes | expect A100-OK |  |
| 2024-12-08_UNetValueEmbedsTweaks | NO SCRIPT | expect A100-OK |  |
| 2024-12-10_MFUTweaks | NO SCRIPT | expect A100-OK |  |
| 2024-12-17_SparsifyEmbeds | NO SCRIPT | expect A100-OK |  |
| 2025-01-04_SoftCap | NO SCRIPT | expect A100-OK |  |
| 2025-01-13_Fp8LmHead | NO SCRIPT | H100-only (fp8/fp4 in name) |  |
| 2025-01-16_Sub3Min | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-01-26_BatchSize | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-02-01_RuleTweak | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-05-09_SkipMLPBlocks | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-05-24_FasterReduce | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-05-24_StableTorch | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-05-25_EvenFasterReduce | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-05-25_MuonWithAuxAdamExample | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-05-30_noallreduce | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-07-12_BosAlign | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-07-13_UpgradeTorch190 | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-07-18_TritonMuon | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-08-23_SparseAttnGate | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-03_FA3 | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-05_SkipMLPBlocks | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-10_Yarn | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-11_VectSigmoidBFloat16 | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-15_AsyncDataLoadAttnFinalWindow | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-18_Smear | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-21_DropAttn | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-23_MuonCustomSizing | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-27_BF16CE | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-29_PolarExpress | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-09-30_CustomBatching | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-10-04_Backout | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-10-24_NorMuon | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-10-27_FixMuonLR | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-10-31_AdamSyncGradientHook | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-11-10_CautiousWD | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-11-18_RefineSkip | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-11-29_BatchSizeSchedule | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-10_SALambdaOnWeights | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-11_NorMuonOptimsAndFixes | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-14_PartialKeyOffset | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-18_CautiousWDAdam | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-19_RetieLMHead | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-21_SmoothedScalars | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-22_MultiTokenPrediction | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-26_LogitRescale | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-29_VeSkipGates | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2025-12-31_GatesToCompiledAdam | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-04_MixedPrecisionInterweavedOptimizer | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-07_PairedHeadAttention | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-10_FusedLinearReLUSquare | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-16_FusedSoftcappedEntropy | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-18_UnifiedOptimizers | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-19_BigramHashEmbedding | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-24_ImprovedLMHead | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-26-UntieValueEmbeddings | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-30_MimeticValueOutput | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-30_VeFused | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-01-31-BigramHashH2D | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-02-02_KernelTuning | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-02-03_VeTuned | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-02-06_SparseBigramGradient | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-02-10_ShortWindow | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-02-12_ParallelResiduals | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-02-16_FlattenForward | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-02-23_CrossEntropyKernel | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-02-28_TransposeCopyBackward | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-03-06_SimplifyHC | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-03-22_VarlenMaxDocs | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-04-04_FuseCEFwdAndBwd | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-04-08_PairedHeadMuon | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-04-22_MuddFormer | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-04-29_XSAGatedLayers | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-05-07_XSAGatedLayers | yes | check: post-FP8 era, likely H100-only parts |  |
| 2026-05-19_FP8MLPUpProj | NO SCRIPT | H100-only (fp8/fp4 in name) |  |
| 2026-05-20_BigramsSignTrick | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-05-27-MuddGatedAndDC | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-06-11_RecursiveFromBest | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-07-13_PrefixTokenPrediction | NO SCRIPT | check: post-FP8 era, likely H100-only parts |  |
| 2026-07-17_FP8DownProjection | NO SCRIPT | H100-only (fp8/fp4 in name) |  |
