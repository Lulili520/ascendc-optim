# AscendC 910B 算子最佳性能

更新时间：2026-09-01（UTC）

- 保留算子：83
- 已获得超过 1% 有效提升：83
- 正式策略训练数据：140 条
- KernelBench910B：45 个算子 / 73 条数据
- Attention910B：29 个算子 / 53 条数据
- MHC910B：9 个算子 / 14 条数据

| Suite | 算子 | 初始 Task Duration (us) | 最佳版本 | 最佳 Task Duration (us) | 总降幅 |
|---|---|---:|---:|---:|---:|
| KernelBench910B | ArgmaxOverADimensionCustom | 2,290,693.972 | `_2` | 9,653.506 | 99.58% |
| KernelBench910B | ArgminOverADimensionCustom | 2,571,796.735 | `_1` | 8,029.181 | 99.69% |
| KernelBench910B | AveragePooling1dCustom | 342,410.018 | `_1` | 6,124.044 | 98.21% |
| KernelBench910B | AveragePooling2dCustom | 819,272.491 | `_4` | 16,781.751 | 97.95% |
| KernelBench910B | AveragePooling3dCustom | 2,914,752.823 | `_3` | 28,944.117 | 99.01% |
| KernelBench910B | BatchedMatrixMultiplicationCustom | 5,544.670 | `_1` | 4,227.849 | 23.75% |
| KernelBench910B | CrossEntropyLossCustom | 45,670.866 | `_1` | 37,067.782 | 18.84% |
| KernelBench910B | CumprodCustom | 702,430.925 | `_2` | 101,363.672 | 85.57% |
| KernelBench910B | CumsumCustom | 466,117.107 | `_3` | 66,516.699 | 85.73% |
| KernelBench910B | CumsumExclusiveCustom | 444,852.932 | `_3` | 109,004.017 | 75.50% |
| KernelBench910B | CumsumReverseCustom | 181,129.701 | `_3` | 101,719.167 | 43.84% |
| KernelBench910B | EluCustom | 10,555.922 | `_1` | 10,151.306 | 3.83% |
| KernelBench910B | FrobeniusNormCustom | 334,270.481 | `_3` | 249,855.469 | 25.25% |
| KernelBench910B | GeluCustom | 15,020.460 | `_1` | 10,931.797 | 27.22% |
| KernelBench910B | GroupNormCustom | 6,074,793.270 | `_3` | 18,086.783 | 99.70% |
| KernelBench910B | InstanceNormCustom | 18,610.828 | `_3` | 12,318.432 | 33.81% |
| KernelBench910B | L2NormCustom | 36,482.291 | `_2` | 14,776.050 | 59.50% |
| KernelBench910B | LayerNormCustom | 914.102 | `_2` | 760.791 | 16.77% |
| KernelBench910B | LeakyReluCustom | 10,624.865 | `_1` | 10,378.515 | 2.32% |
| KernelBench910B | LogSoftmaxCustom | 14,458.111 | `_2` | 10,336.693 | 28.51% |
| KernelBench910B | MaskedCumsumCustom | 178,451.954 | `_1` | 99,422.955 | 44.29% |
| KernelBench910B | MatmulForLowerTriangularMatricesCustom | 37,573.849 | `_1` | 1,661.686 | 95.58% |
| KernelBench910B | MatmulForSymmetricMatricesCustom | 37,424.213 | `_1` | 4,725.369 | 87.37% |
| KernelBench910B | MatmulForUpperTriangularMatricesCustom | 37,528.130 | `_1` | 3,579.103 | 90.46% |
| KernelBench910B | MatmulWithIrregularShapesCustom | 169,638.530 | `_1` | 15,266.550 | 91.00% |
| KernelBench910B | MatmulWithTransposedACustom | 37,339.534 | `_1` | 1,694.308 | 95.46% |
| KernelBench910B | MatmulWithTransposedBothCustom | 37,426.272 | `_1` | 18,713.249 | 50.00% |
| KernelBench910B | MatrixScalarMultiplicationCustom | 7,173.767 | `_1` | 6,775.271 | 5.55% |
| KernelBench910B | MatrixVectorMultiplicationCustom | 1,392,011.208 | `_1` | 231,947.473 | 83.34% |
| KernelBench910B | MaxPooling1dCustom | 365,815.751 | `_4` | 8,164.247 | 97.77% |
| KernelBench910B | MaxPooling2dCustom | 742,807.277 | `_4` | 14,130.865 | 98.10% |
| KernelBench910B | MaxPooling3dCustom | 1,079,452.932 | `_4` | 17,986.159 | 98.33% |
| KernelBench910B | MaxReductionOverADimensionCustom | 785,975.896 | `_1` | 6,071.443 | 99.23% |
| KernelBench910B | MeanReductionOverADimensionCustom | 2,121,735.188 | `_2` | 6,054.922 | 99.71% |
| KernelBench910B | MinGptNewGeluCustom | 650.167 | `_1` | 503.200 | 22.60% |
| KernelBench910B | MinReductionOverADimensionCustom | 6,351.174 | `_2` | 6,052.882 | 4.70% |
| KernelBench910B | RmsNormCustom | 1,587,314.406 | `_2` | 17,787.791 | 98.88% |
| KernelBench910B | SeluCustom | 10,569.069 | `_1` | 10,130.926 | 4.15% |
| KernelBench910B | SigmoidCustom | 10,597.968 | `_1` | 10,070.503 | 4.98% |
| KernelBench910B | SoftmaxCustom | 15,180.317 | `_3` | 10,456.759 | 31.12% |
| KernelBench910B | SoftplusCustom | 10,175.216 | `_1` | 10,072.983 | 1.00% |
| KernelBench910B | SquareMatrixMultiplicationCustom | 37,420.413 | `_1` | 3,213.289 | 91.41% |
| KernelBench910B | StandardMatrixMultiplicationCustom | 37,409.992 | `_1` | 2,455.338 | 93.44% |
| KernelBench910B | SumReductionOverADimensionCustom | 2,171,381.092 | `_1` | 6,054.622 | 99.72% |
| KernelBench910B | TallSkinnyMatrixMultiplicationCustom | 28,306.194 | `_1` | 5,248.510 | 81.46% |
| Attention910B | AxialAttentionCustom | 15,534.601 | `_1` | 1,144.286 | 92.63% |
| Attention910B | BAMCustom | 48.382 | `_1` | 33.062 | 31.66% |
| Attention910B | BlockSparseAttentionCustom | 20,134.165 | `_1` | 9,888.975 | 50.88% |
| Attention910B | CBAMBlockCustom | 2,101.704 | `_2` | 377.995 | 82.01% |
| Attention910B | CoTAttentionCustom | 2,291.632 | `_3` | 400.516 | 82.52% |
| Attention910B | DAModuleCustom | 17,150.605 | `_4` | 121.025 | 99.29% |
| Attention910B | DenseSparseAttentionCustom | 463.419 | `_1` | 240.729 | 48.05% |
| Attention910B | ECAAttentionCustom | 3,016.661 | `_4` | 212.189 | 92.97% |
| Attention910B | GCModuleCustom | 2,082.043 | `_1` | 106.324 | 94.89% |
| Attention910B | GCTCustom | 2,908.736 | `_1` | 260.371 | 91.05% |
| Attention910B | GatedChannelTransformCustom | 2,751.830 | `_1` | 257.551 | 90.64% |
| Attention910B | GlobalFilterCustom | 8,805.452 | `_3` | 247.530 | 97.19% |
| Attention910B | HaloAttentionCustom | 122,422.615 | `_2` | 3,116.145 | 97.45% |
| Attention910B | LCTCustom | 2,998.020 | `_2` | 1,249.590 | 58.32% |
| Attention910B | LocalAttentionCustom | 156,365.831 | `_3` | 19,186.787 | 87.73% |
| Attention910B | LongformerAttentionCustom | 206,840.230 | `_1` | 15,464.798 | 92.52% |
| Attention910B | OutlookAttentionCustom | 282,576.458 | `_3` | 2,755.730 | 99.02% |
| Attention910B | PagedAttentionKVCacheCustom | 11,280.451 | `_3` | 261.650 | 97.68% |
| Attention910B | ParNetAttentionCustom | 109.444 | `_1` | 30.302 | 72.31% |
| Attention910B | ParallelPolarizedSelfAttentionCustom | 11,719.689 | `_2` | 420.377 | 96.41% |
| Attention910B | ResidualAttentionCustom | 642.946 | `_3` | 108.164 | 83.18% |
| Attention910B | S2AttentionCustom | 173.647 | `_2` | 34.142 | 80.34% |
| Attention910B | SEAttentionCustom | 5,213.508 | `_3` | 1,630.085 | 68.73% |
| Attention910B | SKAttentionCustom | 486,137.736 | `_4` | 299.332 | 99.94% |
| Attention910B | SRMCustom | 3,085.384 | `_1` | 153.246 | 95.03% |
| Attention910B | SequentialPolarizedSelfAttentionCustom | 9,190.587 | `_3` | 96.224 | 98.95% |
| Attention910B | ShuffleAttentionCustom | 34,660.985 | `_2` | 109.465 | 99.68% |
| Attention910B | SpatialGroupEnhanceCustom | 3,747.890 | `_3` | 236.310 | 93.69% |
| Attention910B | UnPermuteCustom | 9,889.155 | `_1` | 83.083 | 99.16% |
| MHC910B | FusedMhcKernelsCustom | 22,582.243 | `_2` | 1,699.988 | 92.47% |
| MHC910B | MhcBlockBottleneck2dCustom | 123.485 | `_2` | 52.002 | 57.89% |
| MHC910B | MhcPostBlockCustom | 306.932 | `_2` | 139.986 | 54.39% |
| MHC910B | MhcPreBlockCustom | 117,212.766 | `_2` | 860.275 | 99.27% |
| MHC910B | MhcProjectorCustom | 3,922.077 | `_1` | 2,704.808 | 31.04% |
| MHC910B | MhcUpdateCustom | 35.062 | `_2` | 27.681 | 21.05% |
| MHC910B | StaticMhcHyperConnectionsCustom | 2,090.003 | `_1` | 115.164 | 94.49% |
| MHC910B | StreamMixCustom | 30.341 | `_2` | 22.981 | 24.26% |
| MHC910B | StreamWriteCustom | 31.461 | `_2` | 20.161 | 35.92% |

表中初始性能取同算子 `_0` 的有效 `Task Duration(us)`；最佳性能取正式 `kernel_workspace` 中精度 PASS、源码指纹匹配且存在有效 Task Duration 的最低时延版本。总降幅按 `_0` 到该最低时延计算；未通过后续 5% 迭代门禁的版本若仍是已验证最低时延，依然在本表如实展示，但不作为后续优化父版本。`kernel_eval_workspace` 的模型评测结果不计入本表。
