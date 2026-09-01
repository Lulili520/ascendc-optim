---
name: kernel-precision
description: 编译、部署并验证单个 AscendC KernelBench 910B 算子与 PyTorch reference 的精度。用户要求修改 op_host、op_kernel 后重新验证，比较 NPU 算子与 PyTorch 输出，排查精度误差，或确认 AscendC 算子正确性时使用。
---

# Kernel Precision

只验证一个工作版本；reference 和 `CppExtension` 是只读依赖：

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-precision/scripts/validate_precision.py \
  <OperatorName> --suite <KernelBench910B|Attention910B|MHC910B> \
  --project-dir <OperatorName_version>
```

`--suite` 默认 `KernelBench910B`。Attention/MHC 使用各自唯一映射的本地 PyTorch reference；单输出与 Tensor 序列输出均逐 Tensor 比较并汇总误差。

Attention/MHC 中若导出的 Kernel 只替换整模型内部子图，必须先在
`scripts/subgraph_adapters.py` 定义确定的 C++ arguments 与同边界 expected。
显式适配值优先于通用参数名/别名匹配；适配器只能从固定原始输入、当前
model state 和确定的 reshape/permute/模块计算得到，禁止补随机 Tensor。
已知子图缺少适配器时必须在 OPP 构建前写 `ADAPTER_MISSING`，不得误记为
`PRECISION_FAILED`，也不得进入性能或策略阶段。

入口构建 Host/Kernel、安装独立 vendor，并用同一组原始输入运行 reference 和自定义算子。队列调度时精度与性能共用唯一设备验证槽，不与任何其他 NPU 验证进程并发；首次无 expected 缓存时先执行自定义 Kernel smoke，成功后才运行 reference。只验证一个原始 shape，输出转 FP32，使用 `atol=1e-2, rtol=1e-2`。同一算子的 reference 源码、初始化参数、Torch 版本和 seed 未变化时，在临时目录复用固定输入与 expected；Kernel 源码变化只重跑自定义算子。缓存不进入训练数据，算子队列终止时清理。
缓存使用临时文件原子发布；写入失败或进程异常时立即删除未完成的 `.pt.tmp`，禁止让部分 reference 缓存持续占用磁盘。
落盘前按输入、expected 和 model state 的原始字节数检查可用空间，并预留 512 MiB；空间不足时跳过缓存落盘，但继续当前精度比较。

只有退出码 0、输出 `precision=PASS`、`precision.json` 与 `workspace.json` 的 PASS/指纹一致才能采集性能。构建问题记 `BUILD_FAILED`，运行/reference/数值问题记 `PRECISION_FAILED`，失败也必须保存阶段和日志。
成功报告同时保存实际扩展参数的名称、shape、dtype；策略阶段优先复用这些元数据。临时目录空间不足、设备忙或临时资源不可用记 `ENVIRONMENT_FAILED`，只允许环境重试，禁止触发策略重规划。

默认清理可再生成构建物并保留 `CppExtension/*.so`；只有当前源码已成功构建时才使用 `--skip-build`。
