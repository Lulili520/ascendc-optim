---
name: kernel-precision
description: 编译、部署并验证单个 AscendC KernelBench 910B 算子与 PyTorch reference 的精度。用户要求修改 op_host、op_kernel 后重新验证，比较 NPU 算子与 PyTorch 输出，排查精度误差，或确认 AscendC 算子正确性时使用。
---

# Kernel Precision

只验证一个工作版本；reference 和 `CppExtension` 是只读依赖：

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-precision/scripts/validate_precision.py \
  <OperatorName> --project-dir <OperatorName_version>
```

入口构建 Host/Kernel、安装独立 vendor，并用同一组原始输入运行 reference 和自定义算子。只验证一个原始 shape，输出转 FP32，使用 `atol=1e-2, rtol=1e-2`。

只有退出码 0、输出 `precision=PASS`、`precision.json` 与 `workspace.json` 的 PASS/指纹一致才能采集性能。构建问题记 `BUILD_FAILED`，运行/reference/数值问题记 `PRECISION_FAILED`，失败也必须保存阶段和日志。

默认清理可再生成构建物并保留 `CppExtension/*.so`；只有当前源码已成功构建时才使用 `--skip-build`。
