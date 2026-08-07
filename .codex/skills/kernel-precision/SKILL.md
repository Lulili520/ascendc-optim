---
name: kernel-precision
description: 编译、部署并验证单个 AscendC KernelBench 910B 算子与 PyTorch reference 的精度。用户要求修改 op_host、op_kernel 后重新验证，比较 NPU 算子与 PyTorch 输出，排查精度误差，或确认 AscendC 算子正确性时使用。
---

# AscendC 单算子精度验证

只验证工作区中的一个算子版本。reference 和 `CppExtension` 是只读依赖；普通优化只修改 `op_host/`、`op_kernel/`。

## 执行

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-precision/scripts/validate_precision.py \
  <OperatorName> --project-dir \
  kernel_workspace/KernelBench910B/<level>/<OperatorName>_<version>
```

入口依次构建 Host/Kernel、安装独立 vendor、按需编译扩展，并用同一组原始输入执行 reference 和自定义算子。只使用一个原始 shape；输出转 FP32 统计误差，固定使用 `atol=1e-2, rtol=1e-2`。

本轮已成功构建且源码未变时可加 `--skip-build`。仅为排查构建问题时加 `--keep-artifacts`。

## 成功条件

以下条件必须同时满足：

- 命令退出码为 0；
- 输出明确包含 `precision=PASS`；
- `precision/precision.json` 记录 PASS、shape、dtype、误差和当前源码指纹；
- `workspace.json` 状态为 `PRECISION_PASS`。

否则不得继续性能采集。OPP/Kernel/Host 或扩展构建失败记录为 `BUILD_FAILED`，算子运行、reference 或数值比较失败记录为 `PRECISION_FAILED`。失败也必须生成 `precision/precision.json` 和阶段日志，保存 `failure_stage`、详细 `stage`、退出码、原因及当前源码指纹。

## 清理

默认删除 `build_out/`、`CppExtension/build/` 和本轮 Python 缓存，保留 `CppExtension/*.so` 与已安装 vendor。禁止修改 reference。
