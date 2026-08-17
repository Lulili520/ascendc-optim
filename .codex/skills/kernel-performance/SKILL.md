---
name: kernel-performance
description: 采集单个 AscendC KernelBench 910B 自定义算子的完整 NPU 性能指标并生成客观报告。用户要求 msprof 上板采集、算子硬件性能数据、流水线或存储指标、逐核负载时使用；不运行 PyTorch reference，不计算加速比，不在采集脚本中自动判定瓶颈。
---

# Kernel Performance

只处理当前源码已通过精度的单个工作版本：

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-performance/scripts/collect_performance.py \
  <OperatorName> --device 0 --project-dir <OperatorName_version>
```

入口校验 precision/源码/vendor，探测当前硬件，预热一次，再用独立 `msprof` 进程采集七组正式指标和逐核 cycle。任一组、正数 `Task Duration(us)` 或逐核 cycle 缺失时记 `PERFORMANCE_FAILED`。

结果固定更新 `performance/`；latency 只取 `PipeUtilization`。`performance.json` 保存客观任务、流水、存储、冲突、逐核和精简硬件数据，不输出 Bound 或建议。无法可靠取得的硬件值保持 `null`，禁止按芯片名称猜测。

默认清理中间文件；仅排错时使用 `--keep-intermediates`。
