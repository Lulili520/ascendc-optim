---
name: kernel-performance
description: 测量单个 AscendC KernelBench 910B 工作版本的正式 Task Duration。要求当前源码精度 PASS；在 msprof 外预热同一自定义算子一次，再用一次 PipeUtilization 采集 Task Duration，并相对历史最佳执行 5% 收益门禁。不采其他 metrics、逐核 cycle 或瓶颈信息。
---

# Kernel Task Duration

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-performance/scripts/collect_performance.py \
  <OperatorName> --suite <KernelBench910B|Attention910B|MHC910B> \
  --device 0 --project-dir <OperatorName_version>
```

`--suite` 默认 `KernelBench910B`；不同 suite 的工作版本和历史最佳不得交叉比较。

1. 校验当前源码对应的 precision PASS 和已安装 vendor。
   队列调度时与所有精度/性能命令共用唯一设备验证槽，采集全程不得存在另一 NPU Kernel 验证进程。
2. 在 `msprof` 外执行同一 shape/dtype 的自定义算子一次并同步。
3. 启动一个 `msprof`，显式使用 `--export=on`，只采 `PipeUtilization`，执行算子一次，只读取 `Task Duration(us)`。
4. 原样保存产生该值的 `performance/op_summary_PipeUtilization.csv`，并写 `performance/latency.json`。基线直接完成；子版本相对历史最佳严格下降超过 5% 才置 `PERFORMANCE_DONE`，否则置 `PERFORMANCE_SCREENED_NO_IMPROVEMENT` 并停止算子。

无论成功或失败，都保留 `performance/msprof.log`；CSV 缺失时必须由该日志区分 msprof 执行失败、导出失败和目标任务缺失。

不运行 reference、不采七组指标、不采逐核 cycle、不根据性能生成策略。
