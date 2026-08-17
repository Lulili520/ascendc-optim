---
name: kernel-policy-data
description: 从 AscendC KernelBench 910B 已完成工作版本导出经子版本验证、耗时下降严格大于 1% 的瓶颈与优化策略监督训练数据。用于生成、重建、校验或说明 kernel policy JSONL、ops/input/output 文本格式、policy key 知识或父版本策略到子版本性能收益的数据链；不采集性能、不运行 reference、不修改算子源码或既有报告。
---

# Kernel Policy Data

完整阅读 [data-contract.md](references/data-contract.md)，再从已有父子版本导出：

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-policy-data/scripts/export_policy_data.py \
  --workspace-root kernel_workspace/KernelBench910B \
  --output datasets/kernel_policy_data.jsonl
```

每条 JSONL 只有字符串字段 `ops/input/output`：input 为父版本 `[PERF][HOST][KERNEL]`，output 为全部有序 issues 和对应 strategies 合并的紧凑 `policy[]`。子版本内容、latency 和收益只用于筛选，不进入训练正文。

导出器校验父子 `PERFORMANCE_DONE`、precision/源码指纹、父 bottleneck/strategy、子 implementation 与真实 diff。子版本 bottleneck 用于后续轮次，不作为当前父子样本门禁，也不要求父版本 cause 消失。只有未四舍五入 latency 下降严格大于 1% 时输出训练样本；其他完整执行保留在 `.audit.json` 中且不污染 OPD 正向数据。

不读取 reference，不运行 precision/profiling，不修改任何工作区报告。数据集切分必须按去掉版本后缀的算子名分组。
