---
name: kernel-policy-data
description: 从 AscendC KernelBench 910B 已完成工作版本导出经子版本验证、耗时下降严格大于 1% 的瓶颈与优化策略监督训练数据。用于生成、重建、校验或说明 kernel policy JSONL、system_prompt/input/output/ops 文本格式或父版本策略到子版本性能收益的数据链；不采集性能、不运行 reference、不修改算子源码或既有报告。
---

# Kernel Policy Data

从已有父子版本生成正向 policy 数据：

```text
父版本 manifest 身份 + performance + op_host/op_kernel
  -> 父版本 bottleneck + strategy
  -> 对应子版本 performance
  -> latency 下降严格 > 1%
```

## 数据契约

每条 JSONL 记录只包含四个字符串字段：

```json
{"system_prompt":"<固定契约>","input":"<纯文本>","output":"<紧凑 JSON 字符串>","ops":"<OperatorName>_<parent_version>"}
```

- `ops` 完整保留父版本后缀；`_0` 到 `_3` 的策略分别由紧邻的 `_1` 到 `_4` 验证。
- `input` 只拼接 manifest 身份、精简正式性能和父版本完整 HOST/KERNEL 源码。
- `output` 按 `cause_key` 将父版本瓶颈与策略合并成 `strategies[]`；每项同时包含 bottleneck、cause、evidence、strategy、两阶段 reasoning 和 actions。
- `system_prompt` 包含固定任务、key 映射、operation 槽位与输出契约；所有记录内容完全一致，并另存 `kernel_policy_system_prompt.txt` 便于独立加载与审计。

读取 [references/data-contract.md](references/data-contract.md) 了解完整段落、父子门禁和防泄漏规则。

## 导出条件

1. 只把 `_0`、`_1`、`_2`、`_3` 视为父版本；要求对应的紧邻子版本存在。
2. 父子版本均为 `PERFORMANCE_DONE`，当前源码指纹与 `workspace.json` 及 PASS 的 `precision.json` 一致，且性能 latency 均为有限正数。
3. 父版本 `bottleneck.json`、`strategy.json` 存在且结论非空。
4. 父版本 bottleneck/strategy 必须通过现行校验；子版本 implementation 必须逐项关联全部 action，且其 modified_files、父子真实源码 diff、全部 target 文件三者完全一致。
5. 父版本 `op_host/`、`op_kernel/` 源码及 manifest 算子项存在。
6. 使用未四舍五入 latency 计算：

   ```text
   reduction_percent = (parent - child) / parent * 100
   ```

   仅当 `reduction_percent > 1.0` 时导出；等于 1% 必须排除。

性能收益和子版本信息只写入审计文件，不得进入训练正文。旧工作区不满足新门禁时显式排除，不降级校验。

## System prompt、输入与答案

输入固定顺序：

```text
[OPERATOR]
[PERF]
[HOST]
[KERNEL]
```

共享 system prompt 由导出器直接从现行 cause taxonomy、strategy 推导器和 operation slots 生成，只保留源码优先决策顺序、metric evidence 路径、`cause→bottleneck/evidence/strategy/operation`、operation 槽位、action 字段和唯一输出形状；cause rules 与 operation slots 使用“一次表头 + 多行数组”的无损紧凑表示，不重复字段名、不复制第二套映射，也不使用固定 Bound 阈值筛选源码问题。`PERF` 只保留影响证据所需的比例、核数、shape 和容量。

`output` 是紧凑 JSON 字符串，唯一顶层字段为 `strategies`。合并过程不改写原始 reasoning、evidence、target、operation、edits 或 constraints。

## 使用

运行所有 Python 前激活环境：

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-policy-data/scripts/export_policy_data.py \
  --workspace-root kernel_workspace/KernelBench910B \
  --output datasets/kernel_policy_data.jsonl
```

指定父版本：

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-policy-data/scripts/export_policy_data.py \
  --ops ArgmaxOverADimensionCustom_0 \
  --output /tmp/kernel_policy_sample.jsonl
```

导出器同时生成 `kernel_policy_system_prompt.txt` 和 `<output>.audit.json`；审计记录候选父子对、收益、排除原因及数据、prompt、taxonomy、映射和 slots 的 hash。

## 边界

- 不读取或输入 reference。
- 不输入 bottleneck、strategy、子版本源码、precision、implementation、workspace、日志、CSV、child latency 或收益。
- 不运行 profiling、precision、reference，不修改任何算子或报告。
- 不加入 messages/ChatML；最终字段及顺序固定为 `system_prompt/input/output/ops`。
- 训练/验证/测试按去掉版本后缀的算子名分组，但 `ops` 字段始终保留版本。
