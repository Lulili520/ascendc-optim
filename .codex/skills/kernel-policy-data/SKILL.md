---
name: kernel-policy-data
description: 从 AscendC KernelBench 910B 已完成工作版本导出经子版本验证、耗时下降严格大于 1% 的瓶颈与优化策略监督训练数据。用于生成、重建、校验或说明 kernel policy JSONL、ops/input/output 文本格式、policy key 知识或父版本策略到子版本性能收益的数据链；不采集性能、不运行 reference、不修改算子源码或既有报告。
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

每条 JSONL 记录只包含三个字符串字段：

```json
{"ops":"<OperatorName>_<parent_version>","input":"<纯文本>","output":"<纯文本>"}
```

- `ops` 完整保留父版本后缀；`_0` 策略由 `_1` 验证，`_1` 策略由 `_2` 验证。
- `input` 拼接短指令、紧凑 policy 契约、精简性能字段、父版本完整源码和短输出格式；不按答案 target 裁剪源码。
- `output` 依次拼接父版本原始 `bottleneck.json` 与 `strategy.json`。

读取 [references/data-contract.md](references/data-contract.md) 了解完整段落、父子门禁和防泄漏规则。

## 导出条件

1. 只把 `_0`、`_1` 视为父版本；要求对应 `_1`、`_2` 子版本存在。
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

## 输入输出

输入固定顺序：

```text
[TASK_INSTRUCT]
[POLICY_KNOWLEDGE_JSON]
[OPERATOR_JSON]
[PERFORMANCE_JSON]
[SOURCE_SYMBOLS]
[OP_HOST_SOURCE]
[OP_KERNEL_SOURCE]
[OUTPUT_FORMAT]
```

`POLICY_KNOWLEDGE_JSON` 使用无缩进训练投影，只保留决策顺序、阈值、metric evidence 路径、`cause→bottleneck/evidence/strategy/operation` 和 operation 槽位；完整知识只在导出时校验，不重复塞入样本。`PERFORMANCE_JSON` 只保留诊断所需的比例、核数、shape 和容量。`SOURCE_SYMBOLS` 使用完整 namespace/class 限定名列出全部源码 symbol，不标记答案 target；随后仍输入父版本完整源码。

输出固定顺序：

```text
[BOTTLENECK_JSON]
[STRATEGY_JSON]
```

保留父版本已校验 JSON，不改写 reasoning、evidence、target、operation、edits 或 constraints。

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

导出器同时生成 `<output>.audit.json`，记录所有候选父子对的 latency、下降比例和排除原因。

## 边界

- 不读取或输入 reference。
- 不输入 bottleneck、strategy、子版本源码、precision、implementation、workspace、日志、CSV、child latency 或收益。
- 不运行 profiling、precision、reference，不修改任何算子或报告。
- 不加入 system/messages/ChatML；最终数据保持 `ops/input/output`。
- 训练/验证/测试按去掉版本后缀的算子名分组，但 `ops` 字段始终保留版本。
