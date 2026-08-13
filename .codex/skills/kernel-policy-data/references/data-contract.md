# Kernel Policy Data Contract

## Record

```json
{
  "ops": "ArgmaxOverADimensionCustom_0",
  "input": "[TASK_INSTRUCT]\n...",
  "output": "[BOTTLENECK_JSON]\n..."
}
```

三个字段都是字符串。`ops` 等于父工作版本目录名，版本后缀不可删除。

## Parent and child

```text
parent _0 policy -> child _1 result
parent _1 policy -> child _2 result
```

父版本提供训练输入和答案；子版本只提供对应版本的 latency，用于筛选性能收益。子版本任何内容均不得进入训练正文。

资格条件：

- 父子状态、precision 和源码指纹与当前工作版本一致；
- 父子 performance 报告存在，latency 有限且为正；
- 父 bottleneck/strategy 存在且非空；
- 父 bottleneck/strategy 通过现行校验，子 implementation 按 action_index 完整关联父策略；implementation.modified_files、父子真实源码 diff、全部 target 文件三者完全一致；
- 父版本 manifest 项与 op_host/op_kernel 源码存在；
- `(parent_latency-child_latency)/parent_latency*100 > 1.0`。

旧工作区不满足上述门禁时显式排除，不为兼容性降级校验。

## Input

固定顺序：

```text
[TASK_INSTRUCT]
根据正式性能、源码和固定 policy，输出唯一瓶颈、原因及可实施策略。

[POLICY_KNOWLEDGE_JSON]
<紧凑 policy-knowledge.json；所有样本完全相同>

[OPERATOR_JSON]
{
  "operator": "<manifest key>",
  "level": "<manifest level>",
  "function": "<manifest function>",
  "parameters": ["<manifest parameters>"]
}

[PERFORMANCE_JSON]
<只保留任务规格、pipeline ratio、GM/L2/冲突、逐核和容量字段；不含绝对 latency 或原始核时间>

[SOURCE_SYMBOLS]
<按文件和源码顺序列出全部 namespace/class 完整限定 symbol；不突出答案 target>

[OP_HOST_SOURCE]
--- FILE: op_host/<relative path> ---
<父版本完整文本>

[OP_KERNEL_SOURCE]
--- FILE: op_kernel/<relative path> ---
<父版本完整文本>

[OUTPUT_FORMAT]
<references/output-format.txt>
```

不使用 reference。输入输出 shape/dtype 直接使用 performance.task 的正式采集值。

Policy knowledge 必须列出完整决策顺序、阈值、metric evidence 路径、cause 固定链和 operation 槽位，禁止按当前答案裁剪；省略导出器已校验但训练无需重复学习的长解释。使用无缩进紧凑 JSON，且不得包含当前样本答案。
完整源码不按答案 target 裁剪，避免泄漏；性能报告删除原始时间、FLOPS 和重复带宽等不参与决策的字段。超长样本应在审计中排除，不静默截断源码。

## Output

```text
[BOTTLENECK_JSON]
<父版本完整 bottleneck.json>

[STRATEGY_JSON]
<父版本完整 strategy.json>
```

段标签仅定位两个原始 JSON，不创建新的合并 schema。允许统一缩进，禁止增删或改写字段值。

## Audit only

以下字段只进入 `<output>.audit.json`：

- `child_ops`
- parent/child latency
- `reduction_percent`
- `eligible` 与 `exclusion_reason`
- policy knowledge/hash、input/output 字符数

## Leakage and splitting

- `_0` 输入禁止使用 `_1` 源码；`_1` 输入禁止使用 `_2` 源码。
- 不输入 precision、implementation、workspace、日志、CSV 或收益结果。
- 同一算子的 `_0`、`_1` 按算子名分到同一 train/valid/test 集；最终 `ops` 仍保留后缀。
