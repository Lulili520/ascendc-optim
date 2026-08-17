# Kernel Policy Data Contract

## Record

```json
{
  "ops": "ArgmaxOverADimensionCustom_0",
  "input": "[PERF]\n...\n\n[HOST]\n...\n\n[KERNEL]\n...",
  "output": "{\"policy\":[...]}"
}
```

三个字段都是字符串。`ops` 等于父工作版本目录名，版本后缀不可删除。

## Parent and child

```text
parent _N policy -> child _(N+1) result
```

父版本提供训练输入和答案；子版本只提供对应版本的 latency，用于筛选性能收益。子版本任何内容均不得进入训练正文。

资格条件：

- 父子状态、precision 和源码指纹与当前工作版本一致；
- 父子 performance 报告存在，latency 有限且为正；
- 父 bottleneck/strategy 存在且非空；
- 父 bottleneck/strategy 通过现行校验，子 implementation 按 action_index 完整关联父策略；implementation.modified_files 等于父子真实源码 diff 且覆盖全部 target，可包含 `op_host/op_kernel` 内已记录的必要配套文件；
- 父版本 manifest 项与 op_host/op_kernel 源码存在；
- `(parent_latency-child_latency)/parent_latency*100 > 1.0`。

所有工作版本统一按当前 cause、strategy 和 action 契约校验；不识别或转换历史 key。子版本 bottleneck 只服务下一轮诊断，不进入当前父子样本资格判断；父版本 cause 可以在子版本复检中继续存在。

## Input

固定顺序：

```text
[PERF]
<只保留任务规格、pipeline ratio、GM/L2/冲突、逐核和容量字段；不含绝对 latency 或原始核时间>

[HOST]
--- FILE: op_host/<relative path> ---
<父版本完整文本>

[KERNEL]
--- FILE: op_kernel/<relative path> ---
<父版本完整文本>
```

不使用 reference。输入输出 shape/dtype 直接使用 performance.task 的正式采集值。

Policy knowledge 在导出时校验 cause 固定链和 operation 槽位，但由 skill/训练系统提示提供，不在样本中重复。
完整源码不按答案 target 裁剪，避免泄漏；性能报告删除原始时间、FLOPS 和重复带宽等不参与决策的字段。超长样本应在审计中排除，不静默截断源码。

## Output

```json
{"policy":[{"bottleneck":"...","cause":"...","evidence":["source|fact"],"strategy":"...","reasoning":"...","changes":[{"target":"...","edits":["slot/object：current改为target"],"constraints":["..."]}]}]}
```

`policy` 按顺序包含 bottleneck 的全部 issues 及其一一对应 strategies。每个 strategy 可包含完成自身 cause 必需的多个原子 changes，但不得混入其他 cause；整条子版本收益归因于该有序策略集合。

## Audit only

以下字段只进入 `<output>.audit.json`：

- `child_ops`
- parent/child latency
- `reduction_percent`
- `eligible` 与 `exclusion_reason`
- `result_kind`：`verified_policy` 或 `verified_execution_not_training`
- policy knowledge/hash、input/output 字符数

## Leakage and splitting

- `_N` 输入禁止使用 `_(N+1)` 源码。
- 不输入 precision、implementation、workspace、日志、CSV 或收益结果。
- 同一算子的 `_0`、`_1` 按算子名分到同一 train/valid/test 集；最终 `ops` 仍保留后缀。
