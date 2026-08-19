# Kernel Policy Data Contract

## Record

```json
{
  "system_prompt": "你是 AscendC Kernel 性能策略教师……",
  "input": "[OPERATOR]\n...\n\n[PERF]\n...\n\n[HOST]\n...\n\n[KERNEL]\n...",
  "output": "{\"strategies\":[...]}",
  "ops": "ArgmaxOverADimensionCustom_0"
}
```

四个字段都是字符串。`ops` 等于父工作版本目录名，版本后缀不可删除。固定任务说明和 key 契约直接写入每条记录的 `system_prompt`，并在同目录保留内容完全相同的 `kernel_policy_system_prompt.txt` 供独立加载与审计。

## Parent and child

```text
parent _0 policy -> child _1 result
parent _1 policy -> child _2 result
parent _2 policy -> child _3 result
parent _3 policy -> child _4 result
```

父版本提供训练输入和答案；子版本只提供 latency 以筛选正收益。资格条件：

- 父子状态、precision 和源码指纹与当前版本一致；
- 父子 performance latency 有限且为正；
- 父 bottleneck/strategy 非空并通过现行校验；
- 子 implementation 完整关联父 actions，且声明文件、真实 diff、action targets 完全一致；
- manifest 项与父版本完整 HOST/KERNEL 源码存在；
- `(parent_latency-child_latency)/parent_latency*100 > 1.0`。

旧工作区不满足门禁时显式排除，不做兼容性降级。

## Shared system prompt

导出器从以下 canonical source 动态生成共享 prompt：

- bottleneck cause taxonomy/validator；
- strategy 的 `cause→strategy→operation` 推导器；
- operation slots；
- policy knowledge 中的决策顺序与 action 字段。

Prompt 定义任务、最多 3 个问题、最多 6 个 actions、唯一输出形状和全部固定映射。`cause_rules` 与 `operation_slots` 分别使用一次 `*_fields` 表头和等长行数组，消除重复字段名但不删除任何映射或槽位。每条样本直接携带完全相同的 prompt，审计保存 prompt 与 canonical source 的 SHA-256，防止训练契约静默漂移。

## Input

固定顺序：

```text
[OPERATOR]
<operator、level、function、parameters 的紧凑 JSON>

[PERF]
<任务规格、pipeline ratio、GM/L2/冲突、逐核和可靠容量；不含绝对 latency>

[HOST]
--- FILE: op_host/<relative path> ---
<父版本完整文本>

[KERNEL]
--- FILE: op_kernel/<relative path> ---
<父版本完整文本>
```

不输入 reference、答案提示 symbol、固定知识、输出模板、子版本信息或收益结果。源码不按答案 target 裁剪。

## Output

`output` 是可直接解析的紧凑 JSON 字符串：

```json
{
  "strategies": [
    {
      "bottleneck_key": "...",
      "cause_key": "...",
      "evidence": ["..."],
      "strategy_key": "...",
      "reasoning": ["<bottleneck reasoning>", "<strategy reasoning>"],
      "actions": [
        {
          "target": "...",
          "operation": "...",
          "edits": ["..."],
          "constraints": ["..."]
        }
      ]
    }
  ]
}
```

父 bottleneck 与 strategy 按相同位置和 `cause_key` 双重校验后合并。每个 item 是一个完整监督单元；不另设 bottleneck/strategy 两个答案字段，不改写已校验的 evidence、reasoning 或 actions。

## Audit only

以下内容只进入 `<dataset>.audit.json`：父子 latency、收益、资格和排除原因、input/output 字符数，以及 dataset、system prompt 和 canonical source 的 hash。

同一算子的 `_0` 到 `_3` 必须按去掉版本后缀的算子名进入同一 train/valid/test 分组；最终 `ops` 仍保留版本。
