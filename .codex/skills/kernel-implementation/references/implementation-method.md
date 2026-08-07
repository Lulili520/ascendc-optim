# Strategy 到源码的实施方法

## 职责边界

实施阶段不回答“选哪个策略”，只回答“如何忠实执行已选策略”。固定链路为：

```text
strategy_key
  → change_key[]
  → target + action
  → 具体源码修改
  → attempts（initial + 最多 3 次 repair）
  → precision.json
  → performance.json
```

`strategy.json` 是不可改写的输入，`implementation.json` 是执行记录。两者分离后，策略模型可以只学习瓶颈到策略与修改意图，执行模型可以学习修改意图到源码实现及有界纠错。

## 有界修复

- `attempt=1` 是初次实施，`kind=initial`，不计入修复次数。
- 仅构建或精度失败可以触发源码修复；性能采集失败不触发源码修复。
- 每次修复只基于一个有直接证据的根因，且只能修复已有 `change_key` 的落地。
- 最多 3 次 `repair`，因此 `attempts` 最多 4 项。达到上限仍失败时停止该算子。
- 每次修改后更新本次 attempt，重新校验 `implementation.json`，并从构建开始重新验证。

## 证据到推断

每条实施推断直接引用上游 change 和实际完成的修改：

```text
证据：change_key=vector_reduce.replace_scalar_reduction 的 target/action 已在当前源码完成；推断：记录实际 Vector Reduce 与索引合并实现。
```

不要重新设计策略或要求证明一定提速。仅当 target/action 出现明确硬错误时停止；一个修改若不能对应已有 `change_key`，不得实施。

## 输出示例

```json
{
  "strategy_key": "compute.vectorize_reduction",
  "reasoning": [
    "证据：change_key=vector_reduce.stage_input_in_ub 且源码逐元素直接访问 GM；推断：按 tile 连续搬入 UB。",
    "证据：change_key=vector_reduce.replace_scalar_reduction 且源码主体是标量最大值循环；推断：使用 Vector Reduce 处理主体数据。",
    "证据：change_key=vector_reduce.handle_tail 且输入长度不保证整 tile；推断：保留尾块 mask 与边界判断。"
  ],
  "changes": [
    {
      "change_key": "vector_reduce.stage_input_in_ub",
      "implementation_summary": "增加 UB tile 缓冲并按块 DataCopy。"
    }
  ],
  "modified_files": ["op_kernel/argmax.cpp"],
  "attempts": [
    {
      "attempt": 1,
      "kind": "initial",
      "trigger": null,
      "knowledge_keys": ["api.datacopy", "pattern.vector_reduce_tail"],
      "change_keys": [
        "vector_reduce.stage_input_in_ub",
        "vector_reduce.replace_scalar_reduction",
        "vector_reduce.handle_tail"
      ],
      "implementation_summary": "完成 UB 分块、Vector Reduce 和尾块处理。",
      "modified_files": ["op_kernel/argmax.cpp"],
      "validation": {
        "build": "PASS",
        "precision": "FAILED",
        "performance": "NOT_RUN"
      }
    },
    {
      "attempt": 2,
      "kind": "repair",
      "trigger": {
        "stage": "precision",
        "symptom": "尾块输出索引偏移一个 tile。",
        "evidence": "precision 日志显示首个误差位于尾块起点，源码使用了局部索引。"
      },
      "knowledge_keys": ["pattern.global_index_merge"],
      "change_keys": ["vector_reduce.handle_tail"],
      "implementation_summary": "将尾块局部索引合并为全局索引。",
      "modified_files": ["op_kernel/argmax.cpp"],
      "validation": {
        "build": "PASS",
        "precision": "PASS",
        "performance": "DONE"
      }
    }
  ]
}
```

实际文件必须列出策略中的全部 change，示例为简洁仅展示一项。
