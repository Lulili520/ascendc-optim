# Strategy Action 实施

```text
strategy_key + actions
→ 逐项源码修改
→ implementation actions
→ initial + 最多 3 次 repair
→ precision
→ performance
```

- `strategy.json` 是不可改写输入；按 `action_index` 从 1 开始完整实施。
- 只修改 `op_host/`、`op_kernel/`；不得加入第二策略。
- 只有 build 或 precision 失败可触发原 action 范围内修复；最多 3 次。
- performance 失败不触发源码修复。

`implementation.json` 保存 `strategy_key`、reasoning、逐项 `action_index + implementation_summary`、`modified_files` 和 attempts。attempts 使用 `action_indices` 标识本次涉及的既有 action，并记录 build、precision、performance 门禁状态。
