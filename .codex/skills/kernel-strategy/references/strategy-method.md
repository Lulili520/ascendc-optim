# Cause 到策略与 Action

## 职责

```text
bottleneck_key：哪里受限
cause_key：为什么受限
strategy_key：采用什么优化方向
operation：执行哪类稳定代码变换
edits：当前源码具体改什么
constraints：必须保持什么
```

固定的 `cause_key → strategy_key` 映射以 `scripts/derive_strategy.py::RULES` 为唯一可执行定义。Agent 不重新诊断瓶颈，也不输出 pattern、description 或 change_key。

## Action

每项 action 只包含：

- `target`：当前源码真实存在的 `op_host/...::符号` 或 `op_kernel/...::符号`；
- `operation`：由 strategy 唯一确定的短动词；
- `edits`：1–4 条当前源码的中粒度具体变换；
- `constraints`：1–4 条当前算子特有的 dtype、索引、对齐、容量、tail、同步或数学语义边界。

actions 数量由当前源码决定，采用完成策略所需的最小完整集合。相同 `target + operation` 必须合并；不同符号的独立修改点才拆开。不得保留固定三步模板或混入第二策略。

具体文本边界以 [action-contract.md](action-contract.md) 为准。

## 输出

```json
{
  "reasoning": ["证据：cause_key=...；推断：strategy_key=...。"],
  "strategy": {
    "strategy_key": "parallel.balance_tiling",
    "actions": [
      {"target": "op_kernel/x.cpp::KernelX::Process", "operation": "rebalance_task_partition", "edits": ["rowBegin/rowCount：改为商余数半开区间分配"], "constraints": ["任务无遗漏且不重复"]}
    ]
  }
}
```

`inherent_serial_dependency` 或无法保持接口/数学语义时输出 `strategy:null`。

`redundant_hot_path_overhead` 和 `scalar_address_overhead` 统一映射到 `overhead.reduce_fixed_cost`；该 key 覆盖 kernel 内重复初始化、过细资源生命周期、循环不变量和 Scalar 地址控制开销，不限于 kernel launch。
