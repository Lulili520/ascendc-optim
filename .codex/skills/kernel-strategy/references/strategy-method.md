# Cause 到 Strategy

```text
已排序的全部具体 cause
→ 每个 cause 唯一固定的 strategy_key/operation
→ 按依赖顺序的当前源码 target、edits、constraints
```

固定映射以 `scripts/derive_strategy.py` 为唯一机器定义。若一个 cause 需要在多个修改方向中选择，说明 cause 仍过宽，必须回到 bottleneck taxonomy 拆分；Strategy 不重新诊断或选择候选。

输出：

```json
{"reasoning":["cause 到策略的短说明"],"strategies":[{"cause_key":"serial_pipeline_stages","strategy_key":"pipeline.overlap_stages","actions":[{"target":"op_kernel/x.cpp::KernelX::Process","operation":"overlap_pipeline_stages","edits":["主 tile 循环：改为预取、稳态交错和排空"],"constraints":["每个 tile 恰好处理一次"]}]}]}
```
