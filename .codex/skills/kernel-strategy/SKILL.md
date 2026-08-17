---
name: kernel-strategy
description: 将 kernel-bottleneck 已排序的具体 cause_key 确定性映射为唯一 strategy_key 和 operation，再生成当前源码原子 actions；保持依赖顺序并覆盖所有问题，不重新诊断、不修改源码、不执行优化。
---

# Kernel Strategy

1. 校验 `bottleneck/bottleneck.json`。
2. 用 `scripts/derive_strategy.py` 按原顺序执行 `cause_key → 唯一 strategy_key → 唯一 operation`；`strategies[]` 必须与 `issues[]` 一一对应，不在本阶段选择候选方向。
3. 按 [action-contract.md](references/action-contract.md) 为每个 cause 生成必要的最少 actions；同一 strategy 不混入其他 cause，但全部 strategies 共同覆盖当前版本所有确定问题。若前项会改变后项 target 或公式，在 action 中使用修改后的共同目标结构，禁止删除后项。
4. 仅在 ABI/数学语义必然破坏、API/dtype 明确禁止或容量公式无解时写 `strategy/blocking.json`；其余不确定性交给实施和编译。
5. 运行 `scripts/validate_strategy.py`，统一修复一轮结构、容量、对齐、覆盖、流水或 dtype 路径问题后复检。

不重新诊断、预测收益或修改 bottleneck。
