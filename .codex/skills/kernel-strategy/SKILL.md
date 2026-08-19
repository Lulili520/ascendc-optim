---
name: kernel-strategy
description: 将 kernel-bottleneck 已排序的具体 cause_key 确定性映射为唯一 strategy_key 和 operation，再生成当前源码原子 actions；保持依赖顺序并覆盖所有问题，不重新诊断、不修改源码、不执行优化。
---

# Kernel Strategy

1. 校验 `bottleneck/bottleneck.json`。
2. 用 `scripts/derive_strategy.py` 按原顺序执行 `cause_key → 唯一 strategy_key → 唯一 operation`；`strategies[]` 必须与 `issues[]` 一一对应，不在本阶段选择候选方向。
3. 按 operations 只读取 `../../kernel-knowledge/action-pattern-knowledge.md` 的相关章节，按“算子模式→最大合法有效 tile→搬运 chunk→计算→Buffer→必要同步”形成一套共享设计。AR/ARA/With-Index、Elementwise 和 Cube 使用各自确定公式，不搜索参数；已有 Vector 路径优先扩大合法 work unit，禁止无依据改成 Transpose/Gather。架构 API 细节留给 implementation。
4. 按 [action-contract.md](references/action-contract.md) 将共享设计拆为每个 cause 的最少 actions，全部 strategies 合计不超过 6 个 actions；同一 strategy 不混入其他 cause，但全部 strategies 共同覆盖已输出问题。按“任务与 tiling→搬运→驻留/Buffer→计算→流水/同步→写回/tail”依赖具体化；若前项改变后项 target 或公式，后项必须引用修改后的共同结构。
5. 仅在 ABI/数学语义必然破坏、API/dtype 明确禁止或容量公式无解时写 `strategy/blocking.json`；其余不确定性交给实施和编译。
6. 运行 `scripts/validate_strategy.py`；校验器只检查 cause→strategy→operation、issue 覆盖、真实 target 和非空 edits/constraints。容量、API、对齐和 dtype 的普通不确定性交给实施与编译，只有已证明无解才阻断。

不重新诊断、预测收益或修改 bottleneck。
