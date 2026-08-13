---
name: kernel-strategy
description: 根据 kernel-bottleneck 的唯一 bottleneck_key、cause_key 和证据选择唯一 strategy_key，并结合当前 AscendC 源码生成 target、operation、edits、constraints。用户要求选择优化策略、生成策略训练数据或校验瓶颈到策略推导时使用；不重新诊断瓶颈、不修改源码、不执行优化。
---

# Kernel Strategy

目标：回答“针对已确认原因，采用什么策略，并对当前源码具体改什么”。

## 流程

1. 完整阅读 [references/strategy-method.md](references/strategy-method.md)、[references/action-contract.md](references/action-contract.md) 和 [references/operation-slots.json](references/operation-slots.json)；具体化和硬阻断时再读 [references/strategy-validation.md](references/strategy-validation.md)。
2. 运行 `derive_strategy.py --project-dir <version>`，按 cause 确定唯一 `strategy_key`。
3. 先从源码 symbol 索引定位候选，再阅读对应实现；按 operation 固定槽位生成 1–4 个最少且完整的 `actions`。每项包含真实 `target`、固定 `operation`、1–4 条中粒度 `edits` 和 1–4 条必要 `constraints`。
4. 做一次统一检查；普通问题最多统一修复一轮，不能更换 `strategy_key` 或混入第二策略。
5. 写入 `strategy/strategy.json`，运行 `validate_strategy.py`。

## 固定边界

- 推导链固定为：`bottleneck_key + cause_key → strategy_key → actions[]`。
- `strategy_key` 是稳定优化方向；具体技术模式、边界和实现细节全部进入 action 文本。
- 未命中策略、策略不适用或存在规范硬阻断时输出 `strategy:null`，不得临时补选。
- `strategy` 只包含 `strategy_key` 和 `actions`。每个 action 的 `edits` 只写当前源码变换，`constraints` 只写 dtype、索引、容量、对齐、tail、同步或数学语义边界；不重复通用知识。
- 相同 `target + operation` 必须合并为一个 action，禁止保留固定三步模板。
- 每条 edit 固定写成 `<源码对象>：<明确变换；必要参数>`；禁止只翻译 operation。`constraints` 只写修改后不变式；不得互换，不得引用 `performance.*` 或收益。
- reasoning 固定两条，第二条只引用当前 cause evidence 的 key、source，并连接全部 target 与 operation；不复制 observation。
