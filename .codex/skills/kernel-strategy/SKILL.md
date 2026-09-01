---
name: kernel-strategy
description: 完整分析 AscendC KernelBench 910B 单算子源码中的全部确定问题，按固定类别顺序统一设计并在一轮中全部修复，生成含显式证据链且可独立验证的闭合策略。
---

# Kernel Strategy

1. 完整读取阶段 input、[源码策略方法](references/source-strategy-method.md)、列出的 Host/Kernel、device 与 `knowledge_contract`。只使用这些输入；禁止读取 reference、性能、CMake、validator 源码、其他算子或历史结果，也禁止递归搜索 SDK。
2. 从 shape/dtype 和全部输出路径恢复数学语义、主引擎、任务所有权、数据流与跨 tile 状态；先识别 Elementwise、Prefix/Scan、Reduction、Norm/Softmax、Window/Pooling、contraction/Cube 等数学模式和支配性跨迭代依赖，再按方法文件完成语义、所有权、地址/tail、容量/生命周期、状态、指令/workspace、同步和 ABI 审计及全部性能扫描。Prefix/Scan 必须读取 input 已选择的 `prefix-scan` 与架构 `vector` 契约。首次引入 Cube 只在 input 契约与当前 SDK 足以闭合 API、MNK、格式、L1/L0、重载、所有权、同步和 ABI 时冻结，否则记 unresolved 并继续扫描。若有 `replan_feedback`，将已证伪结构作为硬约束。
3. 只冻结契约确认的 API 家族，参数由 shape、可靠容量、字段范围和任务数唯一推导。所有 changes 共享一致的任务、tile、Buffer、地址、dtype、对齐、tail、同步和 ABI。内部区分 actionable、unresolved 与 clean，仅 actionable 进入既有七种 kinds；全量扫描后没有 actionable 才写 `{"strategy":null}`，该值不得解释为源码绝对无问题。
4. 非空方案依次写 `strategy/planning.json` 和 `strategy/strategy.json`，使用 input 的 `validation_commands` 每轮同时运行两个 validator。纯 schema、格式或引用错误可合并修复并复检，最多 3 次；CLI 调用错误不计次数，相同错误连续两次才停止。不得借校验重选策略，禁止预读 validator。不要修改源码或运行 precision/performance。

本 skill 不修改源码。
