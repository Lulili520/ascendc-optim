---
name: kernel-implementation
description: 实施 AscendC KernelBench 910B 当前版本全部已确认问题的有序 strategies。读取全部原子 actions，创建新版本，只修改 op_host/op_kernel，并记录可审计结果；随后执行源码复检、精度和性能门禁。不重新诊断或改写策略。
---

# Kernel Implementation

1. 阅读 [实施与记录契约](references/implementation-method.md)。控制器模式只使用预建 child、冻结 strategy 和 JSON 骨架；独立模式才创建干净子版本。
2. 只读取阶段输入 `implementation_knowledge.references` 指定的 [通用契约](../../kernel-knowledge/implementation-core.md) 与统一硬件知识文件中精确匹配的 `profile + sections`；禁止读取其他 profile 的规则。修改前按 `sdk-check` 从当前 headers 核对新增或改变 API 的完整声明，不凭记忆补命名空间、dtype 或参数单位；未知架构只查当前 SDK headers。算法结构以冻结 actions 为准。
3. 按顺序实施全部 actions，只修改 child 的 `op_host/`、`op_kernel/`，并在 `implementation.json` 记录核对的 header 与实际调用形式。不重选、遗漏或扩展策略。
4. 运行 `validate_implementation.py`；首次失败后仅允许一次原 action 内的最小修正并终检。只有 ABI/数学语义必坏、API/dtype 明确禁止或容量无解才写 `implementation_blocking.json`。
5. 控制器模式到此结束。独立模式再按精度→性能→重新完整瓶颈分析执行；build/精度失败可按 [失败定位表](../../kernel-knowledge/implementation-diagnosis.md) 最多修复 3 次，性能失败不改源码。
