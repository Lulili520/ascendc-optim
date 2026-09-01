---
name: kernel-implementation
description: 在控制器已创建的 AscendC KernelBench 910B child 中实施冻结的全量源码策略。只修改 op_host/op_kernel，完成全部 changes、记录精简 implementation 并校验源码变更；不创建版本、不重新诊断、不运行精度或性能。
---

# Kernel Implementation

1. 阅读 [实施契约](references/implementation-method.md)，只实施父版本 `strategy/strategy.json` 中冻结的全量策略，并读取已通过校验的内部 `strategy/planning.json` 作为任务、布局、容量、地址和静态工作量证明。
2. 根据 targets 和 changes 读取 [源码优化与实施契约](../../kernel-knowledge/implementation-core.md) 的相关章节；必须实现契约定义的目标结构和静态工作量，不能以等量小粒度循环替代批量搬运或整段 Vector，也不能以 Queue 类型或槽数替代真实流水。编辑前对策略将新增或改变的每个 API、类型、字段和 event 符号，在 input 的 `sdk_include_roots` 中定点检索声明；旧控制器输入尚无该字段时，只在已激活环境的 `$ASCEND_HOME_PATH/x86_64-linux/asc/include` 与 `asc/impl` 中定点检索。核验作用域、参数顺序、字段类型、dtype、count/mask/stride 单位和架构条件；找不到精确声明时不得凭记忆实施该调用形式。
3. 在编辑前闭合同一套任务、tile、容量、地址、dtype、tail、同步和 ABI：逐个片上对象列出容量与生命周期；对每条 Vector 指令的每个操作数验证 `Buffer基址 + subview offset*sizeof(dtype)` 的实际地址对齐，禁止只验证 Buffer 基址；逐个新增/改变 API 核验 header 中的 dtype、参数、count/mask/stride 单位。随后一次完成全部 changes；只修改 child 的 `op_host/`、`op_kernel/`。
4. 写精简 `strategy/implementation.json`，运行 `validate_implementation.py`；该 validator 同时执行已确认 SDK 符号的编译前静态检查。target 可以是函数、文件级常量、class/struct 或 `BEGIN_TILING_DATA_DEF` 声明的 tiling 类型，被策略明确移除的 target 也视为已变化。不得用注释或等价改写伪造 target 变化；错误 target 交还控制器重新规划。
5. 结束并交还控制器。控制器负责精度和 Task Duration。child 构建失败时，控制器可在同一 child、同一冻结 action 范围内发起最多 3 次 `compile_fix`：输入只增加编译器直接证据与 `implementation-diagnosis.md`，只修复 API 签名、符号作用域、字段类型、转换或对象闭环等落地错误。修复用尽或证据表明策略本身不可实施时，才归档失败 child 并反馈父版本重新完整规划。精度失败不由 Implementation 擅自修改数学方案。

不得创建版本、新增策略、跳过 change、运行门禁、读取性能报告重新选方向或修改原始工程。
