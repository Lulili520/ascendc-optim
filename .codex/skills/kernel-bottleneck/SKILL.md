---
name: kernel-bottleneck
description: 直接分析 AscendC KernelBench 910B 单算子的完整 Host/Kernel 源码，识别可唯一决定修改方向的具体 cause，再用正式 performance 补充影响证据；输出去重、排序的 bottleneck_key 与 cause_key。不 profiling、不运行 reference、不修改源码、不生成策略。
---

# Kernel Bottleneck

1. 校验当前 `performance.json` 与源码指纹，完整阅读 [bottleneck-method.md](references/bottleneck-method.md)。
2. 直接读取全部 `op_host/`、`op_kernel/` 源码；由 shape、axis、dtype 和数学语义理解实际分支，再跟踪 tiling、任务映射、搬运、计算、写回、tail、Buffer 与同步。
3. 输出全部确定 issues。`cause_key` 必须具体、排他且有直接源码 evidence，并足以唯一决定一个 `strategy_key`；现象或影响不得重复提升为第二 cause。
4. 按前置依赖、性能影响一致性、热路径乘数和源码顺序去重排序；无问题时输出 `issues=[]`。
5. 只写入 `bottleneck/bottleneck.json` 并运行 `scripts/validate_report.py`。不生成 `coverage.json`、`source_model.json` 或候选处置记录。

禁止从最高 pipeline ratio 或固定阈值直接生成瓶颈；不生成策略或修改建议。
