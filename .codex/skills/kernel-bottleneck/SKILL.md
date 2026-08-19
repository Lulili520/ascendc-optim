---
name: kernel-bottleneck
description: 直接分析 AscendC KernelBench 910B 单算子的完整 Host/Kernel 源码，识别可唯一决定修改方向的具体 cause，再用正式 performance 补充影响证据；输出去重、排序的 bottleneck_key 与 cause_key。不 profiling、不运行 reference、不修改源码、不生成策略。
---

# Kernel Bottleneck

1. 校验当前 `performance.json` 与源码指纹，完整阅读 [分析方法](references/bottleneck-method.md) 和机器定义的 [cause taxonomy](references/cause-taxonomy.json)。
2. 先读取阶段输入中的 `source_facts`（独立调用时运行 `scripts/source_facts.py --project-dir <version>`），再完整读取 `op_host/`、`op_kernel/`。由 shape、axis、dtype 和数学语义理解实际分支，依次完成数学热路径、GM→片上搬运、Scalar/Vector/Cube 计算、写回、片上布局与生命周期、同步与 Queue 流水、Host tiling、task/block/core 映射、tail/dtype/ABI 审查。facts 只存在于阶段输入，不生成额外覆盖文件。
3. 完整扫描后只输出最多 3 个确定且能共享同一优化方向的 issues。LocalTensor `Compare/Select/Reduce` 即使位于 C++ 循环中也是 Vector 计算，禁止仅凭循环标为 `scalar_reduction`。
4. 按源码热路径依次处理：逐元素 GM→碎片/跨步搬运→真正 Scalar 计算→过小 work unit/重复 Vector 指令→GM 往返/物化→写回→多核→流水。小 tile 直接造成 block 过多时只保留 `inefficient_work_unit_size`；单 Queue 槽只提示检查，不足以生成流水 cause。无问题时输出 `issues=[]`。
5. 只写入 `bottleneck/bottleneck.json` 并运行 `scripts/validate_report.py`。`source_facts` 中的 `candidate_causes` 只提示需要阅读的源码形态，不强制生成 issue；校验器只检查 schema、固定 cause 映射和直接源码 target。不生成 `coverage.json`、`source_model.json` 或候选处置记录。

禁止从最高 pipeline ratio 或固定阈值直接生成瓶颈；不生成策略或修改建议。
