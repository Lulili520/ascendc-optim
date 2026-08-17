---
name: kernel-implementation
description: 实施 AscendC KernelBench 910B 当前版本全部已确认问题的有序 strategies。读取全部原子 actions，创建新版本，只修改 op_host/op_kernel，并记录可审计结果；随后执行源码复检、精度和性能门禁。不重新诊断或改写策略。
---

# Kernel Implementation

1. 校验父版本覆盖全部 issues 的有序 strategies，完整阅读 [implementation-method.md](references/implementation-method.md)，创建下一未占用版本并原样复制 `strategy.json`。
2. 按 strategies/actions 的依赖顺序读取 `../../kernel-knowledge/action-api-knowledge.md` 和 `../../kernel-knowledge/action-pattern-knowledge.md` 的相关章节，只修改新版本 `op_host/`、`op_kernel/`；完整实施全部问题，不重选策略或混入未记录优化。
3. ABI/数学语义必然破坏、API/dtype 明确禁止或容量公式无解时，写 `strategy/implementation_blocking.json` 并停止；其余不确定性交给编译。
4. 写入 `strategy/implementation.json` 并运行：

   ```bash
   python .codex/skills/kernel-implementation/scripts/validate_implementation.py \
     --bottleneck <parent>/bottleneck/bottleneck.json \
     --strategy <parent>/strategy/strategy.json \
     --implementation <child>/strategy/implementation.json \
     --parent <parent> --project-dir <child> --require-attempts
   ```

   `strategy.json` 是父版本的不可改写方案，修改后的源码位于子版本；必须显式传入 `--project-dir <child>`，禁止从 strategy 路径推断实施版本。

5. 运行 `scripts/validate_source_effect.py --parent <parent> --child <child>`；全部 action target 的真实 symbol body 必须变化，并逐项复核可静态检查的 cause；文本反模式计数只记 warning。
6. 独立调用本 skill 时，依次使用 `kernel-precision` 和 `kernel-performance`。编译/精度失败时按 `../../kernel-knowledge/implementation-diagnosis.md` 在原 actions 内最小修复，最多 3 次；性能失败不修改源码。性能完成后对新版本重新运行 `kernel-bottleneck`，其结果只用于判断是否需要下一轮；不以父版本 cause 是否仍存在判定本轮执行或训练数据资格。

控制器阶段模式以阶段输入为准：implementation 只创建子版本、修改源码、写入并校验 `implementation.json`；repair 每次只追加一次修复记录并校验。两种模式都不得自行运行精度、性能或下一轮分析，这些门禁由控制器在全新进程中执行。

`implementation.json` 的字段、attempt 状态和门禁顺序以校验器为准。
