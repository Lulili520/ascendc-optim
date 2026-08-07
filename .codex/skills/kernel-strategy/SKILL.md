---
name: kernel-strategy
description: 读取 kernel-bottleneck 生成的唯一主瓶颈及结构化 evidence_keys，按固定推导表确定性生成唯一 strategy_key 和 change_key 契约。用户要求根据瓶颈选择 AscendC 优化策略、验证 bottleneck 到 strategy 的确定转换或生成策略数据时使用；不重新采集性能、不重新认定瓶颈、不修改源码、不执行优化。
---

# AscendC Kernel 策略推导

固定选择策略，并把策略具体化为当前源码可直接执行的修改。`kernel-bottleneck` 回答“瓶颈是什么”，本 skill 回答“选择哪个策略、在哪里改、具体改什么”。

## 流程

1. 要求当前版本存在通过校验的 `bottleneck/bottleneck.json`，且包含唯一非空 `bottleneck`。
2. 完整阅读 [`references/strategy-method.md`](references/strategy-method.md) 和
   [`references/strategy-validation.md`](references/strategy-validation.md)。
3. 运行推导器：

   ```bash
   python .codex/skills/kernel-strategy/scripts/derive_strategy.py \
     --project-dir kernel_workspace/KernelBench910B/<level>/<OperatorName>_<version>
   ```

4. 推导器只依据 `bottleneck_key + evidence_keys` 选择 `strategy_key` 和固定 `change_key`，生成待具体化草稿。
5. 阅读当前版本 `op_host/`、`op_kernel/`，为每个 change 填写：

   - `target`：真实存在的 `op_host/...::符号` 或 `op_kernel/...::符号`；
   - `action`：针对当前源码的明确修改动作，同时说明必要的边界、dtype、索引或容量处理。

   按 change 特征加载 [`../../kernel-knowledge/action-api-knowledge.md`](../../kernel-knowledge/action-api-knowledge.md) 和
   [`../../kernel-knowledge/action-pattern-knowledge.md`](../../kernel-knowledge/action-pattern-knowledge.md) 的相关章节；不要重新读取无关的整套知识。

6. 对具体化草稿执行一次统一 Strategy 校验门禁：同时检查固定映射、JSON 契约、reasoning/change 覆盖、target，以及当前 SDK API/dtype、数据流同步、对齐、容量、尾块、索引和输出 ABI。若无问题，禁止改写草稿；若发现问题，只允许统一修复一轮，然后复检一次。不得按结构检查和语义检查分别修复。
7. 仅在首次检查或单轮修复后的复检确认明确硬阻断时输出 `strategy:null`：找不到任何目标、必须越界修改、当前 SDK 无可替代 API/dtype 路径、所有合理 tiling 均容量不足、无法保持公开接口或数学语义、changes 相互冲突。reasoning 必须记录 `硬阻断：<类型>`。普通问题不得产生 `strategy=null`，复检后也不定义“strategy 生成失败”状态。
8. 将最终结果写入：

   ```text
   strategy/strategy.json
   ```

9. 运行统一门禁中的确定性校验部分，断言固定推导、JSON 契约、change 顺序和 target 文件：

   ```bash
   python .codex/skills/kernel-strategy/scripts/validate_strategy.py \
     --bottleneck <version>/bottleneck/bottleneck.json \
     --strategy <version>/strategy/strategy.json
   ```

10. 更新 `workspace.json` 的独立管理字段；不改变主状态 `PERFORMANCE_DONE`。不额外生成策略 Markdown 报告，也不把校验草稿或修复过程写入训练 JSON。

## 输出约束

- 每次最多输出一个 `strategy_key`。
- 无唯一匹配规则时输出 `strategy=null`。
- 固定规则已匹配但源码不可实施时，硬阻断类型只能是：`target不存在`、`越界修改`、`API不支持`、`dtype不支持`、`容量不足`、`接口语义冲突`、`数学语义冲突` 或 `change冲突`。
- `strategy` 只包含 `strategy_key`、`description` 和 `changes`。
- `reasoning` 必须保留 `bottleneck_key/evidence_keys → strategy_key → change_key → target/action` 的简短证据到推断记录，并引用每个 change key。
- `changes` 的 key 和顺序固定；每项只包含 `change_key`、真实 `target` 和具体 `action`。
- Strategy 校验门禁至多修复一次；首次检查通过时修复次数为零。结构和语义 issue 必须在同一轮处理，单轮修复不得更换策略、增删或重排 change。
- 不输出第二候选，不估计收益，不修改 `op_host/` 或 `op_kernel/`。
- `validate_strategy.py` 是统一门禁的确定性部分，不是第二套 valid 流程；它重新运行固定推导器，保证 `strategy_key`、描述和有序 `change_key` 不可改写，同时检查所有 target 文件真实存在且 action 足够具体。
