---
name: kernel-implementation
description: 实施 AscendC KernelBench 910B 已选优化策略。读取父版本 strategy/strategy.json 中唯一 strategy_key 与完整 actions，创建新优化版本，只修改 op_host/op_kernel，并记录 action 到源码修改的可审计结果；随后执行精度和性能门禁。用户要求执行优化策略或验证策略修改时使用；不重新认定瓶颈、不改写策略、不评价 adopted/rejected。
---

# AscendC 优化策略实施

按上游 actions 中已经具体化的 `target + operation + edits + constraints` 忠实修改源码。策略选择和可执行性筛选由 `kernel-strategy` 完成，本 skill 不重新选择策略，也不重复进行保守的完整可行性审查。

## 输入门禁

1. 父版本必须存在通过校验的 `strategy/strategy.json`，且包含唯一非空 `strategy`。
2. 完整阅读 [`references/implementation-method.md`](references/implementation-method.md)。
3. 创建下一个未占用的优化版本；禁止覆盖历史版本。新版本只继承工程源码、必要配置和策略输入，不继承精度、性能、瓶颈结果或构建中间文件，状态恢复为 `PREPARED`。
4. 将父版本 `strategy/strategy.json` 原样保存在新版本同一路径，作为实施输入。不得修改 `strategy_key`、`actions` 或推导记录。

## 实施流程

1. 按每个 action 的 `target` 定位源码并完整实施。
   同时按 action 特征读取 `../../kernel-knowledge/action-api-knowledge.md` 和
   `../../kernel-knowledge/action-pattern-knowledge.md` 的相关章节，使实施与 Strategy 使用同一 API、同步、对齐、dtype、容量和尾块知识。
2. 只做轻量硬错误检查：target 不存在、源码与 action 明显冲突、API/dtype 明确不支持、容量明确不足或无法保持接口/数学语义时立即停止，不修改源码、不生成 `implementation.json`。不要因为尚未证明性能收益而停止。
3. 每个源码修改必须对应一个已有 action；不得顺手实施第二策略。编译或正确性所必需的配套改动归入相关 action 并在摘要中说明。
4. 只修改新版本的 `op_host/`、`op_kernel/`。禁止修改原始工程、PyTorch reference 和调用层。
5. 写入 `strategy/implementation.json`，然后运行：

   ```bash
   python .codex/skills/kernel-implementation/scripts/validate_implementation.py \
     --bottleneck <parent-version>/bottleneck/bottleneck.json \
     --strategy <new-version>/strategy/strategy.json \
     --implementation <new-version>/strategy/implementation.json \
     --require-attempts
   ```

6. 校验通过后必须使用 `kernel-precision` 验证新版本。构建或精度失败时停止本次验证，禁止采集性能；若尚未用满 3 次修复，则进入下述修复流程。
7. 精度通过后必须使用 `kernel-performance` 采集新版本性能。性能采集失败只按采集流程恢复，不触发源码修复。最终只报告精度结果和客观性能结果，不在本阶段判断策略是否采纳。

初次实施不计入修复次数。编译或精度失败时读取 `../../kernel-knowledge/implementation-diagnosis.md`，依据直接证据形成一个根因假设并做原 change 范围内的一次最小修复；不得借排错更换策略。最多允许 3 次修复，即最多 4 次实施尝试。每次源码变化后更新 `attempts`、重新校验 implementation，并从编译门禁开始验证。修复 3 次后仍未通过则停止该算子，记录最终失败阶段，禁止继续修改或采集性能。

## implementation.json

保留下游实施最终结果和有界尝试轨迹：

- `strategy_key`：必须等于策略中的唯一 `strategy_key`；
- `reasoning`：2–8 条“证据：…；推断：…”记录，说明每个 action 如何由源码事实落地；
- `actions`：按策略 action 顺序逐项记录 `action_index` 和简短 `implementation_summary`；
- `modified_files`：只列相对当前版本的 `op_host/`、`op_kernel/` 文件；
- `attempts`：按顺序记录初次实施及最多 3 次修复。每项固定包含：
  - `attempt`：从 1 连续递增；
  - `kind`：首项为 `initial`，其余为 `repair`；
  - `trigger`：初次实施为 `null`；修复时包含失败 `stage`、直接 `symptom` 和 `evidence`；
  - `knowledge_keys`：本次实际加载的稳定知识主题 key，不保存来源路径；
  - `action_indices`：本次实施或修复涉及的既有 action 序号；
  - `implementation_summary`、`modified_files`：本次具体修改摘要及文件；
  - `validation`：固定记录 `build`、`precision`、`performance` 的 `PASS/FAILED/NOT_RUN` 或 `DONE/FAILED/NOT_RUN` 状态。

校验器确认 strategy 的固定 key 映射和真实 target，再检查全部 action 是否一一覆盖、reasoning 是否逐项引用、`modified_files` 是否覆盖所有 target，并检查尝试顺序、修复上限、失败触发和门禁顺序。
