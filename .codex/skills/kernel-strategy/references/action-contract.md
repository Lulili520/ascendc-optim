# Action 生产契约

Action 是给实施 Agent 的中粒度修改计划，不是性能解释、伪代码全文或通用知识。

## 固定形式

- 每个 action 只对应一个真实 symbol 和一个固定 operation。
- `edits` 使用 1–4 条 `<源码对象>：<明确变换；必要参数>`，每条只表达一个可审计代码变换。
- `constraints` 使用 1–4 条不变式，只保留 dtype、索引覆盖、tail/对齐、容量、同步、公开 ABI 或数学语义。
- 完成策略所需的最小 symbol 集合为 1–4 个 actions；同一 `target + operation` 必须合并。

## 语义边界

Edit 必须以变换动作为主，例如“将连续 ceil 切分改为商余数半开区间切分”。不得仅写保持、确保、禁止、不改变、容量上限、性能结论、收益或 `performance.*` 字段。

“对象”是 target symbol 内可定位的状态、循环、缓冲、API 调用、任务映射或索引公式，例如 `blockDim`、`Process` 的逐元素循环、`GetValue` 读取、Queue slot 或 `rowStart/rowEnd`。target 已给出文件和 symbol，edit 不再重复完整 target 路径。

下列 edit 不合格，因为只是 operation 的翻译：“提高并行度”、“重新平衡任务”、“减少固定开销”、“向量化归约”、“合并小搬运”。

## Operation 固定槽位

固定槽位的唯一机器可读定义是 [operation-slots.json](operation-slots.json)。按其顺序检查，只填写当前源码实际需要的槽位，不为凑数虚构对象。CannBot 中“先用满核再做合并”“标量循环优先向量化”“搬运与计算以真实依赖判断重叠”的原则在此落实为稳定修改对象，不输入卡片编号。

合格示例：`BLOCK_DIM：从固定 24 改为 min(available_aiv_cores, tiles)`。不要猜测源码中不存在的对象。

Constraint 必须表达修改后仍需成立的边界，例如“任务区间完整覆盖 `[0,total)` 且无重复”。不得把“新增/删除/改为/替换”类变换放入 constraints。

## 数字与 API

- 仅使用当前源码、shape 或精简硬件配置可直接证明的数字。
- 数字只在决定 tile、对齐、tail 或容量时保留；可由实施 Agent 重算的偶然中间值不写。
- 只写当前源码已使用或当前 SDK 可确认的 API；不确定时描述代码变换，不猜 API 签名。

## Reasoning

固定两条：

1. `cause_key → strategy_key`；
2. 引用当前 cause evidence 的 `evidence_key + source`，连接全部 target 与固定 operation；不复制 observation。

不复制其他指标，不重新判定 bottleneck，不使用“当前源码存在 N 个修改目标”等无信息句。
