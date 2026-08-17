# Action 生产契约

Action 是给实施 Agent 的中粒度修改计划，不是性能解释、伪代码全文或通用知识。

## 固定形式

- 每个 action 只对应一个真实 symbol 和一个固定 operation。
- `edits` 使用 1–4 条 `<operation_slot>/<源码对象>：<当前实现>改为<目标结构；必要参数>`，每条表达可审计代码变换。`operation-slots.json` 用于检查语义覆盖，不强制拆分数量或顺序；一条 edit 可合并多个相关语义。
- `constraints` 使用 1–4 条不变式，只保留 dtype、索引覆盖、tail/对齐、容量、同步、公开 ABI 或数学语义。
- 完成策略所需的最小 symbol 集合为 1–4 个 actions；同一 `target + operation` 必须合并。

## 语义边界

Edit 必须以变换动作为主，例如“将连续 ceil 切分改为商余数半开区间切分”。不得仅写保持、确保、禁止、不改变、容量上限、性能结论、收益或 `performance.*` 字段。

“对象”是 target symbol 内可定位的状态、循环、缓冲、API 调用、任务映射或索引公式，例如 `blockDim`、`Process` 的逐元素循环、`GetValue` 读取、Queue slot 或 `rowStart/rowEnd`。target 已给出文件和 symbol，edit 不再重复完整 target 路径。

下列 edit 不合格，因为只是 operation 的翻译：“提高并行度”、“重新平衡任务”、“减少固定开销”、“向量化归约”、“合并小搬运”。

## Operation 固定槽位

固定槽位的唯一机器可读定义是 [operation-slots.json](operation-slots.json)。required 槽位必须全部出现，optional 只在当前源码需要时填写，不为凑数虚构对象。场景识别、容量、对齐、覆盖和 ABI 通过稳定修改对象表达，不输入外部知识名称或卡片编号。

合格示例：`BLOCK_DIM：从固定 24 改为 min(available_aiv_cores, tiles)`。不要猜测源码中不存在的对象。

涉及 `resize_work_unit` 时，edits 必须给出当前 work unit、峰值活跃 Buffer 的字节公式、候选选择规则和任务数联动；未经成功子版本验证，不得把经验候选写成确定最优值。固定机器语法为：`fixed_bytes=N,bytes_per_element=N,usable_ub_ratio=R`、`peak_bytes=fixed_bytes+bytes_per_element*tile`、`max_tile=floor((usable_ub_bytes-fixed_bytes)/bytes_per_element)`、`candidate_rule=...`、`total_tasks=...`、`alignment=...`、`valid_len=...`；校验器使用正式硬件 UB 重算预算。

涉及 `coarsen_task_partition` 时，edits 必须给出当前 block/task 粒度、物理核或可靠硬件来源、持久 task 循环及覆盖关系；不能只把 blockDim 改成核数。

涉及 `overlap_pipeline_stages` 时，edits 必须覆盖 prologue、steady、epilogue 与槽所有权；只修改 `InitBuffer` 数量不合格。

涉及 `vectorize_scalar_work` 时，必须明确被替换的 Scalar 循环或 `GetValue/SetValue` 对象、UB tile、Vector 转换/计算序列和 tail；不确定 SDK API 时描述变换并在生成门禁确认，不猜签名。

`pattern_and_tiling` 直接根据当前源码写明 `pattern/branch`，并给出 `fixed_bytes=N,bytes_per_element=N,usable_ub_ratio=R`、`peak_bytes=fixed_bytes+bytes_per_element*chunk`、`chunk_rule=max_chunk=floor((usable_ub_bytes-fixed_bytes)/bytes_per_element)`、`alignment=` 与 `tiling_change=true/false`。校验器使用正式硬件 UB 重算；`tiling_change=true` 时必须包含 Host tiling target。`dtype_tail_abi` 必须给出 `valid_len=`、`input_dtype=`、`compute_dtype=`、`output_dtype=` 和可确认的输出 ABI 路径。

全部 strategies 是一套共享最终设计：相同 target 内的相同源码对象不得出现不同 edit；`pattern/branch/alignment/input_dtype/compute_dtype/output_dtype` 在不同 strategies 中不得冲突。发生冲突时统一重写 actions，不得依赖实施顺序覆盖前一方案。

涉及 `coalesce_global_transfer` 时，必须给出 `block_count=`、`block_len=`、`src_stride=`、`dst_stride=`、`alignment=` 和 `valid_len=`；只有连续或可由合法二维搬运参数表达的地址才能合并。

Constraint 必须表达修改后仍需成立的边界，例如“任务区间完整覆盖 `[0,total)` 且无重复”。不得把“新增/删除/改为/替换”类变换放入 constraints。

## 数字与 API

- 仅使用当前源码、shape 或精简硬件配置可直接证明的数字。
- 数字只在决定 tile、对齐、tail 或容量时保留；可由实施 Agent 重算的偶然中间值不写。
- 只写当前源码已使用或当前 SDK 可确认的 API；不确定时描述代码变换，不猜 API 签名。

## Reasoning

用精简非空文本说明 `cause_key → strategy_key` 和当前源码修复机制。不强制固定句式、条数或重复全部辅助 evidence。

不复制其他指标，不重新判定 bottleneck，不使用“当前源码存在 N 个修改目标”等无信息句。
