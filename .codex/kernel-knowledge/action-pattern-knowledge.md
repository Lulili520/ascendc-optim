# AscendC Action 模式知识

按固定 operation 加载对应章节，用于具体化或 Strategy 校验门禁中的统一修复轮。模式提供实现契约，不改变 `strategy_key`。

## 算子模型与分支

生成 action 前先由 shape、axis、dtype 和数学语义建立最小模型。Reduction 合轴后必须区分 AR、ARA 或多轴，再按峰值活跃 Buffer 与可靠 UB 容量选择 FullLoad 或分载；带索引输出再叠加 With-Index 约束。Elementwise、Broadcast、Conversion 和 Cube 同样先确定布局分支，禁止从 pipeline ratio 直接选实现。

模型不增加训练字段，但必须落入相关 action：`pattern=`、`branch=`、容量公式、任务覆盖、对齐、tail 与输出 ABI。固定 tile/chunk 只有在 shape、硬件或 API 对齐直接证明时才允许。

## Vector reduction 与索引

适用：`vector_reduce.*`、`low_latency_reduce.*`。

- 先判断归约轴在 GM中的连续性和 AR/ARA形态，再决定整段归约或逐行 Compare/Select。
- ARA索引归约先判断 `R×alignedA0` 能否 FullLoad；不能时按 `peak_bytes<=usable_ub` 反推 `R_chunk`，用 DataCopyPad 的 blockCount/stride 批量搬入多行，极值和索引状态跨 chunk 常驻 UB。每次只搬一行仅在容量或依赖确实限制时成立。
- 首索引语义：ArgMax使用“输入 <= 旧最大值则保留”，ArgMin使用“输入 >= 旧最小值则保留”；若 action使用严格 GT/LT更新，也必须明确相等时不更新。
- 索引状态可用 float时，先验证范围；循环行号优先用 float标量累加。ARA优先使用反转 Compare 加 Tensor-Scalar Select，避免每行 Duplicate 索引向量；输出前使用当前 SDK 支持的转换路径。
- action必须同时描述 Host tile/chunk/blockDim、input Queue、极值/索引/mask buffer、全部 API 预留空间、Compare对齐、尾块中性padding、同步和输出 ABI。
- 分片归约必须在合并局部索引时加片起始偏移，并保持跨片首索引语义。

## Vector elementwise

适用：`vector_elementwise.*`。

- 连续输入用 VECIN Queue批量搬入，纯计算临时量用 TBuf，结果用 VECOUT Queue批量搬出。
- action给出 tile、主循环、尾块和 dtype路径；非对齐尾块使用 DataCopyPad或明确mask。
- 多输入操作逐一说明布局、广播与生命周期，避免一个输入被提前复用。

## Batch transfer 与 GM traffic

适用：`batch_transfer.*`、`gm_traffic.*`、`ub_residency.*`。

- 合并前必须证明地址连续或可由合法 blockCount/stride表达；不能把非连续访问描述成连续 DataCopy。
- action写明 blockLen、blockCount、src/dst stride的语义和单位，并处理字段上限。
- 中间结果驻留 UB时，明确生产/消费阶段、生命周期、覆盖时机和容量；删除 GM往返后仍须保持可见性和数学顺序。

## Double Buffer

适用：`double_buffer.*`。

- 仅在已有证据确认可重叠或策略固定要求时实施；input/output Queue使用两份buffer并重算容量。
- action描述 prologue、steady-state、epilogue，或说明 Queue如何自动轮转；必须保证消费者完成后才复用。
- 不把模板 depth误写为buffer数量，不遗漏 event数量和最大 Queue限制。

## UB conflict 与片上布局

适用：`ub_conflict.*`、`cube_reuse.*`。

- padding/stride调整必须同时更新 UB offset、容量、有效长度和访问参数。
- Cube复用明确 L1/L0A/L0B驻留对象、tile形状、重载条件及容量；不能只写“提高复用”。
- 布局变化必须保持输出索引和公开格式，必要转换放在明确边界。

## Cast

适用：`cast.*`。

- 删除或合并 Cast前证明源/目标计算dtype与舍入语义等价。
- action写明每段 `src dtype → dst dtype → RoundMode`，以及尾块count和临时buffer。
- 如果当前 SDK缺少直接转换，允许合法中间dtype路径；不存在语义等价路径才是硬阻断。

## Writeback

适用：`writeback.*`、所有输出为非计算dtype的 change。

- 输出地址、批量粒度和对齐一起设计；非对齐尾块用 DataCopyPad且只提交有效字节。
- Vector结果先在 UB形成目标 ABI布局，再通过 VECOUT/MTE3搬出；避免依赖不受支持的逐元素 GM转换。
- 多核输出必须证明区域不重叠；确有重叠时使用被数学语义允许的原子或确定性所有权方案。

## Parallel 与 tiling

适用：`core_parallelism.*`、`load_balance.*`。

- `used_cores <= min(可用物理核, 独立任务数)`，并明确空核返回。
- 给出核到任务区间映射，证明全覆盖、无重复；余数分散时保持边界。
- blockDim、tile数和尾任务必须与 Kernel索引公式一致；不得只修改 Host而遗漏 Kernel映射。

## Launch cost

适用：`launch_cost.*`。

- 删除初始化前确认其不是同步、状态清零或 ABI要求；循环不变量外提后保持每核/每tile作用域。
- 减少 Scalar LD/ST优先使用局部变量、编译期常量和批量搬运，不把必要状态改成错误共享。
