# AscendC Action 模式知识

按固 operation 加载对应章节，用于 Strategy 生成当前源码的 actions，不改变 `strategy_key`。本文件只保留设计公式和 action 必需约束；精确 API、字段类型、event 和符号作用域由 Implementation 按当前 headers 确定。

## 算子模型与分支

生成 action 前先由 shape、axis、dtype 和数学语义建立最小模型。Reduction 合轴后必须区分 AR、ARA 或多轴，再按峰值活跃 Buffer 与可靠 UB 容量选择 FullLoad 或分载；带索引输出再叠加 With-Index 约束。Elementwise、Broadcast、Conversion 和 Cube 同样先确定布局分支，禁止从 pipeline ratio 直接选实现。

模型不增加训练字段，但必须落入相关 action：`pattern=`、`branch=`、容量公式、任务覆盖、对齐、tail 与输出 ABI。固定 tile/chunk 只有在 shape、硬件或 API 对齐直接证明时才允许。

## Vector reduction 与索引

适用：`vector_reduce.*`、`low_latency_reduce.*`。

- 先合轴并判定 AR/ARA、FullLoad/分载和 With-Index。ARA With-Index 使用连续 A0 lane 的逐行 Compare/Select；它是 Vector 路径，不默认引入 Transpose/Gather。
- ARA 索引归约先在保持 `A1×ceil(A0/tileA0)>=可用AIV核数` 时取容量允许的最大对齐 `tileA0`；再判断 `R×alignedA0` 能否 FullLoad，不能时用剩余 UB 取最大合法 `R_chunk`。用二维批量搬运表达多行，极值和索引状态跨 chunk 常驻片上。
- 首索引语义：ArgMax使用“输入 <= 旧最大值则保留”，ArgMin使用“输入 >= 旧最小值则保留”；若 action使用严格 GT/LT更新，也必须明确相等时不更新。
- 索引状态可用 float时，先验证范围；循环行号优先用 float标量累加。ARA优先使用反转 Compare 加 Tensor-Scalar Select，避免每行 Duplicate 索引向量；输出前使用当前 SDK 支持的转换路径。
- action必须同时描述 Host tile/chunk/blockDim、二维搬运、极值/索引/mask 存储、Compare 计算空间、生产消费顺序、尾块中性 padding 和输出 ABI。
- 分片归约必须在合并局部索引时加片起始偏移，并保持跨片首索引语义。

## Vector elementwise

适用：`vector_elementwise.*`。

- 连续输入、计算临时量和输出分别使用当前架构支持的搬运、片上存储与同步模式。
- action 给出 tile、主循环、尾块和 dtype 路径；非对齐尾块只描述有效范围和语义，具体 API 由架构覆盖层确定。
- 多输入操作逐一说明布局、广播与生命周期，避免一个输入被提前复用。

## Batch transfer 与 GM traffic

适用：`batch_transfer.*`、`gm_traffic.*`、`ub_residency.*`。

- 合并前必须证明地址连续或可由合法 blockCount/stride表达；不能把非连续访问描述成连续 DataCopy。
- action写明有效长度、批量数、src/dst stride 的语义和单位，并要求 Implementation 依当前 API 处理字段上限。
- 中间结果驻留 UB时，明确生产/消费阶段、生命周期、覆盖时机和容量；删除 GM往返后仍须保持可见性和数学顺序。

## Double Buffer

适用：`double_buffer.*`。

- 仅在至少两个 chunk 可交错、搬运与计算独立、两槽容量不迫使主要 tile 缩小时实施；input/output Queue 使用两份 buffer 并重算容量。
- action描述 prologue、steady-state、epilogue，或说明 Queue如何自动轮转；必须保证消费者完成后才复用。
- action 明确所需流水槽数、峰值容量和复用边界；具体 Queue/event 表达由 Implementation 确定。

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

- 输出地址、批量粒度和对齐一起设计；非对齐尾块只提交有效范围。
- Vector 结果先在片上形成目标 ABI 布局，再用当前架构合法路径批量写回；避免依赖未经确认的逐元素 GM 转换。
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
