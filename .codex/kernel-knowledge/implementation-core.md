# AscendC 源码优化与实施契约

Strategy 和 Implementation 只读取当前问题涉及的章节。这里保存跨架构的源码问题到目标结构映射及实施不变量；具体对齐粒度、字段上限、dtype/API 支持和指令语义来自阶段输入指定的架构覆盖层与当前 SDK headers。规则确定目标结构，不提供候选参数，也不增加分析阶段。

## transfer

- 对每个 CopyIn/CopyOut 在内部逐项闭合 `方向、dtype_bytes、GM/片上offset、valid_elems、transfer_elems、compute_elems、allocated_elems、block/stride、full/tail、API`，并证明两端范围不越界；这些事实写入 planning 的 `transfers/buffers/proofs`，不增加训练输出字段。
- `align_elems = 32 / sizeof(dtype)`；普通 `DataCopy` 的 UB 实际子地址和搬运字节数必须满足当前架构对齐。Buffer 总容量对齐不能证明动态 `len`、`coreStart` 或 `LocalTensor[offset]` 对齐，Host 将 tile 截断为 `min(tile,total)` 后必须重新证明。
- 分别确定 `valid_elems`、`transfer_elems`、`compute_elems`、`storage_stride`、`allocated_elems` 和 `writeback_elems`，满足有效范围不超过实际分配范围；二维片上地址只用 `storage_stride` 解码，不能把非对齐的有效长度直接当行跨度。padding 必须使用当前数学主体的单位元或安全值，无效元素不得写回。
- 完整连续对齐块使用普通 `DataCopy`；非对齐尾块或固定非对齐行宽使用当前 headers 确认的 `DataCopyPad`/Ext，并以字节设置 `blockLen`。禁止为方便而让所有完整块走 Pad，也禁止普通 `DataCopy` 向上取整后读写 GM 有效范围之外。
- 连续 Elementwise 优先按 32B data block 划分核间主区间，使普通核的 GM 起点和长度对齐，仅由最后一个有效核处理全局 tail；行/序列所有权不能拆分时按对齐 `storage_stride` 分核并保留每行有效长度。
- 热路径中的逐元素、逐 Vector block 或逐行 GM 搬运，按地址关系确定目标结构：连续区间合并为单次连续搬运；固定行宽和固定行距合并为二维搬运；其余访问只合并每个真实连续段。二维搬运必须明确 `blockCount`、`blockLen`、`srcStride`、`dstStride`、对齐跨度和有效长度；不能把不可表达的地址伪装为连续搬运。
- 批量行数/段数取 UB 峰值容量、API 字段上限和剩余并行任务共同允许的最大值；字段超限时按最大合法批次分段，不退化为逐行调用。比较方案时以 GM 逻辑字节和搬运事务数分别计数：驻留减少前者，合并搬运只减少后者。
- 传入 API/结构体字段的字节数、count 和 stride 必须在容量证明后显式构造为目标字段类型。`sizeof` 参与表达式时不得依赖 `size_t` 向窄类型的隐式转换，例如先形成 `uint32_t copyBytes = elems * static_cast<uint32_t>(sizeof(T))`。

## vector

- 在确定 Vector 链前，为每条实际指令列出每个源/目标的 `Buffer基址 + 动态子视图offset`，按 dtype 换算字节地址并验证该重载要求的对齐。只证明 Buffer 基址、总分配或 tile 起点对齐不够；任何未对齐子视图都必须在 Strategy 阶段改变布局，不能留给 Implementation 或精度失败后修补。
- 对每个 Vector API 从当前 headers 确认输入/输出 dtype、count/mask 单位、repeat/stride 范围、原地限制和临时空间。
- `Select` 的 `selMask` 是 API 定义的位布局，不是默认的逐元素 `uint8_t` 布尔数组。必须从同一组 header 重载闭合 mask dtype、每个输出 lane 对应的 bit、有效 bit 数和生成方式；禁止用 `mask.SetValue(i, 0/1)` 替代位打包，除非该精确重载明确定义为逐 byte mask。边界窗口优先通过正确 padding 布局消除 Select；保留 Select 时左/右边界的每个 tap 都必须有对应有效位证明。
- 热路径中的 Scalar `GetValue/SetValue` 循环或按单个 Vector block 手工循环，若同一 Vector API 能覆盖对齐主区间，则改成一次 count/repeat 调用或字段上限允许的最少次数；只为不能由该调用表达的 tail 保留单独路径。不得把 C++ 循环换成等量的小粒度 Vector API 循环后称为 vectorize。
- `vectorize` 也承载不转移到其他热路径的确定性 Scalar 数学冗余消除。此路径可以不新增 Vector API，但 planning 必须证明完整 launch 的 `scalar_iterations` 严格下降，保持每个保留结果的运算顺序，并证明新增镜像/复用写入不会恢复被消除的数学工作。
- 多段 Vector 链优先在同一片上 tile 完成，只有 dtype、原地限制或 workspace 要求才分配临时对象；每个临时对象计入峰值容量。融合不得改变舍入点、溢出、NaN 或比较语义。
- dtype 转换必须给出完整可编译链；索引表示必须证明精度范围，ArgMax/ArgMin 的 value/index 使用同一比较条件并保持首索引语义。

## reduction

- 先从 shape 和 axis 合并出 outer/reduce/inner，并据此确定每个独立输出、连续 lane 和跨 chunk 状态；禁止按源码现有循环层级直接沿用低效分块。
- 能在 UB 容量内 FullLoad 时一次装入整个归约工作集；否则选择保持足够独立任务的最大合法 reduce chunk。value、index、统计量等跨 chunk 状态常驻片上，片起始偏移在合并局部索引时只加一次。
- ArgMax/ArgMin 明确初值、相等时首索引、NaN 行为、索引 dtype/精度和 tail 中性值。无效 padding 可参与计算但不得胜出；输出只写有效范围。
- ARA 等固定行宽布局优先用二维搬运加载多行，并在完整对齐 inner lane 上执行 Compare/Select；不要为每行重新生成相同 mask、索引向量或循环不变量。
- 独立输出不足而 reduce 维很大时，跨核归约只采用契约确认的闭合结构：第一阶段各核写入按32B跨度隔离的 partial workspace，第二阶段由独立 Kernel/已确认的全局同步机制读取全部 partial 并唯一写回；或使用数学与 dtype 明确允许且已由当前 SDK 确认的 atomic。必须固定 workspace 初始化、partial 索引、可见性边界和 launch 次序。单 Kernel 内没有已确认全局 barrier 时不得假设 block 间可见，此方向记 unresolved，但不得判 clean。

## prefix-scan

- 将 `state=op(state,x[i]); y[i]=state` 识别为 Prefix/Scan，而不是普通 Scalar 循环。先固定方向、inclusive/exclusive、结合算子、单位元、跨 tile carry、特殊值和浮点结合顺序；只有结合律及精度边界允许时才改为分层 scan。
- 优先使用当前 SDK 已确认且语义一致的原生 scan。没有同名高阶 API 不等于不可 Vector 化；结合算子已有合法 Vector 二元 API 时，可用 ping-pong 的 Hillis-Steele/分层结构把逐元素依赖改为 `ceil(log2(valid_elems))` 个 Vector stage，并在 tile 间只保留 carry。
- 每级 shifted operand 必须通过合法对齐布局产生。普通 Vector API 要求32B对齐时，禁止直接把 FP32 `base[1]`、`base[2]` 或 `base[4]` 作为操作数；使用架构契约确认的对齐 Gather/重排链，且 Gather 的 dst/src/offset Tensor 均从对齐基址开始。
- 对齐 Gather 链的 offset 以 API 规定单位生成并钳制到合法范围；shifted operand 的前 `stride` 个元素写结合单位元，其余元素映射到上一级的 `i-stride`。随后从对齐基址对完整有效范围执行二元 Vector 运算。
- UB 峰值必须计入输入、输出、scan ping-pong、shifted operand、offset Tensor、carry 与所有 Queue 槽。先为 scan 主体保留这些对象，再反推 tile；不得保留无收益的输入双缓冲并因此放弃 Vector 主体。
- tail 使用实际 `valid_elems` 控制 offset 生成、Gather、单位元覆盖、Vector 运算和写回。tile 完成后从最后一个有效结果更新 carry，下一 tile 统一应用该 carry；不得把最后一个对齐 padding 当作 carry。
- 独立序列不足且单序列需要跨核时，采用显式三阶段结构：各核对互斥 segment 做 local scan 并写 segment product/sum，独立全局阶段按序计算 segment carry，最后各核对本 segment 做 carry correction。三个阶段必须由独立 Kernel launch 或当前 SDK 已确认的全局同步闭合，workspace 每段按32B跨度隔离；不得在普通单 Kernel 中假设 block 执行顺序。缺少该执行契约时保留为 unresolved，而不是把逐元素串行链判 clean。

## residency

- 同一输入、权重或中间结果在多个 tile/task 间重复访问时，先判断其合法复用范围；容量允许则驻留在 UB/L1/L0，明确装入次数、生产者、消费者、最后使用和覆盖点。只改变缓冲类型而未减少 GM 字节或重复加载次数，不构成驻留优化。
- 循环不变量、常量向量、mask 和地址表在其最外合法作用域生成一次。中间结果若所有消费者都能在片上完成，则删除对应 GM 写回与重读，但保持原数学顺序和可见性。

## lifetime

- 对每个片上对象标明 producer、producer pipeline、可见性边界、consumer pipeline、last-use 和复用点；跨流水数据必须选择 Queue，或显式写出正向可见性 event 和复用前的反向 event。
- LocalTensor→LocalTensor 的 `DataCopy` 仍由 MTE 产生数据；其后直接由 Vector 消费不会因两者同在 UB 而自动建立 RAW 依赖。能由首个 Vector 操作直接写归约目标时，删除该 Local copy；确实需要时必须使用当前架构合法的 Queue/event 闭合 `MTE→V`，并在复用前闭合反向依赖。
- `TBuf` 只分配计算临时内存，不提供 EnQue/DeQue 或自动跨流水同步。GM→片上→计算、计算→片上→GM 等阶段传递优先使用对应 `TQue`；只有完整证明当前架构 event 对时才使用 `TBuf` 手写流水。
- `TQue<VECOUT>` 的 `AllocTensor` 结果用于形成待 EnQue 的输出所有权，不作为跨多级 Scan 的长期 ping Buffer；Prefix 的 ping-pong 使用 `TBuf<VECCALC>`，最终一级再写入输出 Queue。不得为节省一个 tile Buffer 把未 EnQue 的输出 Tensor 跨多级 Gather/Mul/Barrier 复用。
- `PipeBarrier<PIPE_X>` 只解决同一 `PIPE_X` 内的数据依赖，不得代替 MTE2→Vector、Vector→Scalar、Scalar→MTE3 等跨流水 `SetFlag/WaitFlag`。
- Buffer 容量计入全部同时活跃 Queue 槽、临时对象、mask、padding 和 API workspace；流水槽数改变时同步重算容量和生命周期。
- 单槽 Queue 只提供所有权与同步；双槽只有容量容纳两份且至少两个 tile 能交错时才用于隐藏搬运。不得为了双缓冲缩小 tile 而忽略新增搬运、Queue 和循环开销。
- 若双槽使单次多行/多段搬运的 `blockCount` 或 chunk 缩小并增加总搬运事务，则选择事务更少的单槽结构；只有主要 tile/chunk 与事务数不变时才用剩余容量增加槽数。
- 只有源码明确形成 prologue、steady-state、epilogue，且 steady-state 中存在等价于 `CopyIn(i+1) / Compute(i) / CopyOut(i-1)` 的独立槽所有权，才判定为流水优化。仅把 `TBuf` 换成 `TQue`、把 Queue 初始化为两槽或增加 Barrier，不等于阶段重叠。
- 先分析现有 Queue/event 的异步生产消费依赖；源码按 `CopyIn(i+1)、Compute(i)、CopyOut(i)` 排列时，不得仅因缺少 `i-1` 文本就判定阶段串行。pipeline 策略必须指出阻止原路径重叠的直接依赖或槽冲突，并证明修改消除了该阻断；只延长输出 Tensor 生命周期不构成优化。

## tiling

- 参数求解必须从算子计算模式、独立输出和连续计算 lane 开始，不能从现有常数或某个 `kind` 反推。先固定数学主体与跨 tile 状态，再求任务所有权、主 tile/chunk、批量搬运，最后仅在不改变前三者时考虑流水。
- Host 的 blockDim、task/tile 数和 Kernel 解码共用同一组 tiling 字段；证明任务全覆盖、无重复，空核显式返回。
- 先枚举修改后同时活跃的输入、输出、状态、mask、workspace、padding 和流水槽，建立峰值容量式；容量式只证明合法上界，不证明该上界性能最优。tile/chunk 只有在减少主导硬件工作且不删除已有双缓冲、预取、ping-pong或库内部工作空间时，才取约束允许的最大有效值；否则保持当前结构或交回 unresolved，不用经验常数或运行时试参替代证明。
- `used_cores <= min(可靠物理核数, 独立任务数)`。核到任务使用连续均分或 grid-stride 映射，必须覆盖每个任务一次；tile 变大后重新计算 `total_tasks`，不得以耗尽并行任务换取局部大 tile。
- `logical_blocks > physical_cores` 不是空壳证据；多波逻辑任务可能缩短每个block串行链、减小负载尾差或隐藏数据等待。减少blockDim时必须证明完整launch的GM/Cube/Vector工作下降，且每物理核串行任务、驻留和已有等待隐藏不退化。`SetDim(n)`也不单独证明应启动n个block。
- 固定数值必须能回溯到 shape、dtype、容量、对齐或字段上限。例如历史版本的 chunk 数只能作为结构例子，不能跨算子或跨 shape 复用。

## static-work

- 每套闭合方案都计算同一 shape 下的静态工作向量：`GM有效字节、DMA burst、Vector repeat block、Cube运算、Scalar迭代、task/tile生命周期、状态初始化、同步事件、写回burst`。公式来自任务数、循环界限、dtype、每条指令硬件覆盖宽度和搬运 block，不读取 profiler。
- 禁止把源码层一次 `DataCopy/Vector API` 计成一次硬件事务。`DataCopy` 按有效字节与连续 block/burst 估算，Vector 按有效元素与当前 dtype 的 repeat 覆盖量估算；只减少 C++ API、tile 循环、`End`、逻辑block或对象生命周期，而不减少完整launch的burst/repeat、GM字节、数学元素、聚合重载或有直接依赖证据的等待时，不构成确定性能优化。
- 比较当前结构与目标结构时逐项列式。目标不得只降低某个局部量，却使另一项热路径工作量发生数量级增长；例如合并 DMA 不能以成倍增加 Vector 调用或 task 生命周期为代价，大 tile 不能耗尽独立任务，双缓冲不能增加主搬运事务。
- 不得把未知 shape 维度从静态账中删除。必须从 Host tiling、Kernel tiling 或输入 shape 恢复全部维度；无法恢复时不得宣称方案存在结构支配。
- 优先消除数学主体中的数量级冗余，再在等价主体之间依次减少 GM 有效字节、GM 事务、生命周期与同步。只有主要工作量均不退化时，才使用流水隐藏剩余阶段成本。
- 不要求把不同单位强行加权成单一分数。若两个合法布局互有得失，使用可证明的结构支配关系；无法证明支配时保留工作量更稳定、事务更少且实现路径已由当前 SDK 验证的方案，不生成候选试参。

## pattern-solver

- 先由主计算指令确定引擎：纯 Vector 主体使用设备 `aiv_core_count`，已有 Cube 主体使用 `aic_core_count`，真正 MIX 主体才使用混合拓扑。不得因 `SetBlockDim(24)`、SoC 名称或旧注释把 AIC 数量套给 Vector Kernel。
- Elementwise：以连续输出段为独立任务，先让独立输出覆盖可用 AIV，再融合完整 Vector 链；tile 只在减少真实 DMA burst、Vector repeat、同步或重载且不削弱核并行时改变。容量允许的最大 tile 若只减少源码循环次数，不生成 `resize_tile`。当目标的 GM 字节、DMA burst、Vector repeat、数学操作和有效核覆盖均不变时，必须输出 `strategy:null`；不得仅为改 blockDim、tile 常数、Queue 槽数或源码循环生成策略。
- Reduction/Norm/Softmax：从 `outer/reduce/inner` 建模。先最大化一次 Vector 覆盖的连续 inner lane；再用剩余容量最大化 reduce chunk，并让 value/index/stat 状态跨 chunk 常驻；最后决定多行 DMA 和任务映射。不得为了少量 DMA 事务把完整 lane 拆成大量 task 生命周期。
- Prefix/Scan：从独立序列数、序列长度、方向和结合算子建模。先消除逐元素 Scalar 依赖，再以输入/输出、ping-pong、shift、offset 和 carry 的完整容量式求 tile；核间优先分配独立序列，只有独立序列不足且已有合法跨核合并时才拆分单序列。
- Window/Pooling：先在直接窗口、可分离维度归约、滑动状态/前缀状态中求合法 Vector 数学主体，再为数学主体分配中间 Buffer 并反推 tile；禁止先用搬运 tile 占满 UB 后保留热路径 Scalar 计算。逐 tap/inner lane 列出 Vector 实际子地址；子视图不对齐时改为32B对齐 tap 平面或契约确认的重排。峰值容量式包含所有输入平面、中间结果、累加状态、输出、padding 和 Queue 槽；容量不足时缩小 tile，不得删除数学主体 Buffer。非空方案不得只降低 GM/DMA 而保留同数量级 Scalar 数学迭代。
- Scatter/Transposed Conv：先固定无冲突的输出所有权和私有累加范围，再确定输入批量、片上合并与一次写回，禁止用原子或重复清零掩盖所有权问题。
- Cube contraction：先固定 MNK 输出 tile 和核所有权，再联合求 K chunk、L1/L0 驻留和 FixPipe。父源码已有 Cube 时优化现有数据流；父源码为 Scalar/Vector contraction 时，仅在 planning 已闭合当前 SDK 精确 API、MNK、A/B/C 格式、L0A/L0B/L0C、重载次数、输出所有权、tail、同步和 ABI 后首次引入 Cube。当前与目标必须按完整launch分别给出 `A有效字节/B有效字节/C有效字节/A重载次数/B重载次数`；任务展平、核映射或 tile 顺序改变后重新证明外层驻留，禁止以减少 task 生命周期换取 A/B 重载增长。L0 payload 贴满名义容量时还须计入TCubeTiling要求的db系数和内部workspace，不能因单份payload可容纳就判定更大base必优。
- Fused/Network：先删除可在片上直接消费的中间 GM 边，再按各阶段共同 tile 闭合容量、生命周期和同步。

## cube

- 仅优化父源码已经由 `Matmul`、`Mmad` 或 `IterateAll` 承担主计算的路径；不把 Scalar/Vector contraction 首次改写为 Cube。
- 共享确定 MNK/K 分块、A/B/C 格式、L1/L0A/L0B/L0C 驻留、核到输出 tile 的所有权、K 尾块和 FixPipe 写回。tile 同时满足各级容量、指令粒度和足够独立输出任务；改变任一 tile 时同步更新 Host tiling、载入 stride、循环边界和写回地址。
- 权重或输入复用必须表现为更少的 GM/L1/L0 重载次数；仅增大 baseM/baseN/baseK 而未闭合容量、并行度和重载边界，不构成 Cube 优化。
- 不把更长的 `SetTensorA/B→IterateAll→End` 生命周期当作片上驻留证据；只有当前API契约或显式L1/L0数据流证明跨Iterate复用，才能减少目标重载账。只删除中间 `End` 而GM/L1/L0重载与直接等待不变时输出空策略。
- 当前与目标必须同时列出 `A/B/C有效字节、A/B的GM→L1和L1→L0重载次数、Matmul task数、K chunk数、格式转换次数、尾块元素比例`。若目标仅减少 task 或提高活跃核，但增加任一热路径 A/B 重载或格式转换，禁止生成策略。Small-K 的任务上限由 `ceil(M/workM)*ceil(N/workN)` 唯一输出tile数决定；Kernel必须用 `blockIdx` 解码互斥 `(mTile,nTile)` 并同步闭合A/B/C offset与M/N tail，没有该解码时禁止只改 `SetDim/SetBlockDim`。转置输入必须把转置或格式转换计入目标工作。已有 `Matmul/IterateAll` 无结构支配方案时输出 `strategy:null`。

## abi

- 保持公开输入输出、数学语义和工程已有 tiling 注册方式；字段变化同时更新定义、Host 赋值、序列化、Kernel 解析和必要 padding。
- 写回前在片上形成目标 dtype/layout；多核输出必须证明区间互斥，确有重叠时使用数学语义允许的归并方式。

## sdk-check

1. 激活仓库环境。
2. 修改源码前，从当前 SDK headers 定位策略新增或改变的每个 API、字段和符号；逐项确认精确作用域/所属对象、参数顺序与类型、dtype、mask/count/stride 单位、原地限制和架构宏。包括 event 常量在内的符号不得凭记忆添加 `AscendC::` 或其他命名空间。
3. 实施后按修改文件检查局部对象的“声明/获取 → 使用 → last-use/释放”闭环；删除或移动 Buffer、Tensor 或临时量后，不得保留旧标识符引用。
4. 在 `implementation.json` 中记录已核对的 API/header/调用形式；编译器已否定的形式不得再次生成。
5. 用设备文件的可靠容量和拓扑求解参数。硬件 profile 与 headers 冲突时以 headers 为准；未知架构不得套用其他 profile。
