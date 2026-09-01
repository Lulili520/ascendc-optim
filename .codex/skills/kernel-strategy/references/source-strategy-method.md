# 源码策略方法

## 求解顺序

1. 从输入元数据及完整 Host/Kernel 恢复数学语义、固定 shape/dtype、主引擎、独立输出、task 所有权、GM→片上→计算→写回路径和跨 tile 状态。缺失维度必须报告，禁止猜测。
2. 扫描全部输出路径：先完成下文的正确性与资源审计，再识别数学模式和支配性跨迭代依赖，最后按 `数学主体 → 纯冗余 → 重复GM → 碎片搬运 → 已有Cube → 工作粒度 → 任务所有权 → 流水` 扫描性能问题。同类确定问题一起处理；分类只保证覆盖，不决定局部参数。一个问题因契约不足不能冻结时继续扫描其他问题，禁止提前结束整轮。
3. 先确定数学主体、状态和唯一数据布局；随后逐个 CopyIn/CopyOut 闭合 dtype 字节数、GM/UB 实际 offset、valid/transfer/compute/allocated/writeback、full/tail 和 API，再由可靠容量、字段范围和任务数推导 tile、blockDim 与同步。所有 changes 必须共享同一套无冲突设计。
4. 比较当前与目标的完整静态工作：`gm_bytes/dma_bursts/vector_repeat_blocks/cube_ops/scalar_iterations/task_lifecycles/state_initializations/sync_events/writeback_bursts`。所有项按完整 launch 聚合，不能只计算单个新增 task。API 调用、C++ 循环、`End` 或逻辑 block 次数不等于硬件工作；任何局部下降都不能换来其他热路径的数量级增长。
5. 非空方案必须减少数学主体、完整 launch 的 GM/DMA/重载、硬件 Vector/Cube 工作或有直接依赖证据的关键路径。只减少 `task_lifecycles/state_initializations/sync_events` 而 GM、DMA、Vector/Cube、重载和已证明流水均不改善时判空。Elementwise 已经一次读写且Vector主体完整时，只有核数变化且当前确有空闲AIV、独立任务充足、每核工作下降才允许 `parallelize`；只改tile、Queue或源码循环直接判空。已有Cube的任务展开不得增加A/B热路径重载或格式转换。
6. 对每个发现分别判定：`actionable` 表示证据、API、容量、地址、同步和 ABI 均闭合并进入本轮策略；`unresolved` 表示源码低效或风险明确但目标结构缺少必要契约，不得伪造 action，也不得据此宣称源码无问题；`clean` 表示该检查项不存在可消除工作。只把 actionable 导出为既有七种 kinds；全量扫描后没有 actionable 才输出 `{"strategy":null}`，其含义仅为“当前输入下无确定可实施策略”。

## 正确性与资源审计

以下是问题识别维度，不增加输出 kind。每项都检查 current 和 target；target 的证明写入 planning/proofs 与 strategy/guards。

1. **语义与 dtype**：恢复公式、输出 shape/layout、累加 dtype、舍入点、epsilon、NaN/Inf、比较相等与首末索引语义。Softmax/LogSoftmax 保持减 max 的稳定形式；Norm/Variance 不得因重排改变允许误差之外的累加语义。
2. **任务所有权与合并**：证明 `blockIdx→task→output` 全覆盖、无重复、无越界，空核返回；多核 partial 必须闭合 workspace/atomic、初始化、可见性和最终唯一写回。单核全局 Reduction 不能仅因缺少现成跨核代码而判 clean。逻辑 block 数大于物理核数不等于冗余：只有删除逻辑 block 后完整 launch 的热路径工作下降，且每核串行工作、负载尾差、已有等待隐藏和数据驻留均不变差，才允许减少 blockDim。
3. **地址、对齐与 tail**：每个 Copy 和 Vector 操作数分别列出 dtype 字节、GM/片上实际子地址、valid/transfer/compute/allocated/writeback、block/stride、full/tail。Buffer 基址或总容量对齐不能证明动态子视图对齐；padding lane 使用数学中性值且不写回。
4. **容量与生命周期**：峰值容量计入所有同时活跃的输入、输出、Queue 槽、workspace、partial、scalar、cast、mask、padding、ping-pong 和 Cube L1/L0 对象；标明 producer、consumer、last-use 和复用点，禁止隐式依赖 `tile` 足够大或区域互不重叠。容量不等式只证明可行，不证明性能最优；不得仅因更大 tile/baseM/baseN/baseK 填满 UB/L0 就冻结最大值。若增大 tile 删除 Queue 槽、预取、ping-pong 或库内部双缓冲空间，必须证明失去的重叠不在原关键路径，否则该方向为 unresolved。
5. **跨 tile 状态**：Reduction 固定 value/index/stat 合并公式；Prefix/Scan 固定结合算子、方向、inclusive/exclusive、分段语义和 carry；Window 固定 padding 与滑动状态。逐元素 Scalar carry 是已识别问题，只有缺少合法重排/API 时才是 unresolved，不是 clean。
6. **指令与 workspace**：从契约确认精确 API 家族、dtype、原地限制、mask/count/repeat/stride 单位与最小 workspace；比较通用 Reduce 与 Whole/Block/PairReduce 或分层 partial，禁止固定大区域无效二级归约和高频 Vector→Scalar→Vector 串行链。
7. **同步与 Queue**：TQue 承担跨流水所有权与事件，TBuf 只作临时存储；逐对象证明 MTE2→V、V/S、V→MTE3、LocalCopy→V 与复用前反向依赖。缺少显式 barrier 不自动判错，已有 Queue/编译器同步也不自动判为 pipeline 优化。
8. **Host/Kernel ABI**：tiling 字段定义、Host 赋值/序列化、Kernel 解析、单位、窄类型范围、blockDim、workspace 和转置/format 标志一致；任何 task/tile/Buffer 改动同步闭合 ABI。

## 问题与 kind

| 扫描项 | kind | 判定边界 |
|---|---|---|
| 同一数据跨消费者重复访问 GM，能在 UB/L1/L0 保留 | `reuse_onchip` | 必须减少 GM 总字节；只减少事务属于搬运 |
| 连续或规则跨步范围被逐元素/小块搬运 | `batch_transfer` | 逻辑字节基本不变，DMA 事务减少 |
| Scalar 或低效 Vector 承担非 Cube 数学主体 | `vectorize` | 包含以 Vector API 替换主体，以及不转移工作的确定性 Scalar 数学冗余消除；纯复制的 GetValue/SetValue 属于搬运 |
| Cube 承担目标 contraction 主体 | `cube` | 已有 Cube 调整 MNK/K、L1/L0、核映射、FixPipe；首次引入仅在 SDK API、完整 MNK、格式、L1/L0、重载、所有权和 ABI 全部闭合时允许 |
| 所有权、引擎和驻留不变，只调整单 task/chunk | `resize_tile` | tile 必须由容量、并行任务和已有重叠共同推导，不能取容量最大值替代性能证明 |
| 独立输出到物理核存在空核、重复、遗漏或不均衡 | `parallelize` | 只改变 task 大小不属于并行化 |
| 工作量、字节和所有权不变，只改变阶段依赖/重叠 | `pipeline` | 必须证明原重叠阻断和目标新增重叠 |

纯冗余作为配套 change，不单独生成 kind。kinds 按表中顺序去重；辅助 Buffer、tile、blockDim 或 Queue 变化不重复分类。
对齐、tail、容量、生命周期、数值语义、同步和 ABI 是跨 kind 的正确性约束，不单独生成 kind；Reduction/Prefix/Window 是计算模式，数学主体变化仍归 `vectorize`；跨核 partial 归 `parallelize`；减少 GM 字节或事务分别归 `reuse_onchip` 或 `batch_transfer`。

## 计算模式硬约束

- **Elementwise**：连续 tile、批量 DMA、Vector 链融合；每个输入通常只读一次、输出只写一次。
- **Prefix/Scan**：源码出现循环外状态初始化、循环内 `state=op(state,x[i])` 且逐迭代写出该状态时，确定分类为 `prefix_scan`，不得当作普通 Elementwise 或 Reduction。先冻结结合算子、方向、inclusive/exclusive 和跨 tile carry；依次检查契约确认的原生 scan、按 Vector lane 分块的 UB 分层 scan、整 tile 对齐 Gather/重排。没有同名高阶 API 不等于不可实施；契约已确认 Gather 与对应 Vector 二元算子时不得输出 `strategy:null`。不得把未对齐 `base[stride]` 作为 Vector 子视图。先确定局部 scan、block product、block carry 与 correction，再由输入、输出、ping-pong、shift、offset、carry 和 Queue 槽反推 tile，禁止先用搬运 tile 占满 UB。分块方案必须同时统计 `blocks_per_tile*scan_stages` 次局部 Gather/二元指令生命周期和 barrier；不能仅凭元素级 stage 从 `log2(tile)` 降为 `log2(block)` 就判定更优。
- **跨核 Reduction/Scan**：单 Kernel 没有已确认的 block 间全局 barrier 时禁止假设 block 执行顺序或 partial 可见性。Reduction 仅冻结“分核 partial workspace→独立全局归并阶段→唯一写回”或 SDK 已确认且数学允许的 atomic；单序列 Scan 仅冻结“segment local scan→独立 segment carry scan→carry correction”的多阶段结构。workspace 按核/segment 的32B隔离跨度、初始化、launch 顺序和最终所有者全部闭合时才 actionable，否则保留 unresolved。
- **Reduction/Norm/Softmax**：明确 outer/reduce/inner、跨 chunk value/index 状态、相等语义、tail 与稳定公式；避免逐元素 Scalar 主体和无效二级归约。partial 数组只按实际 `partial_count` 分配和二级归约；workspace 按当前 API、dtype 与 tile 参数推导。整行可驻留时优先一次 GM 读完成统计与输出，但若整行驻留需要把输入 Queue 从双槽降为单槽，必须同时比较原 `CopyIn(i+1)/Compute(i)` 重叠与目标减少的 partial/Scalar 工作，不能仅凭 tile 生命周期减少冻结整行方案。不能驻留时明确重复 GM 字节。频繁单元素 Exp/Log、每 tile Scalar readback 或 running state 串行合并必须标为问题并比较向量化 partial merge。
- **Window/Pooling/Conv**：先确定直接窗口、可分离归约或滑动状态的 Vector 数学主体，再计入全部对齐 tap/临时平面求 tile；不得只降低 GM/DMA 而保留同数量级 Scalar 主体。逐 tap 冻结左/右 padding 和有效 lane；使用 `Select` 时必须冻结 header 定义的位布局及生成方式，禁止把逐 byte `0/1` 数组当作逐元素 mask。首 tap 优先由 Vector/Gather 直接初始化归约目标；若使用 LocalTensor `DataCopy` 初始化后再由 Vector 消费，策略必须同时冻结 MTE→V 可见性。
- **Cube contraction**：从 shape 恢复完整 M/N/K；当前与目标分别计算 A/B/C 有效字节、A/B 在 GM/L1/L0 的重载次数、Matmul task、K chunk、格式转换和尾块比例。任务展平不得增加重载；Small-K 的独立任务数固定为 `ceil(M/workM)*ceil(N/workN)`，`used_cores=min(aic_core_count,独立输出tile数)`，每个 `blockIdx` 必须解码唯一 `(mTile,nTile)` 并同步闭合A/B/C offset与M/N tail。现有Kernel没有该任务解码时，禁止只增大 `SetDim/SetBlockDim`。转置输入必须计入转置/格式转换。首次把 Scalar/Vector contraction 改为 Cube 时，planning 必须显式写出“首次引入Cube”，并闭合当前 SDK 的 Matmul/Mmad API、MNK、A/B/C 格式、L0A/L0B/L0C 容量、A/B 重载次数、输出所有权、tail、同步和 Host/Kernel ABI；任一项缺失均为 unresolved。
- **已有 Cube**：Host tiling、MatmulType 与运行时 transpose/format 标志必须一致；`IterateAll/End` 的同步/异步提交边界、C 地址范围和对象生命周期均需证明。`SetDim(n)` 不证明必须启动 n 个 block；blockDim 的改变必须从每个逻辑 task 的真实输出边界和完整 launch 重载推导。增加 used cores 时按所有新增 task 聚合 A/B 的 GM→L1、L1→L0 重载与共享路径竞争；增加任一热路径聚合重载、L2/GM 流量或格式转换时不是确定优化。减少 `End` 只有在源码或契约证明它阻断真实异步提交/片上驻留，且目标确实减少重载或等待时才是 pipeline；不把对象生命周期延长等同于 B/A 保持在 L1/L0。
- **Scatter/Transposed Conv/Fused**：先闭合输出所有权、私有累加、合并写回及中间 GM 消除。

Vector tile 必须满足全部活跃片上对象、32B/Vector 对齐、API 字段上限和 `total_tasks >= usable_cores`（语义上只有少量独立任务除外）。容量允许的最大值不是候选试参，也不是确定最优值；只有更大 tile 减少主导硬件工作且不删除已有有效双缓冲/预取、不拉长主导 Reduction/Vector 串行链时，才选择相应最大有效值，否则保持当前 tile 或判 unresolved。
32B 对齐按 `tile_elems*sizeof(dtype)` 与每个实际 GM/UB 子地址判断，不要求 `tile_elems` 本身是 32 的倍数。不得以 `InitBuffer(AlignUp(...))` 代替动态搬运长度和起始 offset 的证明；普通连续对齐块使用 `DataCopy`，真实非对齐尾块/行宽才使用 headers 已确认的 `DataCopyPad`/Ext。连续 Elementwise 优先按 data block 分核，只让最后一个有效核拥有全局 tail。
连续 Elementwise 的目标若不减少 GM 字节、DMA burst、Vector repeat、数学操作或增加有效核覆盖，直接输出 `strategy:null`；只改 tile/blockDim/Queue 或源码循环不是确定优化。

Cube tile 同时满足：

```text
baseM*baseN*sizeof(accum)*dbL0C <= L0C
baseM*baseK*sizeof(input)*dbL0A <= L0A
baseK*baseN*sizeof(input)*dbL0B <= L0B
ceil(M/workM)*ceil(N/workN) >= usedAICores
```

Strategy 只冻结 `knowledge_contract` 已确认存在的 API 家族；Implementation 再定点核验 header 中的精确签名、dtype、mask/count/stride 单位和架构条件。
`pipeline` 分为两种合法结构：Vector/MTE 流水必须闭合双槽Queue所有权；已有Cube内部流水必须改变包含 `IterateAll/End` 的真实提交边界，并在 current/target 中给出提交次数、被消除的直接等待依赖或重载，并保持输出tile唯一所有权。只减少 `End` 文本次数、延长Matmul对象生命周期或添加无关TQue不得生成 pipeline。

## 输出

非空策略先生成内部 `planning.json`，字段为：

```text
source_fingerprint, pattern, engine, available_cores, used_cores,
shape_model, task_mapping, parameters, buffers, transfers, compute,
work.current, work.target, proofs
```

精确格式由阶段 input 的 `strategy_output_contract` 给出。`pattern` 只取 `elementwise/prefix_scan/reduction/norm_softmax/window_pooling/scatter_transposed/existing_cube/fused_network`，`engine` 只取 `aiv/aic/mixed`；`shape_model/parameters/buffers/transfers/compute/proofs` 均为非空、无换行的字符串数组。

`work.current/target` 必须覆盖九项静态工作量且为同一 shape 的非负整数。`proofs` 证明任务覆盖、容量、地址、dtype、对齐、tail、同步和 ABI；pipeline 还要证明原阻断与目标重叠。

`strategy.json`：

```json
{"strategy":{"kinds":[],"evidence":[],"reasoning":[],"targets":[],"changes":[],"guards":[]}}
```

- `evidence`：`相对文件::symbol | 起始源码事实 | shape/循环/字节/任务公式`，覆盖每个问题；证据位置必须存在，但不要求属于直接修改 targets。
- `reasoning`：恰好六条，依次为 `[任务]`数学语义、`[现状]`当前路径、`[问题]`结构与公式、`[策略]`目标结构、`[推导]`参数与资源公式、`[边界]`修改后不变量。多 kind 子句按 kinds 顺序一一对应；只描述当前源码和本轮目标。
- `targets`：仅列直接编辑的真实 symbol，去重排序。
- `changes`：`相对文件::symbol | 当前结构 -> 目标结构`；当前结构可在 target 正文定位，目标可实施，不写原因或收益。
- `guards`：`约束对象 | 修改后必须成立的不变量`，覆盖数学、所有权、地址、生命周期、tail 和 ABI。

文本保持单行、明确、无 Markdown 和候选词。禁止“可能、尝试、建议、考虑、或许”。
