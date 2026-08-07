# AscendC Action API 知识

本文件是生成和修复 target/action 的仓库内稳定知识。具体签名与架构支持仍须在当前 SDK headers 中只读确认。

## GM 与数据搬运

- GM↔UB API 使用 `GlobalTensor<T>` 与 `LocalTensor<T>`；不得在 action 中要求把裸 `__gm__` 指针直接传给 `DataCopy/DataCopyPad`。
- `DataCopy` 仅用于搬运字节数严格 32B 对齐的路径；非对齐或运行时不确定长度优先使用 `DataCopyPad`。
- `DataCopyPad` 的 UB 起始地址仍须 32B 对齐。action 必须区分有效长度、对齐长度和 UB 行 stride。
- 优先使用 `DataCopyExtParams` 与 `DataCopyPadExtParams<T>`；`blockLen` 使用字节，字段类型以当前 header 为准。
- 逐元素 `GlobalTensor::GetValue/SetValue` 或裸 GM 标量读写只适合诊断或无法批量化的边界；生产 action 优先在 UB 组装后批量搬运。

## TBuf、TQue 与流水线

- MTE2/MTE3 搬运缓冲使用 `TQue<VECIN/VECOUT>`；纯 Vector 临时量使用 `TBuf<VECCALC>`。
- `EnQue/DeQue` 表达生产者—消费者依赖；不得把异步 `DataCopy` 后直接 Vector 计算写成默认安全。
- Double Buffer 由 `InitBuffer(queue, 2, size)` 的 buffer 数量启用，不由 `TQue` 模板 depth 决定。
- 直接使用 `TBuf` 跨 MTE/Vector 时，action 必须写明受支持的 event 或 barrier；不要无依据添加事件。
- Vector 结果被 Scalar 或 MTE3读取前必须存在明确同步/Queue关系；复用同一 UB 前必须确认消费者完成。

## 对齐、尾块与 Mask

- 普通 GM搬运硬约束基于32B；Vector API可能有更强限制，必须检查当前 SDK。
- DAV_2201 上 `Compare` 的 count 空间按256B对齐；FP32通常是64元素倍数。有效长度不能替代硬件计算长度。
- 尾块 action 必须给出：`valid_len`、`aligned_len`、padding值、计算count和实际写回长度。
- padding必须是数学中性值：Max用负无穷/最小值，Min用正无穷/最大值，Sum通常用0；同时确保无效lane不写回。
- `DataCopyPad` 的 padding字段范围和单位以当前 header为准；大 padding需改为先初始化 UB或分段搬运。

## dtype、Cast 与索引

- action 必须给出完整转换链，而不是只写“转换到输出类型”。
- DAV_2201 的 `Select` 路径通常以 half/float保存索引状态；是否支持 int32 dst必须查当前 header。
- float32只能精确表示不超过 `2^24` 的连续整数索引；超过该范围时不得使用 float索引状态。
- AICore Scalar对动态整数/浮点转换有限制。可用 float累加器替代循环内 `uint32→float`，或使用受支持的 Vector `Cast`。
- 输出为 int64而计算路径只支持 int32时，action必须明确 ABI保持方案；优先在 UB 构造正确布局并用 VECOUT/MTE3批量写回，不假设逐元素 float/int32→int64 GM路径可靠。

## 容量与参数范围

- 计算 UB预算时计入 Queue buffer数、输出Queue、所有 TBuf、mask、临时buffer及 API保留空间。
- Double Buffer按两份实际容量计入，不得只计算单tile。
- `blockCount`、repeat次数、stride字段范围以当前 header和架构文档为准；超限时分批。
- 硬件容量未知时 action应保持参数化边界，不得声称已确认充足；只有所有合理 tiling均不足才是硬阻断。

## 当前 SDK 核验

所有会影响可编译性的结论按以下顺序确认：

1. `source /data/lu/activate_evokernel.sh`；
2. 在当前 SDK headers 中搜索 API声明、结构字段和架构宏；
3. 再用本知识文件解释推荐模式；
4. 文档与当前 header冲突时，以当前 header为事实，在 action中选择当前可用替代路径。
