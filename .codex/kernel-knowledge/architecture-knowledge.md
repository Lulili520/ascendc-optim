# AscendC 硬件架构知识

本文件集中保存所有硬件 profile 的架构覆盖知识；不为每种硬件拆分独立文件。控制器先用 `architecture-index.json` 将精确设备身份映射为 profile，Implementation 只读被选 profile 中 actions 涉及的章节。实际重载、字段类型和 API 能力仍以当前 SDK headers 为准。

## profile: dav-2201

适用身份由 `architecture-index.json` 定义；当前映射为 `Ascend910B2C`，以本机 CANN 9.0.0 headers 核验。下列只保留相对通用契约的 profile 差异。

### transfer

- 普通 `DataCopy` 路径要求搬运长度及 UB 端起始地址满足 32B 对齐；非对齐或运行时有效长度使用当前 headers 支持的 Ext/Pad 路径。
- `DataCopyExtParams.blockLen` 表示有效字节数。多行 UB 跨度按 32B 对齐；GM stride 按字节、UB stride 按 32B data block，stride 是相邻 block 之间的间隔。
- CANN 9.0.0 中 `DataCopyExtParams` 的 `blockCount` 为 `uint16_t`，`blockLen/srcStride/dstStride` 为 `uint32_t`；表达式先证明范围再显式构造为字段类型，超限时分批。
- 有效行宽不是 32B 整数倍时，必须把有效元素数与 UB 存储跨度分开；例如 4095 个 FP32 的有效长度是 16380B，逐行 Vector 布局使用至少 4096 个 FP32 的对齐跨度，搬运和写回仍只提交有效范围。

### vector

- `Compare/Select/Cast/Reduce` 的 dtype、mask layout、count/repeat 和原地支持逐项查当前 DAV_2201 headers；Compare 的计算空间不能用有效元素数代替，FP32 常见 256B/64 元素粒度也必须由实际重载确认。
- 不假设直接 INT64 Vector Cast、int32/uint32 Select 或更高版本 MicroAPI 存在。
- 不使用设备端普通 C++ 运行时整型到浮点转换代替已确认的 Vector Cast；小范围行号可在证明 FP32 精确表示后使用 FP32 累加器。
- `Select`、`Not` 和 Compare 结果的 mask dtype/layout 必须来自同一组当前 header 重载，不能默认使用 `uint8_t` mask。

### lifetime

- MTE2 输入通常使用 `TQue<VECIN>`，MTE3 输出使用 `TQue<VECOUT>`，纯 Vector 临时量使用 `TBuf<VECCALC>`。
- 若搬运目标使用 `TBuf<VECCALC>`，CopyIn 后必须使用当前 headers 支持的 `MTE2_V` event；同一 Buffer 被下一轮 MTE2 覆盖前还要证明或加入 `V_MTE2`。`PipeBarrier<PIPE_MTE2>` 不能建立 MTE2→Vector 可见性。
- Vector 结果由 Scalar `GetValue` 消费时使用 `V_S`；Scalar `SetValue` 后由 MTE3 搬出时使用 `S_MTE3`。event ID 由 `FetchEventID/AllocEventID` 获取，Set/Wait 成对出现。
- CANN 9.0.0 中全局 pipe 获取形式是 `GetTPipePtr()`，不是 `AscendC::GetTPipePtr()`；`EVENT_ID0` 等 event 符号使用 header 声明的真实作用域，不默认加 `AscendC::`。Kernel 已持有 `TPipe` 对象时优先由该对象获取 event ID。
- Queue 双缓冲由 `InitBuffer(queue, 2, bytes)` 的 Buffer 数量开启，不由 `TQue` 模板 depth 表示；两份容量和事件资源都必须计入。

### device-error

- 出现 MTE configuration illegal、AIV/AIC memory error、地址未对齐或 Vector 执行异常时，一次复核本轮修改的全部搬运点。
- 同一 PC/症状连续出现说明旧根因假设无效；回到 headers 或已通过精度版本的合法模式，不继续修改无关位置。
