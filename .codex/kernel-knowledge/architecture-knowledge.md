# AscendC 硬件架构知识

本文件集中保存所有硬件 profile 的架构覆盖知识；不为每种硬件拆分独立文件。控制器先用 `architecture-index.json` 将精确设备身份映射为 profile，Implementation 只读当前策略涉及的章节。实际重载、字段类型和 API 能力仍以当前 SDK headers 为准。

## profile: dav-2201

适用身份由 `architecture-index.json` 定义；当前映射为 `Ascend910B2C`，以本机 CANN 9.0.0 headers 核验。下列只保留相对通用契约的 profile 差异。

### transfer

- 普通 `DataCopy` 路径要求搬运长度及 UB 端起始地址满足 32B 对齐；非对齐或运行时有效长度使用当前 headers 支持的 Ext/Pad 路径。
- 对齐按字节而不是按元素名义值判断：`align_elems = 32 / sizeof(dtype)`。FP32、FP16/BF16、INT8 的普通连续搬运长度分别至少按 8、16、32 元素粒度闭合；tile 元素数不必是 32 的倍数。
- `DataCopyExtParams.blockLen` 表示有效字节数。多行 UB 跨度按 32B 对齐；GM stride 按字节、UB stride 按 32B data block，stride 是相邻 block 之间的间隔。
- CANN 9.0.0 中 `DataCopyExtParams` 的 `blockCount` 为 `uint16_t`，`blockLen/srcStride/dstStride` 为 `uint32_t`；表达式先证明范围再显式构造为字段类型，超限时分批。
- 有效行宽不是 32B 整数倍时，必须把有效元素数与 UB 存储跨度分开；例如 4095 个 FP32 的有效长度是 16380B，逐行 Vector 布局使用至少 4096 个 FP32 的对齐跨度，搬运和写回仍只提交有效范围。
- 普通块优先使用 `DataCopy`；只有真实非对齐尾块或非对齐固定行宽使用当前 headers 支持的 `DataCopyPad`/Ext。`DataCopyPad` 不免除 UB 起始地址、目的存储跨度和容量证明，也不得用向上取整的普通 `DataCopy` 越过 GM 有效边界。

### vector

- Vector 地址合法性按每条实际指令的每个操作数检查，不按 Buffer 基址检查。对 `LocalTensor<T> base` 的子视图 `base[offset]`，实际 UB 字节地址为 `base_addr + offset*sizeof(T)`；DAV_2201 要求该重载的操作数地址32B对齐时，FP32 offset 必须是8元素的整数倍。`base[1]` 至 `base[7]` 对应4B至28B偏移，不能直接作为 `Add/Mul/Compare/Select/Reduce` 等 Vector 操作数。
- 滑窗相邻 tap 的元素偏移通常不是32B整数倍，禁止把一份连续输入 Buffer 的 `inputLocal[tap]` 直接当作各 tap 的整段 Vector 源。必须选择当前契约确认的合法布局：把每个 tap/inner lane 搬入独立32B对齐片上平面，或使用 headers 已确认且能表达该重排的 Gather/搬运路径；所有平面、输出和临时量同时计入 UB 峰值。
- Prefix/Scan 的 stride=1/2/4 FP32 子视图同样不满足32B对齐。CANN 9.0.0 DAV_2201 已验证的合法结构是：`ArithProgression<int32_t>` 生成字节 offset，`Maxs<int32_t>` 将负 offset 钳制为0，`Gather<float>` 从对齐 scan 基址生成对齐 shifted Tensor，`Duplicate<float>` 把前 `stride` 个元素覆盖为结合单位元，再由对齐基址 `Mul/Add/Max/Min` 完成整段 stage。offset Tensor 以 `uint32_t` 传给 Gather，所有 Gather offset 必须落在当前有效 scan 范围。
- DAV_2201 上设备错误 507035 且日志明确为 `The UB address accessed by the VEC instruction is not aligned` 时，属于源码子视图布局错误。不得重试环境；定位该 PC 对应 Vector 指令的全部 src/dst 实际字节地址，并改为对齐平面或上述 Gather 链。
- `Compare/Select/Cast/Reduce` 的 dtype、mask layout、count/repeat 和原地支持逐项查当前 DAV_2201 headers；Compare 的计算空间不能用有效元素数代替，FP32 常见 256B/64 元素粒度也必须由实际重载确认。
- 不假设直接 INT64 Vector Cast、int32/uint32 Select 或更高版本 MicroAPI 存在。
- 不使用设备端普通 C++ 运行时整型到浮点转换代替已确认的 Vector Cast；小范围行号可在证明 FP32 精确表示后使用 FP32 累加器。
- `Select`、`Not` 和 Compare 结果的 mask dtype/layout 必须来自同一组当前 header 重载，不能默认使用 `uint8_t` mask。CANN 9.0.0 `Select` 按 `selMask` 的 bit 选择源；普通 `uint8_t` 元素写入值 1 只会置该 byte 的一个 bit，不等于 8 个或一个逐元素布尔 lane。使用 count 重载时仍必须按 header 定义生成完整位布局，或由匹配的 Compare 路径产生 mask。

### lifetime

- MTE2 输入通常使用 `TQue<VECIN>`，MTE3 输出使用 `TQue<VECOUT>`，纯 Vector 临时量使用 `TBuf<VECCALC>`。
- 若搬运目标使用 `TBuf<VECCALC>`，CopyIn 后必须使用当前 headers 支持的 `MTE2_V` event；同一 Buffer 被下一轮 MTE2 覆盖前还要证明或加入 `V_MTE2`。`PipeBarrier<PIPE_MTE2>` 不能建立 MTE2→Vector 可见性。
- UB 内 LocalTensor→LocalTensor `DataCopy` 与后续 Vector API 也是 MTE→V 依赖，不得无 event/Queue 直接消费。滑窗归约的首个 tap 优先用 `Gather(dst, ...)` 直接初始化 dst，后续 tap 再 `Gather(tmp, ...)` 与 `Max(dst,dst,tmp,...)`，避免为初始化引入跨流水 copy。
- Vector 结果由 Scalar `GetValue` 消费时使用 `V_S`；Scalar `SetValue` 后由 MTE3 搬出时使用 `S_MTE3`。event ID 由 `FetchEventID/AllocEventID` 获取，Set/Wait 成对出现。
- CANN 9.0.0 中全局 pipe 获取形式是 `GetTPipePtr()`，不是 `AscendC::GetTPipePtr()`；`EVENT_ID0` 等 event 符号使用 header 声明的真实作用域，不默认加 `AscendC::`。Kernel 已持有 `TPipe` 对象时优先由该对象获取 event ID。
- Queue 双缓冲由 `InitBuffer(queue, 2, bytes)` 的 Buffer 数量开启，不由 `TQue` 模板 depth 表示；两份容量和事件资源都必须计入。

### device-error

- 出现 MTE configuration illegal、AIV/AIC memory error、地址未对齐或 Vector 执行异常时，一次复核本轮修改的全部搬运点。
- 同一 PC/症状连续出现说明旧根因假设无效；回到 headers 或已通过精度版本的合法模式，不继续修改无关位置。
