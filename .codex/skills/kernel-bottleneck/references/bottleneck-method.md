# 直接源码瓶颈分析与确定 Cause

## 推理链

```text
shape/dtype/数学语义
→ 完整 Host/Kernel 执行路径
→ 具体且排他的源码根因
→ 可选 performance 影响证据
→ bottleneck_key + cause_key
```

直接阅读源码并输出问题，不生成 coverage、source model 或候选处置文件。performance 只补充影响、排序和因果一致性；禁止从最高 pipeline ratio 或固定阈值直接产生 cause。

`bottleneck_key` 是问题域，`cause_key` 是已经足以唯一决定修改方向的具体源码机制。若同一 cause 还需要在多个 strategy 中选择，必须在本阶段继续拆分 cause。

## 强制完整分析顺序

1. 从 shape、axis、dtype 和数学语义理解当前执行分支；Reduction 区分 AR/ARA、多轴、FullLoad/Chunked 和 With-Index。
2. 展开数学热路径的循环乘数，先计算每个输出的归约深度、标量运算和访存次数。
3. 跟踪 CopyIn/CopyOut 或裸 GM 访问的连续性、跨步关系、粒度、对齐与 tail。
4. 跟踪 Scalar、Vector、Reduction、Cube 和 FixPipe 主路径。
5. 跟踪写回的连续性、粒度、dtype 转换与公开 ABI。
6. 跟踪 Queue/TBuf 生命周期、片上布局、同步和流水 prologue/steady/epilogue。
7. 最后跟踪 Host tiling、blockDim、参数消费、task 到输出区间的映射、波次和单任务代价；不得用调度问题遮蔽更高乘数的计算或搬运问题。
8. 对每个阶段明确记录“确定问题”或“源码已是合理结构”的内部结论；不得因先发现一个主问题而停止后续阶段。
9. 合并同一源码机制，按“前置依赖、可消除热路径总成本、影响一致性、源码顺序”排序，只输出前 3 个能唯一决定 strategy 的根因。该上限只控制单轮实施面，不限制主体变化数量。

上述结论只存在于本次推理中，不写 `coverage.json`、`source_model.json` 或新的报告字段。

## 静态成本与完整性

- 搬运按 `有效字节 + 重复字节 + 搬运调用次数 × 启动开销` 比较；规则跨步循环必须判断能否由一次二维搬运表达，滑窗必须计算相邻窗口的重叠读取。
- Vector 按有效 lane、repeat/mask、指令链和 Cast/Select/Duplicate 中间量检查；Scalar 按元素、lane、tile 或 task 的实际重复次数检查。C++ 循环中的 LocalTensor `Compare/Select/Reduce` 仍是 Vector 指令，不是逐元素 Scalar 计算。
- Reduction 同时检查 chunk 内归约、chunk 间合并深度、索引伴随状态和同步次数。
- 任务映射计算 `waves=ceil(total_tasks/active_cores)`，并比较最大单核工作量、尾核、每 task 初始化和持久 task 循环。
- 流水只有同时存在可证明的 prologue、steady、epilogue 和正确槽所有权时，才按 `max(MTE2, compute, MTE3)` 理解；否则按未隐藏阶段成本处理。
- performance 只验证源码机制的影响方向。低 ratio 不能否定按高乘数执行的确定源码问题，高 ratio 也不能单独产生 cause。
- work unit、搬运布局、计算路径和向量宽度描述不同成本来源，可以在同一 symbol 上同时成立；只有修改方向与源码机制完全相同才允许去重。

## Cause 体系

生成报告前读取 [cause-taxonomy.json](cause-taxonomy.json)。它是 `cause_key → bottleneck_key + 必需源码 evidence_key` 的唯一事实源；不得自创 key。下面只保留各 cause 的排他判断边界。

## 搬运 Cause 排他边界

- 连续 GM 区间由 `GetValue/SetValue` 逐元素访问：`scalar_global_contiguous_access`。
- 已使用搬运 API，但连续区间被拆成多次小搬运：`fragmented_contiguous_transfer`。
- 循环地址满足固定 `blockLen + srcStride/dstStride`：`fragmented_regular_strided_transfer`。
- 地址由数据索引决定且是读取：`scalar_irregular_gather_access`。
- 地址由数据索引决定且是写入：`scalar_irregular_scatter_access`。
- 主体搬运合理，只有非对齐 tail 仍使用不合法或低效路径：`unaligned_transfer_tail`。
- 输出连续但被逐元素或小块写回：`fragmented_global_writeback`。
- 多任务确实写入重叠地址并使用 Atomic：`atomic_write_contention`；地址互不重叠的多次 Atomic 不属于 contention。

## Buffer、流水与同步边界

- 相同输入跨 tile 重复从 GM 搬入：`repeated_global_transfer`。
- 中间结果写 GM 后又立即读回：`redundant_gm_roundtrip`。
- 片上对象释放过早导致重新加载：`premature_buffer_eviction`。
- 不同时活跃的 Buffer 同时占据 UB：`overextended_buffer_lifetime`。
- UB 地址、stride 或 padding 与并发访问共同构成 bank/group 冲突，并有正式冲突指标佐证：`ub_bank_conflict`。
- Queue/Buffer 生命周期可行但 CopyIn/Compute/CopyOut 没有稳态交错：`serial_pipeline_stages`。
- 同步范围或频率超过真实依赖：`over_synchronization`。
- 槽所有权和复用时点直接迫使阶段串行，且存在至少两个可交错 chunk、独立搬运/计算和足够 UB：`pipeline_slot_reuse_serialization`。单槽本身不是充分证据。

## Scalar、Vector、Reduction 与 Cube 边界

- 循环内重复计算不变量：`recomputed_invariant_scalar_work`。
- LocalTensor 逐 lane 执行可向量化计算：`scalar_local_lane_compute`。
- 完整归约主体由标量 load、标量算术/比较和标量状态完成：`scalar_reduction`。循环体已用 LocalTensor `Compare/Select/Reduce` 更新多 lane 状态时排除此 cause。
- Vector/局部归约已经存在，但 chunk 间按线性依赖链合并：`serial_chunk_reduction`。
- 普通逐元素数学由 Scalar 主循环完成：`scalar_elementwise_compute`。
- 标量或不变量反复 Duplicate 成完整 Tensor：`redundant_vector_materialization`。
- work unit 合理但 repeat/mask 长期只覆盖少量合法 lane：`underutilized_vector_width`。
- Vector 生产者与消费者之间反复落 UB：`vector_ub_bouncing`。
- 连续 Cast 可在保持舍入语义下合并：`excessive_cast_chain`。
- M/N/K tile 与容量、并行或尾块不匹配：`inefficient_cube_tiling`。
- Cube 操作数本可驻留 L0/L1 却重复装载：`low_cube_onchip_reuse`。
- FixPipe layout 与 Cube 输出或公开 ABI 不匹配：`mismatched_fixpipe_layout`。
- FixPipe layout 正确但输出范围被碎片化提交：`fragmented_fixpipe_writeback`。

## 通用判断边界

- `inefficient_work_unit_size` 必须给出当前 work unit、容量允许的更大合法 work unit、任务数、Vector 覆盖和搬运粒度；不能仅因 tile 看起来小而输出。
- `inactive_tiling_parameter` 必须沿 Host 写入到 Kernel 消费证明参数没有改变真实执行结构。
- `overpartitioned_task_mapping` 必须证明 work unit 已合理后仍有过细 block/波次；若小 tile 直接造成 block 过多，只输出 `inefficient_work_unit_size`。
- 双缓冲、增核、常驻、树形归约和布局调整只有在源码适用条件已成立时才产生对应 cause。
- 每个 cause 只出现一次；同一 cause 的多个位置合并为一个 issue 的多条 evidence。
- 同一源码机制的表现和后果写入 observation/reasoning，不重复生成多个 cause。

## 排序

1. 前置依赖：前项修改会改变后项 target、公式或成立条件。
2. 影响一致性：正式 performance 与源码机制直接对应。
3. 总成本：热路径执行次数乘单次搬运、计算、同步或调度成本；不能只按 issue 类型或单次操作大小排序。
4. 稳定并列：按源码出现顺序。

## 输出

```json
{
  "reasoning": ["Process 的 row 循环执行固定 blockLen 的规则跨步小搬运，因此确定为 fragmented_regular_strided_transfer。"],
  "issues": [{
    "evidence": [{
      "evidence_key": "source.kernel.fragmented_regular_strided_transfer",
      "source": "op_kernel/x.cpp::Process",
      "observation": "row 循环每次搬运相同 blockLen，相邻地址具有固定 stride"
    }],
    "bottleneck": {
      "bottleneck_key": "memory.transfer_inefficiency",
      "cause_key": "fragmented_regular_strided_transfer"
    }
  }]
}
```
