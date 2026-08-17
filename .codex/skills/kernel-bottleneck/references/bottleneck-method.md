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

## 分析顺序

1. 从 shape、axis、dtype 和数学语义理解当前执行分支；Reduction 区分 AR/ARA、多轴、FullLoad/Chunked 和 With-Index。
2. 跟踪 Host tiling、blockDim、参数传递与 Kernel 消费。
3. 跟踪 task 到输出区间的映射、余数、波次和单任务代价。
4. 跟踪 CopyIn/CopyOut 的连续性、跨步关系、粒度、对齐与 tail。
5. 跟踪 Scalar、Vector、Reduction、Cube 和 FixPipe 主路径。
6. 跟踪 Queue/TBuf 生命周期、片上布局、同步和流水 prologue/steady/epilogue。
7. 合并同一源码机制，只保留能够唯一决定 strategy 的根因。

## Cause 体系

| bottleneck_key | cause_key |
|---|---|
| `overhead.hot_path_inefficiency` | `repeated_hot_path_overhead`、`overpartitioned_task_mapping` |
| `tiling.execution_inefficiency` | `inactive_tiling_parameter`、`inefficient_work_unit_size` |
| `parallel.core_underuse` | `insufficient_parallelism` |
| `parallel.load_imbalance` | `uneven_task_distribution`、`nonuniform_task_cost` |
| `memory.transfer_inefficiency` | `scalar_global_contiguous_access`、`fragmented_contiguous_transfer`、`fragmented_regular_strided_transfer`、`scalar_irregular_gather_access`、`scalar_irregular_scatter_access`、`unaligned_transfer_tail`、`fragmented_global_writeback`、`atomic_write_contention` |
| `memory.reuse_inefficiency` | `repeated_global_transfer`、`redundant_gm_roundtrip`、`premature_buffer_eviction`、`overextended_buffer_lifetime` |
| `memory.onchip_conflict` | `ub_bank_conflict` |
| `pipeline.overlap_loss` | `serial_pipeline_stages`、`over_synchronization`、`pipeline_slot_reuse_serialization` |
| `compute.scalar_inefficiency` | `recomputed_invariant_scalar_work`、`scalar_local_lane_compute`、`scalar_reduction`、`scalar_elementwise_compute` |
| `compute.reduction_inefficiency` | `serial_chunk_reduction` |
| `compute.vector_dataflow_inefficiency` | `redundant_vector_materialization`、`underutilized_vector_width`、`vector_ub_bouncing`、`excessive_cast_chain` |
| `compute.cube_dataflow_inefficiency` | `inefficient_cube_tiling`、`low_cube_onchip_reuse`、`mismatched_fixpipe_layout`、`fragmented_fixpipe_writeback` |

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
- 槽所有权和复用时点直接迫使阶段串行：`pipeline_slot_reuse_serialization`。

## Scalar、Vector、Reduction 与 Cube 边界

- 循环内重复计算不变量：`recomputed_invariant_scalar_work`。
- LocalTensor 逐 lane 执行可向量化计算：`scalar_local_lane_compute`。
- 完整归约主体由 Scalar 循环完成：`scalar_reduction`。
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

- `inefficient_work_unit_size` 必须给出当前 work unit、峰值 Buffer 容量关系、任务数和搬运粒度；不能仅因 tile 看起来大或小而输出。
- `inactive_tiling_parameter` 必须沿 Host 写入到 Kernel 消费证明参数没有改变真实执行结构。
- `overpartitioned_task_mapping` 必须证明过细 block/波次重复固定控制；任务数大于核数本身不是问题。
- 双缓冲、增核、常驻、树形归约和布局调整只有在源码适用条件已成立时才产生对应 cause。
- 每个 cause 只出现一次；同一 cause 的多个位置合并为一个 issue 的多条 evidence。
- 同一源码机制的表现和后果写入 observation/reasoning，不重复生成多个 cause。

## 排序

1. 前置依赖：前项修改会改变后项 target、公式或成立条件。
2. 影响一致性：正式 performance 与源码机制直接对应。
3. 热路径乘数：按元素、tile、task 重复且覆盖更多数据者优先。
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
