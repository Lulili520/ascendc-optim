# Bottleneck 到 Strategy 的推导

## 1. Strategy Key

| Strategy Key | 名称 | 具体介绍 | CannBot 来源 |
|---|---|---|---|
| `parallel.use_available_cores` | 提高有效并行度 | 根据可并行任务数和物理核数增加实际参与计算的核。 | 核数 Quick Diagnosis |
| `parallel.balance_tiling` | 均衡多核 Tiling | 将尾块或不同代价任务均匀分配，降低最慢核 cycle。 | P4 |
| `overhead.reduce_launch_cost` | 降低固定开销 | 减少小任务下的初始化、栈访问和 Scalar LD/ST 开销。 | P54 |
| `compute.vectorize_reduction` | 标量归约向量化 | 将逐元素标量 Sum/Max/Min/Arg 归约改为 Vector Reduce 路径。 | P94 |
| `compute.vectorize_elementwise` | 标量逐元素向量化 | 将逐元素标量算术循环映射为 Vector API 链。 | P95 |
| `memory.batch_transfer` | 增大搬运粒度 | 合并小粒度搬运，使用连续或批量 DataCopy。 | P10 |
| `memory.improve_l2_reuse` | 改善 L2 复用 | 根据数据复用和容量设置 Cache 策略，减少无效 GM 访问。 | P52/P61 |
| `pipeline.double_buffer` | 搬运计算重叠 | 在已确认串行的迭代流水中使用双缓冲隐藏搬运延迟。 | P1 |
| `memory.reduce_gm_traffic` | 减少 GM 总流量 | 带宽已经饱和时减少搬运总量或提高片上复用。 | MTE2 交叉诊断 |
| `memory.avoid_ub_bank_conflict` | 规避 UB 冲突 | 调整 UB 地址、stride 或 padding，降低 bank/bankgroup 冲突。 | P65 |
| `compute.reduce_cast` | 减少类型转换 | 合并或消除主路径中的冗余 Cast。 | VEC Bound 速查 |
| `compute.low_latency_reduction` | 低延迟 Vector 归约 | 对已向量化的归约路径选择更低延迟的归约组合。 | P68/P84 |
| `memory.keep_intermediate_in_ub` | 中间结果驻留 UB | 将连续计算链的中间结果保留在 UB，避免 GM 往返。 | P69 |
| `cube.improve_onchip_reuse` | 改善 Cube 片上复用 | 提高 L0A/L0B/L1 数据驻留和复用，减少 Cube 等待。 | Cube Bound 速查/P53 |
| `output.align_batch_writeback` | 对齐并合并写回 | 改善 FixPipe/MTE3 输出地址对齐和写回粒度。 | P56/P66 |

## 2. 正式推导表

从上到下匹配第一条成立的规则。`Required Evidence Key=-` 表示只需瓶颈 key。

| Bottleneck Key | Required Evidence Key | Strategy Key |
|---|---|---|
| `parallel.core_underuse` | - | `parallel.use_available_cores` |
| `parallel.load_imbalance` | - | `parallel.balance_tiling` |
| `overhead.small_workload` | - | `overhead.reduce_launch_cost` |
| `pipeline.scalar_bound` | `source.scalar_reduction_loop` | `compute.vectorize_reduction` |
| `pipeline.scalar_bound` | `source.scalar_elementwise_loop` | `compute.vectorize_elementwise` |
| `pipeline.scalar_bound` | `source.init_or_ldst_heavy` | `overhead.reduce_launch_cost` |
| `pipeline.mte2_bound` | `metric.small_transfer` | `memory.batch_transfer` |
| `pipeline.mte2_bound` | `metric.low_l2_hit` | `memory.improve_l2_reuse` |
| `pipeline.mte2_bound` | `source.serial_copy_compute` | `pipeline.double_buffer` |
| `pipeline.mte2_bound` | `metric.bandwidth_saturated` | `memory.reduce_gm_traffic` |
| `pipeline.vector_bound` | `metric.ub_bank_conflict` | `memory.avoid_ub_bank_conflict` |
| `pipeline.vector_bound` | `source.gm_roundtrip` | `memory.keep_intermediate_in_ub` |
| `pipeline.vector_bound` | `metric.excess_cast` | `compute.reduce_cast` |
| `pipeline.vector_bound` | `source.reduction_path` | `compute.low_latency_reduction` |
| `pipeline.cube_bound` | `source.low_onchip_reuse` | `cube.improve_onchip_reuse` |
| `pipeline.fixp_bound` | `source.unaligned_or_small_write` | `output.align_batch_writeback` |
| `pipeline.mte3_bound` | `source.unaligned_or_small_write` | `output.align_batch_writeback` |

同一瓶颈若具备多个标签，表中靠前的规则优先。未命中时返回 `strategy: null`，不得由 Agent 自由补选。

Vector Bound 的固定优先级为：UB 冲突 → GM 往返 → 冗余 Cast → 归约指令。先处理放大流水线耗时的问题，再处理数据流，最后处理局部指令路径。

## 3. 推导链

输出必须能还原：

```text
bottleneck_key
  + matched evidence_key
  → strategy_key
  → change_key[]
```

例如：

```text
pipeline.scalar_bound
  + source.scalar_reduction_loop
  → compute.vectorize_reduction
  → vector_reduce.stage_input_in_ub
  → vector_reduce.replace_scalar_reduction
  → vector_reduce.handle_tail
```

固定映射只决定 `strategy_key` 和有序 `change_key[]`。策略 Agent 随后必须阅读当前源码，为每个 change 生成真实的 `target` 与具体 `action`：

```json
{
  "change_key": "vector_reduce.replace_scalar_reduction",
  "target": "op_kernel/argmax.cpp::ArgmaxKernel::Process",
  "action": "将主体标量最大值循环替换为分块 Vector Reduce，并保持全局索引偏移和相同值取首个索引的语义。"
}
```

若找不到目标，或确认 API、dtype、容量、公开接口、数学语义存在硬冲突，则输出 `strategy: null`。不要把仅有通用描述、需要下游重新猜测的策略作为可执行策略输出。

## 4. Strategy 校验门禁与单轮修复

固定推导完成且 target/action 已具体化后，按 [`strategy-validation.md`](strategy-validation.md) 执行统一校验门禁。固定契约与可实施性在同一次首检中检查，不形成 valid/preflight 两套状态：

```text
具体化草稿 → 统一首次检查（固定契约 + 可实施性）
  ├─ pass → 原样输出，不修复
  ├─ repair_required → 统一修复一次 → 复检一次 → 输出
  └─ hard_blocked → strategy=null
```

修复期间 `strategy_key`、description、有序 `change_key[]` 不可改变。复检后不得进行第二轮修复。普通 API 参数、同步、对齐、尾块、dtype 转换和写回缺陷应在唯一修复轮补齐；只有不存在任何合法替代路径时才构成规范硬阻断。本阶段不定义“strategy 生成失败”状态。`validate_strategy.py` 只承担该门禁的确定性断言，不构成第二轮检查或修复。草稿、issue 和修复轨迹不进入 `strategy.json`。
