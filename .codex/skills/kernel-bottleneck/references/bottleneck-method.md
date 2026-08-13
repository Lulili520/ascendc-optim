# Bottleneck 与 Cause

## 决策链

```text
原始 performance/source evidence
→ fixed cost
→ core underuse
→ load imbalance
→ 正式 pipeline candidate
→ 无支持候选则 bottleneck=null
```

`evidence_key` 表示可复核事实，`bottleneck_key` 表示哪里受限，`cause_key` 表示为什么受限。禁止用 `metric.*_bound` 作为 evidence。

## Bottleneck

| bottleneck_key | 必要原始指标 |
|---|---|
| `overhead.fixed_cost_bound` | `metric.task.head_overhead_ratio` |
| `parallel.core_underuse` | `metric.core.active_count`、`metric.hardware.available_core_count` |
| `parallel.load_imbalance` | `metric.core.active_count`、`metric.core.imbalance_percent` |
| `pipeline.mte2_bound` | `metric.pipeline.mte2_ratio` |
| `pipeline.mte3_bound` | `metric.pipeline.mte3_ratio` |
| `pipeline.cube_bound` | `metric.pipeline.cube_ratio` |
| `pipeline.vector_bound` | `metric.pipeline.vector_ratio` |
| `pipeline.fixp_bound` | `metric.pipeline.fixpipe_ratio` |
| `pipeline.scalar_bound` | `metric.pipeline.scalar_ratio` |

MTE2/Cube ratio >80%，或唯一最大且 >70% 时成为正式候选；其余流水线 ratio >80% 时成为候选。多个候选先取最大值，前两名相差不超过 3 个百分点时按 `MTE2 → Cube → Vector → FixPipe → MTE3 → Scalar` 选择。

Core underuse 还需证明独立任务充足且没有数学串行限制。Load imbalance 要求 active cores >1、imbalance >30%，并由任务映射解释。Fixed cost 必须证明固定路径主导，不能仅凭源码中存在初始化就成立。

决策顺序是标签契约，不是建议顺序。例如 active cores 少于可用核且源码存在足量独立任务时，必须选 `parallel.core_underuse`，不能因某条 pipeline ratio 很高而跳过。实际成功但不符合当前首要标签的子版本，不得导出为该 policy 的正样本。

## 中粒度 Cause

| 父瓶颈 | cause_key（按优先级） |
|---|---|
| Fixed cost | `redundant_hot_path_overhead` |
| Core underuse | `insufficient_parallelism` |
| Load imbalance | `nonuniform_task_cost` → `uneven_task_distribution` |
| MTE2 | `gm_bandwidth_saturation` → `inefficient_gm_transfer` → `low_l2_reuse` → `serial_copy_compute` |
| MTE3 | `inefficient_writeback` |
| Cube | `low_cube_onchip_reuse` |
| Vector | `ub_bank_conflict` → `redundant_gm_roundtrip` → `excessive_cast_chain` → `high_latency_vector_reduction` |
| FixPipe | `inefficient_fixpipe_writeback` |
| Scalar | `inherent_serial_dependency` → `scalar_address_overhead` → `scalar_reduction` → `scalar_elementwise_compute` |

Cause 是可行动根因族。Block Dim、未展平轴、余数、第二执行波、具体 Scalar 循环和 DataCopy 形态写入 observation，不再拆成更多标签。指标 cause 必须满足真实条件；普通聚合数据不能证明 overlap，`serial_copy_compute` 需要 trace 或等价直接证据。

`source.task_distribution` 可引用 `op_host/` 或 `op_kernel/`：它表示任务映射事实，不把同一 cause 按代码所在目录拆成两个 key。

## 输出

```json
{
  "reasoning": ["证据：metric...；推断：bottleneck_key=...、cause_key=...。"],
  "evidence": [
    {"evidence_key": "...", "source": "performance... 或 op_kernel/...::符号", "observation": "..."}
  ],
  "bottleneck": {
    "bottleneck_key": "parallel.load_imbalance",
    "cause_key": "uneven_task_distribution"
  }
}
```

因果事实保存在 evidence/observation，推导保存在 reasoning；不重复保存 causal_explanation、evidence_keys 或 description。有效输入没有同时满足瓶颈条件和直接 cause 证据时输出 `bottleneck:null`。

Source observation 不得包含 `performance.*`、ratio、cycle、百分比、Task Duration 或收益；这些事实必须作为独立 metric evidence。
