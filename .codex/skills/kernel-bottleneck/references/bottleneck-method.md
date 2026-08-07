# CannBot 瓶颈分析方法

本方法整理自 `/data/lu/cannbot-skills/ops/ops-profiling/` 的 `msprof-guide.md`、`csv_fields_reference.md` 与 `optimization_quickref.md`。以实际 CSV 表头为准。

## 目录

1. 分析顺序
2. 主 Bound
3. 交叉验证
4. 逐核负载
5. 源码验证
6. 硬件配置与分析边界
7. 输出契约
8. Bottleneck Key

## 1. 分析顺序

```text
基本信息 → 严重并行问题 → 小任务检查 → 正式主 Bound
        → 相关指标与源码验证 → 唯一主瓶颈
```

`Task Duration(us)` 是整体耗时；AIC/AIV 可能并行，不能相加。先按 `Task Type` 选择有效的 `aic_*`、`aiv_*` 或 MIX 字段。

## 2. 主 Bound

从上到下匹配第一条成立的正式规则：

| 优先级 | 类型 | 正式规则 |
|---:|---|---|
| 1 | MTE2 BOUND | busy >80%，或占比最大且 >70% |
| 2 | CUBE BOUND | busy >80%，或占比最大且 >70% |
| 3 | VEC BOUND | busy >80% |
| 4 | FIXP BOUND | busy >80% |
| 5 | MTE3 BOUND | busy >80% |
| 6 | SCALAR BOUND | Scalar 或 ScalarLDST busy >80% |
| — | 无 Bound | 均不成立 |

VEC >50%、MTE2 >50%、Scalar >30%、FixPipe >15%、ICache miss >15% 仅是异常提示，不替代正式主 Bound。

## 3. 交叉验证

主 Bound 只说明忙在哪里，根因必须由组合信号支持：

| 组合信号 | 根因方向 | 优先检查 |
|---|---|---|
| 高 VEC + 高 bank conflict | UB 冲突放大计算时间 | stride、UB 地址、padding |
| 高 MTE2 + 高带宽利用率 | 可能带宽饱和 | 理论峰值、搬运总量、复用 |
| 高 MTE2 + 低带宽 + 高等待 | 访存延迟/小粒度/依赖 | 搬运粒度、对齐、同步 |
| 高 MTE2 + 低 L2 命中 | Cache miss | CacheMode、布局、复用 |
| 高 FixPipe | 写回或地址对齐 | 512B 对齐、padding |
| 高 Scalar + 低 VEC/CUBE | 控制流/标量循环 | Host 预计算、分支、向量化 |
| MTE2 与 MTE3 都高 | GM 往返过多 | 融合、增大 tile |
| 核间差异大 | Tiling/尾块不均 | Block Dim、尾块分散 |

参考阈值：GM 路径带宽利用率 >60% 为利用较好；<10% 通常不是带宽饱和。L2 总命中率 >80% 良好，<50% 需关注。Vector 总冲突 <5% 良好、>15% 严重；bankgroup/bank/MTE 冲突关注 3%，资源冲突关注 5%。

判断“接近理论带宽/算力”必须使用对应 SoC 的可信峰值。没有硬件配置时只能写相对诊断。

## 4. 逐核负载

```text
imbalance = (max_cycle - min_cycle) / max_cycle × 100%
```

- <10%：均衡；
- 10%～30%：警告；
- >30%：严重。

列出 Top-3 慢核和快核。按 core id 比较前后两半均值，差距 ≥2% 时提示可能的核簇/L2 slice 偏斜，但只标为推断。

## 5. 源码验证

根据指标定向检查，不做无目标代码巡检：

- Scalar：循环、分支、mask、地址计算、逐元素 GM 访问；
- MTE：DataCopy 粒度、连续性、对齐、tile、复用、同步；
- VEC：Cast、重复计算、归约、融合机会；
- Conflict：repeatStride/blockStride、UB 地址和 padding；
- Load imbalance：Block Dim、尾块和工作量映射；
- Pipeline：EnQue/DeQue、buffer 数量和数据依赖。

指标与源码必须相互印证；仅看源码不能生成确定瓶颈。

## 6. 硬件配置与分析边界

`performance.hardware` 只使用 SoC、Cube/Vector 核数、每核 UB/L1/L0 容量、L2 容量和 `gm_peak_bandwidth_gbps_per_core`。频率和原始 Byte/cycle 留在设备文件，正式 Bound 不直接使用。未知字段用 `null`。

GM 带宽必须保持单核口径：AIC/AIV 各自的 main-memory read 与 write 相加，再取两条路径的较大值，与 `gm_peak_bandwidth_gbps_per_core` 相除。优先直接使用采集器生成的 `performance.memory.gm_peak_utilization_percent`。峰值缺失时不得生成带宽饱和证据；利用率 ≥80% 才可标记 `metric.bandwidth_saturated`，60%～80% 只说明利用率较高，不等于接近饱和。

普通 msprof 聚合 CSV 没有可对照的流水气泡时间，不能确认 Double Buffer 重叠率；没有 trace 或等价直接证据时不得生成 `source.serial_copy_compute`。MTE2/MTE3 可能共享 GM 带宽，应合并考虑。采集组来自不同运行，只能交叉诊断，不能混合平均 latency。

## 7. 输出契约

`bottleneck.json` 至少包含：

```json
{
  "reasoning": [
    "证据：metric.scalar_bound 显示 Scalar busy 超过正式阈值；推断：Scalar 是主 Bound。",
    "证据：source.scalar_reduction_loop 显示源码存在逐元素归约；推断：确认 Scalar 根因。"
  ],
  "evidence": [
    {
      "evidence_key": "metric.scalar_bound",
      "source": "performance.pipeline.aiv_scalar_time_ratio",
      "observation": "86.4%，超过 80% 阈值"
    },
    {
      "evidence_key": "source.scalar_reduction_loop",
      "source": "op_kernel/argmax.cpp",
      "observation": "主循环逐元素比较最大值并更新索引"
    }
  ],
  "bottleneck": {
    "bottleneck_key": "pipeline.scalar_bound",
    "description": "主要执行时间消耗在 Scalar 或 ScalarLDST 流水线。",
    "evidence_keys": ["metric.scalar_bound", "source.scalar_reduction_loop"],
    "causal_explanation": "..."
  }
}
```

无法可靠认定时输出 `"bottleneck": null`，不强选瓶颈。

## 8. Bottleneck Key

`bottleneck_key` 是跨版本稳定的机器键，展示名称和解释可以变化，key 不得变化。每次分析最多输出一个 key。

### Key 目录

| Key | 名称 | 具体介绍 |
|---|---|---|
| `parallel.core_underuse` | 并行核利用不足 | 可并行任务充足，但实际活跃核明显少于可用物理核。 |
| `parallel.load_imbalance` | 多核负载不均 | 最慢核决定总时延，逐核 cycle 不均衡度超过 30%。 |
| `overhead.small_workload` | 小任务固定开销 | 工作量不足以支撑流水线分析，启动、初始化或 Scalar LD/ST 主导。 |
| `pipeline.mte2_bound` | MTE2 搬入瓶颈 | GM/L2/L1 到片上存储的搬入流水线满足正式主 Bound 规则。 |
| `pipeline.cube_bound` | Cube 计算瓶颈 | 矩阵计算流水线满足正式主 Bound 规则。 |
| `pipeline.vector_bound` | Vector 计算瓶颈 | 向量计算流水线 busy 超过 80%。 |
| `pipeline.fixp_bound` | FixPipe 写回瓶颈 | Cube 结果转换、搬运或写回流水线 busy 超过 80%。 |
| `pipeline.mte3_bound` | MTE3 搬出瓶颈 | 片上存储向 GM 的搬出流水线 busy 超过 80%。 |
| `pipeline.scalar_bound` | Scalar 流水线瓶颈 | 循环、分支、地址计算或标量 LD/ST 流水线 busy 超过 80%。 |
带宽、L2、冲突、尾块和 overlap 只作为主瓶颈的交叉证据或因果解释，不单独产生 key。新增主类别时只在本目录追加稳定 key；禁止加入算子名、版本号、指标数值或序号。

### 唯一主瓶颈选择顺序

1. 可并行任务充足且活跃核严重不足时，选择 `parallel.core_underuse`。
2. 逐核不均衡度 >30% 时，选择 `parallel.load_imbalance`。
3. `performance.task.head_overhead_ratio > 30%`，且源码确认固定初始化或 Scalar LD/ST 在小工作量下主导时，选择 `overhead.small_workload`。仅头开销超过阈值不能单独定性。
4. 否则按第 2 节正式规则从上到下选择第一个成立的流水线 Bound。
5. 均不成立或证据冲突时，不选择最高值凑数，输出 `bottleneck: null`。

### Evidence Key 目录

每个主瓶颈必须包含对应的正式证据：

| Bottleneck Key | 必要 Evidence Key |
|---|---|
| `parallel.core_underuse` | `metric.core_underuse` |
| `parallel.load_imbalance` | `metric.core_imbalance_over_30` |
| `overhead.small_workload` | `metric.small_workload` |
| `pipeline.mte2_bound` | `metric.mte2_bound` |
| `pipeline.cube_bound` | `metric.cube_bound` |
| `pipeline.vector_bound` | `metric.vector_bound` |
| `pipeline.fixp_bound` | `metric.fixp_bound` |
| `pipeline.mte3_bound` | `metric.mte3_bound` |
| `pipeline.scalar_bound` | `metric.scalar_bound` |

根因 evidence key 用于交叉验证并作为策略规则输入：

只记录已经由指标或源码证明的标签：

| Evidence Key | 含义 |
|---|---|
| `source.scalar_reduction_loop` | 源码存在逐元素标量 Sum/Max/Min/Arg 归约循环 |
| `source.scalar_elementwise_loop` | 源码存在逐元素标量算术循环 |
| `source.init_or_ldst_heavy` | 初始化、栈变量或 Scalar LD/ST 明显 |
| `metric.small_workload` | `head_overhead_ratio > 30%`，其值由 `Task Duration - max(AIC time, AIV time)` 推导 |
| `metric.small_transfer` | 搬运指令平均粒度过小 |
| `metric.low_l2_hit` | L2 命中率低于关注阈值 |
| `metric.bandwidth_saturated` | 可信单核 GM 峰值存在，且 `performance.memory.gm_peak_utilization_percent >= 80%` |
| `source.serial_copy_compute` | trace 或等价直接证据确认搬运与计算串行 |
| `metric.ub_bank_conflict` | UB bank/bankgroup 冲突超过阈值 |
| `metric.excess_cast` | Arithmetic 指标与源码确认 Cast 过多 |
| `source.reduction_path` | 主计算路径是 Vector 归约 |
| `source.gm_roundtrip` | 中间结果存在不必要的 GM 往返 |
| `source.low_onchip_reuse` | Cube 路径的 L0/L1 数据复用不足 |
| `source.unaligned_or_small_write` | 写回地址未达到有效对齐或写回粒度过小 |
