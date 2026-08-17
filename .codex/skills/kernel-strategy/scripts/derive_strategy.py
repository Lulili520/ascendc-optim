#!/usr/bin/env python3
"""Derive one deterministic strategy for every concrete source cause."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STRATEGY_DESCRIPTIONS = {
    "overhead.reduce_hot_path_work": "外提、缓存或删除热路径重复工作",
    "tiling.repair_parameter_flow": "使 Host tiling 参数真实控制 Kernel 执行结构",
    "parallel.increase_parallelism": "增加有效并行任务映射",
    "parallel.balance_task_count": "均衡各核任务数量",
    "parallel.balance_task_cost": "按任务代价均衡各核工作",
    "parallel.coarsen_task_mapping": "减少过细 block、波次与重复控制",
    "tiling.resize_work_unit": "按容量、搬运粒度和并行度调整 work unit",
    "memory.coalesce_transfer": "合并连续的小粒度 GM 搬运",
    "memory.batch_strided_transfer": "把规则跨步循环改为批量二维搬运",
    "memory.vectorize_gather_access": "用偏移表和 Gather 替代离散标量读取",
    "memory.vectorize_scatter_access": "用批量或 Vector Scatter 替代离散标量写入",
    "memory.align_padded_transfer": "用合法对齐和 padding 处理非对齐搬运",
    "memory.privatize_contended_write": "局部私有累加并确定性合并竞争写",
    "memory.keep_data_onchip": "在片上复用数据并删除 GM 往返",
    "memory.reuse_buffer_by_phase": "按真实活跃阶段复用片上 Buffer",
    "memory.relayout_onchip_buffer": "重排 UB 地址、stride 或 padding 规避冲突",
    "pipeline.overlap_stages": "构建搬运、计算和写回的稳态重叠",
    "pipeline.reduce_sync_scope": "把同步缩减到真实依赖范围",
    "pipeline.restructure_slots": "重构 Queue/Buffer 槽所有权以解除串行化",
    "compute.vectorize_scalar_work": "用 Vector 路径替换可并行 Scalar 工作",
    "compute.tree_reduce_chunks": "用分层或树形结构合并 chunk 归约结果",
    "compute.eliminate_materialization": "删除不必要的完整向量物化",
    "compute.retain_vector_intermediate": "保留 Vector 中间表示并消除 UB 往返",
    "compute.simplify_cast_path": "合并或删除连续 Cast",
    "compute.increase_vector_utilization": "扩大合法 Vector 工作粒度和有效 lane",
    "compute.improve_cube_dataflow": "改善 Cube tile、循环顺序与 L0/L1 复用",
    "compute.align_fixpipe_layout": "使 FixPipe 输出布局与 Cube 数据流一致",
    "compute.coalesce_fixpipe_writeback": "合并碎片化 FixPipe 写回",
}

RULES = {
    "repeated_hot_path_overhead": "overhead.reduce_hot_path_work",
    "inactive_tiling_parameter": "tiling.repair_parameter_flow",
    "insufficient_parallelism": "parallel.increase_parallelism",
    "uneven_task_distribution": "parallel.balance_task_count",
    "nonuniform_task_cost": "parallel.balance_task_cost",
    "overpartitioned_task_mapping": "parallel.coarsen_task_mapping",
    "inefficient_work_unit_size": "tiling.resize_work_unit",
    "scalar_global_contiguous_access": "memory.coalesce_transfer",
    "fragmented_contiguous_transfer": "memory.coalesce_transfer",
    "fragmented_regular_strided_transfer": "memory.batch_strided_transfer",
    "scalar_irregular_gather_access": "memory.vectorize_gather_access",
    "scalar_irregular_scatter_access": "memory.vectorize_scatter_access",
    "unaligned_transfer_tail": "memory.align_padded_transfer",
    "fragmented_global_writeback": "memory.coalesce_transfer",
    "atomic_write_contention": "memory.privatize_contended_write",
    "repeated_global_transfer": "memory.keep_data_onchip",
    "redundant_gm_roundtrip": "memory.keep_data_onchip",
    "premature_buffer_eviction": "memory.keep_data_onchip",
    "overextended_buffer_lifetime": "memory.reuse_buffer_by_phase",
    "ub_bank_conflict": "memory.relayout_onchip_buffer",
    "serial_pipeline_stages": "pipeline.overlap_stages",
    "over_synchronization": "pipeline.reduce_sync_scope",
    "pipeline_slot_reuse_serialization": "pipeline.restructure_slots",
    "recomputed_invariant_scalar_work": "overhead.reduce_hot_path_work",
    "scalar_local_lane_compute": "compute.vectorize_scalar_work",
    "scalar_reduction": "compute.vectorize_scalar_work",
    "serial_chunk_reduction": "compute.tree_reduce_chunks",
    "scalar_elementwise_compute": "compute.vectorize_scalar_work",
    "redundant_vector_materialization": "compute.eliminate_materialization",
    "underutilized_vector_width": "compute.increase_vector_utilization",
    "vector_ub_bouncing": "compute.retain_vector_intermediate",
    "excessive_cast_chain": "compute.simplify_cast_path",
    "inefficient_cube_tiling": "compute.improve_cube_dataflow",
    "low_cube_onchip_reuse": "compute.improve_cube_dataflow",
    "mismatched_fixpipe_layout": "compute.align_fixpipe_layout",
    "fragmented_fixpipe_writeback": "compute.coalesce_fixpipe_writeback",
}

OPERATIONS = {
    "overhead.reduce_hot_path_work": "remove_hot_path_overhead",
    "tiling.repair_parameter_flow": "repair_tiling_parameter_flow",
    "parallel.increase_parallelism": "increase_parallelism",
    "parallel.balance_task_count": "rebalance_task_partition",
    "parallel.balance_task_cost": "rebalance_weighted_partition",
    "parallel.coarsen_task_mapping": "coarsen_task_partition",
    "tiling.resize_work_unit": "resize_work_unit",
    "memory.coalesce_transfer": "coalesce_global_transfer",
    "memory.batch_strided_transfer": "batch_strided_transfer",
    "memory.vectorize_gather_access": "vectorize_gather_access",
    "memory.vectorize_scatter_access": "vectorize_scatter_access",
    "memory.align_padded_transfer": "use_aligned_padded_transfer",
    "memory.privatize_contended_write": "privatize_contended_write",
    "memory.keep_data_onchip": "keep_data_on_chip",
    "memory.reuse_buffer_by_phase": "reuse_buffer_by_phase",
    "memory.relayout_onchip_buffer": "relayout_onchip_buffer",
    "pipeline.overlap_stages": "overlap_pipeline_stages",
    "pipeline.reduce_sync_scope": "reduce_sync_scope",
    "pipeline.restructure_slots": "restructure_pipeline_slots",
    "compute.vectorize_scalar_work": "vectorize_scalar_work",
    "compute.tree_reduce_chunks": "tree_reduce_chunks",
    "compute.eliminate_materialization": "eliminate_vector_materialization",
    "compute.retain_vector_intermediate": "retain_vector_intermediate",
    "compute.simplify_cast_path": "simplify_cast_path",
    "compute.increase_vector_utilization": "increase_vector_utilization",
    "compute.improve_cube_dataflow": "improve_cube_dataflow",
    "compute.align_fixpipe_layout": "align_fixpipe_layout",
    "compute.coalesce_fixpipe_writeback": "coalesce_fixpipe_writeback",
}

OPERATION_DESCRIPTIONS = {
    operation: STRATEGY_DESCRIPTIONS[strategy]
    for strategy, operation in OPERATIONS.items()
}


def derive(report: dict) -> dict:
    issues = report.get("issues")
    if not isinstance(issues, list):
        raise RuntimeError("bottleneck report 缺少 issues")
    if not issues:
        raise RuntimeError("issues 为空，当前版本应停止优化而不生成 strategy")
    reasoning = []
    strategies = []
    for index, issue in enumerate(issues, 1):
        bottleneck = issue.get("bottleneck") if isinstance(issue, dict) else None
        cause = bottleneck.get("cause_key") if isinstance(bottleneck, dict) else None
        if cause not in RULES:
            raise RuntimeError(f"未知 cause_key：{cause}")
        key = RULES[cause]
        operation = OPERATIONS[key]
        reasoning.append(
            f"因果映射：第{index}项具体 cause_key={cause} 唯一映射到 "
            f"strategy_key={key}；按 operation={operation} 具体化当前源码变换。"
        )
        strategies.append({
            "cause_key": cause,
            "strategy_key": key,
            "actions": [{"target": None, "operation": operation, "edits": [], "constraints": []}],
        })
    return {"reasoning": reasoning, "strategies": strategies}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.project_dir / "bottleneck/bottleneck.json"
    try:
        result = derive(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise SystemExit(f"STRATEGY_BLOCKED: {error}") from error
    output = args.output or args.project_dir / "strategy/strategy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    blocking = output.parent / "blocking.json"
    if blocking.exists():
        blocking.unlink()
    print(f"strategy={output.resolve()}")


if __name__ == "__main__":
    main()
