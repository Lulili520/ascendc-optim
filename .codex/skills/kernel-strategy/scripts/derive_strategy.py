#!/usr/bin/env python3
"""Select one strategy from one confirmed medium-grained cause."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STRATEGY_DESCRIPTIONS = {
    "parallel.use_available_cores": "提高有效并行度",
    "parallel.balance_tiling": "均衡多核任务分配",
    "overhead.reduce_fixed_cost": "减少热路径固定开销",
    "compute.vectorize_reduction": "将 Scalar 归约改为 Vector Reduce",
    "compute.vectorize_elementwise": "将 Scalar 逐元素计算改为 Vector API",
    "memory.batch_transfer": "合并小粒度搬运",
    "memory.improve_l2_reuse": "改善 L2 数据复用",
    "pipeline.double_buffer": "重叠搬运与计算",
    "memory.reduce_gm_traffic": "减少 GM 总流量",
    "memory.avoid_ub_bank_conflict": "规避 UB bank 冲突",
    "compute.reduce_cast": "合并或删除冗余类型转换",
    "compute.low_latency_reduction": "降低 Vector 归约延迟",
    "memory.keep_intermediate_in_ub": "让中间结果驻留 UB",
    "cube.improve_onchip_reuse": "改善 Cube 片上复用",
    "output.align_batch_writeback": "对齐并合并输出写回",
}

RULES = {
    "insufficient_parallelism": "parallel.use_available_cores",
    "uneven_task_distribution": "parallel.balance_tiling",
    "nonuniform_task_cost": "parallel.balance_tiling",
    "redundant_hot_path_overhead": "overhead.reduce_fixed_cost",
    "scalar_address_overhead": "overhead.reduce_fixed_cost",
    "scalar_reduction": "compute.vectorize_reduction",
    "scalar_elementwise_compute": "compute.vectorize_elementwise",
    "inefficient_gm_transfer": "memory.batch_transfer",
    "low_l2_reuse": "memory.improve_l2_reuse",
    "serial_copy_compute": "pipeline.double_buffer",
    "gm_bandwidth_saturation": "memory.reduce_gm_traffic",
    "ub_bank_conflict": "memory.avoid_ub_bank_conflict",
    "excessive_cast_chain": "compute.reduce_cast",
    "high_latency_vector_reduction": "compute.low_latency_reduction",
    "redundant_gm_roundtrip": "memory.keep_intermediate_in_ub",
    "low_cube_onchip_reuse": "cube.improve_onchip_reuse",
    "inefficient_writeback": "output.align_batch_writeback",
    "inefficient_fixpipe_writeback": "output.align_batch_writeback",
    "inherent_serial_dependency": None,
}

OPERATIONS = {
    "parallel.use_available_cores": "increase_parallelism",
    "parallel.balance_tiling": "rebalance_task_partition",
    "overhead.reduce_fixed_cost": "remove_hot_path_overhead",
    "compute.vectorize_reduction": "replace_scalar_reduction",
    "compute.vectorize_elementwise": "replace_scalar_elementwise",
    "memory.batch_transfer": "merge_small_transfers",
    "memory.improve_l2_reuse": "improve_data_locality",
    "pipeline.double_buffer": "overlap_copy_compute",
    "memory.reduce_gm_traffic": "reduce_gm_traffic",
    "memory.avoid_ub_bank_conflict": "adjust_ub_layout",
    "compute.reduce_cast": "remove_redundant_casts",
    "compute.low_latency_reduction": "shorten_vector_reduction",
    "memory.keep_intermediate_in_ub": "keep_intermediate_on_chip",
    "cube.improve_onchip_reuse": "improve_cube_tiling_reuse",
    "output.align_batch_writeback": "align_batch_writeback",
}

OPERATION_DESCRIPTIONS = {
    "increase_parallelism": "修改并行上限、blockDim 和任务映射以使用更多可用核",
    "rebalance_task_partition": "修改任务区间、余数或 logical block 映射以均衡各核工作量",
    "remove_hot_path_overhead": "外提、缓存、合并或删除热路径内重复固定工作",
    "replace_scalar_reduction": "用 UB 分块和 Vector Reduce 替代 Scalar 归约循环",
    "replace_scalar_elementwise": "用连续搬运和 Vector API 替代逐元素 Scalar 计算",
    "merge_small_transfers": "将频繁小粒度搬运合并为连续或批量搬运",
    "improve_data_locality": "重排 tile 或数据访问以增加 L2 可复用性",
    "overlap_copy_compute": "使用分块和双缓冲重叠搬运与计算",
    "reduce_gm_traffic": "缓存或重用数据以减少 GM 读写总量",
    "adjust_ub_layout": "调整 UB 缓冲布局、步长或 padding 以规避 bank 冲突",
    "remove_redundant_casts": "合并或删除主路径中冗余类型转换",
    "shorten_vector_reduction": "改写 Vector 归约组合以缩短依赖链和归约延迟",
    "keep_intermediate_on_chip": "使中间结果驻留 UB 或其他片上存储并删除 GM 往返",
    "improve_cube_tiling_reuse": "修改 Cube tiling 以提高操作数在 L0/L1 的驻留和复用",
    "align_batch_writeback": "将输出写回调整为对齐、连续或批量形式",
}

if set(OPERATION_DESCRIPTIONS) != set(OPERATIONS.values()):
    raise RuntimeError("OPERATION_DESCRIPTIONS 必须覆盖全部 operation")


def derive(report: dict) -> dict:
    bottleneck = report.get("bottleneck")
    if bottleneck is None:
        return {"reasoning": [
            "证据：bottleneck=null；推断：没有可用于策略选择的原因。",
            "证据：策略必须由固定 cause 规则产生；推断：strategy=null。",
        ], "strategy": None}
    if not isinstance(bottleneck, dict):
        raise RuntimeError("bottleneck 必须是对象或 null")
    cause = bottleneck.get("cause_key")
    if cause not in RULES:
        raise RuntimeError(f"未知 cause_key：{cause}")
    strategy_key = RULES[cause]
    if strategy_key is None:
        return {"reasoning": [
            f"证据：cause_key={cause}；推断：该原因没有保持语义的普通优化策略。",
            "证据：固定策略规则返回空；推断：strategy=null。",
        ], "strategy": None}
    return {"reasoning": [
        f"证据：cause_key={cause}；推断：唯一选择 strategy_key={strategy_key}。",
        f"证据：当前 cause evidence；推断：按 operation={OPERATIONS[strategy_key]} 补充最少且完整的 targets。",
    ], "strategy": {"strategy_key": strategy_key, "actions": [{
        "target": None,
        "operation": OPERATIONS[strategy_key],
        "edits": [],
        "constraints": [],
    }]}}


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
    print(f"strategy={output.resolve()}")


if __name__ == "__main__":
    main()
