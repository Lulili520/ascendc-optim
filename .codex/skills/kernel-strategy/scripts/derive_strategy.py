#!/usr/bin/env python3
"""Derive one stable strategy key from one confirmed bottleneck report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


STRATEGIES = {
    "parallel.use_available_cores": ("提高有效并行度", "根据可并行任务数和物理核数增加实际参与计算的核。"),
    "parallel.balance_tiling": ("均衡多核 Tiling", "将尾块或不同代价任务均匀分配，降低最慢核 cycle。"),
    "overhead.reduce_launch_cost": ("降低固定开销", "减少小任务下的初始化、栈访问和 Scalar LD/ST 开销。"),
    "compute.vectorize_reduction": ("标量归约向量化", "将逐元素标量归约改为 Vector Reduce 路径。"),
    "compute.vectorize_elementwise": ("标量逐元素向量化", "将逐元素标量算术循环映射为 Vector API 链。"),
    "memory.batch_transfer": ("增大搬运粒度", "合并小粒度搬运，使用连续或批量 DataCopy。"),
    "memory.improve_l2_reuse": ("改善 L2 复用", "根据数据复用和容量设置 Cache 策略，减少无效 GM 访问。"),
    "pipeline.double_buffer": ("搬运计算重叠", "在已确认串行的迭代流水中使用双缓冲隐藏搬运延迟。"),
    "memory.reduce_gm_traffic": ("减少 GM 总流量", "带宽已经饱和时减少搬运总量或提高片上复用。"),
    "memory.avoid_ub_bank_conflict": ("规避 UB 冲突", "调整 UB 地址、stride 或 padding，降低 bank 冲突。"),
    "compute.reduce_cast": ("减少类型转换", "合并或消除主路径中的冗余 Cast。"),
    "compute.low_latency_reduction": ("低延迟 Vector 归约", "对已向量化归约路径选择更低延迟的归约组合。"),
    "memory.keep_intermediate_in_ub": ("中间结果驻留 UB", "将连续计算链的中间结果保留在 UB，避免 GM 往返。"),
    "cube.improve_onchip_reuse": ("改善 Cube 片上复用", "提高 L0A、L0B 和 L1 数据驻留与复用。"),
    "output.align_batch_writeback": ("对齐并合并写回", "改善输出地址对齐和写回粒度。"),
}

CHANGES = {
    "parallel.use_available_cores": (
        ("core_parallelism.select_used_cores", "按可用核数与独立任务数确定实际用核数。"),
        ("core_parallelism.cover_all_tasks", "建立核到任务的完整映射。"),
        ("core_parallelism.handle_tail", "保留尾任务与边界处理。"),
    ),
    "parallel.balance_tiling": (
        ("load_balance.distribute_work", "按计算代价均匀分配主体工作。"),
        ("load_balance.distribute_remainder", "将余数任务分散到参与计算的核。"),
        ("load_balance.preserve_coverage", "保证任务无遗漏且不重复。"),
    ),
    "overhead.reduce_launch_cost": (
        ("launch_cost.remove_redundant_init", "删除主路径中的重复初始化。"),
        ("launch_cost.hoist_invariants", "将循环不变量移出热循环。"),
        ("launch_cost.reduce_scalar_ldst", "减少可避免的 Scalar 访存。"),
    ),
    "compute.vectorize_reduction": (
        ("vector_reduce.stage_input_in_ub", "将归约输入分块搬入 UB。"),
        ("vector_reduce.replace_scalar_reduction", "用 Vector Reduce 替换标量归约循环。"),
        ("vector_reduce.handle_tail", "保持尾块与归约语义正确。"),
    ),
    "compute.vectorize_elementwise": (
        ("vector_elementwise.stage_input_in_ub", "将逐元素输入分块搬入 UB。"),
        ("vector_elementwise.replace_scalar_loop", "用 Vector API 替换标量逐元素循环。"),
        ("vector_elementwise.handle_tail", "保持尾块与边界语义正确。"),
    ),
    "memory.batch_transfer": (
        ("batch_transfer.merge_small_copies", "合并连续的小粒度搬运。"),
        ("batch_transfer.use_contiguous_copy", "使用连续或批量 DataCopy。"),
        ("batch_transfer.preserve_alignment_tail", "保持对齐和尾块正确。"),
    ),
    "memory.improve_l2_reuse": (
        ("l2_reuse.identify_reused_region", "识别跨核或跨迭代复用的数据区域。"),
        ("l2_reuse.adjust_cache_policy", "为复用区域设置匹配的 Cache 策略。"),
        ("l2_reuse.reorder_access_for_locality", "调整访问顺序以改善局部性。"),
    ),
    "pipeline.double_buffer": (
        ("double_buffer.allocate_two_buffers", "为流水迭代分配两组片上缓冲。"),
        ("double_buffer.restructure_pipeline", "重排搬入、计算和搬出的流水顺序。"),
        ("double_buffer.recompute_tile_capacity", "按双缓冲容量重新约束 tile。"),
    ),
    "memory.reduce_gm_traffic": (
        ("gm_traffic.keep_intermediate_on_chip", "让可复用中间结果驻留片上。"),
        ("gm_traffic.remove_redundant_roundtrip", "删除可避免的 GM 往返。"),
        ("gm_traffic.preserve_capacity_constraints", "保持片上容量约束。"),
    ),
    "memory.avoid_ub_bank_conflict": (
        ("ub_conflict.adjust_layout_or_stride", "调整 UB 布局或访问 stride。"),
        ("ub_conflict.add_padding_if_needed", "必要时增加最小 padding。"),
        ("ub_conflict.preserve_capacity_alignment", "保持容量和对齐约束。"),
    ),
    "compute.reduce_cast": (
        ("cast.remove_redundant_cast", "删除结果等价的冗余 Cast。"),
        ("cast.merge_cast_stages", "合并相邻类型转换阶段。"),
        ("cast.preserve_compute_dtype", "保持计算与输出 dtype 语义。"),
    ),
    "compute.low_latency_reduction": (
        ("low_latency_reduce.select_instruction_sequence", "选择更低延迟的 Vector 归约组合。"),
        ("low_latency_reduce.allocate_temporary_buffer", "按指令约束配置临时缓冲。"),
        ("low_latency_reduce.preserve_tail_semantics", "保持尾块和归约语义。"),
    ),
    "memory.keep_intermediate_in_ub": (
        ("ub_residency.allocate_intermediate_buffer", "为连续计算链分配 UB 中间缓冲。"),
        ("ub_residency.remove_intermediate_gm_roundtrip", "删除中间结果的 GM 往返。"),
        ("ub_residency.preserve_buffer_lifetime", "保证缓冲生命周期和复用安全。"),
    ),
    "cube.improve_onchip_reuse": (
        ("cube_reuse.adjust_l0_l1_tiling", "调整 L0/L1 tiling 以支持复用。"),
        ("cube_reuse.keep_reused_operand_resident", "让复用操作数保持片上驻留。"),
        ("cube_reuse.preserve_buffer_capacity", "保持各级缓冲容量约束。"),
    ),
    "output.align_batch_writeback": (
        ("writeback.align_output_offset", "对齐输出地址和写回偏移。"),
        ("writeback.merge_small_writes", "合并连续的小粒度写回。"),
        ("writeback.handle_tail_padding", "正确处理尾块和 padding。"),
    ),
}

RULES = (
    ("parallel.core_underuse", None, "parallel.use_available_cores"),
    ("parallel.load_imbalance", None, "parallel.balance_tiling"),
    ("overhead.small_workload", None, "overhead.reduce_launch_cost"),
    ("pipeline.scalar_bound", "source.scalar_reduction_loop", "compute.vectorize_reduction"),
    ("pipeline.scalar_bound", "source.scalar_elementwise_loop", "compute.vectorize_elementwise"),
    ("pipeline.scalar_bound", "source.init_or_ldst_heavy", "overhead.reduce_launch_cost"),
    ("pipeline.mte2_bound", "metric.small_transfer", "memory.batch_transfer"),
    ("pipeline.mte2_bound", "metric.low_l2_hit", "memory.improve_l2_reuse"),
    ("pipeline.mte2_bound", "source.serial_copy_compute", "pipeline.double_buffer"),
    ("pipeline.mte2_bound", "metric.bandwidth_saturated", "memory.reduce_gm_traffic"),
    ("pipeline.vector_bound", "metric.ub_bank_conflict", "memory.avoid_ub_bank_conflict"),
    ("pipeline.vector_bound", "source.gm_roundtrip", "memory.keep_intermediate_in_ub"),
    ("pipeline.vector_bound", "metric.excess_cast", "compute.reduce_cast"),
    ("pipeline.vector_bound", "source.reduction_path", "compute.low_latency_reduction"),
    ("pipeline.cube_bound", "source.low_onchip_reuse", "cube.improve_onchip_reuse"),
    ("pipeline.fixp_bound", "source.unaligned_or_small_write", "output.align_batch_writeback"),
    ("pipeline.mte3_bound", "source.unaligned_or_small_write", "output.align_batch_writeback"),
)


def derive(report: dict) -> dict:
    bottleneck = report.get("bottleneck")
    if bottleneck is None:
        return {
            "reasoning": [
                "证据：bottleneck=null；推断：没有唯一主瓶颈可用于规则匹配。",
                "证据：策略只能由固定瓶颈规则产生；推断：strategy=null。",
            ],
            "strategy": None,
        }
    if not isinstance(bottleneck, dict):
        raise RuntimeError("bottleneck 必须是对象或 null")
    key = bottleneck.get("bottleneck_key")
    tags = set(bottleneck.get("evidence_keys", []))
    for required_key, required_evidence, strategy_key in RULES:
        if key == required_key and (required_evidence is None or required_evidence in tags):
            _, description = STRATEGIES[strategy_key]
            changes = [
                {
                    "change_key": change_key,
                    "target": None,
                    "action": change_description,
                }
                for change_key, change_description in CHANGES[strategy_key]
            ]
            matched = "无需附加根因证据" if required_evidence is None else f"evidence_key={required_evidence}"
            return {
                "reasoning": [
                    f"证据：bottleneck_key={key}；推断：进入该主瓶颈的固定策略规则。",
                    f"证据：{matched}；推断：按固定优先顺序唯一选中 strategy_key={strategy_key}。",
                    f"证据：strategy_key={strategy_key} 的固定修改契约；推断：必须为 {len(changes)} 个 change_key 补充真实 target 和具体 action。",
                ],
                "strategy": {
                    "strategy_key": strategy_key,
                    "description": description,
                    "changes": changes,
                },
            }
    return {
        "reasoning": [
            f"证据：bottleneck_key={key}；推断：检查该 key 的固定策略规则。",
            f"证据：evidence_keys={sorted(tags)} 未满足任何规则；推断：不能可靠生成 strategy_key。",
        ],
        "strategy": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.project_dir / "bottleneck/bottleneck.json"
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
        result = derive(report)
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise SystemExit(f"STRATEGY_BLOCKED: {error}") from error
    output = args.output or args.project_dir / "strategy/strategy.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"strategy={output.resolve()}")


if __name__ == "__main__":
    main()
