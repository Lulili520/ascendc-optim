#!/usr/bin/env python3
"""Validate direct, source-grounded kernel bottleneck reports."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


BOTTLENECK_DESCRIPTIONS = {
    "overhead.hot_path_inefficiency": "热路径包含可消除的重复工作",
    "tiling.execution_inefficiency": "Host tiling 与真实 Kernel 执行结构不匹配",
    "parallel.core_underuse": "独立任务没有充分映射到可用核",
    "parallel.load_imbalance": "参与核之间的任务数量或代价不均",
    "memory.transfer_inefficiency": "GM 搬运形态、对齐或写回粒度低效",
    "memory.reuse_inefficiency": "数据或中间量没有按生命周期片上复用",
    "memory.onchip_conflict": "片上布局造成可规避的访问冲突",
    "pipeline.overlap_loss": "可并行阶段或槽生命周期导致串行化",
    "compute.scalar_inefficiency": "可并行主要工作仍由 Scalar 路径承担",
    "compute.vector_dataflow_inefficiency": "Vector 宽度、物化、转换或中间表示低效",
    "compute.reduction_inefficiency": "分块归约合并结构低效",
    "compute.cube_dataflow_inefficiency": "Cube tile、片上复用或 FixPipe 数据流低效",
}

# cause_key 必须描述足以唯一决定 strategy 的具体源码机制。
CAUSES = {
    "repeated_hot_path_overhead": ("overhead.hot_path_inefficiency", "source.hot_path.redundant_overhead"),
    "inactive_tiling_parameter": ("tiling.execution_inefficiency", "source.host.inactive_tiling_parameter"),
    "insufficient_parallelism": ("parallel.core_underuse", "source.host.parallel_mapping"),
    "uneven_task_distribution": ("parallel.load_imbalance", "source.task_distribution"),
    "nonuniform_task_cost": ("parallel.load_imbalance", "source.kernel.nonuniform_task_cost"),
    "overpartitioned_task_mapping": ("overhead.hot_path_inefficiency", "source.host.overpartitioned_mapping"),
    "inefficient_work_unit_size": ("tiling.execution_inefficiency", "source.host.work_unit_size"),
    "scalar_global_contiguous_access": ("memory.transfer_inefficiency", "source.kernel.scalar_global_contiguous_access"),
    "fragmented_contiguous_transfer": ("memory.transfer_inefficiency", "source.kernel.fragmented_contiguous_transfer"),
    "fragmented_regular_strided_transfer": ("memory.transfer_inefficiency", "source.kernel.fragmented_regular_strided_transfer"),
    "scalar_irregular_gather_access": ("memory.transfer_inefficiency", "source.kernel.scalar_irregular_gather_access"),
    "scalar_irregular_scatter_access": ("memory.transfer_inefficiency", "source.kernel.scalar_irregular_scatter_access"),
    "unaligned_transfer_tail": ("memory.transfer_inefficiency", "source.kernel.unaligned_transfer_tail"),
    "fragmented_global_writeback": ("memory.transfer_inefficiency", "source.kernel.fragmented_global_writeback"),
    "atomic_write_contention": ("memory.transfer_inefficiency", "source.kernel.atomic_write_contention"),
    "repeated_global_transfer": ("memory.reuse_inefficiency", "source.kernel.repeated_global_transfer"),
    "redundant_gm_roundtrip": ("memory.reuse_inefficiency", "source.kernel.redundant_gm_roundtrip"),
    "premature_buffer_eviction": ("memory.reuse_inefficiency", "source.kernel.premature_buffer_eviction"),
    "overextended_buffer_lifetime": ("memory.reuse_inefficiency", "source.kernel.overextended_buffer_lifetime"),
    "ub_bank_conflict": ("memory.onchip_conflict", "source.kernel.ub_bank_conflict"),
    "serial_pipeline_stages": ("pipeline.overlap_loss", "source.kernel.serial_pipeline_stages"),
    "over_synchronization": ("pipeline.overlap_loss", "source.kernel.over_synchronization"),
    "pipeline_slot_reuse_serialization": ("pipeline.overlap_loss", "source.kernel.pipeline_slot_reuse_serialization"),
    "recomputed_invariant_scalar_work": ("compute.scalar_inefficiency", "source.hot_path.recomputed_invariant_scalar_work"),
    "scalar_local_lane_compute": ("compute.scalar_inefficiency", "source.kernel.scalar_local_lane_compute"),
    "scalar_reduction": ("compute.scalar_inefficiency", "source.kernel.scalar_reduction"),
    "serial_chunk_reduction": ("compute.reduction_inefficiency", "source.kernel.serial_chunk_reduction"),
    "scalar_elementwise_compute": ("compute.scalar_inefficiency", "source.kernel.scalar_elementwise_compute"),
    "redundant_vector_materialization": ("compute.vector_dataflow_inefficiency", "source.kernel.redundant_vector_materialization"),
    "underutilized_vector_width": ("compute.vector_dataflow_inefficiency", "source.kernel.underutilized_vector_width"),
    "vector_ub_bouncing": ("compute.vector_dataflow_inefficiency", "source.kernel.vector_ub_bouncing"),
    "excessive_cast_chain": ("compute.vector_dataflow_inefficiency", "source.kernel.excessive_cast_chain"),
    "inefficient_cube_tiling": ("compute.cube_dataflow_inefficiency", "source.host.inefficient_cube_tiling"),
    "low_cube_onchip_reuse": ("compute.cube_dataflow_inefficiency", "source.kernel.low_cube_onchip_reuse"),
    "mismatched_fixpipe_layout": ("compute.cube_dataflow_inefficiency", "source.kernel.mismatched_fixpipe_layout"),
    "fragmented_fixpipe_writeback": ("compute.cube_dataflow_inefficiency", "source.kernel.fragmented_fixpipe_writeback"),
}

CAUSE_DESCRIPTIONS = {cause: source.replace("source.", "").replace(".", " ") for cause, (_, source) in CAUSES.items()}

METRIC_SOURCES = {
    "performance.task.overhead": ("performance.task.head_overhead_ratio",),
    "performance.core.active": ("performance.per_core.active_cores",),
    "performance.core.imbalance": ("performance.per_core.imbalance_percent",),
    "performance.pipeline.mte2": ("performance.pipeline.aic_mte2_ratio", "performance.pipeline.aiv_mte2_ratio"),
    "performance.pipeline.mte3": ("performance.pipeline.aiv_mte3_ratio",),
    "performance.pipeline.cube": ("performance.pipeline.aic_mac_ratio",),
    "performance.pipeline.vector": ("performance.pipeline.aiv_vec_ratio",),
    "performance.pipeline.scalar": ("performance.pipeline.aic_scalar_ratio", "performance.pipeline.aiv_scalar_ratio"),
    "performance.memory.gm": ("performance.memory.gm_peak_utilization_percent",),
    "performance.cache.l2": ("performance.l2_cache.aiv_derived_hit_rate_percent",),
    "performance.memory.ub_conflict": (
        "performance.resource_conflict.aiv_vec_bankgroup_cflt_ratio",
        "performance.resource_conflict.aiv_vec_bank_cflt_ratio",
    ),
}
SOURCE_EVIDENCE = {source for _, source in CAUSES.values()}
EVIDENCE_KEYS = set(METRIC_SOURCES) | SOURCE_EVIDENCE
BOTTLENECK_EVIDENCE = {key: () for key in BOTTLENECK_DESCRIPTIONS}

SOURCE_ANCHORS = {
    "scalar_global_contiguous_access": ("GetValue", "SetValue"),
    "scalar_irregular_gather_access": ("GetValue",),
    "scalar_irregular_scatter_access": ("SetValue",),
    "scalar_local_lane_compute": ("GetValue", "SetValue"),
    "over_synchronization": ("PipeBarrier", "SyncAll", "WaitFlag", "SetFlag"),
    "redundant_vector_materialization": ("Duplicate",),
    "atomic_write_contention": ("Atomic",),
    "excessive_cast_chain": ("Cast",),
    "mismatched_fixpipe_layout": ("Fixpipe", "FixPipe"),
    "fragmented_fixpipe_writeback": ("Fixpipe", "FixPipe"),
}

# 同一 symbol 上这些 cause 是同一机制的不同具体分类，必须只保留一个根因。
EXCLUSIVE_CAUSE_GROUPS = (
    {"inefficient_work_unit_size", "fragmented_contiguous_transfer", "underutilized_vector_width"},
    {"scalar_global_contiguous_access", "fragmented_contiguous_transfer"},
    {"fragmented_contiguous_transfer", "fragmented_regular_strided_transfer"},
    {"serial_pipeline_stages", "pipeline_slot_reuse_serialization"},
    {"scalar_local_lane_compute", "scalar_elementwise_compute"},
    {"premature_buffer_eviction", "repeated_global_transfer"},
    {"mismatched_fixpipe_layout", "fragmented_fixpipe_writeback"},
)


def _mask_comments_and_strings(text: str) -> str:
    pattern = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
    return pattern.sub(lambda match: "".join("\n" if c == "\n" else " " for c in match.group()), text)


def _matching_brace(masked: str, opening: int) -> int:
    depth = 0
    for position in range(opening, len(masked)):
        if masked[position] == "{":
            depth += 1
        elif masked[position] == "}":
            depth -= 1
            if depth == 0:
                return position + 1
    raise RuntimeError("源码 symbol 花括号不闭合")


def _symbol_bodies(text: str, symbol: str) -> list[str]:
    name = re.escape(symbol.rsplit("::", 1)[-1])
    masked = _mask_comments_and_strings(text)
    matches = list(re.finditer(rf"\b{name}\s*\([^;{{}}]*\)[^;{{}}]*\{{", masked, re.S))
    if not matches:
        raise RuntimeError(f"源码 symbol 无法定位到函数体：{symbol}")
    bodies = []
    for match in matches:
        opening = masked.find("{", match.start())
        bodies.append(text[match.start():_matching_brace(masked, opening)])
    return bodies


def _read_path(root: dict[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split(".")[1:]:
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"performance 中不存在字段：{path}")
        value = value[part]
    return value


def _load_performance(report_path: Path) -> dict[str, Any]:
    path = report_path.resolve().parent.parent / "performance/performance.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取同版本 performance.json：{error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("performance.json 顶层必须是对象")
    return value


def _validate_source(project: Path, key: str, cause: str, source: str) -> None:
    if "::" not in source:
        raise RuntimeError("源码 evidence 必须使用 相对文件::符号")
    relative, symbol = source.split("::", 1)
    path = (project / relative).resolve()
    if not relative.startswith(("op_host/", "op_kernel/")) or not path.is_relative_to(project.resolve()) or not path.is_file():
        raise RuntimeError(f"源码 evidence 文件不存在或越界：{relative}")
    if key.startswith("source.host.") and not relative.startswith("op_host/"):
        raise RuntimeError(f"{key} 只能引用 op_host：{relative}")
    if key.startswith(("source.kernel.", "source.hot_path.")) and not relative.startswith("op_kernel/"):
        raise RuntimeError(f"{key} 只能引用 op_kernel：{relative}")
    if not symbol.strip():
        raise RuntimeError(f"源码 evidence 符号不存在：{source}")
    bodies = _symbol_bodies(path.read_text(encoding="utf-8", errors="replace"), symbol)
    anchors = SOURCE_ANCHORS.get(cause)
    if anchors and not any(anchor in body for body in bodies for anchor in anchors):
        raise RuntimeError(f"{cause} 的 target 缺少源码锚点：{'/'.join(anchors)}")


def validate(path: Path) -> None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取报告：{error}") from error
    if not isinstance(report, dict) or set(report) != {"reasoning", "issues"}:
        raise RuntimeError("报告只能包含 reasoning、issues")
    reasoning, issues = report["reasoning"], report["issues"]
    if not isinstance(reasoning, list) or any(not isinstance(x, str) or not x.strip() for x in reasoning):
        raise RuntimeError("reasoning 必须是非空文本数组")
    if not isinstance(issues, list):
        raise RuntimeError("issues 必须是数组")
    if len(reasoning) != (len(issues) if issues else 1):
        raise RuntimeError("reasoning 必须与 issues 逐项对应；无问题时保留一条终止说明")

    performance = _load_performance(path)
    project = path.resolve().parent.parent
    seen_causes: set[str] = set()
    cause_sources: dict[str, set[str]] = {}
    for number, issue in enumerate(issues, 1):
        if not isinstance(issue, dict) or set(issue) != {"evidence", "bottleneck"}:
            raise RuntimeError(f"issue {number} 只能包含 evidence、bottleneck")
        bottleneck = issue["bottleneck"]
        if not isinstance(bottleneck, dict) or set(bottleneck) != {"bottleneck_key", "cause_key"}:
            raise RuntimeError(f"issue {number} bottleneck 字段非法")
        cause = bottleneck["cause_key"]
        if cause not in CAUSES or cause in seen_causes:
            raise RuntimeError(f"issue {number} cause 未知或重复：{cause}")
        seen_causes.add(cause)
        if bottleneck["bottleneck_key"] != CAUSES[cause][0]:
            raise RuntimeError(f"issue {number} bottleneck 与 cause 固定映射不一致")
        evidence = issue["evidence"]
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 4:
            raise RuntimeError(f"issue {number} evidence 必须包含 1–4 条")
        keys: set[str] = set()
        sources: set[str] = set()
        for item in evidence:
            if not isinstance(item, dict) or set(item) != {"evidence_key", "source", "observation"}:
                raise RuntimeError(f"issue {number} evidence 字段非法")
            if any(not isinstance(item[k], str) or not item[k].strip() for k in item):
                raise RuntimeError(f"issue {number} evidence 字段必须非空")
            key, source = item["evidence_key"], item["source"]
            if key not in EVIDENCE_KEYS or key in keys:
                raise RuntimeError(f"issue {number} evidence key 未知或重复：{key}")
            keys.add(key)
            if key.startswith("source."):
                _validate_source(project, key, cause, source)
                sources.add(source)
                if re.search(r"performance\.|ratio|cycle|Task Duration|收益|%", item["observation"], re.I):
                    raise RuntimeError("源码 observation 不得混入性能结论")
            else:
                if source not in METRIC_SOURCES[key]:
                    raise RuntimeError(f"{key} 不能引用 {source}")
                value = _read_path(performance, source)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise RuntimeError(f"{source} 不是有限数值")
        required_source = CAUSES[cause][1]
        if required_source not in keys:
            raise RuntimeError(f"issue {number} 缺少直接源码证据：{required_source}")
        symbols = [source.rsplit("::", 1)[-1] for source in sources]
        if not any(symbol in reasoning[number - 1] for symbol in symbols):
            raise RuntimeError(f"issue {number} reasoning 未引用源码 symbol")
        cause_sources[cause] = sources

    for group in EXCLUSIVE_CAUSE_GROUPS:
        present = [cause for cause in group if cause in cause_sources]
        for index, left in enumerate(present):
            for right in present[index + 1:]:
                if cause_sources[left] & cause_sources[right]:
                    raise RuntimeError(f"同一源码机制不得重复归因：{left}/{right}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        validate(args.report)
    except RuntimeError as error:
        raise SystemExit(f"INVALID_BOTTLENECK: {error}") from error
    print(f"valid={args.report.resolve()}")


if __name__ == "__main__":
    main()
