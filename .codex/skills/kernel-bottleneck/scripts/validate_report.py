#!/usr/bin/env python3
"""Validate one bottleneck inferred from raw metric and source evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


BOTTLENECK_DESCRIPTIONS = {
    "overhead.fixed_cost_bound": "有效工作量不足以摊薄启动、初始化、调度或 Scalar 固定成本",
    "parallel.core_underuse": "独立任务充足，但对应类型的活跃核明显少于可用核",
    "parallel.load_imbalance": "多核参与，但慢核工作尾部显著决定总时延",
    "pipeline.mte2_bound": "搬入流水线是本轮首选执行限制",
    "pipeline.mte3_bound": "搬出流水线是本轮首选执行限制",
    "pipeline.cube_bound": "Cube 是本轮首选执行限制",
    "pipeline.vector_bound": "Vector 是本轮首选执行限制",
    "pipeline.fixp_bound": "FixPipe 是本轮首选执行限制",
    "pipeline.scalar_bound": "Scalar 或 ScalarLDST 是本轮首选执行限制",
}

BOTTLENECK_SELECTION_ORDER = (
    "overhead.fixed_cost_bound",
    "parallel.core_underuse",
    "parallel.load_imbalance",
)
PIPELINE_TIE_ORDER = (
    "pipeline.mte2_bound", "pipeline.cube_bound", "pipeline.vector_bound",
    "pipeline.fixp_bound", "pipeline.mte3_bound", "pipeline.scalar_bound",
)
PIPELINE_TIE_MARGIN_PERCENTAGE_POINTS = 3.0

# Metric evidence names observations; the source is the exact performance path.
METRIC_SOURCES = {
    "metric.task.head_overhead_ratio": ("performance.task.head_overhead_ratio",),
    "metric.core.active_count": ("performance.per_core.active_cores",),
    "metric.hardware.available_core_count": (
        "performance.hardware.aic_core_count", "performance.hardware.aiv_core_count",
    ),
    "metric.core.imbalance_percent": ("performance.per_core.imbalance_percent",),
    "metric.pipeline.mte2_ratio": (
        "performance.pipeline.aic_mte2_ratio", "performance.pipeline.aiv_mte2_ratio",
    ),
    "metric.pipeline.mte3_ratio": ("performance.pipeline.aiv_mte3_ratio",),
    "metric.pipeline.cube_ratio": ("performance.pipeline.aic_mac_ratio",),
    "metric.pipeline.vector_ratio": ("performance.pipeline.aiv_vec_ratio",),
    "metric.pipeline.fixpipe_ratio": ("performance.pipeline.aic_fixpipe_ratio",),
    "metric.pipeline.scalar_ratio": (
        "performance.pipeline.aic_scalar_ratio", "performance.pipeline.aiv_scalar_ratio",
    ),
    "metric.memory.gm_peak_utilization_percent": (
        "performance.memory.gm_peak_utilization_percent",
    ),
    "metric.cache.l2_hit_rate_percent": (
        "performance.l2_cache.aiv_derived_hit_rate_percent",
    ),
    "metric.memory.ub_bank_conflict_ratio": (
        "performance.resource_conflict.aiv_vec_bankgroup_cflt_ratio",
        "performance.resource_conflict.aiv_vec_bank_cflt_ratio",
    ),
}

BOTTLENECK_EVIDENCE = {
    "overhead.fixed_cost_bound": ("metric.task.head_overhead_ratio",),
    "parallel.core_underuse": (
        "metric.core.active_count", "metric.hardware.available_core_count",
    ),
    "parallel.load_imbalance": (
        "metric.core.active_count", "metric.core.imbalance_percent",
    ),
    "pipeline.mte2_bound": ("metric.pipeline.mte2_ratio",),
    "pipeline.mte3_bound": ("metric.pipeline.mte3_ratio",),
    "pipeline.cube_bound": ("metric.pipeline.cube_ratio",),
    "pipeline.vector_bound": ("metric.pipeline.vector_ratio",),
    "pipeline.fixp_bound": ("metric.pipeline.fixpipe_ratio",),
    "pipeline.scalar_bound": ("metric.pipeline.scalar_ratio",),
}

# Medium-grained actionable causes. Concrete code shapes remain in observation.
CAUSES = {
    "redundant_hot_path_overhead": ("overhead.fixed_cost_bound", "source.hot_path.redundant_overhead"),
    "scalar_address_overhead": ("pipeline.scalar_bound", "source.hot_path.scalar_address_overhead"),
    "insufficient_parallelism": ("parallel.core_underuse", "source.host.parallel_mapping"),
    "uneven_task_distribution": ("parallel.load_imbalance", "source.task_distribution"),
    "nonuniform_task_cost": ("parallel.load_imbalance", "source.kernel.nonuniform_task_cost"),
    "inefficient_gm_transfer": ("pipeline.mte2_bound", "source.kernel.inefficient_gm_transfer"),
    "low_l2_reuse": ("pipeline.mte2_bound", "metric.cache.l2_hit_rate_percent"),
    "serial_copy_compute": ("pipeline.mte2_bound", "source.kernel.serial_copy_compute"),
    "gm_bandwidth_saturation": ("pipeline.mte2_bound", "metric.memory.gm_peak_utilization_percent"),
    "inefficient_writeback": ("pipeline.mte3_bound", "source.kernel.inefficient_writeback"),
    "low_cube_onchip_reuse": ("pipeline.cube_bound", "source.kernel.low_cube_onchip_reuse"),
    "ub_bank_conflict": ("pipeline.vector_bound", "metric.memory.ub_bank_conflict_ratio"),
    "redundant_gm_roundtrip": ("pipeline.vector_bound", "source.kernel.redundant_gm_roundtrip"),
    "excessive_cast_chain": ("pipeline.vector_bound", "source.kernel.excessive_cast_chain"),
    "high_latency_vector_reduction": ("pipeline.vector_bound", "source.kernel.high_latency_vector_reduction"),
    "inefficient_fixpipe_writeback": ("pipeline.fixp_bound", "source.kernel.inefficient_fixpipe_writeback"),
    "scalar_reduction": ("pipeline.scalar_bound", "source.kernel.scalar_reduction"),
    "scalar_elementwise_compute": ("pipeline.scalar_bound", "source.kernel.scalar_elementwise_compute"),
    "inherent_serial_dependency": ("pipeline.scalar_bound", "source.kernel.inherent_serial_dependency"),
}

CAUSE_DESCRIPTIONS = {
    "redundant_hot_path_overhead": "热路径存在重复初始化、循环不变量计算或过细资源生命周期",
    "scalar_address_overhead": "热路径主要消耗在 Scalar 索引展开、除法、取模或地址计算",
    "insufficient_parallelism": "可并行任务充足，但任务映射没有使用足够物理核",
    "uneven_task_distribution": "任务数量、余数或执行波次在参与核之间分配不均",
    "nonuniform_task_cost": "各核任务数量可能相近，但单任务计算代价不同",
    "inefficient_gm_transfer": "GM 搬运粒度、连续性或调用频率不合理",
    "low_l2_reuse": "可复用数据未有效命中 L2，造成重复 GM 读取",
    "serial_copy_compute": "搬入与计算串行执行，缺少有效重叠",
    "gm_bandwidth_saturation": "GM 搬入带宽接近可信硬件峰值，成为吞吐上限",
    "inefficient_writeback": "MTE3 输出写回不连续、未对齐或粒度过小",
    "low_cube_onchip_reuse": "Cube 操作数未在 L0/L1 中有效驻留和复用",
    "ub_bank_conflict": "UB 地址、布局或步长导致 bank 或 bank group 冲突",
    "redundant_gm_roundtrip": "连续计算的中间结果不必要地写回 GM 后再读入",
    "excessive_cast_chain": "主路径存在可合并或消除的连续类型转换",
    "high_latency_vector_reduction": "已向量化归约仍采用高延迟组合或多阶段路径",
    "inefficient_fixpipe_writeback": "FixPipe 输出地址、长度或分片不利于高效写回",
    "scalar_reduction": "使用 Scalar 循环执行归约",
    "scalar_elementwise_compute": "可向量化的逐元素计算仍由 Scalar 循环完成",
    "inherent_serial_dependency": "当前元素依赖历史状态，不能直接并行向量化",
}

CAUSE_PRIORITY = {
    "overhead.fixed_cost_bound": ("redundant_hot_path_overhead",),
    "parallel.core_underuse": ("insufficient_parallelism",),
    "parallel.load_imbalance": ("nonuniform_task_cost", "uneven_task_distribution"),
    "pipeline.mte2_bound": ("gm_bandwidth_saturation", "inefficient_gm_transfer", "low_l2_reuse", "serial_copy_compute"),
    "pipeline.mte3_bound": ("inefficient_writeback",),
    "pipeline.cube_bound": ("low_cube_onchip_reuse",),
    "pipeline.vector_bound": ("ub_bank_conflict", "redundant_gm_roundtrip", "excessive_cast_chain", "high_latency_vector_reduction"),
    "pipeline.fixp_bound": ("inefficient_fixpipe_writeback",),
    "pipeline.scalar_bound": ("inherent_serial_dependency", "scalar_address_overhead", "scalar_reduction", "scalar_elementwise_compute"),
}

SOURCE_EVIDENCE = {item[1] for item in CAUSES.values() if item[1].startswith("source.")}
EVIDENCE_KEYS = set(METRIC_SOURCES) | SOURCE_EVIDENCE
BOTTLENECK_FIELDS = {"bottleneck_key", "cause_key"}

if set(CAUSE_DESCRIPTIONS) != set(CAUSES):
    raise RuntimeError("CAUSE_DESCRIPTIONS 必须与 CAUSES 完整对应")
if {cause for ordered in CAUSE_PRIORITY.values() for cause in ordered} != set(CAUSES):
    raise RuntimeError("CAUSE_PRIORITY 必须覆盖且仅覆盖全部 CAUSES")


def read_path(root: dict[str, Any], source: str) -> Any:
    value: Any = root
    for part in source.split(".")[1:]:
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"performance 中不存在字段：{source}")
        value = value[part]
    return value


def load_performance(report_path: Path) -> dict[str, Any]:
    path = report_path.resolve().parent.parent / "performance/performance.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取同版本 performance.json：{error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("performance.json 顶层必须是对象")
    return value


def pipeline_ratios(performance: dict[str, Any]) -> dict[str, float]:
    paths = {
        "pipeline.mte2_bound": ("performance.pipeline.aic_mte2_ratio", "performance.pipeline.aiv_mte2_ratio"),
        "pipeline.mte3_bound": ("performance.pipeline.aiv_mte3_ratio",),
        "pipeline.cube_bound": ("performance.pipeline.aic_mac_ratio",),
        "pipeline.vector_bound": ("performance.pipeline.aiv_vec_ratio",),
        "pipeline.fixp_bound": ("performance.pipeline.aic_fixpipe_ratio",),
        "pipeline.scalar_bound": ("performance.pipeline.aic_scalar_ratio", "performance.pipeline.aiv_scalar_ratio"),
    }
    result = {}
    for key, sources in paths.items():
        values = [read_path(performance, source) for source in sources]
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise RuntimeError(f"{key} 的流水线 ratio 无效")
        result[key] = max(float(value) for value in values)
    return result


def derive_pipeline(performance: dict[str, Any]) -> str | None:
    ratios = pipeline_ratios(performance)
    maximum = max(ratios.values())
    candidates = []
    for key, ratio in ratios.items():
        formal = ratio > 0.8
        if key in {"pipeline.mte2_bound", "pipeline.cube_bound"}:
            formal = formal or (ratio > 0.7 and sum(value == maximum for value in ratios.values()) == 1)
        if formal:
            candidates.append(key)
    if not candidates:
        return None
    candidates.sort(key=lambda key: (-ratios[key], PIPELINE_TIE_ORDER.index(key)))
    top = candidates[0]
    near = [key for key in candidates if (ratios[top] - ratios[key]) * 100 <= PIPELINE_TIE_MARGIN_PERCENTAGE_POINTS]
    return min(near, key=PIPELINE_TIE_ORDER.index)


def validate(path: Path) -> None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取报告：{error}") from error
    if set(report) != {"reasoning", "evidence", "bottleneck"}:
        raise RuntimeError("报告只能包含 reasoning、evidence、bottleneck")
    reasoning = report["reasoning"]
    if not isinstance(reasoning, list) or len(reasoning) != 2:
        raise RuntimeError("reasoning 必须固定包含 2 条记录")
    if any(not isinstance(item, str) or "证据：" not in item or "推断：" not in item for item in reasoning):
        raise RuntimeError("reasoning 每项必须使用“证据：…；推断：…”格式")
    evidence = report["evidence"]
    if not isinstance(evidence, list):
        raise RuntimeError("evidence 必须是数组")
    if len(evidence) > 4:
        raise RuntimeError("evidence 最多保留 4 条必要证据")
    performance = load_performance(path)
    indexed = {}
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"evidence_key", "source", "observation"}:
            raise RuntimeError("每条 evidence 必须包含 evidence_key、source、observation")
        if any(not isinstance(item[field], str) or not item[field].strip() for field in item):
            raise RuntimeError("evidence 字段必须是非空字符串")
        key, source = item["evidence_key"], item["source"]
        if key not in EVIDENCE_KEYS or key in indexed:
            raise RuntimeError(f"未知或重复 evidence_key：{key}")
        if key.startswith("metric."):
            if source not in METRIC_SOURCES[key]:
                raise RuntimeError(f"{key} 不能引用 {source}")
            value = read_path(performance, source)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"{source} 不是有限数值")
            if str(value) not in item["observation"]:
                raise RuntimeError(f"{key} 的 observation 必须包含原始值 {value}")
        elif not source.startswith(("op_host/", "op_kernel/")):
            raise RuntimeError(f"{key} 必须引用 op_host/ 或 op_kernel/")
        else:
            if "::" not in source:
                raise RuntimeError(f"{key} 的 source 必须使用 相对文件::符号")
            relative, symbol = source.split("::", 1)
            if not symbol.strip():
                raise RuntimeError(f"{key} 的源码符号不能为空")
            source_path = path.resolve().parent.parent / relative
            if not source_path.is_file():
                raise RuntimeError(f"源码 evidence 文件不存在：{relative}")
            symbol_leaf = symbol.rsplit("::", 1)[-1]
            if symbol_leaf not in source_path.read_text(encoding="utf-8", errors="replace"):
                raise RuntimeError(f"源码 evidence 符号不存在：{source}")
            observation_lower = item["observation"].lower()
            forbidden = ("performance.", "task duration", "head overhead", "ratio", "cycle", "收益")
            has_percent_metric = re.search(r"\d(?:\.\d+)?\s*%", item["observation"]) is not None
            if any(token in observation_lower for token in forbidden) or has_percent_metric:
                raise RuntimeError(f"{key} observation 只能写源码事实，不得混入指标或收益")
        indexed[key] = item
    bottleneck = report["bottleneck"]
    if bottleneck is None:
        if evidence:
            raise RuntimeError("bottleneck=null 时 evidence 必须为空")
        return
    if not isinstance(bottleneck, dict) or set(bottleneck) != BOTTLENECK_FIELDS:
        raise RuntimeError("bottleneck 只能包含 bottleneck_key、cause_key")
    key, cause = bottleneck["bottleneck_key"], bottleneck["cause_key"]
    if key not in BOTTLENECK_EVIDENCE or cause not in CAUSES:
        raise RuntimeError("未知 bottleneck_key 或 cause_key")
    parent, cause_evidence = CAUSES[cause]
    if parent != key:
        raise RuntimeError(f"{cause} 不属于 {key}")
    for required in (*BOTTLENECK_EVIDENCE[key], cause_evidence):
        if required not in indexed:
            raise RuntimeError(f"{key}+{cause} 缺少必要原始证据 {required}")
    if key == "parallel.core_underuse":
        active = read_path(performance, indexed["metric.core.active_count"]["source"])
        available = read_path(performance, indexed["metric.hardware.available_core_count"]["source"])
        if active >= available:
            raise RuntimeError("active core 不少于 available core，不能认定 core_underuse")
    if key == "parallel.load_imbalance":
        active = read_path(performance, indexed["metric.core.active_count"]["source"])
        imbalance = read_path(performance, indexed["metric.core.imbalance_percent"]["source"])
        if active <= 1 or imbalance <= 30:
            raise RuntimeError("load_imbalance 必须满足 active_cores>1 且 imbalance_percent>30")
    if key.startswith("pipeline."):
        derived = derive_pipeline(performance)
        if derived != key:
            raise RuntimeError(f"流水线原始 ratio 推导结果为 {derived}，不是 {key}")
    text = "\n".join(reasoning)
    for token in (*indexed, key, cause):
        if token not in text:
            raise RuntimeError(f"reasoning 未引用 {token}")
    for evidence_key, item in indexed.items():
        if evidence_key.startswith("metric."):
            value = read_path(performance, item["source"])
            if str(value) not in text:
                raise RuntimeError(f"reasoning 未引用 {evidence_key} 的原始值 {value}")
    metric_required = BOTTLENECK_EVIDENCE[key]
    if any(token not in reasoning[0] for token in metric_required) or f"bottleneck_key={key}" not in reasoning[0]:
        raise RuntimeError("第 1 条 reasoning 必须只完成必要正式指标到 bottleneck 的推断")
    if cause_evidence not in reasoning[1] or f"cause_key={cause}" not in reasoning[1]:
        raise RuntimeError("第 2 条 reasoning 必须完成直接原因 evidence 到 cause 的推断")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        validate(args.report)
    except RuntimeError as error:
        raise SystemExit(f"INVALID_BOTTLENECK_REPORT: {error}") from error
    print(f"valid={args.report.resolve()}")


if __name__ == "__main__":
    main()
