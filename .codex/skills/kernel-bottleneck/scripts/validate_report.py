#!/usr/bin/env python3
"""Validate normalized evidence and the single bottleneck conclusion."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BOTTLENECK_EVIDENCE = {
    "parallel.core_underuse": "metric.core_underuse",
    "parallel.load_imbalance": "metric.core_imbalance_over_30",
    "overhead.small_workload": "metric.small_workload",
    "pipeline.mte2_bound": "metric.mte2_bound",
    "pipeline.cube_bound": "metric.cube_bound",
    "pipeline.vector_bound": "metric.vector_bound",
    "pipeline.fixp_bound": "metric.fixp_bound",
    "pipeline.mte3_bound": "metric.mte3_bound",
    "pipeline.scalar_bound": "metric.scalar_bound",
}

ROOT_CAUSE_EVIDENCE = {
    "source.scalar_reduction_loop", "source.scalar_elementwise_loop",
    "source.init_or_ldst_heavy", "metric.small_transfer",
    "metric.low_l2_hit", "metric.bandwidth_saturated",
    "source.serial_copy_compute", "metric.ub_bank_conflict",
    "metric.excess_cast", "source.reduction_path", "source.gm_roundtrip",
    "source.low_onchip_reuse", "source.unaligned_or_small_write",
}
ROOT_CAUSES_BY_BOTTLENECK = {
    "overhead.small_workload": {"source.init_or_ldst_heavy"},
    "pipeline.mte2_bound": {
        "metric.small_transfer", "metric.low_l2_hit",
        "source.serial_copy_compute", "metric.bandwidth_saturated",
    },
    "pipeline.cube_bound": {"source.low_onchip_reuse"},
    "pipeline.vector_bound": {
        "metric.ub_bank_conflict", "metric.excess_cast",
        "source.reduction_path", "source.gm_roundtrip",
    },
    "pipeline.fixp_bound": {"source.unaligned_or_small_write"},
    "pipeline.mte3_bound": {"source.unaligned_or_small_write"},
    "pipeline.scalar_bound": {
        "source.scalar_reduction_loop", "source.scalar_elementwise_loop",
        "source.init_or_ldst_heavy",
    },
}

EVIDENCE_KEYS = set(BOTTLENECK_EVIDENCE.values()) | ROOT_CAUSE_EVIDENCE
METRIC_SOURCE_PREFIX = {
    "metric.core_underuse": "performance.per_core",
    "metric.core_imbalance_over_30": "performance.per_core.imbalance_percent",
    "metric.small_workload": "performance.task",
    "metric.mte2_bound": "performance.pipeline",
    "metric.cube_bound": "performance.pipeline",
    "metric.vector_bound": "performance.pipeline",
    "metric.fixp_bound": "performance.pipeline",
    "metric.mte3_bound": "performance.pipeline",
    "metric.scalar_bound": "performance.pipeline",
    "metric.small_transfer": "performance.memory",
    "metric.low_l2_hit": "performance.l2_cache",
    "metric.bandwidth_saturated": "performance.memory",
    "metric.ub_bank_conflict": "performance.resource_conflict",
    "metric.excess_cast": "performance.arithmetic",
}
EVIDENCE_FIELDS = {"evidence_key", "source", "observation"}
BOTTLENECK_FIELDS = {
    "bottleneck_key", "description", "evidence_keys", "causal_explanation",
}


def validate_reasoning(reasoning: object) -> None:
    if not isinstance(reasoning, list) or not 2 <= len(reasoning) <= 6:
        raise RuntimeError("reasoning 必须包含 2–6 条证据→推断记录")
    if any(
        not isinstance(item, str) or "证据：" not in item or "推断：" not in item
        for item in reasoning
    ):
        raise RuntimeError("reasoning 每项必须使用“证据：…；推断：…”格式")


def validate(path: Path) -> None:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取报告：{error}") from error
    if set(report) != {"reasoning", "evidence", "bottleneck"}:
        raise RuntimeError("报告只能包含 reasoning、evidence、bottleneck")
    validate_reasoning(report["reasoning"])

    evidence = report["evidence"]
    if not isinstance(evidence, list):
        raise RuntimeError("evidence 必须是数组")
    indexed = {}
    for item in evidence:
        if not isinstance(item, dict) or set(item) != EVIDENCE_FIELDS:
            raise RuntimeError("每条 evidence 必须包含 evidence_key、source、observation")
        if any(not isinstance(item[field], str) or not item[field].strip() for field in EVIDENCE_FIELDS):
            raise RuntimeError("evidence 字段必须是非空字符串")
        key = item["evidence_key"]
        if key not in EVIDENCE_KEYS:
            raise RuntimeError(f"未知 evidence_key：{key}")
        if key in indexed:
            raise RuntimeError(f"重复 evidence_key：{key}")
        expected_source = METRIC_SOURCE_PREFIX.get(key)
        if expected_source and not item["source"].startswith(expected_source):
            raise RuntimeError(f"{key} 必须引用 {expected_source}")
        if key.startswith("source.") and not item["source"].startswith(("op_host/", "op_kernel/")):
            raise RuntimeError(f"{key} 必须引用 op_host/ 或 op_kernel/ 文件")
        if key == "metric.small_workload":
            if item["source"] != "performance.task.head_overhead_ratio":
                raise RuntimeError("metric.small_workload 必须引用 performance.task.head_overhead_ratio")
            values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", item["observation"])]
            if not values or max(values) <= 30.0:
                raise RuntimeError("metric.small_workload 必须记录超过 30% 的头开销占比")
        indexed[key] = item

    bottleneck = report["bottleneck"]
    if bottleneck is None:
        return
    if not isinstance(bottleneck, dict) or set(bottleneck) != BOTTLENECK_FIELDS:
        raise RuntimeError("bottleneck 字段不符合固定契约")
    key = bottleneck["bottleneck_key"]
    if key not in BOTTLENECK_EVIDENCE:
        raise RuntimeError(f"未知 bottleneck_key：{key}")
    used = bottleneck["evidence_keys"]
    if not isinstance(used, list) or not used or len(used) != len(set(used)):
        raise RuntimeError("evidence_keys 必须是非空且不重复的数组")
    if set(used) != set(indexed):
        raise RuntimeError("bottleneck.evidence_keys 必须完整引用 evidence")
    required = BOTTLENECK_EVIDENCE[key]
    if required not in indexed:
        raise RuntimeError(f"{key} 缺少正式主 Bound 证据 {required}")
    used_root_causes = set(used) & ROOT_CAUSE_EVIDENCE
    allowed_root_causes = ROOT_CAUSES_BY_BOTTLENECK.get(key, set())
    unexpected_root_causes = used_root_causes - allowed_root_causes
    if unexpected_root_causes:
        raise RuntimeError(
            f"{key} 包含不匹配的根因证据：{', '.join(sorted(unexpected_root_causes))}"
        )
    if key.startswith("pipeline.") and not used_root_causes:
        raise RuntimeError(f"{key} 必须包含至少一个匹配的根因 evidence_key")
    if key == "overhead.small_workload" and "source.init_or_ldst_heavy" not in used_root_causes:
        raise RuntimeError("overhead.small_workload 必须由源码确认固定初始化或 Scalar LD/ST 根因")
    reasoning_text = "\n".join(report["reasoning"])
    missing_mentions = [evidence_key for evidence_key in used if evidence_key not in reasoning_text]
    if missing_mentions:
        raise RuntimeError("reasoning 未引用 evidence_key：" + ", ".join(missing_mentions))
    for field in ("description", "causal_explanation"):
        if not isinstance(bottleneck[field], str) or not bottleneck[field].strip():
            raise RuntimeError(f"bottleneck.{field} 必须是非空字符串")


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
