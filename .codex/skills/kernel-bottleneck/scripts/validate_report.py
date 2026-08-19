#!/usr/bin/env python3
"""Validate direct, source-grounded kernel bottleneck reports."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "references/cause-taxonomy.json"
CAUSES = {key: tuple(value) for key, value in json.loads(_TAXONOMY_PATH.read_text()).items()}

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
BOTTLENECK_EVIDENCE = {key: () for key, _ in CAUSES.values()}

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
        matches = list(re.finditer(
            rf"\b(?:struct|class)\s+{name}\b[^;{{}}]*\{{", masked, re.S
        ))
    if not matches:
        macro = re.search(
            rf"\bBEGIN_TILING_DATA_DEF\s*\(\s*{name}\s*\)([\s\S]*?)\bEND_TILING_DATA_DEF\b",
            masked,
        )
        if macro:
            return [text[macro.start():macro.end()]]
    if not matches:
        raise RuntimeError(f"源码 symbol 无法定位到函数、类型或 tiling 定义：{symbol}")
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
    if cause == "scalar_reduction" and any(
        re.search(r"\bLocalTensor\s*<", body)
        and re.search(r"\b(?:Compare|Select|Reduce\w*|WholeReduce\w*|BlockReduce\w*)\s*\(", body)
        for body in bodies
    ):
        raise RuntimeError("scalar_reduction 不能引用已使用 LocalTensor Vector 归约状态的 symbol")


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
    if len(issues) > 3:
        raise RuntimeError("每轮最多输出 3 个最高影响 issues")
    if len(reasoning) != (len(issues) if issues else 1):
        raise RuntimeError("reasoning 必须与 issues 逐项对应；无问题时保留一条终止说明")

    performance = _load_performance(path)
    project = path.resolve().parent.parent
    seen_causes: set[str] = set()
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
