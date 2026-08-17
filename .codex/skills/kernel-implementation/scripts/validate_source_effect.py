#!/usr/bin/env python3
"""Require planned targets to change; report heuristic antipattern counts as warnings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BOTTLENECK_SCRIPTS = Path(__file__).resolve().parents[2] / "kernel-bottleneck/scripts"
sys.path.insert(0, str(BOTTLENECK_SCRIPTS))
from validate_report import _symbol_bodies  # noqa: E402


CHECKABLE_PATTERNS = {
    "scalar_local_lane_compute": ("GetValue(", "SetValue("),
    "scalar_global_contiguous_access": ("GetValue(", "SetValue("),
    "redundant_vector_materialization": ("Duplicate(",),
    "over_synchronization": ("PipeBarrier<", "SyncAll(", "WaitFlag<", "SetFlag<"),
    "excessive_cast_chain": ("Cast(",),
    "atomic_write_contention": ("Atomic",),
}
GLOBAL_SCALAR_ANTIPATTERNS = ("GetValue(", "SetValue(")


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON 顶层必须为对象：{path}")
    return value


def source_files(project: Path) -> dict[str, str]:
    return {
        path.relative_to(project).as_posix(): path.read_text(encoding="utf-8", errors="replace")
        for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    }


def validate(parent: Path, child: Path) -> list[str]:
    bottleneck = load(parent / "bottleneck/bottleneck.json")
    strategy = load(parent / "strategy/strategy.json")
    parent_files, child_files = source_files(parent), source_files(child)
    strategies = strategy.get("strategies")
    issues = bottleneck.get("issues")
    if not isinstance(strategies, list) or not isinstance(issues, list) or not issues:
        raise RuntimeError("父版本必须包含非空 issues 和 strategies")
    if len(strategies) != len(issues):
        raise RuntimeError("父版本 strategies 必须完整覆盖全部 issues")
    targets = {
        action["target"].split("::", 1)[0]
        for item in strategies for action in item.get("actions", [])
    }
    unchanged = sorted(path for path in targets if parent_files.get(path) == child_files.get(path))
    if unchanged:
        raise RuntimeError("action target 未发生源码变化：" + ",".join(unchanged))
    unchanged_symbols = []
    for item in strategies:
        for action in item.get("actions", []):
            relative, symbol = action["target"].split("::", 1)
            before_text = parent_files.get(relative)
            after_text = child_files.get(relative)
            if before_text is None or after_text is None:
                raise RuntimeError(f"action target 文件缺失：{action['target']}")
            before = _symbol_bodies(before_text, symbol)
            after = _symbol_bodies(after_text, symbol)
            if before == after:
                unchanged_symbols.append(action["target"])
    if unchanged_symbols:
        raise RuntimeError("action target symbol 未发生源码变化：" + ",".join(sorted(set(unchanged_symbols))))
    warnings: list[str] = []
    for pattern in GLOBAL_SCALAR_ANTIPATTERNS:
        before = sum(text.count(pattern) for text in parent_files.values())
        after = sum(text.count(pattern) for text in child_files.values())
        if after > before:
            warnings.append(f"文本计数新增 {pattern}：{before}->{after}，需结合动态热路径复核")
    for issue, item in zip(issues, strategies):
        cause = issue["bottleneck"]["cause_key"]
        patterns = CHECKABLE_PATTERNS.get(cause)
        if not patterns:
            continue
        target_paths = {action["target"].split("::", 1)[0] for action in item["actions"]}
        before = sum(parent_files.get(path, "").count(pattern) for path in target_paths for pattern in patterns)
        after = sum(child_files.get(path, "").count(pattern) for path in target_paths for pattern in patterns)
        if before <= 0:
            warnings.append(f"{cause} 在父版本 target 中没有可文本计数的锚点")
        if after >= before:
            warnings.append(f"{cause} 文本计数未减少：{before}->{after}，允许主路径转为 tail/辅助路径")
    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--child", required=True, type=Path)
    args = parser.parse_args()
    try:
        warnings = validate(args.parent.resolve(), args.child.resolve())
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise SystemExit(f"INVALID_SOURCE_EFFECT: {error}") from error
    print(f"valid_source_effect={args.child.resolve()}")
    for warning in warnings:
        print(f"warning={warning}")


if __name__ == "__main__":
    main()
