#!/usr/bin/env python3
"""Validate faithful implementation of strategy actions and bounded repairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STRATEGY_SCRIPTS = Path(__file__).resolve().parents[2] / "kernel-strategy/scripts"
sys.path.insert(0, str(STRATEGY_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_strategy import validate as validate_strategy  # noqa: E402
from validate_source_effect import validate as validate_source_effect  # noqa: E402


def load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取{label}：{error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}必须是对象")
    return value


def validate_attempts(attempts: object, expected: set[int], project: Path) -> None:
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 4:
        raise RuntimeError("attempts 必须包含初次实施及最多 3 次修复")
    fields = {"attempt", "kind", "trigger", "knowledge_keys", "action_indices", "implementation_summary", "modified_files", "validation"}
    for number, item in enumerate(attempts, 1):
        if not isinstance(item, dict) or set(item) != fields:
            raise RuntimeError(f"attempt {number} 字段不符合契约")
        if item["attempt"] != number or item["kind"] != ("initial" if number == 1 else "repair"):
            raise RuntimeError("attempt 编号或 kind 非法")
        indices = item["action_indices"]
        if not isinstance(indices, list) or not indices or not set(indices) <= expected:
            raise RuntimeError("action_indices 非法")
        if number == 1 and set(indices) != expected:
            raise RuntimeError("初次实施必须覆盖全部 actions")
        files = item["modified_files"]
        if not isinstance(files, list) or not files or any(not isinstance(p, str) or not p.startswith(("op_host/", "op_kernel/")) or not (project / p).is_file() for p in files):
            raise RuntimeError("attempt modified_files 非法")
        validation = item["validation"]
        if not isinstance(validation, dict) or set(validation) != {"build", "precision", "performance"}:
            raise RuntimeError("attempt validation 非法")
        if validation["build"] not in {"PASS", "FAILED", "NOT_RUN"} or validation["precision"] not in {"PASS", "FAILED", "NOT_RUN"} or validation["performance"] not in {"DONE", "FAILED", "NOT_RUN"}:
            raise RuntimeError("门禁状态非法")


def validate(
    bottleneck_path: Path,
    strategy_path: Path,
    implementation_path: Path,
    require_attempts: bool = False,
    parent_path: Path | None = None,
    project_path: Path | None = None,
) -> None:
    validate_strategy(bottleneck_path, strategy_path)
    primary = load(strategy_path, "策略").get("strategies")
    implementation = load(implementation_path, "实施记录")
    if not isinstance(primary, list) or not primary:
        raise RuntimeError("策略缺少 strategies")
    planned_actions = [action for strategy in primary for action in strategy.get("actions", [])]
    fields = {"strategy_keys", "reasoning", "actions", "modified_files"}
    if set(implementation) not in (fields, fields | {"attempts"}):
        raise RuntimeError("implementation 字段不符合契约")
    if require_attempts and "attempts" not in implementation:
        raise RuntimeError("新实施记录必须包含 attempts")
    expected_keys = [strategy["strategy_key"] for strategy in primary]
    if implementation.get("strategy_keys") != expected_keys:
        raise RuntimeError("strategy_keys 与策略顺序不一致")
    count = len(planned_actions)
    actions = implementation.get("actions")
    if not isinstance(actions, list) or len(actions) != count:
        raise RuntimeError("implementation 必须逐项覆盖 strategy actions")
    indices = []
    for item in actions:
        if not isinstance(item, dict) or set(item) != {"action_index", "implementation_summary"}:
            raise RuntimeError("实施 action 字段非法")
        if not isinstance(item["implementation_summary"], str) or not item["implementation_summary"].strip():
            raise RuntimeError("implementation_summary 不能为空")
        indices.append(item["action_index"])
    if indices != list(range(1, count + 1)):
        raise RuntimeError("action_index 必须从 1 连续覆盖全部 actions")
    reasoning = implementation.get("reasoning")
    if not isinstance(reasoning, list):
        raise RuntimeError("reasoning 必须是数组")
    files = implementation.get("modified_files")
    # strategy.json is the immutable parent-version plan, while
    # implementation.json and the modified sources live in the child version.
    # Do not infer the child from strategy_path when an explicit project is
    # available.  The fallback preserves the standalone legacy invocation.
    project = (project_path or strategy_path.resolve().parent.parent).resolve()
    if not isinstance(files, list) or not files or any(not isinstance(p, str) or not p.startswith(("op_host/", "op_kernel/")) or not (project / p).is_file() for p in files):
        raise RuntimeError("modified_files 非法")
    target_files = {item["target"].split("::", 1)[0] for item in planned_actions}
    if not target_files <= set(files):
        raise RuntimeError("modified_files 未覆盖全部 action target")
    if "attempts" in implementation:
        validate_attempts(implementation["attempts"], set(indices), project)
    if parent_path is not None:
        validate_source_effect(parent_path.resolve(), project.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottleneck", required=True, type=Path)
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--implementation", required=True, type=Path)
    parser.add_argument("--require-attempts", action="store_true")
    parser.add_argument("--parent", type=Path, help="父版本目录；提供时强制执行源码效果复检")
    parser.add_argument("--project-dir", type=Path, help="实施后的子版本目录")
    args = parser.parse_args()
    try:
        validate(
            args.bottleneck,
            args.strategy,
            args.implementation,
            args.require_attempts,
            args.parent,
            args.project_dir,
        )
    except RuntimeError as error:
        raise SystemExit(f"INVALID_IMPLEMENTATION: {error}") from error
    print(f"valid={args.implementation.resolve()}")


if __name__ == "__main__":
    main()
