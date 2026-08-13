#!/usr/bin/env python3
"""Validate faithful implementation of strategy actions and bounded repairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

STRATEGY_SCRIPTS = Path(__file__).resolve().parents[2] / "kernel-strategy/scripts"
sys.path.insert(0, str(STRATEGY_SCRIPTS))
from validate_strategy import validate as validate_strategy  # noqa: E402


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
    previous = None
    for number, item in enumerate(attempts, 1):
        if not isinstance(item, dict) or set(item) != fields:
            raise RuntimeError(f"attempt {number} 字段不符合契约")
        if item["attempt"] != number or item["kind"] != ("initial" if number == 1 else "repair"):
            raise RuntimeError("attempt 编号或 kind 非法")
        trigger = item["trigger"]
        if number == 1 and trigger is not None:
            raise RuntimeError("初次实施 trigger 必须为 null")
        if number > 1:
            if not isinstance(trigger, dict) or set(trigger) != {"stage", "symptom", "evidence"}:
                raise RuntimeError("repair trigger 字段不完整")
            if trigger["stage"] not in {"build", "precision"} or previous is None or previous[trigger["stage"]] != "FAILED":
                raise RuntimeError("repair 必须由前次 build/precision 失败触发")
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
        if validation["build"] != "PASS" and (validation["precision"] != "NOT_RUN" or validation["performance"] != "NOT_RUN"):
            raise RuntimeError("build 未通过不得运行后续门禁")
        if validation["precision"] != "PASS" and validation["performance"] != "NOT_RUN":
            raise RuntimeError("precision 未通过不得运行 performance")
        previous = validation


def validate(bottleneck_path: Path, strategy_path: Path, implementation_path: Path, require_attempts: bool = False) -> None:
    validate_strategy(bottleneck_path, strategy_path)
    primary = load(strategy_path, "策略").get("strategy")
    implementation = load(implementation_path, "实施记录")
    if not isinstance(primary, dict) or not primary.get("actions"):
        raise RuntimeError("策略缺少 actions")
    fields = {"strategy_key", "reasoning", "actions", "modified_files"}
    if set(implementation) not in (fields, fields | {"attempts"}):
        raise RuntimeError("implementation 字段不符合契约")
    if require_attempts and "attempts" not in implementation:
        raise RuntimeError("新实施记录必须包含 attempts")
    if implementation.get("strategy_key") != primary["strategy_key"]:
        raise RuntimeError("strategy_key 与策略不一致")
    count = len(primary["actions"])
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
    if not isinstance(reasoning, list) or not 2 <= len(reasoning) <= 8 or any(not isinstance(x, str) or "证据：" not in x or "推断：" not in x for x in reasoning):
        raise RuntimeError("reasoning 不符合证据到推断格式")
    text = "\n".join(reasoning)
    if any(f"action_index={i}" not in text for i in indices):
        raise RuntimeError("reasoning 必须引用每个 action_index")
    files = implementation.get("modified_files")
    project = strategy_path.resolve().parent.parent
    if not isinstance(files, list) or not files or any(not isinstance(p, str) or not p.startswith(("op_host/", "op_kernel/")) or not (project / p).is_file() for p in files):
        raise RuntimeError("modified_files 非法")
    target_files = {item["target"].split("::", 1)[0] for item in primary["actions"]}
    if not target_files <= set(files):
        raise RuntimeError("modified_files 未覆盖全部 action target")
    if "attempts" in implementation:
        validate_attempts(implementation["attempts"], set(indices), project)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottleneck", required=True, type=Path)
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--implementation", required=True, type=Path)
    parser.add_argument("--require-attempts", action="store_true")
    args = parser.parse_args()
    try:
        validate(args.bottleneck, args.strategy, args.implementation, args.require_attempts)
    except RuntimeError as error:
        raise SystemExit(f"INVALID_IMPLEMENTATION: {error}") from error
    print(f"valid={args.implementation.resolve()}")


if __name__ == "__main__":
    main()
