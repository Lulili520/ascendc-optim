#!/usr/bin/env python3
"""Validate that implementation.json faithfully implements strategy changes."""

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
        raise RuntimeError(f"{label}必须是 JSON 对象")
    return value


def validate_attempts(attempts: object, expected_keys: set[str], project: Path) -> None:
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 4:
        raise RuntimeError("attempts 必须包含初次实施及最多 3 次修复")
    required = {
        "attempt", "kind", "trigger", "knowledge_keys", "change_keys",
        "implementation_summary", "modified_files", "validation",
    }
    previous_validation = None
    for index, item in enumerate(attempts, start=1):
        if not isinstance(item, dict) or set(item) != required:
            raise RuntimeError(f"attempt {index} 字段不完整或包含额外字段")
        if item["attempt"] != index:
            raise RuntimeError("attempt 必须从 1 连续递增")
        expected_kind = "initial" if index == 1 else "repair"
        if item["kind"] != expected_kind:
            raise RuntimeError(f"attempt {index} 的 kind 必须为 {expected_kind}")
        trigger = item["trigger"]
        if index == 1:
            if trigger is not None:
                raise RuntimeError("初次实施的 trigger 必须为 null")
        else:
            if not isinstance(trigger, dict) or set(trigger) != {"stage", "symptom", "evidence"}:
                raise RuntimeError(f"repair attempt {index} 缺少固定 trigger 字段")
            if trigger["stage"] not in {"build", "precision"}:
                raise RuntimeError("源码修复只能由 build 或 precision 失败触发")
            if any(not isinstance(trigger[key], str) or not trigger[key].strip()
                   for key in ("symptom", "evidence")):
                raise RuntimeError(f"repair attempt {index} 的 symptom/evidence 不能为空")
            if previous_validation is None or previous_validation[trigger["stage"]] != "FAILED":
                raise RuntimeError(f"repair attempt {index} 的 trigger 与前次失败状态不一致")
        knowledge = item["knowledge_keys"]
        if not isinstance(knowledge, list) or any(
            not isinstance(key, str) or not key.strip() or "/" in key for key in knowledge
        ):
            raise RuntimeError(f"attempt {index} 的 knowledge_keys 必须是稳定知识 key，不能保存路径")
        change_keys = item["change_keys"]
        if not isinstance(change_keys, list) or not change_keys or len(change_keys) != len(set(change_keys)):
            raise RuntimeError(f"attempt {index} 的 change_keys 必须是非空且不重复的数组")
        if not set(change_keys) <= expected_keys:
            raise RuntimeError(f"attempt {index} 引用了策略外 change_key")
        if index == 1 and set(change_keys) != expected_keys:
            raise RuntimeError("初次实施必须覆盖策略的全部 change_key")
        if not isinstance(item["implementation_summary"], str) or not item["implementation_summary"].strip():
            raise RuntimeError(f"attempt {index} 缺少 implementation_summary")
        files = item["modified_files"]
        if not isinstance(files, list) or not files or any(
            not isinstance(path, str) or not path.startswith(("op_host/", "op_kernel/"))
            for path in files
        ):
            raise RuntimeError(f"attempt {index} 的 modified_files 非法")
        if any(not (project / path).is_file() for path in files):
            raise RuntimeError(f"attempt {index} 的 modified_files 包含不存在文件")
        validation = item["validation"]
        if not isinstance(validation, dict) or set(validation) != {"build", "precision", "performance"}:
            raise RuntimeError(f"attempt {index} 的 validation 字段不完整")
        if validation["build"] not in {"PASS", "FAILED", "NOT_RUN"}:
            raise RuntimeError(f"attempt {index} 的 build 状态非法")
        if validation["precision"] not in {"PASS", "FAILED", "NOT_RUN"}:
            raise RuntimeError(f"attempt {index} 的 precision 状态非法")
        if validation["performance"] not in {"DONE", "FAILED", "NOT_RUN"}:
            raise RuntimeError(f"attempt {index} 的 performance 状态非法")
        if validation["build"] != "PASS" and (
            validation["precision"] != "NOT_RUN" or validation["performance"] != "NOT_RUN"
        ):
            raise RuntimeError(f"attempt {index} 未通过 build 时不得运行后续门禁")
        if validation["precision"] != "PASS" and validation["performance"] != "NOT_RUN":
            raise RuntimeError(f"attempt {index} 未通过 precision 时不得运行 performance")
        if index < len(attempts) and not (
            validation["build"] == "FAILED" or validation["precision"] == "FAILED"
        ):
            raise RuntimeError(f"attempt {index} 未失败，不得追加 repair")
        previous_validation = validation


def validate(
    bottleneck_path: Path, strategy_path: Path, implementation_path: Path,
    require_attempts: bool = False,
) -> None:
    validate_strategy(bottleneck_path, strategy_path)
    strategy = load(strategy_path, "策略")
    implementation = load(implementation_path, "实施记录")
    primary = strategy.get("strategy")
    if not isinstance(primary, dict):
        raise RuntimeError("策略缺少 strategy")
    expected = {item["change_key"]: item for item in primary.get("changes", [])}
    if not expected:
        raise RuntimeError("策略没有 change_key")
    base_fields = {"strategy_key", "reasoning", "changes", "modified_files"}
    allowed_fields = base_fields | {"attempts"}
    if frozenset(implementation) not in {frozenset(base_fields), frozenset(allowed_fields)}:
        raise RuntimeError("implementation 包含训练或实施无关字段")
    if require_attempts and "attempts" not in implementation:
        raise RuntimeError("新实施记录必须包含 attempts")
    if implementation.get("strategy_key") != primary.get("strategy_key"):
        raise RuntimeError("strategy_key 与策略不一致")
    reasoning = implementation.get("reasoning")
    if not isinstance(reasoning, list) or not 2 <= len(reasoning) <= 8:
        raise RuntimeError("reasoning 必须包含 2–8 条记录")
    if any(not isinstance(item, str) or "证据：" not in item or "推断：" not in item for item in reasoning):
        raise RuntimeError("reasoning 每项必须使用“证据：…；推断：…”格式")
    changes = implementation.get("changes")
    if not isinstance(changes, list):
        raise RuntimeError("changes 必须是数组")
    actual = {}
    for item in changes:
        required = {"change_key", "implementation_summary"}
        if not isinstance(item, dict) or set(item) != required:
            raise RuntimeError("每项 change 必须包含 change_key、implementation_summary")
        key = item["change_key"]
        if key in actual:
            raise RuntimeError(f"重复 change_key：{key}")
        if not isinstance(item["implementation_summary"], str) or not item["implementation_summary"].strip():
            raise RuntimeError(f"{key} 缺少 implementation_summary")
        actual[key] = item
    if set(actual) != set(expected):
        raise RuntimeError("implementation change_key 集合与策略不一致")
    reasoning_text = "\n".join(reasoning)
    missing_mentions = [key for key in expected if key not in reasoning_text]
    if missing_mentions:
        raise RuntimeError("reasoning 未引用 change_key：" + ", ".join(missing_mentions))
    files = implementation.get("modified_files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("modified_files 必须是非空数组")
    if any(not isinstance(path, str) or not path.startswith(("op_host/", "op_kernel/")) for path in files):
        raise RuntimeError("modified_files 只能包含 op_host/ 或 op_kernel/ 文件")
    project = strategy_path.resolve().parent.parent
    if any(not (project / path).is_file() for path in files):
        raise RuntimeError("modified_files 包含不存在的文件")
    target_files = {
        item["target"].split("::", 1)[0] for item in expected.values()
    }
    missing_targets = target_files - set(files)
    if missing_targets:
        raise RuntimeError("modified_files 未覆盖 strategy target：" + ", ".join(sorted(missing_targets)))
    if "attempts" in implementation:
        validate_attempts(implementation["attempts"], set(expected), project)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottleneck", required=True, type=Path)
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--implementation", required=True, type=Path)
    parser.add_argument("--require-attempts", action="store_true")
    args = parser.parse_args()
    try:
        validate(
            args.bottleneck, args.strategy, args.implementation,
            require_attempts=args.require_attempts,
        )
    except RuntimeError as error:
        raise SystemExit(f"INVALID_IMPLEMENTATION: {error}") from error
    print(f"valid={args.implementation.resolve()}")


if __name__ == "__main__":
    main()
