#!/usr/bin/env python3
"""Validate only executable strategy structure, mapping, and source targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from derive_strategy import OPERATIONS, RULES

BOTTLENECK_SCRIPTS = Path(__file__).resolve().parents[2] / "kernel-bottleneck/scripts"
sys.path.insert(0, str(BOTTLENECK_SCRIPTS))
from validate_report import validate as validate_bottleneck  # noqa: E402


def _load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取{label}：{error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}必须是对象")
    return value


def _target(project: Path, target: str) -> None:
    if "::" not in target:
        raise RuntimeError(f"target 必须使用 相对文件::符号：{target}")
    relative, symbol = target.split("::", 1)
    path = (project / relative).resolve()
    if (not relative.startswith(("op_host/", "op_kernel/"))
            or not path.is_relative_to(project.resolve()) or not path.is_file()):
        raise RuntimeError(f"target 文件不存在或越界：{target}")
    if not symbol.strip() or symbol.rsplit("::", 1)[-1] not in path.read_text(
        encoding="utf-8", errors="replace"
    ):
        raise RuntimeError(f"target 符号不存在：{target}")


def _validate_action(project: Path, action: object, operation: str, label: str) -> None:
    fields = {"target", "operation", "edits", "constraints"}
    if not isinstance(action, dict) or set(action) != fields:
        raise RuntimeError(f"{label} action 字段非法")
    if not isinstance(action["target"], str):
        raise RuntimeError(f"{label} target 非法")
    _target(project, action["target"])
    if action["operation"] != operation:
        raise RuntimeError(f"{label} operation 与 strategy 不一致")
    for field in ("edits", "constraints"):
        values = action[field]
        if (not isinstance(values, list) or not 1 <= len(values) <= 8
                or any(not isinstance(value, str) or not value.strip() for value in values)):
            raise RuntimeError(f"{label} {field} 必须包含 1–8 项非空文本")


def validate(bottleneck_path: Path, strategy_path: Path) -> None:
    validate_bottleneck(bottleneck_path)
    bottleneck = _load(bottleneck_path, "瓶颈")
    report = _load(strategy_path, "策略")
    if set(report) != {"reasoning", "strategies"}:
        raise RuntimeError("strategy.json 只能包含 reasoning、strategies")
    reasoning, strategies = report["reasoning"], report["strategies"]
    if not isinstance(reasoning, list) or any(not isinstance(x, str) for x in reasoning):
        raise RuntimeError("strategy reasoning 必须是文本数组")
    if not isinstance(strategies, list):
        raise RuntimeError("strategies 必须是数组")
    causes = [item["bottleneck"]["cause_key"] for item in bottleneck["issues"]]
    if not causes:
        if strategies:
            raise RuntimeError("issues 为空时 strategies 必须为空")
        return
    if len(strategies) != len(causes):
        raise RuntimeError("strategies 必须按顺序完整覆盖全部 issues")
    if sum(len(item.get("actions", [])) for item in strategies if isinstance(item, dict)) > 6:
        raise RuntimeError("本轮全部 strategies 合计最多 6 个 actions")
    project = bottleneck_path.resolve().parent.parent
    for index, (cause, strategy) in enumerate(zip(causes, strategies), 1):
        if not isinstance(strategy, dict) or set(strategy) != {"cause_key", "strategy_key", "actions"}:
            raise RuntimeError(f"strategy {index} 字段非法")
        if strategy["cause_key"] != cause:
            raise RuntimeError(f"strategy {index} cause 顺序错误")
        expected = RULES[cause]
        if strategy["strategy_key"] != expected:
            raise RuntimeError(f"strategy {index} key 与固定映射不一致")
        actions = strategy["actions"]
        if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
            raise RuntimeError(f"strategy {index} actions 必须包含 1–8 项")
        identities = set()
        for action in actions:
            _validate_action(project, action, OPERATIONS[expected], f"strategy {index}")
            identity = (action["target"], action["operation"])
            if identity in identities:
                raise RuntimeError(f"strategy {index} 重复 target+operation")
            identities.add(identity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bottleneck", required=True, type=Path)
    parser.add_argument("--strategy", required=True, type=Path)
    args = parser.parse_args()
    try:
        validate(args.bottleneck, args.strategy)
    except RuntimeError as error:
        raise SystemExit(f"INVALID_STRATEGY: {error}") from error
    print(f"valid={args.strategy.resolve()}")


if __name__ == "__main__":
    main()
