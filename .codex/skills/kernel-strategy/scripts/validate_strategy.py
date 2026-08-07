#!/usr/bin/env python3
"""Validate fixed strategy selection and source-specific executable changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from derive_strategy import derive


BOTTLENECK_SCRIPTS = Path(__file__).resolve().parents[2] / "kernel-bottleneck/scripts"
sys.path.insert(0, str(BOTTLENECK_SCRIPTS))
from validate_report import validate as validate_bottleneck  # noqa: E402


def load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取{label}：{error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}必须是 JSON 对象")
    return value


def validate_reasoning(reasoning: object) -> None:
    if not isinstance(reasoning, list) or not 2 <= len(reasoning) <= 8:
        raise RuntimeError("reasoning 必须包含 2–8 条记录")
    if any(
        not isinstance(item, str) or "证据：" not in item or "推断：" not in item
        for item in reasoning
    ):
        raise RuntimeError("reasoning 每项必须使用“证据：…；推断：…”格式")


HARD_BLOCKERS = (
    "target不存在", "越界修改", "API不支持", "dtype不支持",
    "容量不足", "接口语义冲突", "数学语义冲突", "change冲突",
)


def target_file(project: Path, target: str) -> str:
    if "::" not in target:
        raise RuntimeError(f"target 必须使用 相对文件::符号 格式：{target}")
    relative, symbol = target.split("::", 1)
    if not relative.startswith(("op_host/", "op_kernel/")) or not symbol.strip():
        raise RuntimeError(f"target 超出允许范围或缺少符号：{target}")
    candidate = (project / relative).resolve()
    if not candidate.is_relative_to(project.resolve()) or not candidate.is_file():
        raise RuntimeError(f"target 文件不存在：{relative}")
    return relative


def validate(bottleneck_path: Path, strategy_path: Path) -> None:
    validate_bottleneck(bottleneck_path)
    bottleneck = load(bottleneck_path, "瓶颈")
    report = load(strategy_path, "策略")
    if set(report) != {"reasoning", "strategy"}:
        raise RuntimeError("strategy.json 只能包含 reasoning 和 strategy")
    validate_reasoning(report["reasoning"])
    expected = derive(bottleneck)
    expected_strategy = expected["strategy"]
    actual = report["strategy"]
    if actual is None:
        if expected_strategy is None:
            return
        reasoning_text = "\n".join(report["reasoning"]).replace(" ", "")
        if "strategy=null" not in reasoning_text:
            raise RuntimeError("停止策略必须在 reasoning 中明确推导 strategy=null")
        if not any(f"硬阻断：{blocker}" in reasoning_text for blocker in HARD_BLOCKERS):
            raise RuntimeError("strategy=null 必须说明一个受支持的硬阻断类型")
        return
    if expected_strategy is None:
        raise RuntimeError("固定规则没有可选策略")
    if not isinstance(actual, dict) or set(actual) != {"strategy_key", "description", "changes"}:
        raise RuntimeError("strategy 字段不符合固定契约")
    for field in ("strategy_key", "description"):
        if actual[field] != expected_strategy[field]:
            raise RuntimeError(f"{field} 与固定策略选择不一致")
    changes = actual["changes"]
    expected_changes = expected_strategy["changes"]
    if not isinstance(changes, list) or len(changes) != len(expected_changes):
        raise RuntimeError("changes 数量与固定策略不一致")
    project = bottleneck_path.resolve().parent.parent
    expected_keys = [item["change_key"] for item in expected_changes]
    actual_keys = []
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"change_key", "target", "action"}:
            raise RuntimeError("每项 change 必须包含 change_key、target、action")
        key = change["change_key"]
        actual_keys.append(key)
        target = change["target"]
        action = change["action"]
        if not isinstance(target, str) or not target.strip():
            raise RuntimeError(f"{key} 缺少具体 target")
        target_file(project, target)
        if not isinstance(action, str) or len(action.strip()) < 12:
            raise RuntimeError(f"{key} 缺少足够具体的 action")
    if actual_keys != expected_keys:
        raise RuntimeError("change_key 或顺序与固定策略不一致")
    reasoning_text = "\n".join(report["reasoning"])
    if actual["strategy_key"] not in reasoning_text:
        raise RuntimeError("reasoning 未引用 strategy_key")
    missing = [key for key in actual_keys if key not in reasoning_text]
    if missing:
        raise RuntimeError("reasoning 未引用 change_key：" + ", ".join(missing))


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
