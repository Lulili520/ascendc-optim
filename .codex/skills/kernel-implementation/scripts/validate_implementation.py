#!/usr/bin/env python3
"""Validate faithful implementation of one compact strategy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STRATEGY_SCRIPTS = Path(__file__).resolve().parents[2] / "kernel-strategy/scripts"
sys.path.insert(0, str(STRATEGY_SCRIPTS))
from validate_strategy import validate as validate_strategy  # noqa: E402
from validate_source_effect import validate as validate_source_effect  # noqa: E402

VAGUE = ("可能", "尝试", "建议", "考虑", "或许", "maybe", "try ", "consider")
KNOWN_INVALID_SDK_FORMS = {
    r"\bAscendC::GetTPipePtr\s*\(": "GetTPipePtr 在当前 SDK 中是全局函数",
    r"\bAscendC::event_t\b": "event_t 在当前 SDK 中不属于 AscendC 命名空间",
    r"\bAscendC::EVENT_ID\d+\b": "EVENT_ID 常量在当前 SDK 中不属于 AscendC 命名空间",
}


def text(value: object, field: str, minimum: int = 5, maximum: int = 500) -> str:
    if not isinstance(value, str) or value != value.strip() or not minimum <= len(value) <= maximum:
        raise RuntimeError(f"{field} 必须是长度 {minimum}–{maximum} 的规范文本")
    lowered = value.lower()
    if "\n" in value or "```" in value or value.lstrip().startswith("#") or any(x in lowered for x in VAGUE):
        raise RuntimeError(f"{field} 必须单行、无 Markdown、无模糊措辞")
    return value


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON 顶层必须为对象：{path}")
    return value


def valid_files(files: object, project: Path) -> bool:
    return (isinstance(files, list) and bool(files)
            and all(isinstance(item, str) and item.startswith(("op_host/", "op_kernel/"))
                    and (project / item).is_file() for item in files))


def validate_known_sdk_forms(project: Path, files: list[str]) -> None:
    """Reject only context-free forms disproved by the pinned CANN 9.0.0 SDK."""
    for relative in files:
        if not relative.startswith("op_kernel/"):
            continue
        source = (project / relative).read_text(encoding="utf-8", errors="replace")
        source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.S)
        for pattern, reason in KNOWN_INVALID_SDK_FORMS.items():
            if re.search(pattern, source):
                raise RuntimeError(f"{relative} 包含已知非法 SDK 形式：{reason}")


def validate(strategy_path: Path, implementation_path: Path, parent: Path, project: Path, require_attempts: bool = False) -> None:
    validate_strategy(strategy_path, parent)
    strategy = load(strategy_path)["strategy"]
    if not isinstance(strategy, dict):
        raise RuntimeError("空 strategy 不得创建 child")
    record = load(implementation_path)
    if set(record) != {"strategy_kinds", "summary", "modified_files", "attempts"}:
        raise RuntimeError("implementation 字段非法")
    if record["strategy_kinds"] != strategy["kinds"]:
        raise RuntimeError("strategy_kinds 不一致")
    if not isinstance(record["summary"], list) or not record["summary"]:
        raise RuntimeError("summary 必须包含规范文本")
    if not valid_files(record["modified_files"], project):
        raise RuntimeError("modified_files 非法")
    target_files = {target.split("::", 1)[0] for target in strategy["targets"]}
    strategy_targets = set(strategy["targets"])
    summary_targets = set()
    for summary in record["summary"]:
        text(summary, "summary", 15)
        parts = [part.strip() for part in summary.split(" | ")]
        if len(parts) != 2 or parts[0] not in strategy_targets or not parts[1]:
            raise RuntimeError("summary 必须使用 strategy target | 已实施事实")
        summary_targets.add(parts[0])
    if summary_targets != strategy_targets:
        raise RuntimeError("summary 必须覆盖全部 strategy targets")
    if not target_files <= set(record["modified_files"]):
        raise RuntimeError("modified_files 未覆盖 targets")
    validate_known_sdk_forms(project, record["modified_files"])
    attempts = record["attempts"]
    if require_attempts and not attempts:
        raise RuntimeError("缺少 attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= 4:
        raise RuntimeError("attempts 必须包含初次实施及最多3次修复")
    fields = {"attempt", "kind", "trigger", "summary", "modified_files"}
    for number, attempt in enumerate(attempts, 1):
        if not isinstance(attempt, dict) or set(attempt) != fields:
            raise RuntimeError("attempt 字段非法")
        if attempt["attempt"] != number or attempt["kind"] != ("initial" if number == 1 else "repair"):
            raise RuntimeError("attempt 编号或kind非法")
        text(attempt["summary"], "attempt.summary", 15)
        summary_parts = [part.strip() for part in attempt["summary"].split(" | ")]
        if len(summary_parts) != 2 or summary_parts[0] not in strategy_targets or not summary_parts[1] or not valid_files(attempt["modified_files"], project):
            raise RuntimeError("attempt 摘要或文件非法")
        if number == 1 and attempt["trigger"] is not None:
            raise RuntimeError("initial attempt 的 trigger 必须为 null")
        if number > 1:
            trigger = attempt["trigger"]
            if not isinstance(trigger, dict) or set(trigger) != {"stage", "symptom", "evidence"}:
                raise RuntimeError("repair trigger 必须为 stage/symptom/evidence")
            for field in ("stage", "symptom", "evidence"):
                text(trigger[field], f"trigger.{field}", 2)
    validate_source_effect(strategy_path, parent, project)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--implementation", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--require-attempts", action="store_true")
    args = parser.parse_args()
    try:
        validate(args.strategy.resolve(), args.implementation.resolve(), args.parent.resolve(), args.project_dir.resolve(), args.require_attempts)
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise SystemExit(f"INVALID_IMPLEMENTATION: {error}") from error
    print(f"valid={args.implementation.resolve()}")


if __name__ == "__main__":
    main()
