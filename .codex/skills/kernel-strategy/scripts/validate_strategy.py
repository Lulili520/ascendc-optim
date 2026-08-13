#!/usr/bin/env python3
"""Validate cause-to-strategy selection and source-specific actions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from derive_strategy import OPERATIONS, RULES

OPERATION_SLOTS = json.loads(
    (Path(__file__).resolve().parents[1] / "references/operation-slots.json").read_text(encoding="utf-8")
)
if set(OPERATION_SLOTS) != set(OPERATIONS.values()):
    raise RuntimeError("operation-slots.json 必须覆盖全部 operation")

BOTTLENECK_SCRIPTS = Path(__file__).resolve().parents[2] / "kernel-bottleneck/scripts"
sys.path.insert(0, str(BOTTLENECK_SCRIPTS))
from validate_report import CAUSES, validate as validate_bottleneck  # noqa: E402


EDIT_CONSTRAINT_PREFIXES = (
    "保持", "确保", "禁止", "不得", "不改变", "证明",
    "UB 总", "总 UB", "单缓冲总量", "总量低于", "容量低于",
)
CONSTRAINT_EDIT_PREFIXES = (
    "新增", "删除", "改为", "替换", "将", "把", "增加", "移除", "改写",
)
TRANSFORM_TERMS = (
    "改为", "替换", "外提", "合并", "删除", "新增", "增加", "移除",
    "重排", "拆分", "缓存", "扩大", "缩小", "从", "修改", "重用",
)
SOURCE_OBJECT_TERMS = (
    "blockdim", "block", "tiling", "task", "row", "tile", "loop", "循环", "区间",
    "索引", "偏移", "映射", "余数", "queue", "tque", "tbuf", "buffer", "缓冲",
    "getvalue", "setvalue", "datacopy", "reduce", "vector", "scalar", "gm", "ub", "l1",
    "weight", "bias", "input", "output", "workspace", "stride", "padding", "mask", "repeat",
    "dtype", "shape", "count", "length", "start", "end", "base", "process", "init",
)
GENERIC_OPERATION_EDITS = {
    "提高并行度", "增加并行度", "使用更多核", "使用可用核",
    "重新平衡任务", "平衡任务划分", "均衡多核负载",
    "减少固定开销", "删除冗余开销", "优化热路径",
    "向量化归约", "将标量归约向量化", "向量化逐元素计算",
    "合并小搬运", "优化数据搬运", "调整ub布局",
}


def validate_edit(index: int, edit: str) -> None:
    normalized = re.sub(r"[\s。，；,;:_\-]", "", edit).lower()
    if normalized in GENERIC_OPERATION_EDITS:
        raise RuntimeError(f"action {index} edit 只重复 operation，缺少当前源码对象：{edit}")
    if "：" not in edit and ":" not in edit:
        raise RuntimeError(f"action {index} edit 必须使用 <源码对象>：<明确变换>：{edit}")
    object_text, transform_text = re.split(r"[：:]", edit, maxsplit=1)
    if not object_text.strip() or not transform_text.strip():
        raise RuntimeError(f"action {index} edit 的源码对象或变换为空：{edit}")
    if not any(term in transform_text for term in TRANSFORM_TERMS):
        raise RuntimeError(f"action {index} edit 缺少明确变换动作：{edit}")
    lowered = object_text.lower()
    has_identifier = re.search(r"[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?", object_text) is not None
    has_source_object = has_identifier or any(term in lowered for term in SOURCE_OBJECT_TERMS)
    if not has_source_object:
        raise RuntimeError(f"action {index} edit 缺少 target symbol 内可定位的源码对象：{edit}")


def load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取{label}：{error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}必须是 JSON 对象")
    return value


def target_file(project: Path, target: str) -> str:
    if "::" not in target:
        raise RuntimeError(f"target 必须使用 相对文件::符号：{target}")
    relative, symbol = target.split("::", 1)
    candidate = (project / relative).resolve()
    if not relative.startswith(("op_host/", "op_kernel/")) or not symbol.strip():
        raise RuntimeError(f"target 超出范围或缺少符号：{target}")
    if not candidate.is_relative_to(project.resolve()) or not candidate.is_file():
        raise RuntimeError(f"target 文件不存在：{relative}")
    symbol_leaf = symbol.rsplit("::", 1)[-1]
    if symbol_leaf not in candidate.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"target 符号不存在：{target}")
    return relative


def validate(bottleneck_path: Path, strategy_path: Path) -> None:
    validate_bottleneck(bottleneck_path)
    bottleneck_report = load(bottleneck_path, "瓶颈")
    report = load(strategy_path, "策略")
    if set(report) != {"reasoning", "strategy"}:
        raise RuntimeError("strategy.json 只能包含 reasoning 和 strategy")
    reasoning = report["reasoning"]
    if not isinstance(reasoning, list) or len(reasoning) != 2:
        raise RuntimeError("reasoning 必须固定包含 2 条记录")
    if any(not isinstance(item, str) or "证据：" not in item or "推断：" not in item for item in reasoning):
        raise RuntimeError("reasoning 必须使用证据到推断格式")
    bottleneck = bottleneck_report.get("bottleneck")
    cause = bottleneck.get("cause_key") if isinstance(bottleneck, dict) else None
    expected_key = RULES.get(cause)
    actual = report["strategy"]
    if actual is None:
        if expected_key is not None:
            raise RuntimeError("固定规则存在策略，不能输出 strategy=null")
        return
    if expected_key is None:
        raise RuntimeError("固定规则没有可选策略")
    if not isinstance(actual, dict) or set(actual) != {"strategy_key", "actions"}:
        raise RuntimeError("strategy 只能包含 strategy_key 和 actions")
    if actual["strategy_key"] != expected_key:
        raise RuntimeError("strategy_key 与 cause 固定映射不一致")
    actions = actual["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 4:
        raise RuntimeError("actions 必须包含 1–4 项")
    project = bottleneck_path.resolve().parent.parent
    identities = set()
    for index, action in enumerate(actions, 1):
        fields = {"target", "operation", "edits", "constraints"}
        if not isinstance(action, dict) or set(action) != fields:
            raise RuntimeError(f"action {index} 字段必须为 target、operation、edits、constraints")
        if not isinstance(action["target"], str):
            raise RuntimeError(f"action {index} target 非法")
        target_file(project, action["target"])
        if action["operation"] != OPERATIONS[expected_key]:
            raise RuntimeError(f"action {index} operation 与 strategy_key 不一致")
        identity = (action["target"], action["operation"])
        if identity in identities:
            raise RuntimeError(f"action {index} 与前项 target+operation 重复，应合并")
        identities.add(identity)
        edits = action["edits"]
        constraints = action["constraints"]
        if not isinstance(edits, list) or not 1 <= len(edits) <= 4:
            raise RuntimeError(f"action {index} edits 必须包含 1–4 项")
        if not isinstance(constraints, list) or not 1 <= len(constraints) <= 4:
            raise RuntimeError(f"action {index} constraints 必须包含 1–4 项")
        if any(not isinstance(text, str) or not 8 <= len(text.strip()) <= 320 for text in edits):
            raise RuntimeError(f"action {index} edit 必须是 8–320 字符的具体修改")
        if any(not isinstance(text, str) or not 4 <= len(text.strip()) <= 240 for text in constraints):
            raise RuntimeError(f"action {index} constraint 必须是 4–240 字符的边界")
        for edit in edits:
            if edit.strip().startswith(EDIT_CONSTRAINT_PREFIXES):
                raise RuntimeError(f"action {index} edit 只写了不变式，应移入 constraints：{edit}")
            validate_edit(index, edit)
        for constraint in constraints:
            if constraint.strip().startswith(CONSTRAINT_EDIT_PREFIXES):
                raise RuntimeError(f"action {index} constraint 包含源码变换，应移入 edits：{constraint}")
        if any("performance." in item for item in (*edits, *constraints)):
            raise RuntimeError(f"action {index} 不得引用 performance.* 或把性能证据写入 action")
    text = "\n".join(reasoning)
    for token in (f"cause_key={cause}", f"strategy_key={expected_key}"):
        if token not in text:
            raise RuntimeError(f"reasoning 未引用 {token}")
    operation = OPERATIONS[expected_key]
    if f"operation={operation}" not in text:
        raise RuntimeError(f"reasoning 未将当前源码与 operation={operation} 连接")
    bottleneck_evidence = bottleneck_report.get("evidence", [])
    cause_evidence_key = CAUSES[cause][1]
    cause_evidence = next(
        (item for item in bottleneck_evidence if item.get("evidence_key") == cause_evidence_key),
        None,
    )
    if cause_evidence is None:
        raise RuntimeError(f"缺少 cause evidence：{cause_evidence_key}")
    grounded = reasoning[1]
    for token in (cause_evidence_key, cause_evidence["source"]):
        if token not in grounded:
            raise RuntimeError("第 2 条 reasoning 必须引用当前 cause evidence 的 key 和 source")
    if cause_evidence["observation"] in grounded:
        raise RuntimeError("第 2 条 reasoning 不得复制 cause evidence observation")
    for action in actions:
        if action["target"] not in grounded:
            raise RuntimeError(f"第 2 条 reasoning 未引用 action target：{action['target']}")


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
