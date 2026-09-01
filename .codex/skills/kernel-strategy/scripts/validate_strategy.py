#!/usr/bin/env python3
"""Validate one compact source-derived strategy."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

KINDS = ("reuse_onchip", "batch_transfer", "vectorize", "cube", "resize_tile", "parallelize", "pipeline")
KIND_SET = set(KINDS)
CUBE_MARKERS = ("Matmul<", "Matmul ", "Mmad(", ".Mmad(", "IterateAll(", ".IterateAll(")
VAGUE = ("可能", "尝试", "建议", "考虑", "或许", "maybe", "try ", "consider")
REASONING_LABELS = ("[任务]", "[现状]", "[问题]", "[策略]", "[推导]", "[边界]")


def has_scalar_prefix_chain(source: str) -> bool:
    """Recognize a per-element state recurrence that also writes every prefix."""
    state_update = re.search(
        r"\b(?P<state>carry|running\w*|prefix\w*|state\w*)\s*"
        r"(?:\*=|\+=|-=|=\s*(?P=state)\s*[+*])",
        source,
        re.I,
    )
    if not state_update or not re.search(r"\bfor\s*\(", source):
        return False
    state = re.escape(state_update.group("state"))
    return bool(re.search(rf"\w+\s*\([^)]*[ijc]\w*[^)]*\)\s*=\s*{state}\b", source, re.I))


def clean_text(item: object, field: str, minimum: int, maximum: int) -> str:
    if not isinstance(item, str) or item != item.strip() or not minimum <= len(item) <= maximum:
        raise RuntimeError(f"{field} 文本长度必须为 {minimum}–{maximum} 且无首尾空白")
    lowered = item.lower()
    if "\n" in item or "```" in item or item.lstrip().startswith("#"):
        raise RuntimeError(f"{field} 必须是无 Markdown 的单行文本")
    if any(token in lowered for token in VAGUE):
        raise RuntimeError(f"{field} 不得使用不确定措辞")
    return item


def project_source(project: Path) -> str:
    paths = sorted(
        path for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*")
        if path.is_file() and path.suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
    )
    return "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in paths)


def symbol_body(source: str, symbol: str) -> str:
    name = re.escape(symbol.rsplit("::", 1)[-1])
    macro = re.search(rf"\bBEGIN_TILING_DATA_DEF\s*\(\s*{name}\s*\)", source)
    if macro:
        end = re.search(r"\bEND_TILING_DATA_DEF\s*;", source[macro.end():])
        if not end:
            raise RuntimeError(f"target tiling data 宏不闭合：{symbol}")
        return source[macro.start():macro.end() + end.end()]
    constant = re.search(
        rf"^[ \t]*(?:(?:static|inline)\s+)*(?:constexpr|const)\b[^;\n]*\b{name}\b[^;]*;",
        source,
        re.M,
    )
    if constant:
        return constant.group(0)
    match = re.search(rf"\b(?:struct|class)\s+{name}\b[^;{{}}]*\{{", source, re.S)
    if not match:
        match = re.search(rf"\b{name}\s*\([^;{{}}]*\)[^;{{}}]*\{{", source, re.S)
    if not match:
        raise RuntimeError(f"无法定位 target symbol 正文：{symbol}")
    start = source.find("{", match.start())
    depth = 0
    for index in range(start, len(source)):
        depth += source[index] == "{"
        depth -= source[index] == "}"
        if depth == 0:
            return source[match.start():index + 1]
    raise RuntimeError(f"target symbol 花括号不闭合：{symbol}")


def load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取策略：{error}") from error
    if not isinstance(value, dict) or set(value) != {"strategy"}:
        raise RuntimeError("strategy.json 只能包含 strategy")
    return value


def validate(strategy_path: Path, project: Path) -> None:
    value = load(strategy_path)["strategy"]
    if value is None:
        source = project_source(project)
        if has_scalar_prefix_chain(source):
            raise RuntimeError("仍存在逐元素 Scalar Prefix/Scan 状态链，不得输出 strategy=null")
        pooling = any(token in project.name.lower() for token in ("pool", "window"))
        vector_markers = ("AscendC::Add(", "AscendC::Muls(", "AscendC::ReduceSum(",
                          "AscendC::WholeReduceSum(", "AscendC::Adds(")
        if pooling and ".GetValue(" in source and not any(marker in source for marker in vector_markers):
            raise RuntimeError("Window/Pooling 仍存在纯 Scalar GetValue 数学主体，不得输出 strategy=null")
        return
    fields = {"kinds", "evidence", "reasoning", "targets", "changes", "guards"}
    if not isinstance(value, dict) or set(value) != fields:
        raise RuntimeError("strategy 字段必须为 kinds/evidence/reasoning/targets/changes/guards")
    kinds = value["kinds"]
    if (not isinstance(kinds, list) or not kinds or len(kinds) != len(set(kinds))
            or any(kind not in KIND_SET for kind in kinds)):
        raise RuntimeError("strategy kinds 必须是非空、去重的合法分类数组")
    if kinds != sorted(kinds, key=KINDS.index):
        raise RuntimeError("strategy kinds 顺序非法")
    if "cube" in kinds and not any(marker in project_source(project) for marker in CUBE_MARKERS):
        try:
            planning = json.loads((strategy_path.parent / "planning.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"首次引入 Cube 缺少可读 planning：{error}") from error
        proof = "\n".join(
            str(item) for field in ("shape_model", "parameters", "buffers", "compute", "proofs")
            for item in planning.get(field, [])
        )
        required = ("首次引入Cube", "MNK", "L0A", "L0B", "L0C", "A重载次数", "B重载次数", "输出所有权", "SDK")
        missing = [token for token in required if token not in proof]
        if planning.get("engine") not in {"aic", "mixed"} or missing:
            detail = ",".join(missing) if missing else "engine必须为aic/mixed"
            raise RuntimeError(f"首次引入 Cube 的闭合证明不完整：{detail}")
    if "pipeline" in kinds:
        try:
            planning = json.loads((strategy_path.parent / "planning.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"pipeline 缺少可读 planning：{error}") from error
        proof = "\n".join(planning.get("proofs", []))
        if "原路径重叠阻断" not in proof or "目标新增重叠" not in proof:
            raise RuntimeError("pipeline 必须在 planning 证明原路径重叠阻断和目标新增重叠")
    limits = {"evidence": (1, None), "reasoning": (6, 6), "targets": (1, None), "changes": (1, None), "guards": (1, None)}
    for field, (minimum, maximum) in limits.items():
        items = value[field]
        if not isinstance(items, list) or len(items) < minimum or (maximum is not None and len(items) > maximum):
            suffix = f"–{maximum}" if maximum is not None else " 条以上"
            raise RuntimeError(f"{field} 数量必须为 {minimum}{suffix}")
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise RuntimeError(f"{field} 必须是非空文本数组")
    text_limits = {"evidence": (15, 500), "reasoning": (15, 600), "targets": (5, 240),
                   "changes": (15, 500), "guards": (10, 400)}
    for field, (minimum, maximum) in text_limits.items():
        for item in value[field]:
            clean_text(item, field, minimum, maximum)
    for item, label in zip(value["reasoning"], REASONING_LABELS, strict=True):
        if not item.startswith(label + " ") or len(item.removeprefix(label).strip()) < 10:
            raise RuntimeError("reasoning 必须按 [任务]/[现状]/[问题]/[策略]/[推导]/[边界] 各一条且正文完整")
        if any(other in item for other in REASONING_LABELS if other != label):
            raise RuntimeError("每条 reasoning 只能包含自己的固定标签")
    if len("".join(value["reasoning"])) > 6000:
        raise RuntimeError("reasoning 过长")
    for target in value["targets"]:
        if "::" not in target:
            raise RuntimeError(f"target 必须使用 相对文件::symbol：{target}")
        relative, symbol = target.split("::", 1)
        path = (project / relative).resolve()
        if (not relative.startswith(("op_host/", "op_kernel/"))
                or not path.is_relative_to(project.resolve()) or not path.is_file()):
            raise RuntimeError(f"target 文件不存在或越界：{target}")
        source = path.read_text(encoding="utf-8", errors="replace")
        if not symbol.strip() or symbol.rsplit("::", 1)[-1] not in source:
            raise RuntimeError(f"target symbol 不存在：{target}")
        symbol_body(source, symbol)
    targets = set(value["targets"])
    for item in value["evidence"]:
        parts = [part.strip() for part in item.split(" | ")]
        if len(parts) != 3 or any(not part for part in parts):
            raise RuntimeError("evidence 必须使用 文件::symbol | 源码事实 | 静态成本公式")
        if "::" not in parts[0]:
            raise RuntimeError(f"evidence 位置必须使用 相对文件::symbol：{parts[0]}")
        relative, symbol = parts[0].split("::", 1)
        path = (project / relative).resolve()
        if (not relative.startswith(("op_host/", "op_kernel/"))
                or not path.is_relative_to(project.resolve()) or not path.is_file()
                or not symbol.strip() or symbol.rsplit("::", 1)[-1] not in path.read_text(encoding="utf-8", errors="replace")):
            raise RuntimeError(f"evidence 源码位置不存在或越界：{parts[0]}")
    for item in value["changes"]:
        parts = [part.strip() for part in item.split(" | ")]
        if len(parts) != 2 or parts[0] not in targets or " -> " not in parts[1]:
            raise RuntimeError("changes 必须使用 target | 当前结构 -> 目标结构，且 target 属于 targets")
        before, after = [part.strip() for part in parts[1].split(" -> ", 1)]
        if not before or not after:
            raise RuntimeError("changes 的当前结构和目标结构不得为空")
        relative, symbol = parts[0].split("::", 1)
        source = symbol_body((project / relative).read_text(encoding="utf-8", errors="replace"), symbol)
        numeric_facts = re.findall(r"(?<![A-Za-z_])\d+(?![A-Za-z_])", before)
        if numeric_facts and not any(re.search(rf"(?<![A-Za-z_]){re.escape(number)}(?![A-Za-z_])", source)
                                     for number in numeric_facts):
            raise RuntimeError(f"changes 当前结构的数值不在 target symbol 正文中，疑似把间接影响列为直接 target：{parts[0]}")
    mutation_words = ("新增", "删除", "替换", "改为", "调整", "修改")
    for item in value["guards"]:
        parts = [part.strip() for part in item.split(" | ")]
        if len(parts) != 2 or any(not part for part in parts):
            raise RuntimeError("guards 必须使用 约束对象 | 修改后不变量")
        if any(word in parts[1] for word in mutation_words):
            raise RuntimeError("guards 只能描述不变量，不得描述修改动作")
    forbidden = ("performance.", "pipeline ratio", "task duration", "latency",
                 "msprof", "性能报告", "性能指标", "候选", "试参")
    if any(token in "\n".join(value["reasoning"]).lower() for token in forbidden):
        raise RuntimeError("reasoning 不得依赖性能报告、候选或试参")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--project-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        validate(args.strategy.resolve(), args.project_dir.resolve())
    except RuntimeError as error:
        raise SystemExit(f"INVALID_STRATEGY: {error}") from error
    print(f"valid={args.strategy.resolve()}")


if __name__ == "__main__":
    main()
