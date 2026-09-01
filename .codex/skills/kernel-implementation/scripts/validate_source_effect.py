#!/usr/bin/env python3
"""Require every planned target symbol to change in the child."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def symbol_body(text: str, symbol: str) -> str:
    name = re.escape(symbol.rsplit("::", 1)[-1])
    macro = re.search(rf"\bBEGIN_TILING_DATA_DEF\s*\(\s*{name}\s*\)", text)
    if macro:
        end = re.search(r"\bEND_TILING_DATA_DEF\s*;", text[macro.end():])
        if not end:
            raise RuntimeError(f"tiling data 宏不闭合：{symbol}")
        return text[macro.start():macro.end() + end.end()]
    constant = re.search(
        rf"^[ \t]*(?:(?:static|inline)\s+)*(?:constexpr|const)\b[^;\n]*\b{name}\b[^;]*;",
        text,
        re.M,
    )
    if constant:
        return constant.group(0).strip()
    match = re.search(rf"\b(?:struct|class)\s+{name}\b[^;{{}}]*\{{", text, re.S)
    if not match:
        match = re.search(rf"\b{name}\s*\([^;{{}}]*\)[^;{{}}]*\{{", text, re.S)
    if not match:
        raise RuntimeError(f"无法定位 symbol：{symbol}")
    start = text.find("{", match.start())
    depth = 0
    for index in range(start, len(text)):
        depth += text[index] == "{"
        depth -= text[index] == "}"
        if depth == 0:
            return text[match.start():index + 1]
    raise RuntimeError(f"symbol 花括号不闭合：{symbol}")


def validate(strategy_path: Path, parent: Path, child: Path) -> None:
    strategy = json.loads(strategy_path.read_text(encoding="utf-8")).get("strategy")
    if not isinstance(strategy, dict):
        raise RuntimeError("父版本缺少非空 strategy")
    changed_bodies = []
    for target in strategy["targets"]:
        relative, symbol = target.split("::", 1)
        before, after = parent / relative, child / relative
        if not before.is_file() or not after.is_file():
            raise RuntimeError(f"target 文件缺失：{target}")
        before_body = symbol_body(before.read_text(errors="replace"), symbol)
        try:
            after_body = symbol_body(after.read_text(errors="replace"), symbol)
        except RuntimeError as error:
            if not str(error).startswith("无法定位 symbol："):
                raise
            continue
        if before_body == after_body:
            raise RuntimeError(f"target symbol 未变化：{target}")
        changed_bodies.append(after_body)

    # Targets prove that every frozen edit happened.  API-family checks prove
    # the final child structure, so they must inspect the complete Host/Kernel;
    # an unchanged CopyIn or Cube body may legitimately support a Host-only edit.
    source_parts = []
    for folder in ("op_host", "op_kernel"):
        root = child / folder
        if root.is_dir():
            source_parts.extend(
                path.read_text(encoding="utf-8", errors="replace")
                for path in sorted(root.rglob("*")) if path.is_file()
            )
    source = "\n".join(source_parts)
    changed_source = "\n".join(changed_bodies)
    kinds = set(strategy.get("kinds", []))
    reasoning = strategy.get("reasoning", [])
    strategy_text = reasoning[3] if isinstance(reasoning, list) and len(reasoning) == 6 else ""
    if "batch_transfer" in kinds:
        if not re.search(r"\bDataCopy(?:Pad)?\s*\(", source):
            raise RuntimeError("batch_transfer 未在最终源码中形成 DataCopy 路径")
        if "二维DMA" in strategy_text:
            variable_block_count = re.search(r"\.blockCount\s*=", source)
            multi_block_initializer = re.search(
                r"DataCopyExtParams\s+\w+\s*\{\s*(?!1\s*[,}])[^,}]+[,}]", source)
            if not variable_block_count and not multi_block_initializer:
                raise RuntimeError("二维DMA仍只有固定 blockCount=1，未减少规则多行搬运事务")
    if "vectorize" in kinds and not re.search(
            r"\b(?:Abs|Add|Cast|Compare|Div|Duplicate|Exp|Max|Min|Mul|Reduce|Select|Sub)\s*\(", source):
        # `vectorize` owns the non-Cube mathematical body.  A valid action can
        # either replace Scalar work with Vector APIs or remove provably
        # redundant Scalar mathematics altogether (for example, compute only
        # one triangle of a symmetric product).  In the latter case the private
        # planning proof must show a strict full-launch Scalar-work reduction.
        try:
            planning = json.loads(
                (strategy_path.parent / "planning.json").read_text(encoding="utf-8"))
            current = planning["work"]["current"]["scalar_iterations"]
            target = planning["work"]["target"]["scalar_iterations"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            current = target = 0
        scalar_body = "\n".join(reasoning).lower()
        removes_scalar_math = (
            isinstance(current, int) and isinstance(target, int) and 0 <= target < current
            and ("scalar" in scalar_body or "标量" in scalar_body)
            and bool(changed_source.strip())
        )
        if not removes_scalar_math:
            raise RuntimeError(
                "vectorize 既未形成 Vector 指令链，也无 planning 证明的 Scalar 数学主体减少")
    if "pipeline" in kinds:
        has_two_slots = re.search(r"InitBuffer\s*\([^,]+,\s*2\s*,", source)
        has_stages = all(token in source for token in ("AllocTensor", "DeQue", "FreeTensor"))
        vector_pipeline = bool(has_two_slots and has_stages)
        cube_pipeline = bool(
            re.search(r"\b(?:Matmul|Mmad|IterateAll)\b", source)
            and re.search(r"\b(?:IterateAll|End)\s*\(", changed_source)
        )
        if not vector_pipeline and not cube_pipeline:
            raise RuntimeError("pipeline 既无双槽生产消费闭环，也无变更target中的Cube提交边界")
    if "cube" in kinds and not re.search(r"\b(?:Matmul|Mmad|IterateAll)\b", source):
        raise RuntimeError("cube 未保留已有 Cube 主计算路径")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True, type=Path)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--child", required=True, type=Path)
    args = parser.parse_args()
    try:
        validate(args.strategy.resolve(), args.parent.resolve(), args.child.resolve())
    except (OSError, json.JSONDecodeError, RuntimeError) as error:
        raise SystemExit(f"INVALID_SOURCE_EFFECT: {error}") from error
    print(f"valid_source_effect={args.child.resolve()}")


if __name__ == "__main__":
    main()
