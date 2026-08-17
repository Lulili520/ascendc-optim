#!/usr/bin/env python3
"""Validate ordered strategies and grounded actions for every source issue."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from derive_strategy import OPERATIONS, RULES

BOTTLENECK_SCRIPTS = Path(__file__).resolve().parents[2] / "kernel-bottleneck/scripts"
sys.path.insert(0, str(BOTTLENECK_SCRIPTS))
from validate_report import CAUSES, validate as validate_bottleneck  # noqa: E402

OPERATION_SLOTS = json.loads((Path(__file__).resolve().parents[1] / "references/operation-slots.json").read_text())
if set(OPERATION_SLOTS) != set(OPERATIONS.values()):
    raise RuntimeError("operation slots 未覆盖全部 operation")

TRANSFORMS = ("改为", "替换", "外提", "合并", "删除", "新增", "增加", "移除", "重排", "缓存", "修改", "复用")
UB_FORMULA = re.compile(
    r"fixed_bytes=(\d+),bytes_per_element=(\d+),usable_ub_ratio=(0(?:\.\d+)?|1(?:\.0+)?)"
)
VECTOR_TILING_FIELDS = ("pattern=", "branch=", "peak_bytes=", "chunk_rule=", "alignment=")
VECTOR_ABI_FIELDS = ("valid_len=", "input_dtype=", "compute_dtype=", "output_dtype=")
TRANSFER_FIELDS = ("block_count=", "block_len=", "src_stride=", "dst_stride=", "alignment=", "valid_len=")
VECTOR_BUDGET = re.compile(
    r"fixed_bytes=(\d+),bytes_per_element=(\d+),usable_ub_ratio=(0(?:\.\d+)?|1(?:\.0+)?)"
)
SHARED_FIELDS = ("pattern", "branch", "alignment", "input_dtype", "compute_dtype", "output_dtype")


def validate_edit(index: int, edit: str) -> None:
    """Public action-contract check retained for tests and callers."""
    if edit in {"提高并行度", "增加并行度", "合并小搬运", "向量化归约"}:
        raise RuntimeError(f"action {index} edit 只重复 operation")
    if not isinstance(edit, str) or not 8 <= len(edit.strip()) <= 320:
        raise RuntimeError(f"action {index} edit 长度非法")
    if not re.search(r"[：:]", edit):
        raise RuntimeError(f"action {index} edit 必须使用 operation_slot/源码对象：明确变换")
    if not any(term in edit for term in TRANSFORMS):
        raise RuntimeError(f"action {index} edit 缺少明确变换动作")


def edit_slot(edit: str) -> str:
    head = re.split(r"[：:]", edit, maxsplit=1)[0].strip()
    if "/" not in head or not head.split("/", 1)[1].strip():
        raise RuntimeError("edit 必须使用 <operation_slot>/<源码对象>：<当前实现>改为<目标结构>")
    return head.split("/", 1)[0]


def _load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"无法读取{label}：{error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label}必须是对象")
    return value


def _validate_resize_work_unit(project: Path, edits: list[str]) -> None:
    by_slot = {edit_slot(edit): edit for edit in edits}
    budget = by_slot["active_buffer_budget"]
    match = UB_FORMULA.search(budget)
    if match is None or "peak_bytes=fixed_bytes+bytes_per_element*tile" not in budget:
        raise RuntimeError(
            "resize_work_unit 必须给出 fixed_bytes,bytes_per_element,usable_ub_ratio 和 peak_bytes 公式"
        )
    fixed_bytes, bytes_per_element = int(match.group(1)), int(match.group(2))
    ratio = float(match.group(3))
    if bytes_per_element <= 0 or not 0 < ratio <= 1:
        raise RuntimeError("resize_work_unit UB 公式参数非法")
    performance = _load(project / "performance/performance.json", "性能")
    ub_bytes = (performance.get("hardware") or {}).get("ub_bytes_per_core")
    if not isinstance(ub_bytes, int) or ub_bytes <= 0:
        raise RuntimeError("resize_work_unit 缺少可靠 hardware.ub_bytes_per_core")
    usable = int(ub_bytes * ratio)
    if fixed_bytes >= usable or (usable - fixed_bytes) // bytes_per_element <= 0:
        raise RuntimeError("resize_work_unit UB 预算没有合法 tile 空间")
    candidate = by_slot["candidate_work_unit"]
    if "candidate_rule=" not in candidate or "max_tile=floor((usable_ub_bytes-fixed_bytes)/bytes_per_element)" not in candidate:
        raise RuntimeError("resize_work_unit 必须输出可重算 candidate_rule 与 max_tile 公式")
    if re.search(r"(?:取|选择|设为|改为)\s*\d+", candidate):
        raise RuntimeError("实施前不得把未经子版本验证的候选 tile 写成确定值")
    if "total_tasks=" not in by_slot["task_parallelism"]:
        raise RuntimeError("resize_work_unit 必须给出 total_tasks 公式")
    tail = by_slot["alignment_and_tail"]
    if "alignment=" not in tail or "valid_len=" not in tail:
        raise RuntimeError("resize_work_unit 必须给出 alignment 与 valid_len 规则")


def _field(text: str, name: str) -> str | None:
    match = re.search(rf"(?:^|[,;，；\s]){re.escape(name)}=([^,;，；\s]+)", text)
    return match.group(1) if match else None


def _validate_vectorize_scalar_work(project: Path, actions: list[dict]) -> None:
    by_slot = {edit_slot(edit): edit for action in actions for edit in action["edits"]}
    tiling = by_slot["pattern_and_tiling"]
    missing = [field for field in VECTOR_TILING_FIELDS if field not in tiling]
    if missing:
        raise RuntimeError("vectorize_scalar_work 缺少场景/tiling字段：" + ",".join(missing))
    if not _field(tiling, "pattern") or not _field(tiling, "branch"):
        raise RuntimeError("vectorize_scalar_work 必须从当前源码明确 pattern/branch")
    budget = VECTOR_BUDGET.search(tiling)
    if budget is None or "peak_bytes=fixed_bytes+bytes_per_element*chunk" not in tiling:
        raise RuntimeError("vectorize_scalar_work 必须给出可重算 fixed_bytes/bytes_per_element/usable_ub_ratio/peak_bytes")
    fixed, per_element, ratio = int(budget.group(1)), int(budget.group(2)), float(budget.group(3))
    performance = _load(project / "performance/performance.json", "性能")
    ub_bytes = (performance.get("hardware") or {}).get("ub_bytes_per_core")
    if not isinstance(ub_bytes, int) or ub_bytes <= 0 or per_element <= 0 or not 0 < ratio <= 1:
        raise RuntimeError("vectorize_scalar_work 缺少可靠 UB 或预算参数非法")
    usable = int(ub_bytes * ratio)
    if fixed >= usable or (usable - fixed) // per_element <= 0:
        raise RuntimeError("vectorize_scalar_work UB 预算无合法 chunk")
    if "chunk_rule=max_chunk=floor((usable_ub_bytes-fixed_bytes)/bytes_per_element)" not in tiling:
        raise RuntimeError("vectorize_scalar_work chunk_rule 不可重算")
    abi = by_slot["dtype_tail_abi"]
    missing = [field for field in VECTOR_ABI_FIELDS if field not in abi]
    if missing:
        raise RuntimeError("vectorize_scalar_work 缺少 dtype/tail/ABI字段：" + ",".join(missing))
    if "int64" in abi.lower() and not any(
        token in abi for token in ("Cast", "DataCopy", "批量转换", "LocalTensor标量扩宽")
    ):
        raise RuntimeError("INT64 输出必须给出可确认的转换和写回路径")
    tiling_change = _field(tiling, "tiling_change")
    if tiling_change not in {"true", "false"}:
        raise RuntimeError("vectorize_scalar_work 必须声明 tiling_change=true/false")
    if tiling_change == "true" and not any("op_host/" in action["target"] for action in actions):
        raise RuntimeError("vectorize_scalar_work 改变 tiling 时必须包含 Host target")


def _validate_coalesce_global_transfer(actions: list[dict]) -> None:
    text = "\n".join(edit for action in actions for edit in action["edits"])
    missing = [field for field in TRANSFER_FIELDS if field not in text]
    if missing:
        raise RuntimeError("coalesce_global_transfer 缺少搬运字段：" + ",".join(missing))


def _validate_cross_strategy(strategies: list[dict]) -> None:
    writes: dict[tuple[str, str], tuple[int, str]] = {}
    shared: dict[str, tuple[int, str]] = {}
    for index, strategy in enumerate(strategies, 1):
        for action in strategy["actions"]:
            for edit in action["edits"]:
                head = re.split(r"[：:]", edit, maxsplit=1)[0].strip()
                source_object = head.split("/", 1)[1]
                identity = (action["target"], source_object)
                previous = writes.get(identity)
                if previous and previous[1] != edit:
                    raise RuntimeError(
                        f"strategy {previous[0]}/{index} 对同一源码对象给出冲突变换：{identity[0]}::{source_object}"
                    )
                writes[identity] = (index, edit)
                for field in SHARED_FIELDS:
                    value = _field(edit, field)
                    if value is None:
                        continue
                    previous_value = shared.get(field)
                    if previous_value and previous_value[1] != value:
                        raise RuntimeError(
                            f"strategy {previous_value[0]}/{index} 的共享参数 {field} 不一致"
                        )
                    shared[field] = (index, value)


def _target(project: Path, target: str) -> str:
    if "::" not in target:
        raise RuntimeError(f"target 必须使用 相对文件::符号：{target}")
    relative, symbol = target.split("::", 1)
    path = (project / relative).resolve()
    if not relative.startswith(("op_host/", "op_kernel/")) or not path.is_relative_to(project.resolve()) or not path.is_file():
        raise RuntimeError(f"target 文件不存在或越界：{target}")
    if not symbol.strip() or symbol.rsplit("::", 1)[-1] not in path.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError(f"target 符号不存在：{target}")
    return relative


def _validate_action(project: Path, action: object, operation: str, label: str) -> None:
    fields = {"target", "operation", "edits", "constraints"}
    if not isinstance(action, dict) or set(action) != fields:
        raise RuntimeError(f"{label} action 字段非法")
    if not isinstance(action["target"], str):
        raise RuntimeError(f"{label} target 非法")
    _target(project, action["target"])
    if action["operation"] != operation:
        raise RuntimeError(f"{label} operation 与 strategy 不一致")
    edits, constraints = action["edits"], action["constraints"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 4:
        raise RuntimeError(f"{label} edits 必须包含 1–4 项")
    if not isinstance(constraints, list) or not 1 <= len(constraints) <= 4:
        raise RuntimeError(f"{label} constraints 必须包含 1–4 项")
    for edit in edits:
        validate_edit(1, edit)
    for constraint in constraints:
        if not isinstance(constraint, str) or not 4 <= len(constraint.strip()) <= 240:
            raise RuntimeError(f"{label} constraint 非法")
    if any("performance." in text for text in (*edits, *constraints)):
        raise RuntimeError(f"{label} action 不得引用 performance")


def validate(bottleneck_path: Path, strategy_path: Path) -> None:
    validate_bottleneck(bottleneck_path)
    bottleneck = _load(bottleneck_path, "瓶颈")
    report = _load(strategy_path, "策略")
    if set(report) != {"reasoning", "strategies"}:
        raise RuntimeError("strategy.json 只能包含 reasoning、strategies")
    reasoning, strategies = report["reasoning"], report["strategies"]
    if not isinstance(reasoning, list) or any(
        not isinstance(x, str) or not x.strip() for x in reasoning
    ):
        raise RuntimeError("strategy reasoning 必须是非空文本")
    if not isinstance(strategies, list):
        raise RuntimeError("strategies 必须是数组")
    causes = [item["bottleneck"]["cause_key"] for item in bottleneck["issues"]]
    if not causes:
        if strategies or reasoning:
            raise RuntimeError("issues 为空时 strategy reasoning/strategies 也必须为空")
        return
    if len(strategies) != len(causes):
        raise RuntimeError("strategies 必须按顺序完整覆盖全部 issues")
    if len(reasoning) != len(strategies):
        raise RuntimeError("strategy reasoning 必须与 strategies 逐项对应")
    project = bottleneck_path.resolve().parent.parent
    seen: set[str] = set()
    for index, (cause, strategy) in enumerate(zip(causes, strategies), 1):
        if not isinstance(strategy, dict) or set(strategy) != {"cause_key", "strategy_key", "actions"}:
            raise RuntimeError(f"strategy {index} 字段非法")
        if strategy["cause_key"] != cause or cause in seen:
            raise RuntimeError(f"strategy {index} cause 顺序错误或重复")
        seen.add(cause)
        expected = RULES[cause]
        if strategy["strategy_key"] != expected:
            raise RuntimeError(f"strategy {index} key 与固定映射不一致")
        if cause not in reasoning[index - 1] or expected not in reasoning[index - 1]:
            raise RuntimeError(f"strategy {index} reasoning 未引用 cause/strategy")
        actions = strategy["actions"]
        if not isinstance(actions, list) or not 1 <= len(actions) <= 4:
            raise RuntimeError(f"strategy {index} actions 必须包含 1–4 项")
        identities = set()
        used_slots: list[str] = []
        source_objects: list[str] = []
        for action in actions:
            _validate_action(project, action, OPERATIONS[expected], f"strategy {index}")
            identity = (action["target"], action["operation"])
            if identity in identities:
                raise RuntimeError(f"strategy {index} 重复 target+operation")
            identities.add(identity)
            for edit in action["edits"]:
                used_slots.append(edit_slot(edit))
                head = re.split(r"[：:]", edit, maxsplit=1)[0].strip()
                source_objects.append(head.split("/", 1)[1])
        slot_contract = OPERATION_SLOTS[OPERATIONS[expected]]
        required_slots = slot_contract["required"]
        optional_slots = slot_contract["optional"]
        allowed_order = required_slots + optional_slots
        if any(slot not in allowed_order for slot in used_slots):
            raise RuntimeError(f"strategy {index} 使用未知 operation slot")
        if len(used_slots) != len(set(used_slots)):
            raise RuntimeError(f"strategy {index} operation slots 重复")
        missing_slots = [slot for slot in required_slots if slot not in used_slots]
        if missing_slots:
            raise RuntimeError(f"strategy {index} 缺少 required operation slots：{','.join(missing_slots)}")
        if OPERATIONS[expected] == "resize_work_unit":
            _validate_resize_work_unit(
                project,
                [edit for action in actions for edit in action["edits"]],
            )
        if OPERATIONS[expected] == "vectorize_scalar_work":
            _validate_vectorize_scalar_work(project, actions)
        if OPERATIONS[expected] == "coalesce_global_transfer":
            _validate_coalesce_global_transfer(actions)
        if not any(obj in reasoning[index - 1] for obj in source_objects):
            raise RuntimeError(f"strategy {index} reasoning 未引用 edits 中的源码对象")
    _validate_cross_strategy(strategies)


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
