#!/usr/bin/env python3
"""Export validated, performance-positive KernelBench policy JSONL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


INSTRUCTION = "根据正式性能、源码和固定 policy，输出唯一瓶颈、原因及可实施策略。"

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
VERSION_RE = re.compile(r"^(?P<operator>.+)_(?P<version>[01])$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def skill_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return value


def pretty_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def compact_performance(report: dict[str, Any]) -> dict[str, Any]:
    """Keep fields needed for diagnosis and source-grounded action planning."""
    task = report.get("task", {})
    pipeline = report.get("pipeline", {})
    memory = report.get("memory", {})
    l2 = report.get("l2_cache", {})
    conflict = report.get("resource_conflict", {})
    hardware = report.get("hardware", {})
    return {
        "task": {key: task.get(key) for key in (
            "Task Type", "Block Num", "Input Shapes", "Input Data Types",
            "Output Shapes", "Output Data Types", "head_overhead_ratio",
        )},
        "pipeline": {key: pipeline.get(key) for key in (
            "aic_mac_ratio", "aic_scalar_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
            "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio",
            "aic_icache_miss_rate", "aiv_icache_miss_rate",
        )},
        "memory": {"gm_peak_utilization_percent": memory.get("gm_peak_utilization_percent")},
        "l2_cache": {"aiv_derived_hit_rate_percent": l2.get("aiv_derived_hit_rate_percent")},
        "resource_conflict": {key: conflict.get(key) for key in (
            "aiv_vec_bankgroup_cflt_ratio", "aiv_vec_bank_cflt_ratio", "aiv_vec_resc_cflt_ratio",
        )},
        "per_core": {key: (report.get("per_core") or {}).get(key) for key in (
            "active_cores", "imbalance_percent",
        )},
        "hardware": {key: hardware.get(key) for key in (
            "aic_core_count", "aiv_core_count", "ub_bytes_per_core", "l1_bytes_per_core",
            "l0a_bytes_per_core", "l0b_bytes_per_core", "l0c_bytes_per_core", "l2_bytes",
        )},
    }


def training_policy_knowledge(knowledge: dict[str, Any]) -> dict[str, Any]:
    """Project the validated canonical contract to the non-redundant training surface."""
    root = repo_root()
    bottleneck = load_module(
        "kernel_bottleneck_training_contract",
        root / ".codex/skills/kernel-bottleneck/scripts/validate_report.py",
    )
    strategy = load_module(
        "kernel_strategy_training_contract",
        root / ".codex/skills/kernel-strategy/scripts/derive_strategy.py",
    )
    slots = load_json(root / ".codex/skills/kernel-strategy/references/operation-slots.json")
    metric_sources = {key: "|".join(sources) for key, sources in bottleneck.METRIC_SOURCES.items()}
    cause_rules = {
        cause: "|".join(filter(None, (
            parent, evidence, strategy.RULES[cause], strategy.OPERATIONS.get(strategy.RULES[cause]),
        )))
        for cause, (parent, evidence) in bottleneck.CAUSES.items()
    }
    return {
        "decision_order": knowledge["decision_order"],
        "thresholds": {
            "load_imbalance_percent_gt": 30,
            "pipeline_ratio_gt": 0.8,
            "mte2_cube_unique_max_gt": 0.7,
            "pipeline_tie_margin_points": bottleneck.PIPELINE_TIE_MARGIN_PERCENTAGE_POINTS,
        },
        "metric_evidence_sources": metric_sources,
        "cause_rule_fields": "bottleneck|cause_evidence|strategy|operation",
        "cause_rules": cause_rules,
        "operation_slots": {key: ">".join(value) for key, value in slots.items()},
    }


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载模块：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_policy_knowledge(knowledge: dict[str, Any]) -> None:
    root = repo_root()
    bottleneck = load_module(
        "kernel_bottleneck_contract",
        root / ".codex/skills/kernel-bottleneck/scripts/validate_report.py",
    )
    strategy = load_module(
        "kernel_strategy_contract",
        root / ".codex/skills/kernel-strategy/scripts/derive_strategy.py",
    )
    got_bottlenecks = knowledge.get("bottleneck_keys", {})
    if set(got_bottlenecks) != set(bottleneck.BOTTLENECK_EVIDENCE):
        raise ValueError("policy knowledge 的 bottleneck_keys 与官方 validator 不一致")
    if knowledge.get("cause_strategies") != strategy.RULES:
        raise ValueError("policy knowledge 的 cause→strategy 映射与官方推导器不一致")
    if knowledge.get("strategy_operations") != strategy.OPERATIONS:
        raise ValueError("policy knowledge 的 strategy→operation 映射与官方推导器不一致")
    if knowledge.get("strategy_key_meanings") != strategy.STRATEGY_DESCRIPTIONS:
        raise ValueError("policy knowledge 的 strategy key 定义与官方推导器不一致")
    if knowledge.get("operation_meanings") != strategy.OPERATION_DESCRIPTIONS:
        raise ValueError("policy knowledge 的 operation 定义与官方推导器不一致")
    if knowledge.get("pipeline_tie_order") != list(bottleneck.PIPELINE_TIE_ORDER):
        raise ValueError("policy knowledge 的流水线平局顺序与官方 validator 不一致")
    if knowledge.get("pipeline_tie_margin_percentage_points") != bottleneck.PIPELINE_TIE_MARGIN_PERCENTAGE_POINTS:
        raise ValueError("policy knowledge 的流水线平局 margin 与官方 validator 不一致")
    expected_cause_priority = {key: list(value) for key, value in bottleneck.CAUSE_PRIORITY.items()}
    if knowledge.get("cause_priority") != expected_cause_priority:
        raise ValueError("policy knowledge 的 cause 优先级与官方 validator 不一致")
    meanings = knowledge.get("evidence_key_meanings", {})
    if set(meanings) != bottleneck.EVIDENCE_KEYS or any(
        not isinstance(value, str) or not value.strip() for value in meanings.values()
    ):
        raise ValueError("policy knowledge 的 evidence key 定义不完整")
    cause_meanings = knowledge.get("cause_key_meanings", {})
    if set(cause_meanings) != set(bottleneck.CAUSE_DESCRIPTIONS) or any(
        not isinstance(value, str) or not value.strip() for value in cause_meanings.values()
    ):
        raise ValueError("policy knowledge 的 cause key 定义不完整")
    selection = knowledge.get("selection_rules", {})
    if set(selection) != {"fixed_cost_bound", "core_underuse", "load_imbalance", "pipeline_candidate", "priority"}:
        raise ValueError("policy knowledge 缺少完整瓶颈选择规则")
    action_fields = knowledge.get("action_field_meanings", {})
    if set(action_fields) != {"target", "operation", "edits", "constraints"}:
        raise ValueError("policy knowledge 缺少 action 字段语义")
    slots = load_json(root / ".codex/skills/kernel-strategy/references/operation-slots.json")
    if set(slots) != set(strategy.OPERATIONS.values()) or any(
        not isinstance(value, list) or not 2 <= len(value) <= 4
        or any(not isinstance(item, str) or not item for item in value)
        for value in slots.values()
    ):
        raise ValueError("operation slots 未完整覆盖固定 operation")


def validate_parent_reports(parent: Path) -> None:
    root = repo_root()
    bottleneck_scripts = root / ".codex/skills/kernel-bottleneck/scripts"
    strategy_scripts = root / ".codex/skills/kernel-strategy/scripts"
    for directory in (str(bottleneck_scripts), str(strategy_scripts)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    bottleneck_validator = load_module(
        "kernel_policy_bottleneck_validator",
        bottleneck_scripts / "validate_report.py",
    )
    strategy_validator = load_module(
        "kernel_policy_strategy_validator",
        strategy_scripts / "validate_strategy.py",
    )
    bottleneck_path = parent / "bottleneck/bottleneck.json"
    strategy_path = parent / "strategy/strategy.json"
    bottleneck_validator.validate(bottleneck_path)
    strategy_validator.validate(bottleneck_path, strategy_path)


def validate_completed_version(project: Path, label: str) -> dict[str, Any]:
    workspace = load_json(project / "workspace.json")
    precision = load_json(project / "precision/precision.json")
    fingerprint = source_fingerprint(project)
    if workspace.get("status") != "PERFORMANCE_DONE":
        raise ValueError(f"{label} workspace 不是 PERFORMANCE_DONE")
    if workspace.get("source_fingerprint") != fingerprint:
        raise ValueError(f"{label} workspace 源码指纹不匹配")
    if precision.get("status") != "PASS" or precision.get("exit_code") != 0:
        raise ValueError(f"{label} precision 未 PASS")
    if precision.get("source_fingerprint") != fingerprint:
        raise ValueError(f"{label} precision 源码指纹不匹配")
    return workspace


def source_file_bytes(project: Path) -> dict[str, bytes]:
    return {
        path.relative_to(project).as_posix(): path.read_bytes()
        for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    }


def validate_implementation_link(parent: Path, parent_strategy: dict[str, Any], child: Path) -> None:
    implementation = load_json(child / "strategy/implementation.json")
    strategy = parent_strategy.get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError("父版本 strategy 为空")
    if implementation.get("strategy_key") != strategy.get("strategy_key"):
        raise ValueError("子版本 implementation.strategy_key 与父版本策略不一致")
    modified = implementation.get("modified_files")
    if not isinstance(modified, list) or not modified:
        raise ValueError("子版本 implementation 缺少 modified_files")
    target_files = {action["target"].split("::", 1)[0] for action in strategy.get("actions", [])}
    parent_files = source_file_bytes(parent)
    child_files = source_file_bytes(child)
    actual_modified = {
        path for path in set(parent_files) | set(child_files)
        if parent_files.get(path) != child_files.get(path)
    }
    if set(modified) != actual_modified:
        raise ValueError("implementation.modified_files 与父子源码真实 diff 不一致")
    if target_files != actual_modified:
        raise ValueError("真实源码 diff 必须恰好覆盖全部 action target，且不得混入未声明文件")
    actions = implementation.get("actions")
    expected_indices = list(range(1, len(strategy.get("actions", [])) + 1))
    if not isinstance(actions, list) or [item.get("action_index") for item in actions if isinstance(item, dict)] != expected_indices:
        raise ValueError("implementation.actions 未按 action_index 完整关联父策略")


def source_fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*")
        if path.is_file()
    )
    if not files:
        raise ValueError("op_host/op_kernel 中没有文件")
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def render_sources(project: Path, folder: str) -> str:
    paths = sorted(
        path for path in (project / folder).rglob("*")
        if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"{folder} 中没有可导出的源码")
    chunks = []
    for path in paths:
        relative = path.relative_to(project).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace").rstrip()
        chunks.append(f"--- FILE: {relative} ---\n{text}")
    return "\n\n".join(chunks)


def matching_brace(text: str, opening: int) -> int:
    depth = 0
    for position in range(opening, len(text)):
        if text[position] == "{":
            depth += 1
        elif text[position] == "}":
            depth -= 1
            if depth == 0:
                return position
    return len(text)


def mask_comments_and_strings(text: str) -> str:
    pattern = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', re.S)
    return pattern.sub(lambda match: "".join("\n" if char == "\n" else " " for char in match.group()), text)


def file_symbols(text: str) -> list[str]:
    masked = mask_comments_and_strings(text)
    scopes = []
    for match in re.finditer(r"\b(namespace|class|struct)\s+([A-Za-z_]\w*)[^;{]*\{", masked):
        opening = masked.find("{", match.start())
        scopes.append((opening, matching_brace(masked, opening), match.group(2), match.group(1)))
    symbols = []
    function_pattern = re.compile(
        r"(?:^|\n)\s*(?:template\s*<[^;{}]+>\s*)?"
        r"(?:[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*[\s*&<>:,]+)+"
        r"((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{",
        re.S,
    )
    for match in function_pattern.finditer(masked):
        name = match.group(1)
        if name in {"if", "for", "while", "switch", "catch"}:
            continue
        parents = [scope for scope in scopes if scope[0] < match.start(1) < scope[1]]
        prefix = "::".join(scope[2] for scope in sorted(parents, key=lambda item: item[0]))
        if "::" not in name and prefix:
            name = f"{prefix}::{name}"
        symbols.append(name)
    return list(dict.fromkeys(symbols))


def source_symbols(project: Path) -> str:
    """List every visible source symbol without selecting or ranking an answer target."""
    rows = []
    for folder in ("op_host", "op_kernel"):
        for path in sorted(
            item for item in (project / folder).rglob("*")
            if item.is_file() and item.suffix.lower() in SOURCE_SUFFIXES
        ):
            relative = path.relative_to(project).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
            for name in file_symbols(text):
                rows.append(f"{relative}::{name}")
    if not rows:
        raise ValueError("op_host/op_kernel 中未提取到源码 symbol")
    return "\n".join(rows)


def latency(performance: dict[str, Any]) -> float:
    report_value = performance.get("kernel_latency_us")
    if not isinstance(report_value, (int, float)) or not math.isfinite(report_value) or report_value <= 0:
        raise ValueError("performance kernel_latency_us 不是有限正数")
    return float(report_value)


def operator_json(operator: str, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "operator": operator,
        "level": entry.get("level"),
        "function": entry.get("function"),
        "parameters": entry.get("parameters"),
    }


def evaluate_candidate(
    parent: Path,
    manifest: dict[str, Any],
    knowledge: dict[str, Any],
    output_format: str,
    min_reduction: float,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    match = VERSION_RE.fullmatch(parent.name)
    if match is None:
        raise ValueError("目录名不是 <OperatorName>_[01]")
    operator = match.group("operator")
    version = int(match.group("version"))
    child = parent.with_name(f"{operator}_{version + 1}")
    audit: dict[str, Any] = {"ops": parent.name, "child_ops": child.name, "eligible": False}
    if operator not in manifest:
        raise ValueError(f"manifest 中不存在算子 {operator}")
    if not child.is_dir():
        raise ValueError("对应子版本不存在")

    validate_completed_version(parent, "父版本")
    validate_completed_version(child, "子版本")

    parent_perf = load_json(parent / "performance/performance.json")
    child_perf = load_json(child / "performance/performance.json")
    parent_latency = latency(parent_perf)
    child_latency = latency(child_perf)
    reduction = (parent_latency - child_latency) / parent_latency * 100.0
    audit.update({
        "parent_latency_us": parent_latency,
        "child_latency_us": child_latency,
        "reduction_percent": reduction,
        "threshold_percent": min_reduction,
    })

    bottleneck = load_json(parent / "bottleneck/bottleneck.json")
    strategy = load_json(parent / "strategy/strategy.json")
    if bottleneck.get("bottleneck") is None or strategy.get("strategy") is None:
        raise ValueError("父版本 bottleneck 或 strategy 为空")
    validate_parent_reports(parent)
    validate_implementation_link(parent, strategy, child)
    if not reduction > min_reduction:
        audit["exclusion_reason"] = f"latency 下降 {reduction:.12g}% 未严格大于 {min_reduction}%"
        return None, audit

    input_text = "\n\n".join([
        "[TASK_INSTRUCT]\n" + INSTRUCTION,
        "[POLICY_KNOWLEDGE_JSON]\n" + compact_json(training_policy_knowledge(knowledge)),
        "[OPERATOR_JSON]\n" + compact_json(operator_json(operator, manifest[operator])),
        "[PERFORMANCE_JSON]\n" + compact_json(compact_performance(parent_perf)),
        "[SOURCE_SYMBOLS]\n" + source_symbols(parent),
        "[OP_HOST_SOURCE]\n" + render_sources(parent, "op_host"),
        "[OP_KERNEL_SOURCE]\n" + render_sources(parent, "op_kernel"),
        "[OUTPUT_FORMAT]\n" + output_format.strip(),
    ])
    output_text = "\n\n".join([
        "[BOTTLENECK_JSON]\n" + compact_json(bottleneck),
        "[STRATEGY_JSON]\n" + compact_json(strategy),
    ])
    sample = {"ops": parent.name, "input": input_text, "output": output_text}
    audit.update({
        "eligible": True,
        "policy_knowledge_sha256": hashlib.sha256(pretty_json(knowledge).encode()).hexdigest(),
        "input_chars": len(input_text),
        "output_chars": len(output_text),
    })
    return sample, audit


def discover(root: Path, requested_ops: set[str], levels: set[str]) -> list[Path]:
    projects = [
        path for path in root.glob("level*/*")
        if path.is_dir() and VERSION_RE.fullmatch(path.name)
        and (not levels or path.parent.name in levels)
    ]
    if requested_ops:
        projects = [path for path in projects if path.name in requested_ops]
        missing = requested_ops - {path.name for path in projects}
        if missing:
            raise SystemExit("未找到 ops：" + ", ".join(sorted(missing)))
    return sorted(projects, key=lambda path: (path.parent.name, path.name))


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=root / "kernel_workspace/KernelBench910B")
    parser.add_argument("--manifest", type=Path, default=root / "kernel/KernelBench910B/manifest.json")
    parser.add_argument("--output", type=Path, default=root / "datasets/kernel_policy_data.jsonl")
    parser.add_argument("--ops", action="append", default=[])
    parser.add_argument(
        "--level", action="append", choices=("level1", "level2", "level3"), default=[],
        help="仅导出指定 level；可重复传入",
    )
    parser.add_argument("--min-reduction-percent", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.min_reduction_percent) or args.min_reduction_percent < 0:
        raise SystemExit("--min-reduction-percent 必须是有限非负数")
    workspace_root = args.workspace_root.resolve()
    manifest = load_json(args.manifest.resolve())
    knowledge = load_json(skill_root() / "references/policy-knowledge.json")
    validate_policy_knowledge(knowledge)
    output_format = (skill_root() / "references/output-format.txt").read_text(encoding="utf-8")
    projects = discover(workspace_root, set(args.ops), set(args.level))
    samples: list[dict[str, str]] = []
    audit_items: list[dict[str, Any]] = []

    for project in projects:
        try:
            sample, audit = evaluate_candidate(
                project, manifest, knowledge, output_format,
                min_reduction=args.min_reduction_percent,
            )
            audit_items.append(audit)
            if sample is None:
                continue
            samples.append(sample)
        except Exception as error:
            audit_items.append({
                "ops": project.name,
                "eligible": False,
                "exclusion_reason": str(error),
            })

    if len({sample["ops"] for sample in samples}) != len(samples):
        raise ValueError("导出结果包含重复 ops")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit_path.write_text(pretty_json({
        "workspace_root": str(workspace_root),
        "output": str(output),
        "minimum_reduction_percent_exclusive": args.min_reduction_percent,
        "candidates": len(projects),
        "exported": len(samples),
        "excluded": len(projects) - len(samples),
        "items": audit_items,
    }), encoding="utf-8")
    print(f"exported={len(samples)} excluded={len(projects) - len(samples)} output={output}")
    print(f"audit={audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
