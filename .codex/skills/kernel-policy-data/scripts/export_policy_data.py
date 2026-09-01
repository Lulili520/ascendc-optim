#!/usr/bin/env python3
"""Export compact source-to-strategy records filtered by real Task Duration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / ".codex"))
from suite_config import SUITES, load_suite_manifest, suite_paths
SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}
VERSION_RE = re.compile(r"^(?P<operator>.+)_(?P<version>[0-4])$")
TERMINAL_STATES = {
    "completed_max_rounds", "stopped_no_strategy", "stopped_no_improvement",
    "build_failed", "precision_failed", "performance_failed", "agent_failed",
    "implementation_blocked", "prepare_failed", "environment_failed", "stopped_by_user",
}
STRATEGY_FIELDS = {"kinds", "evidence", "reasoning", "targets", "changes", "guards"}
REASONING_LABELS = ["任务", "现状", "问题", "策略", "推导", "边界"]


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层不是对象：{path}")
    return value


def compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for folder in ("op_host", "op_kernel") for path in (project / folder).rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def valid_precision(project: Path) -> bool:
    value = load(project / "precision/precision.json")
    return value.get("status") == "PASS" and value.get("exit_code") == 0 and value.get("source_fingerprint") == fingerprint(project)


def latency(project: Path) -> float:
    path = project / "performance/latency.json"
    value = load(path)
    number = value.get("task_duration_us")
    if value.get("source_fingerprint") != fingerprint(project):
        raise ValueError(f"Task Duration源码指纹失效：{path}")
    if value.get("accepted") is not True:
        raise ValueError(f"Task Duration未通过收益门禁：{path}")
    if not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
        raise ValueError(f"无有效 Task Duration：{path}")
    return float(number)


def best_before(parent: Path) -> float:
    match = VERSION_RE.fullmatch(parent.name)
    if match is None:
        raise ValueError("父版本名非法")
    operator, version = match.group("operator"), int(match.group("version"))
    values = [latency(parent.parent / f"{operator}_{index}") for index in range(version + 1)]
    return min(values)


def strip_comments(text: str) -> str:
    text = re.sub(r"\A\s*/\*.*?\*/\s*", "", text, count=1, flags=re.S)
    text = re.sub(r"(?m)^\s*//[^\n]*\n", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def sources(project: Path, folder: str) -> str:
    paths = sorted(path for path in (project / folder).rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_SUFFIXES)
    if not paths:
        raise ValueError(f"{folder} 没有源码")
    return "\n\n".join(
        f"--- FILE: {path.relative_to(project).as_posix()} ---\n{strip_comments(path.read_text(encoding='utf-8', errors='replace'))}"
        for path in paths
    )


def validate_pair(parent: Path, child: Path) -> None:
    commands = [
        [sys.executable, str(ROOT / ".codex/skills/kernel-strategy/scripts/validate_planning.py"), "--planning", str(parent / "strategy/planning.json"), "--project-dir", str(parent)],
        [sys.executable, str(ROOT / ".codex/skills/kernel-strategy/scripts/validate_strategy.py"), "--strategy", str(parent / "strategy/strategy.json"), "--project-dir", str(parent)],
        [sys.executable, str(ROOT / ".codex/skills/kernel-implementation/scripts/validate_implementation.py"), "--strategy", str(parent / "strategy/strategy.json"), "--implementation", str(child / "strategy/implementation.json"), "--parent", str(parent), "--project-dir", str(child), "--require-attempts"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise ValueError(result.stdout.strip()[-800:])


def system_prompt() -> str:
    return "\n".join([
        "你是AscendC源码优化策略教师。仅根据OP和完整HOST/KERNEL源码，找出全部有直接证据且可确定修复的问题，生成直达目标、可编译、语义闭合且覆盖全部问题的统一策略。训练样本只包含已有验证收益的非空策略。",
        "先识别计算模式、独立输出、数学主体和跨tile状态，再由shape、容量、字段范围与任务数确定唯一布局。成本按GM字节、DMA burst、Vector repeat、Cube运算、Scalar迭代、task生命周期、状态初始化、同步和写回burst计算；非空方案须减少静态工作，或在每输出工作不增加时提高有效核覆盖。完整Vector化且一次读写的Elementwise只改tile或Queue时不生成策略。",
        "kinds只记直接变化并去重：GM总字节减少=reuse_onchip；仅GM事务减少=batch_transfer；改变Scalar/Vector数学表达=vectorize；调整已有Cube主路径=cube；只改单任务工作量=resize_tile；改变核间所有权=parallelize；只改阶段依赖或重叠=pipeline。配套变化不另记；cube仅限源码已有Matmul/Mmad/IterateAll主计算，禁止将Scalar/Vector contraction转换为Cube。",
        "Cube由M/N/K推导输出tile和blockIdx，禁止只增大SetDim/SetBlockDim；记录A/B重载、K chunk、转换和tail。pipeline区分双槽Vector/MTE与IterateAll/End边界。",
        "evidence使用‘文件::symbol | 源码事实 | shape/循环/字节/任务公式’；targets覆盖修改位置；changes使用‘target | 当前结构 -> 目标结构’；guards记录不变量。",
        "reasoning固定六条：[任务]语义、[现状]数据流、[问题]结构与公式、[策略]目标结构、[推导]参数与资源公式、[边界]不变量。同一kind使用相同术语和句式；多kind按顺序以分号一一对应。",
        "只输出单行紧凑JSON：{\"strategy\":{\"kinds\":[],\"evidence\":[],\"reasoning\":[],\"targets\":[],\"changes\":[],\"guards\":[]}}。changes按依赖顺序表达直达目标结构的修改；文本须单行、明确、无Markdown，不输出性能信息或额外字段。",
    ])


def discover_parents(workspace_root: Path, requested: set[str]) -> list[Path]:
    return sorted(path for path in workspace_root.glob("*")
                  if path.is_dir() and VERSION_RE.fullmatch(path.name)
                  and (not requested or path.name in requested))


def queue_states(queue_path: Path) -> dict[str, str] | None:
    """Return managed operator states, or None when no managed queue exists."""
    if not queue_path.is_file():
        return None
    queue = load(queue_path)
    items = queue.get("items")
    if not isinstance(items, list):
        raise ValueError(f"队列缺少items：{queue_path}")
    return {
        item["operator"]: item["state"] for item in items
        if isinstance(item, dict)
        and isinstance(item.get("operator"), str)
        and isinstance(item.get("state"), str)
    }


def input_text(operator: str, project: Path, entry: dict) -> str:
    precision = load(project / "precision/precision.json")
    return "\n\n".join([
        "[OP]\n" + compact({"operator": operator, "function": entry.get("function"),
                             "parameters": entry.get("parameters"),
                             "output_shape": precision.get("output_shape"),
                             "output_dtype": precision.get("output_dtype")}),
        "[HOST]\n" + sources(project, "op_host"),
        "[KERNEL]\n" + sources(project, "op_kernel"),
    ])


def evaluate_transition(parent: Path, manifest: dict, threshold: float) -> tuple[dict | None, dict]:
    match = VERSION_RE.fullmatch(parent.name)
    operator, version = match.group("operator"), int(match.group("version"))
    child = parent.with_name(f"{operator}_{version + 1}")
    audit = {"ops": parent.name, "child_ops": child.name, "eligible": False}
    if operator not in manifest or not child.is_dir():
        raise ValueError("manifest或child缺失")
    if not valid_precision(parent) or not valid_precision(child):
        raise ValueError("父子精度门禁无效")
    validate_pair(parent, child)
    before, after = best_before(parent), latency(child)
    reduction = (before - after) / before * 100.0
    audit.update({"best_parent_task_duration_us": before, "child_task_duration_us": after, "reduction_percent": reduction, "threshold_percent": threshold})
    if reduction <= threshold:
        audit["exclusion_reason"] = "Task Duration收益未严格超过阈值"
        return None, audit
    entry = manifest[operator]
    source_input = input_text(operator, parent, entry)
    strategy = load(parent / "strategy/strategy.json").get("strategy")
    if not isinstance(strategy, dict):
        raise ValueError("父版本没有可导出的非空strategy")
    audit.update({"eligible": True, "input_chars": len(source_input)})
    return {"input": source_input, "strategy": strategy}, audit


def terminal_null_project(operator: str, workspace_root: Path) -> str | None:
    versions = sorted(
        path for path in workspace_root.glob(f"{operator}_[0-4]")
        if path.is_dir() and VERSION_RE.fullmatch(path.name)
    )
    for project in reversed(versions):
        try:
            strategy = load(project / "strategy/strategy.json").get("strategy")
            if strategy is not None or not valid_precision(project):
                continue
            latency(project)
            return project.name
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            continue
    return None


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def load_terminal_policy(parent: Path, final_child_ops: str) -> dict | None:
    """Load an independently normalized initial-to-final policy when present.

    Raw per-round strategies are deliberately never composed here: their
    evidence and current-state reasoning belong to different source versions.
    """
    path = parent / "strategy/terminal_policy.json"
    if not path.is_file():
        return None
    value = load(path)
    final = parent.with_name(final_child_ops)
    if value.get("source_ops") != parent.name:
        raise ValueError(f"终态策略起始版本失效：{path}")
    if value.get("final_child_ops") != final_child_ops:
        raise ValueError(f"终态策略目标版本失效：{path}")
    if value.get("source_fingerprint") != fingerprint(parent):
        raise ValueError(f"终态策略起始源码指纹失效：{path}")
    if not final.is_dir() or value.get("final_source_fingerprint") != fingerprint(final):
        raise ValueError(f"终态策略最终源码指纹失效：{path}")
    strategy = value.get("strategy")
    if not isinstance(strategy, dict) or set(strategy) != STRATEGY_FIELDS:
        raise ValueError(f"终态策略字段不完整：{path}")
    if any(not isinstance(strategy[field], list) for field in STRATEGY_FIELDS):
        raise ValueError(f"终态策略字段必须为数组：{path}")
    labels = [re.match(r"^\[([^]]+)\]", line).group(1)
              if isinstance(line, str) and re.match(r"^\[([^]]+)\]", line) else None
              for line in strategy["reasoning"]]
    if labels != REASONING_LABELS:
        raise ValueError(f"终态策略必须重新生成规范六段reasoning：{path}")
    if len(compact({"strategy": strategy})) > 6000:
        raise ValueError(f"终态策略超过6000字符：{path}")
    return strategy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITES), default="KernelBench910B")
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--manifest", type=Path, help="仅用于兼容自定义 keyed manifest")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/kernel_policy_data.jsonl")
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--min-reduction", type=float, default=1.0)
    parser.add_argument("--ops", action="append", default=[])
    args = parser.parse_args()
    _, default_workspace, _ = suite_paths(ROOT, args.suite)
    args.workspace_root = args.workspace_root or default_workspace
    args.queue = args.queue or (args.workspace_root / "optimization_queue.json")
    manifest = load(args.manifest) if args.manifest else load_suite_manifest(ROOT, args.suite)
    prompt = system_prompt()
    requested = set(args.ops)
    managed_states = queue_states(args.queue)
    parents = discover_parents(args.workspace_root, set())
    transitions, audits = {}, []
    for parent in parents:
        try:
            transition, audit = evaluate_transition(parent, manifest, args.min_reduction)
            if transition:
                match = VERSION_RE.fullmatch(parent.name)
                transitions[(match.group("operator"), int(match.group("version")))] = transition
            audits.append(audit)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            audits.append({"ops": parent.name, "eligible": False, "exclusion_reason": str(error)})
    records, seen = [], set()
    for (operator, version), transition in sorted(transitions.items()):
        # Operators absent from the current targeted queue are durable history.
        # Operators managed by this queue become stable only at a terminal state.
        if (managed_states is not None and operator in managed_states
                and managed_states[operator] not in TERMINAL_STATES):
            continue
        if requested and f"{operator}_{version}" not in requested:
            continue
        cursor = version
        while (operator, cursor) in transitions:
            cursor += 1
        chain_length = cursor - version
        strategy = transition["strategy"]
        composition_mode = "single_step"
        if chain_length > 1:
            normalized = load_terminal_policy(
                args.workspace_root / f"{operator}_{version}", f"{operator}_{cursor}"
            )
            if normalized is not None:
                strategy = normalized
                composition_mode = "terminal_normalized"
            else:
                # A verified immediate transition is safer training data than
                # a mechanically concatenated answer with cross-version facts.
                cursor = version + 1
                composition_mode = "single_step_fallback"
        output_text = compact({"strategy": strategy})
        ops = f"{operator}_{version}" if args.suite == "KernelBench910B" else f"{args.suite}/{operator}_{version}"
        record = {"system_prompt": prompt, "input": transition["input"],
                  "output": output_text, "ops": ops}
        identity = hashlib.sha256((record["input"] + "\0" + output_text).encode()).hexdigest()
        if identity not in seen:
            seen.add(identity)
            records.append(record)
        for audit in audits:
            if audit.get("ops") == f"{operator}_{version}" and audit.get("eligible"):
                audit.update(source_strategies=(chain_length if composition_mode == "terminal_normalized" else 1),
                             available_chain_steps=chain_length,
                             composition_mode=composition_mode,
                             final_child_ops=f"{operator}_{cursor}", output_chars=len(output_text))
                break
    operators = sorted({match.group("operator") for path in parents
                        if (match := VERSION_RE.fullmatch(path.name))})
    excluded_strategy_null = []
    for operator in operators:
        if managed_states is not None and operator in managed_states:
            if managed_states[operator] != "stopped_no_strategy":
                continue
        project_name = terminal_null_project(operator, args.workspace_root)
        if project_name is not None:
            excluded_strategy_null.append(project_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output, "".join(compact(record) + "\n" for record in records))
    prompt_path = args.output.parent / "kernel_policy_system_prompt.txt"
    atomic_write(prompt_path, prompt)
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.json")
    atomic_write(audit_path, json.dumps({"records": len(records),
                                         "excluded_strategy_null": excluded_strategy_null,
                                         "system_prompt_chars": len(prompt),
                                         "system_prompt_sha256": sha(prompt_path),
                                         "items": audits}, ensure_ascii=False, indent=2) + "\n")
    print(f"records={len(records)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
