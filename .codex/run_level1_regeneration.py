#!/usr/bin/env python3
"""Run the Level1 optimization queue with one fresh Codex context per operator."""

from __future__ import annotations

import json
import hashlib
import fcntl
import math
import os
import shutil
import subprocess
import sys
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
QUEUE = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_queue.json"
RUNS = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_runs"
LOG = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_progress.tsv"
LEVEL = ROOT / "kernel_workspace/KernelBench910B/level1"
ARCHIVE_ROOT = ROOT / "train_result/level1_regeneration_archives"
ASCENDC_TMPDIR = Path("/data/lu/ascendc-tmp")
LOCK = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration.lock"
MAX_AGENT_ATTEMPTS = 2
MAX_REPAIRS = 3
MAX_OPTIMIZATION_ROUNDS = 4
MIN_IMPROVEMENT_PERCENT = 1.0
TERMINAL_STATES = {
    "completed_max_rounds_with_issues", "stopped_no_issue", "build_failed",
    "stopped_no_improvement", "stopped_by_user",
    "precision_failed", "performance_failed", "strategy_blocked",
    "implementation_blocked", "prepare_failed", "agent_failed", "incomplete",
}


def save(queue: dict) -> None:
    temporary = QUEUE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, QUEUE)


def load_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def source_fingerprint(project: Path) -> str | None:
    files = sorted(
        path for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def precision_gate(project: Path, workspace: dict | None) -> tuple[bool, str]:
    if not workspace or workspace.get("status") not in {
        "PRECISION_PASS", "PERFORMANCE_DONE", "PERFORMANCE_SCREENED_NO_IMPROVEMENT",
    }:
        return False, "workspace status has not passed precision"
    fingerprint = source_fingerprint(project)
    if not fingerprint or workspace.get("source_fingerprint") != fingerprint:
        return False, "workspace source fingerprint mismatch"
    precision = load_json(project / "precision/precision.json")
    if not precision or precision.get("status") != "PASS" or precision.get("exit_code") != 0:
        return False, "precision gate is not PASS"
    if precision.get("source_fingerprint") != fingerprint:
        return False, "precision source fingerprint mismatch"
    return True, "precision durable gate passed"


def performance_gate(project: Path, workspace: dict | None) -> tuple[bool, str]:
    if not workspace or workspace.get("status") != "PERFORMANCE_DONE":
        return False, "workspace status is not PERFORMANCE_DONE"
    passed, reason = precision_gate(project, workspace)
    if not passed:
        return False, reason
    performance = load_json(project / "performance/performance.json")
    required_sections = {
        "task", "pipeline", "arithmetic", "memory", "memory_l0", "memory_ub",
        "l2_cache", "resource_conflict", "per_core", "hardware",
    }
    if not performance or not required_sections <= set(performance):
        return False, "performance report is incomplete"
    required_files = {
        "op_summary_PipeUtilization.csv", "op_summary_ArithmeticUtilization.csv",
        "op_summary_Memory.csv", "op_summary_MemoryL0.csv", "op_summary_MemoryUB.csv",
        "op_summary_L2Cache.csv", "op_summary_ResourceConflictRatio.csv", "per_core_cycles.csv",
    }
    missing = sorted(name for name in required_files if not (project / "performance" / name).is_file())
    if missing:
        return False, "performance artifacts missing: " + ",".join(missing)
    latency = performance.get("kernel_latency_us") if performance else None
    if not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency <= 0:
        return False, "performance latency is not a finite positive number"
    return True, "all durable gates passed"


def performance_screening(project: Path, workspace: dict | None) -> dict | None:
    if not workspace or workspace.get("status") != "PERFORMANCE_SCREENED_NO_IMPROVEMENT":
        return None
    screening = load_json(project / "performance/screening.json")
    fingerprint = source_fingerprint(project)
    if (
        not screening or screening.get("status") != "NO_IMPROVEMENT"
        or not fingerprint or screening.get("source_fingerprint") != fingerprint
    ):
        return None
    latency = screening.get("kernel_latency_us")
    improvement = screening.get("improvement_percent")
    if not isinstance(latency, (int, float)) or latency <= 0:
        return None
    if not isinstance(improvement, (int, float)) or improvement > MIN_IMPROVEMENT_PERCENT:
        return None
    return screening


def cleanup_reference_cache(operator: str) -> None:
    cache = ASCENDC_TMPDIR / "kernel_precision_reference_cache"
    if not cache.is_dir():
        return
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", operator)
    for path in cache.glob(f"{safe}.*.pt*"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def kernel_latency(project: Path) -> float | None:
    performance = load_json(project / "performance/performance.json") or {}
    value = performance.get("kernel_latency_us")
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None


def improvement_percent(parent: Path, child: Path) -> float | None:
    before, after = kernel_latency(parent), kernel_latency(child)
    if before is None or after is None:
        return None
    return (before - after) / before * 100.0


def best_prior_project(operator: str, version: int) -> tuple[int, Path] | None:
    candidates = [
        (candidate, path, kernel_latency(path))
        for candidate, path in operator_projects(operator) if candidate < version
    ]
    valid = [(candidate, path, latency) for candidate, path, latency in candidates if latency is not None]
    if not valid:
        return None
    candidate, path, _ = min(valid, key=lambda item: item[2])
    return candidate, path


def non_improving_result(operator: str) -> tuple[int, int, float] | None:
    """Find the earliest completed version that failed to beat prior best."""
    for version, project in operator_projects(operator):
        if version <= 0 or kernel_latency(project) is None:
            continue
        best = best_prior_project(operator, version)
        if best is None:
            continue
        best_version, parent = best
        gain = improvement_percent(parent, project)
        if gain is not None and gain <= MIN_IMPROVEMENT_PERCENT:
            return version, best_version, gain
    return None


def bottleneck_gate(project: Path) -> tuple[bool, str]:
    report = load_json(project / "bottleneck/bottleneck.json")
    if not report or set(report) != {"reasoning", "issues"}:
        return False, "bottleneck report is missing or has stale top-level fields"
    if not isinstance(report["reasoning"], list) or not isinstance(report["issues"], list):
        return False, "bottleneck reasoning/issues are not lists"
    expected_reasoning = len(report["issues"]) if report["issues"] else 1
    if len(report["reasoning"]) != expected_reasoning:
        return False, "bottleneck reasoning/issues length mismatch"
    validation = subprocess.run(
        [sys.executable, str(ROOT / ".codex/skills/kernel-bottleneck/scripts/validate_report.py"),
         str(project / "bottleneck/bottleneck.json")],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if validation.returncode != 0:
        return False, "bottleneck validator failed: " + validation.stdout.strip()[-500:]
    return True, "bottleneck durable gate passed"


def completed_round_gate(parent: Path, child: Path) -> tuple[bool, str]:
    parent_workspace = load_json(parent / "workspace.json")
    child_workspace = load_json(child / "workspace.json")
    if not performance_gate(parent, parent_workspace)[0]:
        return False, "parent performance gate failed"
    if not performance_gate(child, child_workspace)[0]:
        return False, "child performance gate failed"
    if not bottleneck_gate(child)[0]:
        return False, "child bottleneck gate failed"
    strategy = parent / "strategy/strategy.json"
    implementation = child / "strategy/implementation.json"
    commands = [
        [sys.executable, str(ROOT / ".codex/skills/kernel-strategy/scripts/validate_strategy.py"),
         "--bottleneck", str(parent / "bottleneck/bottleneck.json"), "--strategy", str(strategy)],
        [sys.executable, str(ROOT / ".codex/skills/kernel-implementation/scripts/validate_implementation.py"),
         "--bottleneck", str(parent / "bottleneck/bottleneck.json"),
         "--strategy", str(strategy), "--implementation", str(implementation),
         "--parent", str(parent), "--project-dir", str(child), "--require-attempts"],
    ]
    for command in commands:
        validation = subprocess.run(
            command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if validation.returncode != 0:
            return False, "round validator failed: " + validation.stdout.strip()[-500:]
    return True, "round durable gates passed"


def strategy_gate(project: Path) -> tuple[bool, str]:
    command = [
        sys.executable, str(ROOT / ".codex/skills/kernel-strategy/scripts/validate_strategy.py"),
        "--bottleneck", str(project / "bottleneck/bottleneck.json"),
        "--strategy", str(project / "strategy/strategy.json"),
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0, result.stdout.strip()[-500:]


def implementation_gate(parent: Path, child: Path) -> tuple[bool, str]:
    command = [
        sys.executable,
        str(ROOT / ".codex/skills/kernel-implementation/scripts/validate_implementation.py"),
        "--bottleneck", str(parent / "bottleneck/bottleneck.json"),
        "--strategy", str(parent / "strategy/strategy.json"),
        "--implementation", str(child / "strategy/implementation.json"),
        "--parent", str(parent), "--project-dir", str(child), "--require-attempts",
    ]
    result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    return result.returncode == 0, result.stdout.strip()[-500:]


def operator_projects(operator: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(operator)}_(\d+)$")
    projects = []
    for path in LEVEL.glob(f"{operator}_*"):
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            projects.append((int(match.group(1)), path))
    return sorted(projects)


def resume_summary(operator: str) -> str:
    """Build the compact durable-state snapshot passed to a content Agent."""
    lines = []
    for version, project in operator_projects(operator):
        workspace = load_json(project / "workspace.json") or {}
        precision_ok = precision_gate(project, workspace)[0]
        performance_ok = performance_gate(project, workspace)[0]
        bottleneck_ok = bottleneck_gate(project)[0] if performance_ok else False
        bottleneck = load_json(project / "bottleneck/bottleneck.json") if bottleneck_ok else None
        issue_count = len(bottleneck.get("issues", [])) if bottleneck else None
        lines.append(
            f"_{version}:status={workspace.get('status', 'MISSING')},"
            f"precision={'PASS' if precision_ok else 'NO'},"
            f"performance={'DONE' if performance_ok else 'NO'},"
            f"bottleneck={'DONE' if bottleneck_ok else 'NO'},"
            f"issues={issue_count if issue_count is not None else 'NA'}"
        )
    return "; ".join(lines) if lines else "no workspace version"


def next_durable_stage(operator: str) -> str:
    """Identify the first stage that can change durable state."""
    projects = operator_projects(operator)
    if not projects:
        return "prepare _0"
    for version, project in projects:
        workspace = load_json(project / "workspace.json") or {}
        if performance_screening(project, workspace):
            return f"stop at _{version}: screened no improvement"
        if not precision_gate(project, workspace)[0]:
            return f"validate or repair _{version} precision"
        if not performance_gate(project, workspace)[0]:
            return f"collect _{version} formal performance"
        if not bottleneck_gate(project)[0]:
            return f"analyze _{version} bottleneck"
        bottleneck = load_json(project / "bottleneck/bottleneck.json") or {}
        if not bottleneck.get("issues"):
            return f"stop at _{version}: issues=[]"
        if version >= MAX_OPTIMIZATION_ROUNDS:
            return f"stop at _{MAX_OPTIMIZATION_ROUNDS}: version limit reached"
        strategy = load_json(project / "strategy/strategy.json")
        if not strategy:
            return f"derive _{version} strategy"
        child = LEVEL / f"{operator}_{version + 1}"
        if not child.exists():
            return f"implement _{version} strategy into _{version + 1}"
    latest = projects[-1][0]
    return f"resume latest _{latest} incomplete transition"


def repair_count(project: Path) -> int:
    implementation = load_json(project / "strategy/implementation.json") or {}
    return sum(1 for attempt in implementation.get("attempts", [])
               if isinstance(attempt, dict) and attempt.get("kind") == "repair")


def completed_chain(operator: str, final_version: int) -> bool:
    """Require every parent->child transition through final_version to be valid."""
    return all(
        completed_round_gate(
            LEVEL / f"{operator}_{version - 1}",
            LEVEL / f"{operator}_{version}",
        )[0]
        for version in range(1, final_version + 1)
    )


def failure_summary(project: Path) -> dict:
    """Return a bounded failure record; never inject a complete build log."""
    precision = load_json(project / "precision/precision.json") or {}
    summary = {
        "failure_stage": precision.get("failure_stage"),
        "stage": precision.get("stage"),
        "reason": precision.get("reason") or precision.get("message"),
        "exit_code": precision.get("exit_code"),
    }
    log_value = precision.get("log_path") or precision.get("log")
    if isinstance(log_value, str):
        log_path = Path(log_value)
        if not log_path.is_absolute():
            log_path = project / log_path
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            summary["log_tail"] = lines[-80:]
        except OSError:
            summary["log_tail"] = []
    return {key: value for key, value in summary.items() if value not in (None, [], "")}


def _strategy_operations(strategy: dict | None) -> set[str]:
    return {
        str(action.get("operation", ""))
        for item in (strategy or {}).get("strategies", [])
        for action in item.get("actions", []) if isinstance(action, dict)
    } - {""}


def _knowledge_sections(operations: set[str]) -> list[str]:
    sections = {"sdk-check"}
    for operation in operations:
        if any(token in operation for token in ("transfer", "gather", "scatter", "writeback")):
            sections.add("transfer")
            sections.add("lifetime")
        if any(token in operation for token in ("vector", "reduce", "cast", "cube")):
            sections.add("vector")
        if any(token in operation for token in (
            "buffer", "pipeline", "sync", "resident", "retain", "reuse", "on_chip", "privatize"
        )):
            sections.add("lifetime")
        if any(token in operation for token in ("tiling", "parallel", "partition", "work_unit", "cube")):
            sections.add("tiling")
        if any(token in operation for token in ("writeback", "cast", "fixpipe", "scatter")):
            sections.add("abi")
    return sorted(sections)


def implementation_knowledge(
    strategy: dict | None, device_id: int = 0, include_device_error: bool = False
) -> dict:
    """Select compact knowledge from frozen actions and exact device identity."""
    operations = _strategy_operations(strategy)
    sections = _knowledge_sections(operations)
    core = ROOT / ".codex/kernel-knowledge/implementation-core.md"
    references = [{"path": str(core), "sections": sections}]

    device_path = LEVEL.parent / "hardware" / f"device_{device_id}.json"
    device = load_json(device_path) or {}
    soc = (device.get("identity") or {}).get("soc")
    index = load_json(ROOT / ".codex/kernel-knowledge/architecture-index.json") or {}
    architecture = index.get(soc) if isinstance(soc, str) else None
    if isinstance(architecture, dict):
        reference = ROOT / ".codex/kernel-knowledge/architecture-knowledge.md"
        arch_sections = [name for name in sections if name in {"transfer", "vector", "lifetime"}]
        if include_device_error:
            arch_sections.append("device-error")
        if reference.is_file() and arch_sections:
            references.append({
                "path": str(reference),
                "profile": architecture.get("profile"),
                "sections": arch_sections,
            })

    return {
        "device_soc": soc,
        "architecture": architecture.get("profile") if isinstance(architecture, dict) else None,
        "references": references,
    }


def initial_implementation(strategy: dict) -> dict:
    strategies = strategy.get("strategies") or []
    actions = [action for item in strategies for action in item.get("actions", [])]
    indices = list(range(1, len(actions) + 1))
    return {
        "strategy_keys": [item["strategy_key"] for item in strategies],
        "reasoning": ["由 implementation Agent 填写实际源码落地摘要。"],
        "actions": [
            {"action_index": index, "implementation_summary": "由 implementation Agent 填写。"}
            for index in indices
        ],
        "modified_files": sorted({action["target"].split("::", 1)[0] for action in actions}),
        "attempts": [{
            "attempt": 1,
            "kind": "initial",
            "trigger": None,
            "knowledge_keys": [],
            "action_indices": indices,
            "implementation_summary": "由 implementation Agent 填写。",
            "modified_files": sorted({action["target"].split("::", 1)[0] for action in actions}),
            "validation": {"build": "NOT_RUN", "precision": "NOT_RUN", "performance": "NOT_RUN"},
        }],
    }


def prepare_child(parent: Path, child: Path, operator: str) -> None:
    """Create a clean PREPARED child; content Agents never copy versions."""
    if child.exists():
        raise RuntimeError(f"child already exists: {child}")
    strategy = load_json(parent / "strategy/strategy.json")
    parent_workspace = load_json(parent / "workspace.json") or {}
    if not strategy:
        raise RuntimeError("parent strategy is missing")
    shutil.copytree(
        parent, child, symlinks=True,
        ignore=shutil.ignore_patterns(
            "precision", "performance", "bottleneck", "strategy", "workspace.json",
            "build", "build_out", "CMakeFiles", "*.log",
        ),
    )
    strategy_dir = child / "strategy"
    strategy_dir.mkdir()
    shutil.copy2(parent / "strategy/strategy.json", strategy_dir / "strategy.json")
    (strategy_dir / "implementation.json").write_text(
        json.dumps(initial_implementation(strategy), ensure_ascii=False, indent=2) + "\n"
    )
    version = int(child.name.rsplit("_", 1)[1])
    workspace = {
        "operator": operator,
        "version": version,
        "source_project": parent_workspace.get("source_project"),
        "parent_version": int(parent.name.rsplit("_", 1)[1]),
        "vendor": parent_workspace.get("vendor"),
        "status": "PREPARED",
        "source_fingerprint": source_fingerprint(child),
        "precision": None,
        "performance": None,
        "bottleneck": None,
        "strategy": {"status": "IMPLEMENTING", "result": "strategy/strategy.json"},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (child / "workspace.json").write_text(json.dumps(workspace, ensure_ascii=False, indent=2) + "\n")


def durable_action(operator: str) -> dict:
    """Choose exactly one state-changing action from durable artifacts."""
    projects = operator_projects(operator)
    if not projects:
        return {"kind": "terminal", "state": "prepare_failed", "reason": "missing _0 workspace"}
    version, project = projects[-1]
    workspace = load_json(project / "workspace.json") or {}
    screening = performance_screening(project, workspace)
    if screening is not None:
        return {
            "kind": "terminal", "state": "stopped_no_improvement",
            "reason": (
                f"_{version} vs_best=_{screening.get('best_version')} "
                f"improvement={float(screening['improvement_percent']):.4f}%<=1%; "
                f"best=_{screening.get('best_version')}"
            ),
        }
    rejected = non_improving_result(operator)
    if rejected is not None:
        rejected_version, best_version, gain = rejected
        return {"kind": "terminal", "state": "stopped_no_improvement",
                "reason": f"_{rejected_version} vs_best=_{best_version} improvement={gain:.4f}%<=1%; best=_{best_version}"}
    if version > 0:
        parent = LEVEL / f"{operator}_{version - 1}"
        if not strategy_gate(parent)[0]:
            return {"kind": "terminal", "state": "incomplete",
                    "reason": f"_{version - 1} strategy uses stale contract; reinitialize queue"}
        blocking = load_json(project / "strategy/implementation_blocking.json")
        if blocking and blocking.get("reason"):
            return {"kind": "terminal", "state": "implementation_blocked",
                    "reason": f"_{version}: {blocking['reason']}"}
        if not implementation_gate(parent, project)[0]:
            return {"kind": "agent", "stage": "implementation_fix", "version": version,
                    "project": project, "parent": parent}
    if not precision_gate(project, workspace)[0]:
        status = workspace.get("status")
        if status in {"BUILD_FAILED", "PRECISION_FAILED"}:
            precision = load_json(project / "precision/precision.json") or {}
            # A repair changed source after the recorded failure.  The old
            # failure is stale, so rebuild/validate before considering another
            # repair attempt.
            if precision.get("source_fingerprint") != source_fingerprint(project):
                return {"kind": "command", "stage": "precision", "version": version,
                        "project": project}
            repairs = repair_count(project)
            implementation = project / "strategy/implementation.json"
            parent = LEVEL / f"{operator}_{version - 1}"
            if version > 0 and implementation.is_file() and parent.is_dir() and repairs < MAX_REPAIRS:
                return {"kind": "agent", "stage": "repair", "version": version,
                        "project": project, "parent": parent, "repair": repairs + 1}
            return {"kind": "terminal", "state": status.lower(),
                    "reason": f"_{version}={status}; repairs={repairs}/{MAX_REPAIRS}"}
        return {"kind": "command", "stage": "precision", "version": version, "project": project}
    if not performance_gate(project, workspace)[0]:
        if workspace.get("status") == "PERFORMANCE_FAILED":
            return {"kind": "terminal", "state": "performance_failed",
                    "reason": f"_{version}=PERFORMANCE_FAILED"}
        return {"kind": "command", "stage": "performance", "version": version, "project": project}
    if not bottleneck_gate(project)[0]:
        return {"kind": "agent", "stage": "planning", "version": version, "project": project}
    bottleneck = load_json(project / "bottleneck/bottleneck.json") or {}
    if bottleneck.get("issues") == []:
        return {"kind": "terminal", "state": "stopped_no_issue",
                "reason": f"_{version}=PERFORMANCE_DONE,issues=[]"}
    if version >= MAX_OPTIMIZATION_ROUNDS:
        if completed_chain(operator, version):
            return {"kind": "terminal", "state": "completed_max_rounds_with_issues",
                    "reason": f"_{version}=PERFORMANCE_DONE,residual_issues={len(bottleneck.get('issues', []))}"}
        return {"kind": "terminal", "state": "incomplete",
                "reason": f"_{version} round-chain validation failed"}
    blocking = load_json(project / "strategy/blocking.json")
    if blocking and blocking.get("reason"):
        return {"kind": "terminal", "state": "strategy_blocked",
                "reason": f"_{version}: {blocking['reason']}"}
    strategy = project / "strategy/strategy.json"
    if not strategy.is_file() or not strategy_gate(project)[0]:
        return {"kind": "agent", "stage": "planning", "version": version, "project": project}
    child = LEVEL / f"{operator}_{version + 1}"
    if child.exists():
        return {"kind": "terminal", "state": "incomplete",
                "reason": f"_{version + 1} exists but was not selected as latest workspace"}
    return {"kind": "agent", "stage": "implementation", "version": version,
            "project": project, "child": child}


def classify_result(operator: str, returncode: int) -> tuple[str, str]:
    """Classify by durable workspace gates, never by agent exit code alone."""
    projects = operator_projects(operator)
    reports = [(version, project, load_json(project / "workspace.json")) for version, project in projects]
    for version, project, workspace in reversed(reports):
        if version <= 0 or not performance_gate(project, workspace)[0]:
            continue
        best = best_prior_project(operator, version)
        best_version, parent = best if best is not None else (version - 1, LEVEL / f"{operator}_{version - 1}")
        gain = improvement_percent(parent, project)
        if gain is not None and gain <= MIN_IMPROVEMENT_PERCENT:
            return "stopped_no_improvement", f"_{version} vs_best=_{best_version} improvement={gain:.4f}%<=1%; best=_{best_version}"
    final = next(((project, report) for version, project, report in reports
                  if version == MAX_OPTIMIZATION_ROUNDS), None)
    if final is not None:
        project, report = final
        if completed_chain(operator, MAX_OPTIMIZATION_ROUNDS):
            bottleneck = load_json(project / "bottleneck/bottleneck.json") or {}
            if bottleneck.get("issues") == []:
                return "stopped_no_issue", f"_{MAX_OPTIMIZATION_ROUNDS}=PERFORMANCE_DONE,issues=[]"
            return ("completed_max_rounds_with_issues",
                    f"_{MAX_OPTIMIZATION_ROUNDS}=PERFORMANCE_DONE,residual_issues={len(bottleneck.get('issues', []))}")
    for version, project, workspace in reversed(reports):
        if not performance_gate(project, workspace)[0]:
            continue
        bottleneck = load_json(project / "bottleneck/bottleneck.json")
        parent = next((path for candidate, path in projects if candidate == version - 1), None)
        round_ok = version == 0 or (parent is not None and completed_round_gate(parent, project)[0])
        if round_ok and bottleneck_gate(project)[0] and bottleneck.get("issues") == []:
            return "stopped_no_issue", f"_{version}=PERFORMANCE_DONE,issues=[]"
    latest_existing = projects[-1] if projects else None
    for version, project in ([latest_existing] if latest_existing else []):
        blocking = load_json(project / "strategy/blocking.json")
        if blocking and blocking.get("stage") == "strategy" and blocking.get("cause_key") and blocking.get("reason"):
            return "strategy_blocked", f"_{version}: {blocking['cause_key']}: {blocking['reason']}"
        blocking = load_json(project / "strategy/implementation_blocking.json")
        if blocking and blocking.get("stage") == "implementation" and blocking.get("cause_key") and blocking.get("reason"):
            return "implementation_blocked", f"_{version}: {blocking['cause_key']}: {blocking['reason']}"
    latest = next(((version, report) for version, _, report in reversed(reports) if report), None)
    if latest:
        version, workspace = latest
        status = workspace.get("status")
        project = LEVEL / f"{operator}_{version}"
        screening = performance_screening(project, workspace)
        if screening is not None:
            return (
                "stopped_no_improvement",
                f"_{version} vs_best=_{screening.get('best_version')} "
                f"improvement={float(screening['improvement_percent']):.4f}%<=1%",
            )
        if status in {"BUILD_FAILED", "PRECISION_FAILED", "PERFORMANCE_FAILED"}:
            return status.lower(), f"_{version}={status}"
    if returncode != 0:
        return "agent_failed", f"codex_returncode={returncode}"
    return "incomplete", "agent exited without a terminal workspace result"


def initialize() -> None:
    """Archive derived decisions/versions, preserve every _0 source and valid baseline report."""
    archive = ARCHIVE_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if archive.exists():
        raise SystemExit(f"archive already exists: {archive}")
    archive.mkdir(parents=True)
    base_projects = sorted(LEVEL.glob("*_0"), key=lambda path: path.name)
    if not base_projects:
        raise SystemExit("no level1 _0 projects")
    for base in base_projects:
        operator = base.name[:-2]
        operator_archive = archive / operator
        moved = False
        for version, derived in operator_projects(operator):
            if version == 0:
                continue
            if derived.exists():
                operator_archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(derived), str(operator_archive / derived.name))
                moved = True
        for folder in ("bottleneck", "strategy"):
            stale = base / folder
            if stale.exists():
                operator_archive.mkdir(parents=True, exist_ok=True)
                shutil.move(str(stale), str(operator_archive / f"{base.name}_{folder}"))
                moved = True
        if not moved and operator_archive.exists():
            operator_archive.rmdir()
    items = []
    for position, base in enumerate(base_projects, 1):
        workspace = load_json(base / "workspace.json")
        performance_ok, _ = performance_gate(base, workspace)
        precision_ok, _ = precision_gate(base, workspace)
        items.append({
            "position": position,
            "operator": base.name[:-2],
            "state": "pending",
            "baseline": "reuse_performance" if performance_ok else (
                "reuse_precision_collect_performance" if precision_ok else "revalidate_precision_and_performance"
            ),
            "stage_attempts": {},
        })
    queue = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "controller_state_machine_one_thread_per_operator_max_four_rounds",
        "archive": str(archive),
        "max_agent_attempts_per_stage": MAX_AGENT_ATTEMPTS,
        "max_repairs_per_version": MAX_REPAIRS,
        "max_optimization_rounds": MAX_OPTIMIZATION_ROUNDS,
        "min_improvement_percent": MIN_IMPROVEMENT_PERCENT,
        "items": items,
    }
    save(queue)
    LOG.write_text("timestamp\tposition\toperator\tstate\treturncode\n")
    print(f"initialized={len(items)} queue={QUEUE} archive={archive}")


def rebuild() -> None:
    """Rebuild scheduling state without moving or overwriting any workspace version."""
    base_projects = sorted(LEVEL.glob("*_0"), key=lambda path: path.name)
    if not base_projects:
        raise SystemExit("no level1 _0 projects")
    items = []
    for position, base in enumerate(base_projects, 1):
        workspace = load_json(base / "workspace.json")
        performance_ok, _ = performance_gate(base, workspace)
        precision_ok, _ = precision_gate(base, workspace)
        versions = [version for version, _ in operator_projects(base.name[:-2])]
        items.append({
            "position": position,
            "operator": base.name[:-2],
            "state": "pending",
            "baseline": "reuse_performance" if performance_ok else (
                "reuse_precision_collect_performance" if precision_ok else "revalidate_precision_and_performance"
            ),
            "existing_versions": versions,
            "stage_attempts": {},
        })
    queue = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "controller_state_machine_one_thread_per_operator_max_four_rounds",
        "preserve_existing_versions": True,
        "max_agent_attempts_per_stage": MAX_AGENT_ATTEMPTS,
        "max_repairs_per_version": MAX_REPAIRS,
        "max_optimization_rounds": MAX_OPTIMIZATION_ROUNDS,
        "min_improvement_percent": MIN_IMPROVEMENT_PERCENT,
        "items": items,
    }
    save(queue)
    LOG.write_text("timestamp\tposition\toperator\tstate\treturncode\n")
    print(f"rebuilt={len(items)} queue={QUEUE} existing_versions_preserved=true")


def reconcile() -> None:
    """Reconcile stopped queue bookkeeping from durable artifacts only."""
    queue = json.loads(QUEUE.read_text())
    counts: dict[str, int] = {}
    for item in queue["items"]:
        previous_state = item.get("state")
        state, reason = classify_result(item["operator"], 0)
        if state in {"incomplete", "agent_failed"}:
            if previous_state == "running":
                item["state"] = "pending"
                current_stage = item.get("current_stage")
                if current_stage and current_stage in item.get("stage_attempts", {}):
                    item["stage_attempts"][current_stage] = max(
                        0, item["stage_attempts"][current_stage] - 1
                    )
                elif "attempts" in item:  # migrate queues created by the old runner
                    item["attempts"] = max(0, item.get("attempts", 0) - 1)
                item["result_reason"] = "controller stopped; resume from durable stage"
                item["interrupted_at"] = datetime.now(timezone.utc).isoformat()
        else:
            item["state"] = state
            item["result_reason"] = reason
            item["recovered_at"] = datetime.now(timezone.utc).isoformat()
        counts[item["state"]] = counts.get(item["state"], 0) + 1
    save(queue)
    print("reconciled=" + json.dumps(counts, ensure_ascii=False, sort_keys=True))


def stage_key(action: dict) -> str:
    """Give each bounded repair its own retry budget."""
    suffix = str(action["stage"])
    if suffix == "repair":
        suffix += str(action["repair"])
    return f"v{action['version']}:{suffix}"


def stage_input(operator: str, action: dict, attempt: int) -> Path:
    """Persist the small hand-off contract for one operator Agent turn."""
    project = action["project"]
    payload = {
        "operator": operator,
        "stage": action["stage"],
        "version": action["version"],
        "project": str(project),
    }
    if action["stage"] == "planning":
        payload.update({
            "host_files": [str(path) for path in sorted((project / "op_host").rglob("*")) if path.is_file()],
            "kernel_files": [str(path) for path in sorted((project / "op_kernel").rglob("*")) if path.is_file()],
        })
        facts = subprocess.run(
            [sys.executable, str(ROOT / ".codex/skills/kernel-bottleneck/scripts/source_facts.py"),
             "--project-dir", str(project)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        if facts.returncode != 0:
            raise RuntimeError("source facts failed: " + facts.stdout.strip()[-500:])
        payload["source_facts"] = json.loads(facts.stdout)
    if action["stage"] == "implementation":
        strategy_data = load_json(project / "strategy/strategy.json")
        child = action["child"]
        payload.update({
            "parent": str(project),
            "child": str(child),
            "bottleneck": str(project / "bottleneck/bottleneck.json"),
            "strategy": str(project / "strategy/strategy.json"),
            "implementation": str(child / "strategy/implementation.json"),
            "host_files": [str(path) for path in sorted((child / "op_host").rglob("*")) if path.is_file()],
            "kernel_files": [str(path) for path in sorted((child / "op_kernel").rglob("*")) if path.is_file()],
            "implementation_knowledge": implementation_knowledge(strategy_data),
        })
    elif action["stage"] in {"repair", "implementation_fix"}:
        strategy_data = load_json(action["parent"] / "strategy/strategy.json")
        payload.update({
            "parent": str(action["parent"]),
            "bottleneck": str(action["parent"] / "bottleneck/bottleneck.json"),
            "strategy": str(action["parent"] / "strategy/strategy.json"),
            "implementation": str(project / "strategy/implementation.json"),
            "host_files": [str(path) for path in sorted((project / "op_host").rglob("*")) if path.is_file()],
            "kernel_files": [str(path) for path in sorted((project / "op_kernel").rglob("*")) if path.is_file()],
            "implementation_knowledge": implementation_knowledge(
                strategy_data, include_device_error=action["stage"] == "repair"
            ),
        })
        if action["stage"] == "repair":
            payload.update({
                "repair": action["repair"],
                "repairs_remaining_after_this": MAX_REPAIRS - action["repair"],
                "failure": failure_summary(project),
            })
    directory = RUNS / f"{operator}.stages"
    directory.mkdir(parents=True, exist_ok=True)
    label = str(action["stage"])
    if label == "repair":
        label += str(action["repair"])
    path = directory / f"v{action['version']}.{label}.attempt{attempt}.input.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def agent_prompt(input_path: Path, action: dict) -> str:
    stage = action["stage"]
    common = f"""只处理阶段输入 {input_path}。仓库为 {ROOT}，严格遵守 AGENTS.md。
阶段输入是权威最小状态；不遍历其他算子、历史版本、完整 CSV/日志，不读取 validator 源码。precision、performance、msprof 和 reference 均由控制器执行。
"""
    if stage == "planning":
        return common + """依次使用 kernel-bottleneck、kernel-strategy，完整读取两个 SKILL.md 并按其引用按需加载知识。源码只读一次；source_facts 中的排除项必须遵守。按算子模式快速确定最多 3 个兼容 issues，再按最大合法 tile→搬运 chunk→计算→Buffer→必要同步生成 strategy；不建立成本账本、不搜索参数。校验两个 JSON，不修改源码或生成辅助分析文件。"""
    if stage == "implementation":
        return common + """只使用 kernel-implementation。child、冻结 strategy 和 implementation 骨架已创建，不得创建版本。仅读取 implementation_knowledge.references 指定章节；architecture=null 时不使用架构常数。实施全部 actions，填写骨架；validator 总计最多两次，中间只允许一次最小修正。"""
    if stage == "implementation_fix":
        return common + """只使用 kernel-implementation。子版本已存在；按 validator 错误修正记录或补全冻结 action，复验一次。不得新增优化或运行门禁。"""
    return common + """只使用 kernel-implementation。依据 bounded failure 在冻结 actions 内做一次最小修复，并追加一个 repair attempt。设备异常读取当前架构的 device-error/transfer 节；未知架构查 headers，并复核本轮全部搬运点。校验后结束，不运行门禁。"""


def thread_id_from_log(path: Path) -> str | None:
    """Read the persisted Codex thread id from a JSONL run log."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
            return event["thread_id"]
    return None


def agent_command(thread_id: str | None, final_message: Path, prompt: str) -> list[str]:
    common = ["--json", "--dangerously-bypass-approvals-and-sandbox",
              "-o", str(final_message)]
    if thread_id:
        return ["codex", "exec", "resume", *common, thread_id, prompt]
    return ["codex", "exec", *common, "-C", str(ROOT), prompt]


def run_agent_stage(item: dict, action: dict) -> int:
    stage_attempts = item.setdefault("stage_attempts", {})
    key = stage_key(action)
    attempt = stage_attempts.get(key, 0) + 1
    stage_attempts[key] = attempt
    if action["stage"] == "implementation":
        prepare_child(action["project"], action["child"], item["operator"])
    input_path = stage_input(item["operator"], action, attempt)
    label = str(action["stage"])
    if label == "repair":
        label += str(action["repair"])
    stem = f"{item['position']:03d}_{item['operator']}.v{action['version']}.{label}.attempt{attempt}"
    run_log = RUNS / f"{stem}.jsonl"
    final_message = RUNS / f"{stem}.final.txt"
    with run_log.open("w") as output:
        child_env = os.environ.copy()
        child_env["ASCENDC_TMPDIR"] = str(ASCENDC_TMPDIR)
        child_env["TMPDIR"] = str(ASCENDC_TMPDIR)
        result = subprocess.run(
            agent_command(item.get("agent_thread_id"), final_message,
                          agent_prompt(input_path, action)),
            cwd=ROOT, env=child_env, stdout=output, stderr=subprocess.STDOUT, text=True,
        )
    if not item.get("agent_thread_id"):
        thread_id = thread_id_from_log(run_log)
        if thread_id:
            item["agent_thread_id"] = thread_id
    return result.returncode


def run_gate(item: dict, action: dict) -> int:
    operator = item["operator"]
    project = action["project"]
    stage = action["stage"]
    script = ROOT / f".codex/skills/kernel-{stage}/scripts" / (
        "validate_precision.py" if stage == "precision" else "collect_performance.py"
    )
    command = (
        "source /data/lu/activate_evokernel.sh && "
        f"python {shlex.quote(str(script))} {shlex.quote(operator)} "
        f"--project-dir {shlex.quote(str(project))}"
    )
    if stage == "performance":
        command += " --device 0"
    stem = f"{item['position']:03d}_{operator}.v{action['version']}.{stage}"
    log_path = RUNS / f"{stem}.log"
    with log_path.open("w") as output:
        result = subprocess.run(["bash", "-lc", command], cwd=ROOT,
                                stdout=output, stderr=subprocess.STDOUT, text=True)
    return result.returncode


def run_state_machine(queue: dict) -> None:
    """Advance each operator one durable stage at a time."""
    queue.pop("acceptance", None)
    queue["mode"] = "controller_state_machine_one_thread_per_operator_max_four_rounds"
    queue["max_agent_attempts_per_stage"] = MAX_AGENT_ATTEMPTS
    queue["max_repairs_per_version"] = MAX_REPAIRS
    queue["max_optimization_rounds"] = MAX_OPTIMIZATION_ROUNDS
    queue["min_improvement_percent"] = MIN_IMPROVEMENT_PERCENT
    index = 0
    while index < len(queue["items"]):
        item = queue["items"][index]
        if item["state"] in TERMINAL_STATES:
            cleanup_reference_cache(item["operator"])
            index += 1
            continue
        action = durable_action(item["operator"])
        if action["kind"] == "terminal":
            item["state"] = action["state"]
            item["result_reason"] = action["reason"]
            item["finished_at"] = datetime.now(timezone.utc).isoformat()
            cleanup_reference_cache(item["operator"])
            save(queue)
            index += 1
            continue
        item["state"] = "running"
        item["current_stage"] = stage_key(action)
        item["started_at"] = datetime.now(timezone.utc).isoformat()
        save(queue)
        if action["kind"] == "command":
            returncode = run_gate(item, action)
        else:
            key = item["current_stage"]
            if item.get("stage_attempts", {}).get(key, 0) >= MAX_AGENT_ATTEMPTS:
                item["state"] = "agent_failed"
                item["result_reason"] = f"{key} Agent attempt limit exhausted"
                save(queue)
                index += 1
                continue
            returncode = run_agent_stage(item, action)
        item["returncode"] = returncode
        item["last_stage_finished_at"] = datetime.now(timezone.utc).isoformat()
        item["state"] = "pending"
        if returncode != 0:
            if action["kind"] == "agent":
                key = item["current_stage"]
                if item.get("stage_attempts", {}).get(key, 0) >= MAX_AGENT_ATTEMPTS:
                    item["state"] = "agent_failed"
                    item["result_reason"] = f"{key} failed with returncode={returncode}"
            else:
                item["result_reason"] = f"{item['current_stage']} gate returncode={returncode}"
        save(queue)
        with LOG.open("a") as progress:
            progress.write(
                f"{item['last_stage_finished_at']}\t{item['position']}\t{item['operator']}\t"
                f"{item['current_stage']}:{item['state']}\t{returncode}\n"
            )
        if item["state"] in TERMINAL_STATES:
            index += 1


def requested_mode(argv: list[str]) -> str:
    modes = {"--initialize": "initialize", "--rebuild": "rebuild",
             "--reconcile": "reconcile", "--show-stages": "show-stages"}
    if argv == ["--help"]:
        return "help"
    if not argv:
        return "run"
    if len(argv) != 1 or argv[0] not in modes:
        raise SystemExit("usage: run_level1_regeneration.py [--initialize|--rebuild|--reconcile|--show-stages]")
    return modes[argv[0]]


def main() -> None:
    mode = requested_mode(sys.argv[1:])
    if mode == "help":
        print("usage: run_level1_regeneration.py [--initialize|--rebuild|--reconcile|--show-stages]")
        return
    if mode == "initialize":
        initialize()
        return
    if mode == "rebuild":
        rebuild()
        return
    if mode == "reconcile":
        reconcile()
        return
    if mode == "show-stages":
        queue = json.loads(QUEUE.read_text())
        for item in queue["items"]:
            action = durable_action(item["operator"])
            compact = {key: str(value) if isinstance(value, Path) else value
                       for key, value in action.items()}
            print(json.dumps({"operator": item["operator"], **compact}, ensure_ascii=False))
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    ASCENDC_TMPDIR.mkdir(parents=True, exist_ok=True)
    if not os.access(ASCENDC_TMPDIR, os.W_OK):
        raise SystemExit(f"temporary directory is not writable: {ASCENDC_TMPDIR}")
    lock_handle = LOCK.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"level1 queue is already running: {LOCK}")
    queue = json.loads(QUEUE.read_text())
    run_state_machine(queue)


if __name__ == "__main__":
    main()
