#!/usr/bin/env python3
"""Clean stale Level1 optimization artifacts and build a resumable operator queue."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
SOURCE = ROOT / "kernel/KernelBench910B/level1"
WORKSPACE = ROOT / "kernel_workspace/KernelBench910B/level1"
QUEUE = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_queue.json"
CLEANUP = ROOT / "kernel_workspace/KernelBench910B/level1_cleanup_manifest.json"


def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    for folder in ("op_host", "op_kernel"):
        for path in sorted((project / folder).rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(project).as_posix().encode())
                digest.update(b"\0")
                digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    operators = sorted(path.name for path in SOURCE.iterdir() if path.is_dir())
    conflicts = []
    for operator in operators:
        base = WORKSPACE / f"{operator}_0"
        if not base.is_dir() or fingerprint(SOURCE / operator) != fingerprint(base):
            conflicts.append(operator)
    if conflicts:
        raise RuntimeError(f"_0 missing or conflicts with original source: {conflicts}")

    removed: list[str] = []
    for project in sorted(WORKSPACE.iterdir()):
        if not project.is_dir() or "_" not in project.name:
            continue
        try:
            version = int(project.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        if version > 0:
            removed.append(str(project.relative_to(ROOT)))
            shutil.rmtree(project)

    for operator in operators:
        base = WORKSPACE / f"{operator}_0"
        for relative in ("bottleneck", "strategy", "build_out", "CppExtension/build"):
            target = base / relative
            if target.exists():
                removed.append(str(target.relative_to(ROOT)))
                shutil.rmtree(target) if target.is_dir() else target.unlink()
        for cache in base.rglob("__pycache__"):
            if cache.is_dir():
                removed.append(str(cache.relative_to(ROOT)))
                shutil.rmtree(cache)

    now = datetime.now(timezone.utc).isoformat()
    items = []
    for position, operator in enumerate(operators, 1):
        base = WORKSPACE / f"{operator}_0"
        state = json.loads((base / "workspace.json").read_text())
        reusable = (
            state.get("status") == "PERFORMANCE_DONE"
            and (base / "precision/precision.json").is_file()
            and (base / "performance/performance.json").is_file()
        )
        items.append({
            "position": position,
            "operator": operator,
            "state": "pending",
            "initial_result": "reuse_performance" if reusable else "resume_from_precision",
            "attempts": 0,
        })

    CLEANUP.write_text(json.dumps({
        "generated_at": now,
        "removed_count": len(removed),
        "removed": removed,
        "recovery": "deleted artifacts are not recoverable from this workspace; original kernel and all _0 sources/results were retained",
    }, ensure_ascii=False, indent=2) + "\n")
    QUEUE.write_text(json.dumps({
        "generated_at": now,
        "mode": "single_concurrency_one_fresh_agent_context_per_operator",
        "items": items,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"operators={len(items)} removed={len(removed)} queue={QUEUE}")


if __name__ == "__main__":
    main()
