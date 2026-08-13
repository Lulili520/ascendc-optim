#!/usr/bin/env python3
"""Create one clean Level 2 optimization child from an explicit parent version."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
LEVEL = ROOT / "kernel_workspace/KernelBench910B/level2"


def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    )
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operator")
    parser.add_argument("--parent-version", type=int, required=True)
    args = parser.parse_args()

    parent = LEVEL / f"{args.operator}_{args.parent_version}"
    if not parent.is_dir():
        raise RuntimeError(f"父版本不存在：{parent}")
    parent_state = json.loads((parent / "workspace.json").read_text(encoding="utf-8"))
    if parent_state.get("status") != "PERFORMANCE_DONE":
        raise RuntimeError("父版本不是 PERFORMANCE_DONE")
    strategy = parent / "strategy/strategy.json"
    if not strategy.is_file() or json.loads(strategy.read_text(encoding="utf-8")).get("strategy") is None:
        raise RuntimeError("父版本缺少非空 strategy.json")

    occupied = []
    for path in LEVEL.glob(f"{args.operator}_*"):
        suffix = path.name.removeprefix(f"{args.operator}_")
        if path.is_dir() and suffix.isdigit():
            occupied.append(int(suffix))
    version = max(occupied) + 1
    target = LEVEL / f"{args.operator}_{version}"
    if target.exists():
        raise RuntimeError(f"目标版本已存在：{target}")

    target.mkdir(parents=False)
    for name in ("CMakeLists.txt", "CMakePresets.json", "build.sh"):
        shutil.copy2(parent / name, target / name)
    for name in ("op_host", "op_kernel"):
        shutil.copytree(parent / name, target / name, symlinks=True)
    shutil.copytree(
        parent / "CppExtension",
        target / "CppExtension",
        symlinks=True,
        ignore=shutil.ignore_patterns("build", "*.o", "*.d", "__pycache__"),
    )
    (target / "strategy").mkdir()
    shutil.copy2(strategy, target / "strategy/strategy.json")

    state = {
        "operator": args.operator,
        "version": version,
        "source_project": parent_state["source_project"],
        "parent_version": args.parent_version,
        "vendor": parent_state["vendor"],
        "status": "PREPARED",
        "source_fingerprint": fingerprint(target),
        "precision": None,
        "performance": None,
        "bottleneck": None,
        "strategy": {"status": "INPUT", "result": "strategy/strategy.json"},
        "implementation": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (target / "workspace.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared={target}")


if __name__ == "__main__":
    main()
