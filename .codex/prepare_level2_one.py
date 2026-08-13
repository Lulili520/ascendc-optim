#!/usr/bin/env python3
"""Prepare exactly one planned Level 2 operator as its _0 workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
LEVEL_NAME = "level2"
SOURCE = ROOT / "kernel/KernelBench910B/level2"
WORKSPACE = ROOT / "kernel_workspace/KernelBench910B/level2"
MANIFEST = ROOT / "kernel/KernelBench910B/manifest.json"
QUEUE = ROOT / "kernel_workspace/KernelBench910B/level2_queue.json"


def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    )
    if not files:
        raise RuntimeError("op_host/op_kernel 中没有源码文件")
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def same_source(left: Path, right: Path) -> bool:
    for folder in ("op_host", "op_kernel"):
        lroot, rroot = left / folder, right / folder
        left_files = sorted(p.relative_to(lroot) for p in lroot.rglob("*") if p.is_file())
        right_files = sorted(p.relative_to(rroot) for p in rroot.rglob("*") if p.is_file())
        if left_files != right_files or any(
            (lroot / rel).read_bytes() != (rroot / rel).read_bytes() for rel in left_files
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operator")
    args = parser.parse_args()
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    planned = [item for item in queue["items"] if item["operator"] == args.operator]
    if len(planned) != 1:
        raise RuntimeError("算子不在唯一规划队列中")
    earlier = [item for item in queue["items"] if item["position"] < planned[0]["position"]]
    unfinished = [item["operator"] for item in earlier if item["status"] not in {
        "COMPLETED", "FAILED_PRECISION", "FAILED_BUILD", "STOPPED_NO_BOTTLENECK",
        "STOPPED_NO_STRATEGY", "STOPPED_IMPLEMENTATION",
    }]
    if unfinished:
        raise RuntimeError(f"前序算子尚未结束：{unfinished[0]}")
    source = SOURCE / args.operator
    target = WORKSPACE / f"{args.operator}_0"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    item = manifest.get(args.operator)
    if not source.is_dir() or not isinstance(item, dict) or item.get("level") != LEVEL_NAME:
        raise RuntimeError("原始工程或 manifest 非法")
    if target.exists():
        workspace_file = target / "workspace.json"
        if not workspace_file.is_file() or not same_source(source, target):
            raise RuntimeError("已存在的 _0 与原始工程冲突")
        print(f"reused={target}")
        return
    shutil.copytree(source, target, symlinks=True)
    state = {
        "operator": args.operator,
        "version": 0,
        "source_project": f"kernel/KernelBench910B/level2/{args.operator}",
        "parent_version": None,
        "vendor": item["vendor"],
        "status": "PREPARED",
        "source_fingerprint": fingerprint(target),
        "precision": None, "performance": None, "bottleneck": None,
        "strategy": None, "implementation": None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (target / "workspace.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared={target}")


if __name__ == "__main__":
    main()
