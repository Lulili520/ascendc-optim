#!/usr/bin/env python3
"""Prepare the unregistered level1 _0 workspaces without overwriting content."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
SOURCE = ROOT / "kernel/KernelBench910B/level1"
WORKSPACE = ROOT / "kernel_workspace/KernelBench910B/level1"
MANIFEST = ROOT / "kernel/KernelBench910B/manifest.json"
QUEUE = ROOT / "kernel_workspace/KernelBench910B/level1_remaining_queue.json"
TARGETS = [
    "MatrixScalarMultiplicationCustom",
    "MatrixVectorMultiplicationCustom",
    "MaxPooling1dCustom",
    "MaxPooling2dCustom",
    "MaxPooling3dCustom",
    "MaxReductionOverADimensionCustom",
    "MeanReductionOverADimensionCustom",
    "MinGptNewGeluCustom",
    "MinReductionOverADimensionCustom",
    "ReluCustom",
    "RmsNormCustom",
    "SeluCustom",
    "SigmoidCustom",
    "SoftmaxCustom",
    "SoftplusCustom",
    "SoftsignCustom",
    "SquareMatrixMultiplicationCustom",
    "StandardMatrixMultiplicationCustom",
    "SumReductionOverADimensionCustom",
    "SwishCustom",
    "TallSkinnyMatrixMultiplicationCustom",
    "TanhCustom",
    "TripletMarginLossCustom",
]


def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path
        for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*")
        if path.is_file()
    )
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def same_tree(left: Path, right: Path) -> bool:
    left_files = sorted(p.relative_to(left) for p in left.rglob("*") if p.is_file())
    right_files = sorted(p.relative_to(right) for p in right.rglob("*") if p.is_file())
    return left_files == right_files and all(
        (left / rel).read_bytes() == (right / rel).read_bytes() for rel in left_files
    )


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    registered = {
        json.loads(path.read_text())["operator"]
        for path in WORKSPACE.glob("*_*/workspace.json")
    }
    queue = [name for name in TARGETS if name not in registered]
    prepared = []
    for name in queue:
        source = SOURCE / name
        target = WORKSPACE / f"{name}_0"
        if target.exists():
            if not same_tree(source, target):
                raise RuntimeError(f"{name}: existing _0 conflicts with original project")
        else:
            shutil.copytree(source, target, symlinks=True)
        item = manifest[name]
        now = datetime.now(timezone.utc).isoformat()
        workspace = {
            "operator": name,
            "version": 0,
            "source_project": f"kernel/KernelBench910B/level1/{name}",
            "parent_version": None,
            "vendor": item["vendor"],
            "status": "PREPARED",
            "source_fingerprint": fingerprint(target),
            "precision": None,
            "performance": None,
            "bottleneck": None,
            "strategy": None,
            "implementation": None,
            "updated_at": now,
        }
        (target / "workspace.json").write_text(
            json.dumps(workspace, ensure_ascii=False, indent=2) + "\n"
        )
        prepared.append({"operator": name, "version": 0, "status": "PREPARED"})
    all_items = []
    for name in TARGETS:
        state = json.loads((WORKSPACE / f"{name}_0/workspace.json").read_text())
        all_items.append({"operator": name, "version": 0, "status": state["status"]})
    QUEUE.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "items": all_items},
                   ensure_ascii=False, indent=2) + "\n"
    )
    print(f"prepared_queue={len(prepared)}")
    for item in prepared:
        print(item["operator"])


if __name__ == "__main__":
    main()
