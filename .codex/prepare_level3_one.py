#!/usr/bin/env python3
"""Prepare exactly one planned Level 3 operator as its _0 workspace."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
MODULE_PATH = ROOT / ".codex/prepare_level2_one.py"
SPEC = importlib.util.spec_from_file_location("prepare_level_one", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载：{MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MODULE.LEVEL_NAME = "level3"
MODULE.SOURCE = ROOT / "kernel/KernelBench910B/level3"
MODULE.WORKSPACE = ROOT / "kernel_workspace/KernelBench910B/level3"
MODULE.QUEUE = ROOT / "kernel_workspace/KernelBench910B/level3_queue.json"

if __name__ == "__main__":
    MODULE.main()
