#!/usr/bin/env python3
"""Create one clean Level 3 optimization child from an explicit parent."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
MODULE_PATH = ROOT / ".codex/prepare_next_level2_version.py"
SPEC = importlib.util.spec_from_file_location("prepare_next_level_version", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"无法加载：{MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MODULE.LEVEL = ROOT / "kernel_workspace/KernelBench910B/level3"

if __name__ == "__main__":
    MODULE.main()
