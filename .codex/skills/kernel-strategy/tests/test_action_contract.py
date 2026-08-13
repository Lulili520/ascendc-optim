#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".codex/skills/kernel-strategy/scripts"
sys.path.insert(0, str(SCRIPTS))


def load_validator():
    path = SCRIPTS / "validate_strategy.py"
    spec = importlib.util.spec_from_file_location("kernel_strategy_validator_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ActionContractTest(unittest.TestCase):
    def test_edit_accepts_source_object_and_transform(self) -> None:
        validator = load_validator()
        validator.validate_edit(
            1,
            "Process 中逐元素 GetValue 累加循环：改为 UB 分块搬入和 Vector Reduce",
        )

    def test_edit_rejects_operation_paraphrase(self) -> None:
        validator = load_validator()
        with self.assertRaisesRegex(RuntimeError, "只重复 operation"):
            validator.validate_edit(1, "提高并行度")

    def test_edit_rejects_missing_transform(self) -> None:
        validator = load_validator()
        with self.assertRaisesRegex(RuntimeError, "缺少明确变换"):
            validator.validate_edit(1, "Process 中的 GetValue 标量归约循环：保持原实现")


if __name__ == "__main__":
    unittest.main()
