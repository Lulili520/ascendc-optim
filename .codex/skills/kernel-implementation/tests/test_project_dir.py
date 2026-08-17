#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_implementation.py"


def module():
    spec = importlib.util.spec_from_file_location("implementation_project_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ExplicitProjectDirTest(unittest.TestCase):
    def test_parent_strategy_validates_child_sources(self) -> None:
        validator = module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            (parent / "strategy").mkdir(parents=True)
            (parent / "bottleneck").mkdir()
            (child / "strategy").mkdir(parents=True)
            (child / "op_kernel").mkdir()
            (child / "op_kernel/x.cpp").write_text("void Process(){new_work();}\n")
            bottleneck = parent / "bottleneck/bottleneck.json"
            strategy = parent / "strategy/strategy.json"
            implementation = child / "strategy/implementation.json"
            bottleneck.write_text("{}")
            strategy.write_text(json.dumps({"strategies": [{
                "strategy_key": "compute.vectorize_scalar_work",
                "actions": [{"target": "op_kernel/x.cpp::Process"}],
            }]}))
            implementation.write_text(json.dumps({
                "strategy_keys": ["compute.vectorize_scalar_work"],
                "reasoning": ["implemented"],
                "actions": [{"action_index": 1, "implementation_summary": "changed Process"}],
                "modified_files": ["op_kernel/x.cpp"],
            }))
            with patch.object(validator, "validate_strategy"), patch.object(
                validator, "validate_source_effect"
            ) as source_effect:
                validator.validate(
                    bottleneck, strategy, implementation,
                    parent_path=parent, project_path=child,
                )
            source_effect.assert_called_once_with(parent.resolve(), child.resolve())


if __name__ == "__main__":
    unittest.main()
