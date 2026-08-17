#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_source_effect.py"


def module():
    spec = importlib.util.spec_from_file_location("source_effect_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class SourceEffectTest(unittest.TestCase):
    def test_target_change_is_hard_but_pattern_count_is_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent, child = Path(directory) / "Op_0", Path(directory) / "Op_1"
            for project in (parent, child):
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_host").mkdir()
            (parent / "bottleneck").mkdir()
            (parent / "strategy").mkdir()
            (parent / "op_kernel/x.cpp").write_text("void Process(){x.GetValue(0);}\n")
            (child / "op_kernel/x.cpp").write_text("void Process(){x.GetValue(0); x.GetValue(1);}\n")
            (parent / "bottleneck/bottleneck.json").write_text(json.dumps({
                "issues": [{"bottleneck": {"cause_key": "scalar_local_lane_compute"}}]
            }))
            (parent / "strategy/strategy.json").write_text(json.dumps({
                "strategies": [{"actions": [{"target": "op_kernel/x.cpp::Process"}]}]
            }))
            warnings = module().validate(parent, child)
            self.assertTrue(any("GetValue" in warning for warning in warnings))

    def test_comment_only_change_outside_target_symbol_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent, child = Path(directory) / "Op_0", Path(directory) / "Op_1"
            for project in (parent, child):
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_host").mkdir()
            (parent / "bottleneck").mkdir()
            (parent / "strategy").mkdir()
            (parent / "op_kernel/x.cpp").write_text("void Process(){run();}\n")
            (child / "op_kernel/x.cpp").write_text("// changed\nvoid Process(){run();}\n")
            (parent / "bottleneck/bottleneck.json").write_text(json.dumps({
                "issues": [{"bottleneck": {"cause_key": "repeated_hot_path_overhead"}}]
            }))
            (parent / "strategy/strategy.json").write_text(json.dumps({
                "strategies": [{"actions": [{"target": "op_kernel/x.cpp::Process"}]}]
            }))
            with self.assertRaisesRegex(RuntimeError, "symbol 未发生"):
                module().validate(parent, child)


if __name__ == "__main__":
    unittest.main()
