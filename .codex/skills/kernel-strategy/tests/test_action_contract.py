#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / ".codex/skills/kernel-strategy/scripts"
sys.path.insert(0, str(SCRIPTS))


def validator():
    spec = importlib.util.spec_from_file_location("strategy_validator_test", SCRIPTS / "validate_strategy.py")
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


class ActionContractTest(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        project = root / "Op_0"
        for folder in ("bottleneck", "strategy", "performance", "op_kernel", "op_host"):
            (project / folder).mkdir(parents=True, exist_ok=True)
        (project / "op_kernel/x.cpp").write_text("void Process(){ run(); }\n")
        (project / "performance/performance.json").write_text("{}")
        bottleneck = {
            "reasoning": ["Process 存在标量归约。"],
            "issues": [{
                "evidence": [{"evidence_key": "source.kernel.scalar_reduction",
                              "source": "op_kernel/x.cpp::Process", "observation": "循环内标量累加"}],
                "bottleneck": {"bottleneck_key": "compute.scalar_inefficiency",
                               "cause_key": "scalar_reduction"},
            }],
        }
        strategy = {
            "reasoning": ["向量化归约。"],
            "strategies": [{
                "cause_key": "scalar_reduction",
                "strategy_key": "compute.vectorize_scalar_work",
                "actions": [{"target": "op_kernel/x.cpp::Process",
                             "operation": "vectorize_scalar_work",
                             "edits": ["将标量循环改为向量归约"],
                             "constraints": ["保持数学语义和输出 ABI"]}],
            }],
        }
        bp, sp = project / "bottleneck/bottleneck.json", project / "strategy/strategy.json"
        bp.write_text(json.dumps(bottleneck, ensure_ascii=False))
        sp.write_text(json.dumps(strategy, ensure_ascii=False))
        return bp, sp

    def test_minimal_executable_strategy_passes_without_text_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            bp, sp = self.fixture(Path(directory))
            validator().validate(bp, sp)

    def test_wrong_fixed_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bp, sp = self.fixture(Path(directory))
            value = json.loads(sp.read_text())
            value["strategies"][0]["strategy_key"] = "memory.coalesce_transfer"
            sp.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "固定映射"):
                validator().validate(bp, sp)

    def test_missing_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bp, sp = self.fixture(Path(directory))
            value = json.loads(sp.read_text())
            value["strategies"][0]["actions"][0]["target"] = "op_kernel/missing.cpp::Process"
            sp.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "target 文件"):
                validator().validate(bp, sp)

    def test_derive_covers_issues_in_order(self):
        from derive_strategy import derive
        report = {"issues": [
            {"bottleneck": {"cause_key": "scalar_reduction"}},
            {"bottleneck": {"cause_key": "serial_pipeline_stages"}},
        ]}
        self.assertEqual(
            [item["cause_key"] for item in derive(report)["strategies"]],
            ["scalar_reduction", "serial_pipeline_stages"],
        )

    def test_more_than_six_actions_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            bp, sp = self.fixture(Path(directory))
            value = json.loads(sp.read_text())
            action = value["strategies"][0]["actions"][0]
            value["strategies"][0]["actions"] = [
                {**action, "target": f"op_kernel/x.cpp::Process{i}"} for i in range(7)
            ]
            source = "void Process(){ run(); }\n" + "\n".join(
                f"void Process{i}(){{ run(); }}" for i in range(7)
            )
            (bp.parent.parent / "op_kernel/x.cpp").write_text(source)
            sp.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "最多 6"):
                validator().validate(bp, sp)


if __name__ == "__main__":
    unittest.main()
