#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_report.py"


def module():
    spec = importlib.util.spec_from_file_location("source_first_validator", SCRIPT)
    assert spec and spec.loader
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class SourceFirstContractTest(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        project = root / "Op_0"
        for folder in ("bottleneck", "performance", "op_kernel", "op_host"):
            (project / folder).mkdir(parents=True, exist_ok=True)
        (project / "op_kernel/x.cpp").write_text(
            "void Process() { for (int i=0;i<n;i++) { auto v=x.GetValue(i); } }\n"
        )
        (project / "op_host/x.cpp").write_text("void TilingFunc() { int tile = 64; }\n")
        (project / "performance/performance.json").write_text(json.dumps({
            "pipeline": {"aiv_scalar_ratio": 0.01},
        }))
        report = {
            "reasoning": [
                "Process 的生产循环按连续索引逐元素读取 GlobalTensor，具体根因为 scalar_global_contiguous_access。"
            ],
            "issues": [{
                "evidence": [{
                    "evidence_key": "source.kernel.scalar_global_contiguous_access",
                    "source": "op_kernel/x.cpp::Process",
                    "observation": "生产循环按连续索引逐元素调用 GetValue",
                }],
                "bottleneck": {
                    "bottleneck_key": "memory.transfer_inefficiency",
                    "cause_key": "scalar_global_contiguous_access",
                },
            }],
        }
        path = project / "bottleneck/bottleneck.json"
        path.write_text(json.dumps(report, ensure_ascii=False))
        return path

    def test_report_requires_no_auxiliary_analysis_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory))
            module().validate(path)
            self.assertFalse((path.parent / "coverage.json").exists())
            self.assertFalse((path.parent / "source_model.json").exists())

    def test_source_issue_does_not_require_pipeline_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            module().validate(self.fixture(Path(directory)))

    def test_rejects_ambiguous_legacy_cause(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory))
            report = json.loads(path.read_text())
            report["issues"][0]["bottleneck"]["cause_key"] = "scalar_tensor_access"
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "cause 未知"):
                module().validate(path)

    def test_evidence_target_only_requires_a_real_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory))
            project = path.parent.parent
            (project / "op_kernel/x.cpp").write_text(
                "void Process() { run(); }\nvoid Debug() { x.GetValue(0); }\n"
            )
            module().validate(path)

    def test_multiple_distinct_causes_on_same_symbol_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory))
            report = json.loads(path.read_text())
            report["issues"].append({
                "evidence": [{
                    "evidence_key": "source.kernel.fragmented_contiguous_transfer",
                    "source": "op_kernel/x.cpp::Process",
                    "observation": "同一连续区间被拆成多次小搬运",
                }],
                "bottleneck": {
                    "bottleneck_key": "memory.transfer_inefficiency",
                    "cause_key": "fragmented_contiguous_transfer",
                },
            })
            report["reasoning"].append("Process 同一连续访问被重复归因。")
            path.write_text(json.dumps(report, ensure_ascii=False))
            module().validate(path)

    def test_more_than_three_issues_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory))
            report = json.loads(path.read_text())
            additions = [
                ("fragmented_contiguous_transfer", "source.kernel.fragmented_contiguous_transfer"),
                ("scalar_elementwise_compute", "source.kernel.scalar_elementwise_compute"),
                ("over_synchronization", "source.kernel.over_synchronization"),
            ]
            for cause, key in additions:
                report["issues"].append({
                    "evidence": [{"evidence_key": key, "source": "op_kernel/x.cpp::Process", "observation": cause}],
                    "bottleneck": {"bottleneck_key": module().CAUSES[cause][0], "cause_key": cause},
                })
                report["reasoning"].append(cause)
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError, "最多输出 3"):
                module().validate(path)

    def test_symbol_locator_supports_struct_and_tiling_macro(self) -> None:
        value = module()
        self.assertTrue(value._symbol_bodies("struct Tiling : Base { int x; };", "Tiling"))
        self.assertTrue(value._symbol_bodies(
            "BEGIN_TILING_DATA_DEF(MyTiling)\nTILING_DATA_FIELD_DEF(uint32_t, x);\nEND_TILING_DATA_DEF;",
            "MyTiling",
        ))

    def test_rejects_scalar_reduction_on_vector_state_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.fixture(Path(directory))
            project = path.parent.parent
            (project / "op_kernel/x.cpp").write_text(
                "void Process(){ LocalTensor<float> row,max; LocalTensor<uint8_t> mask; "
                "for(int r=0;r<n;++r){ Compare(mask,row,max,0,64); Select(max,mask,row,max,0,64); }}\n"
            )
            report = json.loads(path.read_text())
            report["reasoning"] = ["Process 使用标量归约。"]
            report["issues"] = [{
                "evidence": [{"evidence_key": "source.kernel.scalar_reduction",
                              "source": "op_kernel/x.cpp::Process", "observation": "逐行更新状态"}],
                "bottleneck": {"bottleneck_key": "compute.scalar_inefficiency",
                               "cause_key": "scalar_reduction"},
            }]
            path.write_text(json.dumps(report, ensure_ascii=False))
            with self.assertRaisesRegex(RuntimeError, "LocalTensor Vector"):
                module().validate(path)


if __name__ == "__main__":
    unittest.main()
