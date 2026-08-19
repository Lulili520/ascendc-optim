#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/source_facts.py"


def module():
    spec = importlib.util.spec_from_file_location("source_facts_test", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(value)
    return value


class SourceFactsTest(unittest.TestCase):
    def test_many_small_tasks_require_mapping_and_work_unit_causes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for folder in ("op_host", "op_kernel", "performance"):
                (project / folder).mkdir()
            (project / "op_host/x.cpp").write_text(
                "static constexpr uint32_t TILE_INNER = 64;\n"
            )
            (project / "op_kernel/x.cpp").write_text(
                "void Process(){ auto x = GetBlockIdx(); }\n"
            )
            (project / "performance/performance.json").write_text(json.dumps({
                "task": {"Block Num": "8192", "Input Shapes": "128,4096,4095"},
                "per_core": {"active_cores": 48},
                "hardware": {"ub_bytes_per_core": 196352, "aiv_core_count": 48},
            }))
            facts = module().extract(project)
            self.assertEqual(facts["waves"], 171)
            self.assertEqual(module().candidate_causes(facts), {
                "inefficient_work_unit_size"
            })

    def test_scalar_strided_reduction_and_writeback_are_not_hidden_by_tiling(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for folder in ("op_host", "op_kernel", "performance"):
                (project / folder).mkdir()
            (project / "op_host/x.cpp").write_text(
                "static constexpr uint32_t BATCH=128, REDUCE_DIM=4096, INNER_DIM=4095, TILE_INNER=64;\n"
            )
            (project / "op_kernel/x.cpp").write_text(
                "__gm__ const float* xPtr; __gm__ int64_t* yPtr;\n"
                "void Process(){ for(unsigned j=0;j<64;++j){ float curMax=-1; unsigned curArg=0; "
                "for(unsigned r=0;r<4096;++r){ float v=xPtr[r*4095+j]; "
                "if(v>curMax){curMax=v;curArg=r;} } yPtr[j]=(int64_t)curArg; }}\n"
            )
            (project / "performance/performance.json").write_text(json.dumps({
                "task": {"Block Num": "8192", "Input Shapes": "128,4096,4095"},
                "per_core": {"active_cores": 48},
                "hardware": {"ub_bytes_per_core": 196352, "aiv_core_count": 48},
            }))
            facts = module().extract(project)
            self.assertTrue(facts["has_scalar_reduction"])
            self.assertTrue(facts["gm_scalar_strided_reads"])
            self.assertTrue(facts["gm_scalar_writes"])
            self.assertTrue({
                "fragmented_regular_strided_transfer", "scalar_reduction",
                "fragmented_global_writeback",
            } <= module().candidate_causes(facts))

    def test_vector_compare_select_loop_is_not_scalar_reduction(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for folder in ("op_host", "op_kernel", "performance"):
                (project / folder).mkdir()
            (project / "op_host/x.cpp").write_text(
                "static constexpr uint32_t BATCH=128, REDUCE_DIM=4096, INNER_DIM=4095, TILE_INNER=64;\n"
            )
            (project / "op_kernel/x.cpp").write_text(
                "void Process(){ LocalTensor<float> row,max; LocalTensor<uint8_t> mask; "
                "for(unsigned r=0;r<4096;++r){ Compare(mask,row,max,CMPMODE::GT,64); "
                "Select(max,mask,row,max,SELMODE::VSEL_TENSOR_TENSOR_MODE,64); }}\n"
            )
            (project / "performance/performance.json").write_text(json.dumps({
                "task": {"Block Num": "8192", "Input Shapes": "128,4096,4095"},
                "per_core": {"active_cores": 48},
                "hardware": {"ub_bytes_per_core": 196352, "aiv_core_count": 48},
            }))
            facts = module().extract(project)
            self.assertTrue(facts["has_vector_state_update"])
            self.assertTrue(facts["scalar_reduction_excluded"])
            self.assertFalse(facts["has_scalar_reduction"])
            self.assertNotIn("scalar_reduction", module().candidate_causes(facts))


if __name__ == "__main__":
    unittest.main()
