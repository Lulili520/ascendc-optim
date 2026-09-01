import hashlib, json, sys, tempfile, unittest
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from validate_planning import validate

KEYS = ("gm_bytes", "dma_bursts", "vector_repeat_blocks", "cube_ops", "scalar_iterations", "task_lifecycles", "state_initializations", "sync_events", "writeback_bursts")

def fingerprint_for(project):
    digest = hashlib.sha256()
    for path in sorted(path for folder in ("op_host", "op_kernel") for path in (project / folder).rglob("*") if path.is_file()):
        digest.update(path.relative_to(project).as_posix().encode()); digest.update(b"\0"); digest.update(path.read_bytes())
    return digest.hexdigest()

def planning_value(digest, pattern, current, target):
    return {"source_fingerprint": digest, "pattern": pattern, "engine": "aic" if pattern == "existing_cube" else "aiv", "available_cores": 48, "used_cores": 24, "shape_model": ["N=16"], "task_mapping": "task=tile", "parameters": ["tile=16 | N"], "buffers": ["x | fp16 | bytes=32 | Process"], "transfers": ["GM->UB | bytes=32 | tx=1"], "compute": ["Vector Add | count=16"], "work": {"current": current, "target": target}, "proofs": ["capacity | 32<=UB"]}

class PlanningTest(unittest.TestCase):
    def test_prefix_scan_must_replace_element_scalar_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory); (project / "op_host").mkdir(); (project / "op_kernel").mkdir()
            (project / "op_host/h.cpp").write_text("void H(){}")
            (project / "op_kernel/k.cpp").write_text("void K(){}")
            current = {key: 100 for key in KEYS}; current["vector_repeat_blocks"] = 0
            target = dict(current); target["scalar_iterations"] = 10; target["vector_repeat_blocks"] = 20
            value = planning_value(fingerprint_for(project), "prefix_scan", current, target)
            path = project / "planning.json"; path.write_text(json.dumps(value)); validate(path, project)
            value["work"]["target"]["scalar_iterations"] = 90
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "分块 carry"):
                validate(path, project)

    def test_complete_work_vector_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory); (project / "op_host").mkdir(); (project / "op_kernel").mkdir()
            (project / "op_host/h.cpp").write_text("void H(){}")
            (project / "op_kernel/k.cpp").write_text("void K(){}")
            digest = hashlib.sha256()
            for path in sorted([project / "op_host/h.cpp", project / "op_kernel/k.cpp"]):
                digest.update(path.relative_to(project).as_posix().encode()); digest.update(b"\0"); digest.update(path.read_bytes())
            keys = ("gm_bytes", "dma_bursts", "vector_repeat_blocks", "cube_ops", "scalar_iterations", "task_lifecycles", "state_initializations", "sync_events", "writeback_bursts")
            current = {key: 1 for key in keys}; target = dict(current); target["dma_bursts"] = 0
            value = {"source_fingerprint": digest.hexdigest(), "pattern": "elementwise", "engine": "aiv", "available_cores": 48, "used_cores": 48, "shape_model": ["N=16"], "task_mapping": "task=tile", "parameters": ["tile=16 | N"], "buffers": ["x | fp16 | bytes=32 | Process"], "transfers": ["GM->UB | bytes=32 | tx=1"], "compute": ["Vector Add | count=16"], "work": {"current": current, "target": target}, "proofs": ["capacity | 32<=UB"]}
            path = project / "planning.json"; path.write_text(json.dumps(value)); validate(path, project)
            value["work"]["target"] = dict(current); path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "不得与当前完全相同"): validate(path, project)

    def test_pooling_must_reduce_scalar_math_not_only_dma(self):
        with tempfile.TemporaryDirectory(prefix="AveragePooling") as directory:
            project = Path(directory); (project / "op_host").mkdir(); (project / "op_kernel").mkdir()
            (project / "op_host/h.cpp").write_text("void H(){}")
            (project / "op_kernel/k.cpp").write_text("void K(){}")
            digest = fingerprint_for(project)
            keys = ("gm_bytes", "dma_bursts", "vector_repeat_blocks", "cube_ops", "scalar_iterations", "task_lifecycles", "state_initializations", "sync_events", "writeback_bursts")
            current = {key: 100 for key in keys}; current["vector_repeat_blocks"] = 0
            target = dict(current); target["dma_bursts"] = 10
            value = planning_value(digest, "window_pooling", current, target)
            path = project / "planning.json"; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "只优化 GM/DMA"): validate(path, project)

    def test_cube_requires_complete_abc_reuse_account(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory); (project / "op_host").mkdir(); (project / "op_kernel").mkdir()
            (project / "op_host/h.cpp").write_text("void H(){}")
            (project / "op_kernel/k.cpp").write_text("void K(){}")
            keys = ("gm_bytes", "dma_bursts", "vector_repeat_blocks", "cube_ops", "scalar_iterations", "task_lifecycles", "state_initializations", "sync_events", "writeback_bursts")
            current = {key: 100 for key in keys}; target = dict(current); target["task_lifecycles"] = 50
            value = planning_value(fingerprint_for(project), "existing_cube", current, target)
            path = project / "planning.json"; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "A/B/C 复用账"): validate(path, project)


if __name__ == "__main__": unittest.main()
