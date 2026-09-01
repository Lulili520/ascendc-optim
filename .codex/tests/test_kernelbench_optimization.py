import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("queue", ROOT / ".codex/run_kernelbench_optimization.py")
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class QueueContractTest(unittest.TestCase):
    def test_limits_are_small_and_explicit(self):
        self.assertEqual(QUEUE.MAX_ROUNDS, 4)
        self.assertEqual(QUEUE.MAX_CONCURRENCY, 4)
        self.assertEqual(QUEUE.MAX_COMMAND_ATTEMPTS, 2)
        self.assertEqual(QUEUE.MAX_AGENT_ATTEMPTS, 3)
        self.assertEqual(QUEUE.MAX_IMPLEMENTATION_FIXES, 1)
        self.assertEqual(QUEUE.MAX_REPLANS, 3)
        self.assertEqual(QUEUE.MIN_IMPROVEMENT, 5.0)

    def test_precision_and_performance_share_one_device_validation_slot(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn("device_validation_busy", source)
        self.assertIn('action["kind"] == "command" and device_validation_busy', source)

    def test_children_use_process_groups_and_stage_scoped_contexts(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn("start_new_session=True", source)
        self.assertIn("os.killpg(process.pid, signal.SIGTERM)", source)
        self.assertIn('context_key = f"planning:v{action[\'version\']}"', source)
        self.assertIn('context_key = f"implementation:v{context_version}"', source)

    def test_non_null_strategy_requires_private_planning(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn("validate_planning.py", source)
        self.assertIn('"planning_output": str(project / "strategy/planning.json")', source)

    def test_precision_gate_requires_deployed_vendor(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn('VENDORS_ROOT / vendor / "bin/set_env.bash"', source)
        self.assertIn("vendor_env.is_file()", source)

    def test_strategy_null_is_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            old_workspace, old_vendors = QUEUE.WORKSPACE, QUEUE.VENDORS_ROOT
            QUEUE.WORKSPACE = Path(directory)
            QUEUE.VENDORS_ROOT = Path(directory) / "vendors"
            try:
                project = QUEUE.WORKSPACE / "Op_0"
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_host").mkdir()
                (project / "op_kernel/k.cpp").write_text("void K() {}")
                vendor_env = QUEUE.VENDORS_ROOT / "test_vendor/bin/set_env.bash"
                vendor_env.parent.mkdir(parents=True)
                vendor_env.write_text("")
                (project / "workspace.json").write_text(json.dumps({"vendor": "test_vendor"}))
                (project / "strategy").mkdir()
                (project / "strategy/strategy.json").write_text('{"strategy":null}')
                (project / "precision").mkdir()
                fp = QUEUE.fingerprint(project)
                (project / "precision/precision.json").write_text(json.dumps(
                    {"status": "PASS", "exit_code": 0, "source_fingerprint": fp}))
                (project / "performance").mkdir()
                (project / "performance/latency.json").write_text(json.dumps(
                    {"task_duration_us": 10, "source_fingerprint": fp, "accepted": True}))
                action = QUEUE.next_action("Op")
                self.assertEqual(action["state"], "stopped_no_strategy")
                self.assertIn("no currently closed source strategy", action["reason"])
            finally:
                QUEUE.WORKSPACE, QUEUE.VENDORS_ROOT = old_workspace, old_vendors

    def test_prompt_excludes_profiling_diagnosis(self):
        value = QUEUE.prompt(Path("/tmp/input.json"), {"stage": "planning"})
        self.assertIn("kernel-strategy SKILL.md", value)
        self.assertIn("一次闭合全部确定问题", value)
        self.assertIn("knowledge_contract", value)
        self.assertIn("replan_feedback", value)
        self.assertIn("planning.json与strategy.json", value)
        self.assertIn("不运行precision、performance", value)
        self.assertLess(len(value), 700)
        self.assertNotIn("bottleneck", value)
        self.assertNotIn("GM有效字节、DMA burst", value)

    def test_planning_sources_exclude_build_files_and_contract_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_host").mkdir(); (project / "op_kernel").mkdir()
            (project / "op_host/CMakeLists.txt").write_text("build")
            (project / "op_host/x.cpp").write_text("void TilingFunc(){}")
            (project / "op_kernel/k.cpp").write_text("void Process(){ ArgMax(x); }")
            self.assertEqual([path.name for path in QUEUE.source_files(project, "op_host")], ["x.cpp"])
            contract = QUEUE.planning_contract(project, "dav-2201")
            self.assertIn("reduction", contract["core"])
            self.assertNotIn("cube", contract["core"])
            self.assertEqual(set(contract["architecture"]), {"transfer", "vector", "lifetime"})

    def test_prefix_scan_contract_and_pattern_are_injected(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_host").mkdir(); (project / "op_kernel").mkdir()
            (project / "op_host/x.cpp").write_text("void TilingFunc(){}")
            (project / "op_kernel/k.cpp").write_text(
                "void Process(){float carry=1; for(int i=0;i<n;++i){carry*=x(i); y(i)=carry;}}")
            contract = QUEUE.planning_contract(project, "dav-2201")
            self.assertIn("prefix-scan", contract["core"])
            source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
            self.assertIn('"prefix_scan", "reduction"', source)

    def test_scalar_contraction_receives_cube_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_host").mkdir(); (project / "op_kernel").mkdir()
            (project / "op_host/x.cpp").write_text("void TilingFunc(){}")
            (project / "op_kernel/k.cpp").write_text(
                "void Process(){for(int i=0;i<m;i++)for(int k=0;k<n;k++)acc += a[i][k]*b[k];}")
            contract = QUEUE.planning_contract(project, "dav-2201")
            self.assertIn("cube", contract["core"])

    def test_reference_metadata_recovers_matmul_k_without_allocating_tensors(self):
        entry = {"parameters": ["A", "B"],
                 "reference_candidates": ["level1/3_Batched_matrix_multiplication.py"]}
        metadata = QUEUE.reference_input_metadata(entry)
        self.assertEqual(metadata["inputs"], [
            {"name": "A", "shape": [128, 512, 1024], "dtype": "torch.float32"},
            {"name": "B", "shape": [128, 1024, 2048], "dtype": "torch.float32"},
        ])

    def test_precision_metadata_overrides_static_reference_metadata(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn('precision.get("input_metadata") or []', source)
        self.assertIn('"inputs": metadata["inputs"]', source)

    def test_queue_uses_source_projects_as_whitelist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, workspace = root / "kernel", root / "workspace"
            source.mkdir(); workspace.mkdir()
            (source / "Wanted").mkdir()
            (workspace / "Wanted_0").mkdir()
            (workspace / "Stale_0").mkdir()
            old_source, old_workspace = QUEUE.SOURCE_PROJECTS, QUEUE.WORKSPACE
            QUEUE.SOURCE_PROJECTS, QUEUE.WORKSPACE = source, workspace
            try:
                self.assertEqual([item["operator"] for item in QUEUE.queue_items()], ["Wanted"])
                (workspace / "Wanted_0").rmdir()
                with self.assertRaisesRegex(SystemExit, "missing KernelBench910B _0"):
                    QUEUE.queue_items()
            finally:
                QUEUE.SOURCE_PROJECTS, QUEUE.WORKSPACE = old_source, old_workspace

    def test_single_operator_validation_has_dedicated_queue_mode(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn('sys.argv[1] == "--initialize-op"', source)
        self.assertIn('"mode": "single_operator_validation"', source)

    def test_command_retry_key_changes_with_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_host").mkdir(); (project / "op_kernel").mkdir()
            source = project / "op_kernel/k.cpp"
            source.write_text("void K(){int x=0;}")
            action = {"kind": "command", "stage": "precision", "version": 1, "project": project}
            before = QUEUE.stage_key(action)
            source.write_text("void K(){int x=1;}")
            after = QUEUE.stage_key(action)
            self.assertNotEqual(before, after)

    def test_known_csv_export_failure_is_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "workspace.json").write_text(json.dumps({
                "status": "PERFORMANCE_FAILED",
                "performance": {"status": "FAILED", "reason": "未生成 PipeUtilization CSV"},
            }))
            self.assertTrue(QUEUE.recoverable_performance_failure(project))

    def test_environment_failure_reads_detailed_build_log(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "precision").mkdir()
            (project / "precision/precision.json").write_text(json.dumps({
                "status": "BUILD_FAILED", "reason": "opp_build exited 1",
                "log": "precision/opp_build.log",
            }))
            (project / "precision/opp_build.log").write_text(
                "Not enough space left in TMPDIR (0 KB) to decompress package")
            self.assertTrue(QUEUE.environment_failure(project))

    def test_environment_recovery_has_dedicated_queue_mode(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn('"--recover-environment-failures": recover_environment_failures', source)
        self.assertIn('item.setdefault("stage_attempts", {}).pop(stage_key(action), None)', source)
        self.assertIn('status in {"ADAPTER_MISSING", "BUILD_FAILED", "PRECISION_FAILED", "ENVIRONMENT_FAILED"}', source)

    def test_current_device_maps_to_dav_2201(self):
        profile, refs = QUEUE.architecture_profile()
        self.assertEqual(profile, "dav-2201")
        self.assertTrue(any(path.endswith("architecture-knowledge.md") for path in refs))

    def test_reconcile_can_reopen_stale_terminal_item(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn("item.get(\"state\") in TERMINAL", source)
        self.assertIn('item.get("state") == "agent_failed"', source)
        self.assertIn('item["agent_thread_ids"] = {}', source)

    def test_terminal_cleanup_removes_only_matching_operator_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            old_tmp = QUEUE.ASCENDC_TMPDIR
            QUEUE.ASCENDC_TMPDIR = Path(directory)
            try:
                cache = QUEUE.ASCENDC_TMPDIR / "kernel_precision_reference_cache"
                cache.mkdir()
                own = cache / "Op.abc.pt"
                partial = cache / "Op.def.pt.tmp"
                other = cache / "Other.abc.pt"
                for path in (own, partial, other):
                    path.write_text("x")
                QUEUE.cleanup_cache("Op")
                self.assertFalse(own.exists())
                self.assertFalse(partial.exists())
                self.assertTrue(other.exists())
            finally:
                QUEUE.ASCENDC_TMPDIR = old_tmp

    def test_policy_export_is_terminal_triggered_not_performance_triggered(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        finish = source[source.index("def finish_item"):source.index("def export_policy_data")]
        run_queue = source[source.index("def run_queue"):source.index("def source_bases")]
        self.assertIn("export_policy_data()", finish)
        self.assertNotIn('action["stage"] == "performance" and code == 0', run_queue)

    def test_replan_request_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "strategy").mkdir()
            (project / "strategy/replan.json").write_text(json.dumps({
                "reason": "冻结tile耗尽UB且Vector子地址不对齐",
                "evidence": ["inputLocal[1]偏移4B而Vector要求32B对齐"],
                "required_design_changes": ["缩小tile并增加对齐重排Buffer"]}))
            self.assertIsNotNone(QUEUE.replan_request(project))

    def test_failed_child_returns_to_planning_without_repair_stage(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertNotIn('"stage": "repair"', source)
        self.assertIn('"kind": "replan"', source)

    def test_replan_history_has_a_hard_limit(self):
        source = (ROOT / ".codex/run_kernelbench_optimization.py").read_text()
        self.assertIn('len(item.get("replan_history", [])) >= MAX_REPLANS', source)
        self.assertIn('"implementation_blocked"', source)


if __name__ == "__main__":
    unittest.main()
