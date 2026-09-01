import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from validate_strategy import validate


def reasoning():
    return [
        "[任务] 算子对连续输入执行逐元素数学计算并输出相同shape的结果。",
        "[现状] Process使用标量循环逐个读取输入元素并立即写回对应输出。",
        "[问题] 数学主体由Scalar循环执行，每个有效元素重复一次，总次数等于totalElements。",
        "[策略] 将数学主体改为覆盖连续范围的Vector指令链并保持相同元素映射。",
        "[推导] 有效范围为totalElements，对齐范围由UB容量和tail共同确定，Vector调用覆盖全部元素。",
        "[边界] 有效元素、输出所有权、地址范围和尾块语义均与原实现一致。",
    ]


def batch_vector_reasoning():
    items = reasoning()
    items[2] = "[问题] 连续范围被拆成totalElements次小粒度搬运事务；数学主体由Scalar循环执行totalElements次。"
    items[3] = "[策略] 将连续范围改为blockCount和blockLen确定的二维DMA；将数学主体改为覆盖连续范围的Vector指令链。"
    items[4] = "[推导] 搬运事务由totalElements降为tileCount；有效范围为totalElements、对齐范围为alignedElements，Vector调用覆盖全部元素。"
    return items


class StrategyTest(unittest.TestCase):
    def test_scalar_prefix_chain_cannot_stop_with_null_strategy(self):
        with tempfile.TemporaryDirectory(prefix="Cumprod") as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text(
                "void Process(){float carry=1; for(int i=0;i<n;++i){carry*=x(i); y(i)=carry;}}")
            path = project / "strategy.json"
            path.write_text('{"strategy":null}')
            with self.assertRaisesRegex(RuntimeError, "Scalar Prefix/Scan"):
                validate(path, project)

    def test_indirect_numeric_effect_is_not_a_direct_target(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text(
                "static constexpr int kTile=8192; void Process(){for(int t=0;t<tiles;++t){Compute(t);}}")
            value = {"strategy": {"kinds": ["resize_tile"],
                "evidence": ["op_kernel/k.cpp::Process | Process按tiles循环执行Compute | 生命周期为ceil(total/8192)"],
                "reasoning": [
                    "[任务] 算子按连续tile处理全部输入并输出相同shape结果。",
                    "[现状] Process按现有工作粒度循环执行计算与写回。",
                    "[问题] 当前工作粒度=8192产生ceil(total/8192)个任务生命周期。",
                    "[策略] 在所有权与主体不变时把工作粒度改为16384。",
                    "[推导] 峰值Buffer容量允许16384且任务数量覆盖可用物理核。",
                    "[边界] 数学语义、任务所有权、地址范围和tail保持一致。"],
                "targets": ["op_kernel/k.cpp::Process"],
                "changes": ["op_kernel/k.cpp::Process | 8192元素工作粒度 -> 16384元素工作粒度"],
                "guards": ["输出范围 | 全部有效元素只写回一次"]}}
            path = project / "strategy.json"; path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "target symbol 正文"):
                validate(path, project)

    def test_compact_strategy_and_null(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text("void Process() {}")
            path = project / "strategy.json"
            path.write_text(json.dumps({"strategy": {"kinds": ["vectorize"],
                "evidence": ["op_kernel/k.cpp::Process | 逐元素循环执行标量计算 | 循环次数为totalElements"],
                "reasoning": reasoning(),
                "targets": ["op_kernel/k.cpp::Process"],
                "changes": ["op_kernel/k.cpp::Process | 逐元素标量循环 -> 连续Vector计算"],
                "guards": ["输出范围 | 有效元素与tail均只写回原输出范围"]}}))
            validate(path, project)
            path.write_text('{"strategy":null}')
            validate(path, project)

    def test_scalar_pooling_cannot_stop_with_null_strategy(self):
        with tempfile.TemporaryDirectory(prefix="AveragePooling") as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text(
                "void Process(){for(int i=0;i<n;++i){sum += inputLocal.GetValue(i);}}")
            path = project / "strategy.json"
            path.write_text('{"strategy":null}')
            with self.assertRaisesRegex(RuntimeError, "纯 Scalar GetValue"):
                validate(path, project)

    def test_multiple_kinds_are_supported_in_deterministic_order(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text("void Process() {}")
            value = {"strategy": {"kinds": ["batch_transfer", "vectorize"],
                "evidence": ["op_kernel/k.cpp::Process | 小块搬运后逐元素计算 | 两者均执行totalElements次"],
                "reasoning": batch_vector_reasoning(), "targets": ["op_kernel/k.cpp::Process"],
                "changes": ["op_kernel/k.cpp::Process | 小块搬运和标量循环 -> 批量搬运和Vector计算"],
                "guards": ["输出范围 | 有效元素与tail均只写回原输出范围"]}}
            path = project / "strategy.json"
            path.write_text(json.dumps(value))
            validate(path, project)
            value["strategy"]["kinds"].reverse()
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "kinds 顺序非法"):
                validate(path, project)

    def test_method_contains_deterministic_kind_boundaries(self):
        method = (Path(__file__).resolve().parents[1] / "references/source-strategy-method.md").read_text()
        for rule in ("减少 GM 总字节", "DMA 事务减少", "独立输出到物理核",
                     "单 task/chunk", "阶段依赖/重叠"):
            self.assertIn(rule, method)
        self.assertIn("首次引入仅在 SDK API", method)
        self.assertIn("首次把 Scalar/Vector contraction 改为 Cube", method)
        self.assertIn("扫描全部输出路径", method)
        for problem in ("纯冗余", "重复GM", "碎片搬运", "数学主体", "已有Cube",
                        "工作粒度", "任务所有权", "流水"):
            self.assertIn(problem, method)
        self.assertIn("同类确定问题一起处理", method)
        self.assertNotIn("静态执行总次数降序", method)
        self.assertNotIn("严格覆盖条件", method)
        self.assertIn("不得把未对齐 `base[stride]`", method)
        for audit in ("语义与 dtype", "任务所有权与合并", "地址、对齐与 tail",
                      "容量与生命周期", "跨 tile 状态", "指令与 workspace",
                      "同步与 Queue", "Host/Kernel ABI"):
            self.assertIn(audit, method)
        for outcome in ("actionable", "unresolved", "clean"):
            self.assertIn(outcome, method)
        self.assertIn("不得据此宣称源码无问题", method)
        self.assertIn("partial_count", method)
        self.assertIn("单元素 Exp/Log", method)
        self.assertIn("分核 partial workspace", method)
        self.assertIn("segment local scan", method)
        for regression_guard in ("容量不等式只证明可行，不证明性能最优",
                                 "逻辑 block 数大于物理核数不等于冗余",
                                 "`SetDim(n)` 不证明必须启动 n 个 block",
                                 "只减少 `End` 文本次数"):
            self.assertIn(regression_guard, method)
        implementation = (Path(__file__).resolve().parents[3] / "kernel-knowledge/implementation-core.md").read_text()
        architecture = (Path(__file__).resolve().parents[3] / "kernel-knowledge/architecture-knowledge.md").read_text()
        self.assertIn("Buffer基址 + 动态子视图offset", implementation)
        self.assertIn("logical_blocks > physical_cores", implementation)
        self.assertIn("不把更长的 `SetTensorA/B→IterateAll→End` 生命周期当作片上驻留证据", implementation)
        self.assertIn("base[1]` 至 `base[7]", architecture)

    def test_first_cube_requires_closed_planning_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            source = project / "op_kernel/k.cpp"
            source.write_text("void Process() {}")
            strategy = {"strategy": {"kinds": ["cube"],
                "evidence": ["op_kernel/k.cpp::Process | 已有Matmul主路径 | MNK循环次数由shape确定"],
                "reasoning": [
                    "[任务] 算子对两个矩阵执行乘法并输出目标矩阵。",
                    "[现状] Process使用已有Matmul主路径完成矩阵乘法。",
                    "[问题] 已有Cube主路径存在MNK分块和重复装载成本。",
                    "[策略] 保持已有Cube主体并联合调整MNK、L1、L0、核映射和FixPipe。",
                    "[推导] baseM、baseN、baseK同时满足L0容量和独立任务公式。",
                    "[边界] 输出元素、数据类型和矩阵乘数学语义保持一致。",
                ],
                "targets": ["op_kernel/k.cpp::Process"],
                "changes": ["op_kernel/k.cpp::Process | 当前MNK分块 -> 容量约束下的目标MNK分块"],
                "guards": ["数学语义 | 输出元素与原矩阵乘定义一致"]}}
            path = project / "strategy.json"
            path.write_text(json.dumps(strategy))
            with self.assertRaisesRegex(RuntimeError, "缺少可读 planning"):
                validate(path, project)
            (project / "planning.json").write_text(json.dumps({
                "engine": "aic",
                "shape_model": ["首次引入Cube固定MNK并保持输出所有权"],
                "parameters": ["当前SDK确认Matmul API"],
                "buffers": ["L0A、L0B、L0C容量均闭合"],
                "compute": ["A重载次数和B重载次数均按完整launch计算"],
                "proofs": ["输出所有权、tail、同步和ABI均闭合"],
            }))
            validate(path, project)
            source.write_text("void Process() { Matmul<int> mm; mm.IterateAll(); }")
            validate(path, project)

    def test_evidence_must_link_target_fact_and_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text("void Process() {}")
            value = {"strategy": {"kinds": ["vectorize"],
                "evidence": ["这是一个长度足够但缺少固定分隔结构的源码证据"],
                "reasoning": reasoning(),
                "targets": ["op_kernel/k.cpp::Process"],
                "changes": ["op_kernel/k.cpp::Process | 标量循环 -> Vector循环"],
                "guards": ["输出语义 | 全部有效输出保持一致"]}}
            path = project / "strategy.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "evidence 必须使用"):
                validate(path, project)

    def test_reasoning_labels_are_fixed_and_ordered(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text("void Process() {}")
            items = reasoning()
            items[0], items[1] = items[1], items[0]
            value = {"strategy": {"kinds": ["vectorize"],
                "evidence": ["op_kernel/k.cpp::Process | 逐元素执行标量计算 | 循环次数为totalElements"],
                "reasoning": items, "targets": ["op_kernel/k.cpp::Process"],
                "changes": ["op_kernel/k.cpp::Process | 标量循环 -> Vector循环"],
                "guards": ["输出语义 | 全部有效输出保持一致"]}}
            path = project / "strategy.json"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(RuntimeError, "reasoning 必须按"):
                validate(path, project)

    def test_resize_tile_does_not_require_fixed_chinese_keywords(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text("void Process() { constexpr int tile = 64; }")
            items = reasoning()
            items[2] = "[问题] 当前工作粒度为64个元素并产生ceil(total/64)个任务生命周期。"
            items[3] = "[策略] 在所有权和计算主体不变时将工作粒度调整为128个元素。"
            items[4] = "[推导] tile从64扩大到128以减少循环次数并保持地址连续。"
            value = {"strategy": {"kinds": ["resize_tile"],
                "evidence": ["op_kernel/k.cpp::Process | 每个task只处理64个元素 | 总task数为ceil(total/64)"],
                "reasoning": items, "targets": ["op_kernel/k.cpp::Process"],
                "changes": ["op_kernel/k.cpp::Process | 每task处理64个元素 -> 每task处理128个元素"],
                "guards": ["输出覆盖 | 全部有效元素均被唯一任务覆盖"]}}
            path = project / "strategy.json"
            path.write_text(json.dumps(value))
            validate(path, project)

    def test_reasoning_accepts_clear_equivalent_terminology(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            (project / "op_kernel").mkdir()
            (project / "op_kernel/k.cpp").write_text("void Process() {}")
            items = batch_vector_reasoning()
            items[3] = items[3].replace("二维DMA", "批量复制")
            value = {"strategy": {"kinds": ["batch_transfer", "vectorize"],
                "evidence": ["op_kernel/k.cpp::Process | 小块搬运后执行Scalar循环 | 两者均执行totalElements次"],
                "reasoning": items, "targets": ["op_kernel/k.cpp::Process"],
                "changes": ["op_kernel/k.cpp::Process | 小块搬运和Scalar循环 -> 二维DMA和Vector指令链"],
                "guards": ["输出语义 | 全部有效输出保持一致"]}}
            path = project / "strategy.json"
            path.write_text(json.dumps(value))
            validate(path, project)


if __name__ == "__main__":
    unittest.main()
