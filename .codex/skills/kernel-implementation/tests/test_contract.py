import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_implementation.py"
SOURCE_EFFECT_SCRIPT = SCRIPT.with_name("validate_source_effect.py")
SOURCE_EFFECT_SPEC = importlib.util.spec_from_file_location("validate_source_effect", SOURCE_EFFECT_SCRIPT)
SOURCE_EFFECT = importlib.util.module_from_spec(SOURCE_EFFECT_SPEC)
SOURCE_EFFECT_SPEC.loader.exec_module(SOURCE_EFFECT)


class ImplementationContractTest(unittest.TestCase):
    def test_cube_pipeline_accepts_changed_iterate_submit_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent, child = root / "Op_0", root / "Op_1"
            for project, body in (
                (parent, "void Process(){for(int i=0;i<2;i++){mm.IterateAll(out);mm.End();}}"),
                (child, "void Process(){for(int i=0;i<2;i++){mm.IterateAll(out);}mm.End();}"),
            ):
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_kernel/k.cpp").write_text("Matmul mm;" + body)
            strategy = root / "strategy.json"
            strategy.write_text(json.dumps({"strategy": {
                "kinds": ["pipeline"], "targets": ["op_kernel/k.cpp::Process"],
                "reasoning": ["", "", "", "[策略] 合并Cube IterateAll与End提交边界。", "", ""],
            }}))
            SOURCE_EFFECT.validate(strategy, parent, child)

    def test_final_structure_may_live_outside_changed_host_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for project, tile in ((parent, 128), (child, 256)):
                (project / "op_host").mkdir(parents=True)
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_host/x.cpp").write_text(
                    f"void TilingFunc(){{unsigned tile={tile};}}")
                (project / "op_kernel/x.cpp").write_text(
                    "void CopyIn(){DataCopy(dst,src,n);} void K(){Matmul mm;mm.IterateAll(out);}")
            strategy = root / "strategy.json"
            strategy.write_text(json.dumps({"strategy": {
                "kinds": ["batch_transfer", "cube"],
                "evidence": ["op_host/x.cpp::TilingFunc | 当前tile为128 | 单任务处理128元素"],
                "reasoning": [
                    "[任务] Host为已有Cube主路径下发固定任务粒度并保持输出所有权。",
                    "[现状] Kernel已有DataCopy和Matmul IterateAll路径且Host固定tile为128。",
                    "[问题] 当前Host粒度使规则连续块产生多余搬运事务和Cube任务生命周期。",
                    "[策略] Host将统一粒度改为256并复用Kernel中已有搬运与Cube主路径。",
                    "[推导] tile由容量和任务覆盖共同确定且最终源码保留DataCopy与Matmul路径。",
                    "[边界] Host Kernel ABI、地址范围、Cube语义和输出所有权保持一致。",
                ],
                "targets": ["op_host/x.cpp::TilingFunc"],
                "changes": ["op_host/x.cpp::TilingFunc | tile为128 -> tile为256"],
                "guards": ["最终结构 | 完整源码保留DataCopy与Cube主路径"],
            }}))
            record = root / "implementation.json"
            summary = "op_host/x.cpp::TilingFunc | 已将统一任务粒度从128改为256"
            record.write_text(json.dumps({
                "strategy_kinds": ["batch_transfer", "cube"], "summary": [summary],
                "modified_files": ["op_host/x.cpp"], "attempts": [{
                    "attempt": 1, "kind": "initial", "trigger": None,
                    "summary": summary, "modified_files": ["op_host/x.cpp"],
                }],
            }))
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--strategy", str(strategy),
                "--implementation", str(record), "--parent", str(parent),
                "--project-dir", str(child), "--require-attempts",
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_file_scope_constant_target_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for project, body in ((parent, "static constexpr unsigned kBlocks = 24;"),
                                  (child, "static constexpr unsigned kBlocks = 48;")):
                (project / "op_host").mkdir(parents=True)
                (project / "op_host/x.cpp").write_text(body)
            strategy = root / "strategy.json"
            strategy.write_text(json.dumps({"strategy": {"kinds": ["parallelize"],
                "evidence": ["op_host/x.cpp::kBlocks | 当前固定使用24核 | 设备AIV核数为48"],
                "reasoning": ["[任务] 算子将独立元素连续划分到Vector核并输出对应结果。",
                    "[现状] Host使用文件级常量kBlocks固定下发24个Vector任务。",
                    "[问题] 当前任务所有权只使用24个任务，未覆盖48个可用AIV。",
                    "[策略] 将文件级核数常量改为48，建立覆盖全部可用AIV的任务所有权。",
                    "[推导] blockDim等于min(48,N)，每个任务拥有互斥连续区间。",
                    "[边界] 全部元素保持唯一所有权且Host与Kernel核数契约一致。"],
                "targets": ["op_host/x.cpp::kBlocks"],
                "changes": ["op_host/x.cpp::kBlocks | 固定24核 -> 固定48核"],
                "guards": ["任务所有权 | 连续区间互斥并覆盖全部元素"]}}))
            record = root / "implementation.json"
            summary = "op_host/x.cpp::kBlocks | 已将文件级任务核数常量从24改为48"
            record.write_text(json.dumps({"strategy_kinds": ["parallelize"], "summary": [summary],
                "modified_files": ["op_host/x.cpp"], "attempts": [{"attempt": 1,
                    "kind": "initial", "trigger": None, "summary": summary,
                    "modified_files": ["op_host/x.cpp"]}]}))
            result = subprocess.run([sys.executable, str(SCRIPT), "--strategy", str(strategy),
                "--implementation", str(record), "--parent", str(parent),
                "--project-dir", str(child), "--require-attempts"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_changed_target_and_compact_record_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for project, body in ((parent, "void K(){int x=0;}"), (child, "void K(){Add(dst,src0,src1,n);}")):
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_kernel/k.cpp").write_text(body)
            strategy = root / "strategy.json"
            strategy.write_text(json.dumps({"strategy": {"kinds": ["vectorize"],
                "evidence": ["op_kernel/k.cpp::K | 标量循环逐元素计算 | 执行n次"],
                "reasoning": [
                    "[任务] 算子对连续输入执行逐元素数学计算并输出相同shape的结果。",
                    "[现状] K使用标量循环逐个读取输入元素并立即写回对应输出。",
                    "[问题] 数学主体由Scalar循环执行，每个有效元素重复一次，静态次数等于n。",
                    "[策略] 将数学主体改为覆盖连续范围的Vector指令链并保持相同元素映射。",
                    "[推导] 有效范围为n，对齐范围由UB容量和tail确定，Vector调用覆盖全部元素。",
                    "[边界] 有效元素、输出所有权、地址范围和尾块语义均与原实现一致。",
                ],
                "targets": ["op_kernel/k.cpp::K"],
                "changes": ["op_kernel/k.cpp::K | 标量循环 -> 连续Vector计算"],
                "guards": ["访问范围 | 输入输出均保持在有效边界内"]}}))
            record = root / "implementation.json"
            summary = "op_kernel/k.cpp::K | 已将标量循环实现替换为连续Vector实现"
            record.write_text(json.dumps({"strategy_kinds": ["vectorize"], "summary": [summary],
                "modified_files": ["op_kernel/k.cpp"], "attempts": [{"attempt": 1,
                "kind": "initial", "trigger": None, "summary": summary,
                "modified_files": ["op_kernel/k.cpp"]}]}))
            result = subprocess.run([sys.executable, str(SCRIPT), "--strategy", str(strategy),
                "--implementation", str(record), "--parent", str(parent),
                "--project-dir", str(child), "--require-attempts"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_vectorize_accepts_proven_scalar_math_elimination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for project, body in (
                (parent, "void K(){for(int i=0;i<8;i++)for(int j=0;j<8;j++)a[i][j]=dot(i,j);}"),
                (child, "void K(){for(int i=0;i<8;i++)for(int j=i;j<8;j++){a[i][j]=dot(i,j);a[j][i]=a[i][j];}}"),
            ):
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_kernel/k.cpp").write_text(body)
            strategy_dir = root / "strategy"
            strategy_dir.mkdir()
            strategy = strategy_dir / "strategy.json"
            strategy.write_text(json.dumps({"strategy": {
                "kinds": ["vectorize"],
                "evidence": ["op_kernel/k.cpp::K | 对称矩阵完整计算上下三角 | Scalar dot由64次降到36次"],
                "reasoning": [
                    "[任务] 算子计算固定八阶对称矩阵并输出全部六十四个元素。",
                    "[现状] K为上下三角的每个位置分别执行一次Scalar dot计算。",
                    "[问题] 对称位置重复执行相同Scalar数学主体并产生六十四次dot。",
                    "[策略] 只计算上三角Scalar dot并将同一结果镜像到下三角。",
                    "[推导] 完整launch的Scalar迭代由512下降为288且不转移到其他路径。",
                    "[边界] 每个保留dot的累加顺序、全部输出和矩阵对称语义保持一致。",
                ],
                "targets": ["op_kernel/k.cpp::K"],
                "changes": ["op_kernel/k.cpp::K | 完整8乘8独立dot -> 36个上三角dot并镜像"],
                "guards": ["数学语义 | 对称位置共享同一个已计算结果"],
            }}))
            (strategy_dir / "planning.json").write_text(json.dumps({
                "work": {"current": {"scalar_iterations": 512},
                         "target": {"scalar_iterations": 288}},
            }))
            record = root / "implementation.json"
            summary = "op_kernel/k.cpp::K | 已仅计算上三角dot并镜像下三角"
            record.write_text(json.dumps({
                "strategy_kinds": ["vectorize"], "summary": [summary],
                "modified_files": ["op_kernel/k.cpp"], "attempts": [{
                    "attempt": 1, "kind": "initial", "trigger": None,
                    "summary": summary, "modified_files": ["op_kernel/k.cpp"],
                }],
            }))
            SOURCE_EFFECT.validate(strategy, parent, child)

    def test_macro_generated_tiling_target_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            before = "BEGIN_TILING_DATA_DEF(OpTilingData)\nTILING_DATA_FIELD_DEF(uint32_t, oldField);\nEND_TILING_DATA_DEF;"
            after = "BEGIN_TILING_DATA_DEF(OpTilingData)\nTILING_DATA_FIELD_DEF(uint32_t, newField);\nEND_TILING_DATA_DEF;"
            for project, body in ((parent, before), (child, after)):
                (project / "op_host").mkdir(parents=True)
                (project / "op_host/tiling.h").write_text(body)
            strategy = root / "strategy.json"
            strategy.write_text(json.dumps({"strategy": {"kinds": ["resize_tile"],
                "evidence": ["op_host/tiling.h::OpTilingData | tiling字段包含oldField | 每次launch传递1个字段"],
                "reasoning": [
                    "[任务] 算子使用Host tiling向Kernel传递单任务处理参数并保持输出语义。",
                    "[现状] OpTilingData声明oldField并由当前任务划分读取该字段。",
                    "[问题] 当前工作粒度字段未表达容量与任务数共同确定的目标粒度。",
                    "[策略] 在所有权和主体不变时将工作粒度改为newField并保持ABI一致。",
                    "[推导] newField由UB容量和总任务数共同确定并保持任务数覆盖可用核。",
                    "[边界] 字段顺序、字段类型、任务覆盖和Host Kernel契约保持一致。",
                ], "targets": ["op_host/tiling.h::OpTilingData"],
                "changes": ["op_host/tiling.h::OpTilingData | oldField -> newField"],
                "guards": ["Host Kernel ABI | tiling字段定义与读取保持一致"]}}))
            record = root / "implementation.json"
            summary = "op_host/tiling.h::OpTilingData | 已将oldField替换为newField"
            record.write_text(json.dumps({"strategy_kinds": ["resize_tile"], "summary": [summary],
                "modified_files": ["op_host/tiling.h"], "attempts": [{"attempt": 1,
                "kind": "initial", "trigger": None, "summary": summary,
                "modified_files": ["op_host/tiling.h"]}]}))
            result = subprocess.run([sys.executable, str(SCRIPT), "--strategy", str(strategy),
                "--implementation", str(record), "--parent", str(parent),
                "--project-dir", str(child), "--require-attempts"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_two_dimensional_dma_rejects_fixed_single_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for project, body in ((parent, "void K(){int x=0;}"),
                                  (child, "void K(){DataCopyExtParams p{1,32,0,0,0};DataCopyPad(dst,src,p);}")):
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_kernel/k.cpp").write_text(body)
            strategy = root / "strategy.json"
            strategy.write_text(json.dumps({"strategy": {"kinds": ["batch_transfer"],
                "evidence": ["op_kernel/k.cpp::K | 连续行被逐行搬运 | 搬运调用次数为rows"],
                "reasoning": [
                    "[任务] 算子搬运连续输入行并保持相同输出布局。",
                    "[现状] K对每个输入行分别执行一次小粒度搬运。",
                    "[问题] 连续行被拆成rows次小粒度搬运事务。",
                    "[策略] 将连续行改为blockCount和blockLen确定的二维DMA。",
                    "[推导] 搬运事务由rows降为ceil(rows/chunkRows)，批量大小受容量和任务数约束。",
                    "[边界] 每个有效字节只搬运一次且输入输出地址均不越界。"],
                "targets": ["op_kernel/k.cpp::K"],
                "changes": ["op_kernel/k.cpp::K | 逐行搬运 -> 多行二维DMA"],
                "guards": ["地址范围 | 全部有效行均被完整覆盖"]}}))
            record = root / "implementation.json"
            summary = "op_kernel/k.cpp::K | 已声明使用二维DMA处理连续行"
            record.write_text(json.dumps({"strategy_kinds": ["batch_transfer"], "summary": [summary],
                "modified_files": ["op_kernel/k.cpp"], "attempts": [{"attempt": 1,
                "kind": "initial", "trigger": None, "summary": summary,
                "modified_files": ["op_kernel/k.cpp"]}]}))
            result = subprocess.run([sys.executable, str(SCRIPT), "--strategy", str(strategy),
                "--implementation", str(record), "--parent", str(parent),
                "--project-dir", str(child), "--require-attempts"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("blockCount=1", result.stdout + result.stderr)

    def test_known_invalid_sdk_namespace_is_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent, child = root / "Op_0", root / "Op_1"
            for project, body in (
                (parent, "void K(){int x=0;}"),
                (child, "void K(){auto *pipe=AscendC::GetTPipePtr();Add(dst,src0,src1,n);}"),
            ):
                (project / "op_kernel").mkdir(parents=True)
                (project / "op_kernel/k.cpp").write_text(body)
            strategy = root / "strategy.json"
            strategy.write_text(json.dumps({"strategy": {
                "kinds": ["vectorize"],
                "evidence": ["op_kernel/k.cpp::K | 当前逐元素执行加法 | 标量迭代次数为n"],
                "reasoning": [
                    "[任务] 算子对连续输入逐元素执行加法并写回相同shape的输出。",
                    "[现状] K使用标量路径逐个处理输入元素并写回对应位置。",
                    "[问题] 数学主体执行n次Scalar迭代且没有使用Vector覆盖连续范围。",
                    "[策略] 将连续主区间改为Vector指令并为不能整段覆盖的tail保留边界处理。",
                    "[推导] 有效范围为全部输入元素，对齐范围由指令字段上限和tail共同确定。",
                    "[边界] 输入输出映射、有效元素范围和tail数学语义保持不变。",
                ],
                "targets": ["op_kernel/k.cpp::K"],
                "changes": ["op_kernel/k.cpp::K | 标量加法 -> 连续Vector Add"],
                "guards": ["访问范围 | Vector主区间和tail恰好覆盖有效元素"],
            }}))
            record = root / "implementation.json"
            summary = "op_kernel/k.cpp::K | 已将标量加法替换为连续Vector Add"
            record.write_text(json.dumps({
                "strategy_kinds": ["vectorize"], "summary": [summary],
                "modified_files": ["op_kernel/k.cpp"], "attempts": [{
                    "attempt": 1, "kind": "initial", "trigger": None,
                    "summary": summary, "modified_files": ["op_kernel/k.cpp"],
                }],
            }))
            result = subprocess.run([
                sys.executable, str(SCRIPT), "--strategy", str(strategy),
                "--implementation", str(record), "--parent", str(parent),
                "--project-dir", str(child), "--require-attempts",
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("GetTPipePtr 在当前 SDK 中是全局函数", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
