# AscendC Optim

AscendC Optim 是一套面向 AscendC 910B 算子的源码优化与 OPD self-distill 数据生成流程。它从完整 Host/Kernel 源码中推导确定性优化策略，实施后使用精度和 `Task Duration(us)` 验证结果，最后导出适合下游 Qwen3-8B 学习的紧凑策略数据。

## 当前规模

| Suite | 有效算子 | 策略数据 |
|---|---:|---:|
| KernelBench910B | 45 | 73 |
| Attention910B | 29 | 53 |
| MHC910B | 9 | 14 |
| 合计 | **83** | **140** |

完整的初始性能、最佳版本、最佳时延和总降幅见 [算子优化汇总](docs/operator_optimization_summary.md)。

## 优化闭环

```text
baseline 精度/时延
        ↓
完整 Host/Kernel 源码分析
        ↓
planning.json + strategy.json
        ↓
干净 child 实施 + implementation.json
        ↓
精度 PASS
        ↓
Task Duration 改善 > 5%
        ↓
继续迭代（最多到 `_4`）+ 导出策略数据
```

主要原则：

- 每个算子使用独立 Agent 上下文。
- 规划和实施可并发，精度与性能验证共用唯一设备槽并严格串行。
- 策略必须基于可定位的源码证据，并闭合任务、tile、Buffer、地址、dtype、tail、同步和 ABI。
- 不使用候选参数试跑代替静态推导。
- 精度不通过时不采集性能。
- `strategy:null`、退化、无效版本和不完整链路不进入正式训练数据。

## 数据格式

正式 JSONL 位于 `datasets/`：

- `kernel_policy_data.jsonl`
- `attention910b_policy_data.jsonl`
- `mhc910b_policy_data.jsonl`

每条数据只包含四个字符串字段：

```json
{
  "system_prompt": "稳定的策略任务说明",
  "input": "[OP] + [HOST] + [KERNEL]",
  "output": "{\"strategy\":{...}}",
  "ops": "OperatorName_version"
}
```

`output.strategy` 由以下字段组成：

- `kinds`：优化类型，可取 `reuse_onchip`、`batch_transfer`、`vectorize`、`cube`、`resize_tile`、`parallelize`、`pipeline`。
- `evidence`：可从输入源码定位的问题证据。
- `reasoning`：任务、现状、问题、策略、推导和边界六段统一推理。
- `targets`：需要变更的真实源码 symbol。
- `changes`：每个 target 的 current-to-target 修改。
- `guards`：数学、所有权、容量、地址、tail、同步和 ABI 不变量。

精度、时延、收益和工作区状态只用于数据资格判定，不进入模型的 input/output。

## 仓库结构

```text
.
├── .codex/
│   ├── kernel-knowledge/          # 实施与架构知识
│   ├── skills/                    # strategy/implementation/precision/performance/data
│   ├── run_kernelbench_optimization.py
│   └── prepare_suite_workspace.py
├── datasets/                     # OPD self-distill JSONL 及审计文件
├── docs/                         # 性能汇总
├── AGENTS.md                     # 完整工作规范
└── README.md
```

以下大型本地目录默认不上传 Git：

- `kernel/`：原始算子与 PyTorch reference。
- `kernel_workspace/`：可构建的版本、精度和性能证据。
- `kernel_eval_workspace/`：下游模型评测工作区。
- `train_result/`：优化归档。

## 环境

需要可用的 CANN、Torch-NPU、AscendC SDK 和 910B 设备。运行任何 Python、Torch-NPU 或 CANN 命令前，先激活环境：

```bash
source /data/lu/activate_evokernel.sh
```

## 常用命令

初始化 Attention 或 MHC suite 的本地 `_0` 工作区：

```bash
python .codex/prepare_suite_workspace.py --suite Attention910B
```

重建现有队列：

```bash
python .codex/run_kernelbench_optimization.py --suite KernelBench910B --rebuild
```

启动优化：

```bash
python .codex/run_kernelbench_optimization.py --suite KernelBench910B
```

从正式最佳版本重开指定算子：

```bash
python .codex/run_kernelbench_optimization.py --suite KernelBench910B \
  --reopen-ops 'OperatorA,OperatorB'
```

重新导出策略数据：

```bash
python .codex/skills/kernel-policy-data/scripts/export_policy_data.py \
  --suite KernelBench910B --output datasets/kernel_policy_data.jsonl
```

详细的版本、并发、精度、性能和数据契约见 [AGENTS.md](AGENTS.md)。

## 测试

```bash
source /data/lu/activate_evokernel.sh
python -m unittest discover -s .codex/tests -p 'test_*.py'
python -m unittest discover -s .codex/skills/kernel-strategy/tests -p 'test_*.py'
python -m unittest discover -s .codex/skills/kernel-implementation/tests -p 'test_*.py'
python -m unittest discover -s .codex/skills/kernel-performance/tests -p 'test_*.py'
python -m unittest discover -s .codex/skills/kernel-policy-data/tests -p 'test_*.py'
python -m unittest discover -s .codex/skills/kernel-precision/tests -p 'test_*.py'
```

## 说明

本项目的策略只针对当前输入 shape、dtype、SDK 契约和可靠硬件容量中能静态闭合的修改。`strategy:null` 只表示“当前输入下没有确定可实施策略”，不表示算子在所有 shape 和架构上都没有优化空间。
