---
name: kernel-evaluation
description: 评测 AscendC KernelBench 910B 策略模型的结构化输出与真实优化效果。用户要求用训练同格式数据测试模型、解析 bottleneck/strategy、让当前 Agent 实施预测策略、在 kernel_workspace_eval 隔离工作区执行精度与正式性能门禁、统计真实 latency 收益或生成评测报告时使用；不把标准答案泄漏给实施 Agent，不修改原始工程或常规优化工作区。
---

# Kernel Evaluation

评测两层能力：离线 policy 结构正确性，以及预测策略经真实实施后的精度与性能效果。

## 输入契约

接受与 `kernel-policy-data` 完全相同的 JSONL：每行仅含字符串 `ops`、`input`、`output`。模型只接收 `input`；预测文件每行含 `ops` 和 `output`（或 `prediction`）。

`input` 已包含基线 `PERFORMANCE_JSON`、完整 `OP_HOST_SOURCE` 和 `OP_KERNEL_SOURCE`。必须从数据还原待优化源码，禁止从历史工作版本读取源码。只允许从 `kernel/KernelBench910B` 复制 CMake、CppExtension 等非训练源码构建脚手架，并从原始 manifest 取得独立 vendor。

需要检查详细解析、安全落盘或报告字段时，读取 [references/evaluation-contract.md](references/evaluation-contract.md)。

## 流程

1. 可选：按去除 `_0/_1` 的算子名稳定切分数据，禁止同算子泄漏。
2. 用模型对每条 `input` 生成 `[BOTTLENECK_JSON]` 与 `[STRATEGY_JSON]`。
3. 先运行离线结构评测，报告 JSON 可解析率、key/change/target/action 契约指标；该结果不能替代真实效果。
4. 初始化 effect run，将标准 `output` 隐藏到 run 的 `.gold/`，实施 Agent 不得读取。
5. 严格按队列逐条 materialize。每个算子在 `kernel_workspace_eval/KernelBench910B/<level>/<ops>/source` 保存 input 恢复源码，各快照候选为同目录的 `eval_stepN`；任何时刻只创建当前 case。
6. 当前 Agent 把预测 strategy 当作被测输入，按 target/action 的修改意图实施，随后依次使用 `kernel-precision` 和 `kernel-performance`。不要求预测先通过固定策略推导校验或与 gold 一致；精度失败不得采性能，性能失败不修源码。
7. finalize 当前 case，再处理下一条。最后生成真实精度通过率、可评测数、严格大于 1% 的改善率和平均 latency 降幅。

训练输出若已经包含 `dataset_split_ops.json` 和 `eval_samples/<epoch>_<step>.jsonl`，先运行 `scripts/import_training_output.py`。必须使用文件中保存的 eval input 和模型 output，并仅从 gold dataset 补标准 output。所有保存的 step 都必须分别进入 Agent 效果评测，禁止只选最终或 gold 指标最好的快照。

## 命令

所有 Python/CANN 命令先执行：

```bash
source /data/lu/activate_evokernel.sh
```

离线评分：

```bash
python .codex/skills/kernel-evaluation/scripts/evaluate_policy_model.py \
  --dataset <eval.jsonl> --predictions <predictions.jsonl> --report <report.json>
```

导入训练输出快照：

```bash
python .codex/skills/kernel-evaluation/scripts/import_training_output.py \
  --training-output <training-output-dir> \
  --gold-dataset datasets/kernel_policy_data.jsonl \
  --output-dir kernel_workspace_eval/results/<run-name>
```

每个 step 固定选择一个 train probe 和一个 eval 样本时：

```bash
python .codex/skills/kernel-evaluation/scripts/import_fixed_pair.py \
  --training-output <training-output-dir> \
  --train-ops <TrainOperator_version> --eval-ops <EvalOperator_version> \
  --gold-dataset datasets/kernel_policy_data.jsonl \
  --output-dir kernel_workspace_eval/results/<run-name>
```

导入器固定同 step 内 train 在前、eval 在后，并要求每个 step 两者都唯一存在。

真实效果队列：

```bash
python .codex/skills/kernel-evaluation/scripts/policy_effect_eval.py init \
  --dataset <eval.jsonl> --predictions <predictions.jsonl> \
  --run-dir kernel_workspace_eval/runs/<run-name>
python .codex/skills/kernel-evaluation/scripts/policy_effect_eval.py materialize \
  --run-dir kernel_workspace_eval/runs/<run-name> --ops <Operator_version>
```

多个训练 step 使用统一 campaign；初始化只生成元数据和策略队列，不创建算子工作区：

```bash
python .codex/skills/kernel-evaluation/scripts/agent_effect_campaign.py init \
  --training-eval kernel_workspace_eval/results/<run-name>/training_output_eval.json \
  --campaign-dir kernel_workspace_eval/campaigns/<run-name>
python .codex/skills/kernel-evaluation/scripts/agent_effect_campaign.py next \
  --campaign-dir kernel_workspace_eval/campaigns/<run-name>
python .codex/skills/kernel-evaluation/scripts/agent_effect_campaign.py materialize-next \
  --campaign-dir kernel_workspace_eval/campaigns/<run-name>
```

当前 Agent 按生成的 `AGENT_TASK.md` 完成实施、精度和性能后，运行 `finalize-current`。实施硬阻断或性能采集失败时运行 `fail-current`。只有当前 case 终止后才能 materialize 下一项。最终运行 `report`，分别输出每个 step 与整体结果。

绘制各训练 step 的真实 Task Duration 时运行 `scripts/plot_task_duration.py`，输入 campaign report 与 policy-data audit；不可实施或未采纳点落在 baseline，并用红点和虚线连接相邻有效点。

实施及门禁完成后：

```bash
python .codex/skills/kernel-evaluation/scripts/policy_effect_eval.py finalize \
  --run-dir kernel_workspace_eval/runs/<run-name> --ops <Operator_version>
python .codex/skills/kernel-evaluation/scripts/policy_effect_eval.py report \
  --run-dir kernel_workspace_eval/runs/<run-name>
```

## 门禁与边界

- `kernel_workspace_eval` 是唯一可写算子评测工作区；禁止写入 `kernel_workspace` 和 `kernel/KernelBench910B`。
- 只处理当前队首 case；失败记录阶段并继续下一条，不提前生成后续工作区。
- campaign 固定按 step 升序、同 step 内按训练 eval ops 顺序执行；任何时刻全 campaign 最多一个 `MATERIALIZED` case。
- 实施 Agent 禁止读取 `.gold` 或训练 `output`，也不得以标准答案修正预测。
- 预测为空或确实无法采集策略时记录失败。效果评测优先解析标准标签，也兼容两个连续 JSON 和 `bottleneck_json/strategy_json` 包装；格式不规范只影响离线格式分，不阻断 Agent。能采集时直接交给 Agent；仅在 target 不存在、API/dtype/容量明确不支持或无法保持数学语义时停止实施，不替换为 gold 策略。
- 基线 latency 直接取训练 `input`；候选 latency 只取正式 `kernel-performance` 报告。
- 不以 exact match 代替真实优化效果，也不因 exact match 不一致而拒绝可实施预测。
