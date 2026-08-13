# Evaluation Contract

## Source restoration

按固定段落读取 `OPERATOR_JSON -> PERFORMANCE_JSON -> OP_HOST_SOURCE -> OP_KERNEL_SOURCE -> OUTPUT_FORMAT`。源码段由一个或多个标记组成：

```text
--- FILE: op_host/<relative path> ---
<complete source>
```

只接受相对路径，拒绝绝对路径、`..`、重复路径以及与当前段目录前缀不一致的路径。评测工作区中的 `op_host/`、`op_kernel/` 必须完全由这些段落创建。

## Isolation

```text
kernel_workspace_eval/
├── KernelBench910B/<level>/<ops>/
│   ├── source/
│   └── eval_stepN/
└── runs/<run-name>/
    ├── queue.json
    ├── cases/<ops>/{input.txt,prediction.txt,policy/,AGENT_TASK.md}
    └── .gold/<ops>.json
```

`.gold` 只供最终离线审计，实施 Agent 不得读取。候选工作区不得引用常规 `kernel_workspace`。

## Effect result

每个 case 至少记录：预测解析状态、精度状态、基线 latency、候选 latency、降幅百分比及 `reduction_percent > 1.0`。构建/精度失败不可进入性能统计；性能报告缺失或失败必须单独记录，不能伪装成零收益。

汇总至少包含：状态计数、精度通过率、效果可评测数、严格大于 1% 的数量与比例、平均降幅。离线 policy 指标与真实效果指标分别报告。

## Multi-step campaign

训练输出中的每个 `<epoch>_<step>.jsonl` 都建立独立子队列，campaign 再按 step 升序串联所有子队列。相同 `ops` 的 `source` 必须与各 step 保存 input 中的源码逐文件一致；每个 `eval_stepN` 都从 `source` 独立复制，禁止继承前一 step 的修改。campaign 初始化不得提前创建 candidate；只有 `materialize-next` 创建当前一项。
