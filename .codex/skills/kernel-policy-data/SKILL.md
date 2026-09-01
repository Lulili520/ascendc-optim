---
name: kernel-policy-data
description: 从 AscendC KernelBench 910B 连续有效版本链导出源码策略 self-distill 数据。输入只含某个起始版本的算子规格与完整 Host/Kernel 源码，将从该版本到正式最佳版本的有效策略合成为一份统一 strategy；不输入性能、不采集性能、不修改源码。
---

# Kernel Source Policy Data

```bash
python .codex/skills/kernel-policy-data/scripts/export_policy_data.py \
  --suite <KernelBench910B|Attention910B|MHC910B> --output <suite.jsonl>
```

三个 suite 独立读取工作区、队列、manifest/reference 映射和版本链；非 KernelBench 的 `ops` 使用 `Suite/Operator_version`，禁止跨 suite 合并版本。

按 [数据契约](references/data-contract.md) 导出 `system_prompt/input/output/ops` 四个字符串字段。

- 输入：算子规格、去除版权头和纯注释后的完整 Host/Kernel 源码。
- 输出：`{"strategy":{"kinds","evidence","reasoning","targets","changes","guards"}}`。
- 资格：当前队列管理的算子必须已进入终态；不在当前目标队列中的历史版本链可按持久结果导出。每个相邻步骤的父策略和子实施通过现行校验、父子精度有效、子版本相对历史最佳 Task Duration 严格下降超过 1%。正式训练集只保留非空有效策略；运行中、待重试和 `strategy:null` 均不导出，null 终态只记录在 audit。
- 合成：每个有效父版本生成一条记录。多轮链只有存在针对该起始源码独立生成并校验的 `strategy/terminal_policy.json` 时，才使用其中的 initial-to-final 统一策略；禁止直接拼接不同版本的 evidence、reasoning、targets、changes 或 guards。缺少合格终态策略时降级为当前父子之间的单轮有效策略。不把多个版本源码拼入 input，不跨越无效步骤，不引入未经验证的新修改。
- 性能只用于筛选，不进入 input/output。

不生成 bottleneck、cause、operation 或性能解释；源码证据只保留在紧凑 `evidence` 中。
