---
name: kernel-bottleneck
description: 分析已完成性能采集的 AscendC KernelBench 910B 单算子，读取 performance.json 和 op_host/op_kernel 源码，以规范 evidence_key 认定唯一 bottleneck_key 并交叉验证根因。用户要求分析瓶颈、规范性能与源码证据、判断计算/访存/Scalar/冲突/负载问题或生成瓶颈数据时使用；不执行 profiling、不运行 PyTorch reference、不修改算子源码、不生成优化建议。
---

# AscendC 单算子瓶颈分析

分析已有证据，不采集、不构建、不修改源码、不生成优化建议。`kernel-performance` 产生事实；本 skill 只认定瓶颈并解释因果链。

## 执行流程

1. 直接读取 `performance/performance.json`。它必须包含任务、流水线、算术、各级存储、L2、资源冲突、逐核统计和硬件配置；字段缺失时停止，不用部分数据强推。
2. 完整阅读 [`references/bottleneck-method.md`](references/bottleneck-method.md)，严格按“严重并行问题 → 小任务固定开销 → 正式主 Bound”顺序选择唯一主瓶颈。
3. 阅读 `op_host/`、`op_kernel/`，用源码交叉验证指标根因。
4. 从方法文档选择固定 `evidence_key`，每条证据统一写为 `evidence_key + source + observation`。指标证据必须引用 `performance.*`，源码证据必须引用 `op_host/` 或 `op_kernel/`。
5. 只有对应主 Bound 的必要 `evidence_key` 存在时才能选择 `bottleneck_key`；证据不足直接输出 `bottleneck: null`。
   `overhead.small_workload` 还必须满足 `performance.task.head_overhead_ratio > 30%`，并由 `source.init_or_ldst_heavy` 确认小工作量下的固定初始化或 Scalar LD/ST 根因。
6. 把证据到结论压缩为 2–6 条 `reasoning`，每条使用“证据：<evidence_key>…；推断：…”格式，并引用所有使用的 evidence key。
7. 将结论写入当前版本：

   ```text
   bottleneck/bottleneck.json
   ```

8. 校验结构化报告：

   ```bash
   python .codex/skills/kernel-bottleneck/scripts/validate_report.py \
     kernel_workspace/KernelBench910B/<level>/<OperatorName>_<version>/bottleneck/bottleneck.json
   ```

9. 更新 `workspace.json` 的管理字段，不改变主状态 `PERFORMANCE_DONE`。

## 报告要求

只给唯一主瓶颈、证据链和源码根因。当前阶段不输出额外 Markdown 报告、次级瓶颈、候选项、优化建议、收益估计或复测指标。

`bottleneck.json` 只包含 `reasoning`、`evidence`、唯一 `bottleneck`。`bottleneck` 包含 `bottleneck_key`、`description`、`evidence_keys` 和 `causal_explanation`；无法认定时为 `null`。

校验器强制检查：key 来自固定目录、主 Bound 必要证据存在、证据来源类型正确、`evidence_keys` 完整引用证据、reasoning 引用全部 evidence key。

普通 msprof 聚合数据不能证明流水气泡或 Double Buffer 重叠率。无法证明时明确写限制，不补造数值。
