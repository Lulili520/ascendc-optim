---
name: kernel-bottleneck
description: 分析已完成正式性能采集的 AscendC KernelBench 910B 单算子，以原始性能和源码证据确定唯一 bottleneck_key 与中粒度 cause_key。用户要求判断固定开销、并行、流水线瓶颈或生成瓶颈训练数据时使用；不 profiling、不运行 reference、不修改源码、不生成策略。
---

# Kernel Bottleneck

目标：根据有效证据稳定选择本轮优先优化入口，输出一个瓶颈及一个可验证原因。

## 流程

1. 读取完整且源码指纹匹配的 `performance/performance.json`。
2. 完整阅读 [references/bottleneck-method.md](references/bottleneck-method.md)。
3. 从报告提取原始数值 evidence，严格按决策树判断：固定开销 → 核利用不足 → 负载不均 → 稳态流水线 Bound。只有当前优先级无“正式指标 + 直接源码根因”时才继续；禁止把 `*_bound` 或 `*_underuse` 结论伪装成 evidence。
4. 读取定向的 `op_host/`、`op_kernel/`，记录可定位的源码事实 evidence，确定一个可直接指导策略的中粒度 `cause_key`。
5. 写入 `bottleneck/bottleneck.json`，运行：

   ```bash
   python .codex/skills/kernel-bottleneck/scripts/validate_report.py <bottleneck.json>
   ```

## 固定边界

- `bottleneck_key` 只表示固定开销、并行损失或执行流水线 Bound。
- `evidence_key` 只表示原始指标或源码事实，`bottleneck_key` 表示受限位置，`cause_key` 表示可行动根因族；具体代码形态留在 observation，不继续拆 key。
- Cache、冲突、搬运粒度、同步、循环形态属于 `cause_key`，不单列主瓶颈。
- 报告不重复保存 description、evidence_keys 或 causal_explanation；事实留在 evidence，推导留在 reasoning。
- 每个 key 的含义、父子关系和必要证据以 reference 与 validator 为唯一规范。
- 输入报告不完整、字段无效或源码指纹不匹配时停止，不输出 `bottleneck:null`。
- 有效输入下没有同时满足正式瓶颈条件和直接 cause 证据的候选时，输出 `bottleneck:null`，停止该算子优化。
- 不输出候选策略、收益估计或修改建议。
- 当 `active_cores < available_cores` 时，必须先阅读任务映射并判断独立任务是否足够；未完成该检查不得跳到 pipeline Bound。
- source observation 只写源码可复核事实，不写 ratio、cycle、百分比、Task Duration 或“排除其他瓶颈”。
- 非空结论只保留 2–4 条必要 evidence；reasoning 固定两条：第一条由正式指标确定 bottleneck，第二条由源码或直接原因 evidence 确定 cause。禁止逐条复述全部 evidence。
