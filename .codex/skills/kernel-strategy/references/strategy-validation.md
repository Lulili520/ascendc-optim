# Strategy 校验

一次检查，普通问题最多统一修复一轮：

1. `cause_key → strategy_key → operation` 与唯一固定映射一致；不接受候选策略或同 cause 多方向；
2. 每个 target 文件和符号真实存在且仅位于 `op_host/`、`op_kernel/`；
3. actions 是完成策略所需的 1–4 个最小完整集合，不混入第二策略；
4. operation 与 strategy 固定映射一致，edits 描述源码变换，constraints 只承载相关边界；
5. action 之间可以同时成立；同一源码对象没有互斥变换，共享 pattern、tiling、对齐和 dtype 一致；
6. 容量公式由正式硬件值重算，不能只检查字段存在。

Reasoning 用非空精简文本说明 `cause_key → strategy_key` 与当前源码修复机制。固定中文前缀、文本长度和辅助 performance evidence 重复引用不是硬门禁。所有 action 均要有 constraints。

校验器同时拒绝纯不变式 edit、包含代码变换的 constraint，以及 action 中的 `performance.*` 或收益证据。具体边界见 [action-contract.md](action-contract.md)。

普通缺漏统一补充后复检一次。仅在 target 无法定位到任何真实源码对象、API/dtype 明确禁止、容量公式证明无解、或 ABI/数学语义必然破坏时写 `strategy/blocking.json`。不确定 API 和尚未证明收益不是阻断理由。
