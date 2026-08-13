# Strategy 校验

一次检查，普通问题最多统一修复一轮：

1. `cause_key → strategy_key` 与固定映射一致；
2. 每个 target 文件和符号真实存在且仅位于 `op_host/`、`op_kernel/`；
3. actions 是完成策略所需的 1–4 个最小完整集合，不混入第二策略；
4. operation 与 strategy 固定映射一致，edits 描述源码变换，constraints 只承载相关边界；
5. action 之间可以同时成立。

Reasoning 固定两条：第一条完成 `cause_key → strategy_key`；第二条只引用当前 cause evidence 的 key、source，并连接全部 target 与 operation，不复制 observation。所有 action 均要有 constraints。

校验器同时拒绝纯不变式 edit、包含代码变换的 constraint，以及 action 中的 `performance.*` 或收益证据。具体边界见 [action-contract.md](action-contract.md)。

普通缺漏统一补充后复检一次。仅在 target 不存在、越界修改、API/dtype 明确不支持、容量明确不足、接口或数学语义冲突时输出 `strategy:null`。尚未证明性能收益不是阻断理由。
