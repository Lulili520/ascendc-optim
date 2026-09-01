# Source Policy 数据契约

```json
{"system_prompt":"...","input":"...","output":"...","ops":"Operator_0"}
```

`input`：

```text
[OP]
算子名、function、parameters

[HOST]
完整有效 op_host C/C++/H 源码

[KERNEL]
完整有效 op_kernel C/C++/H 源码
```

只机械删除版权头、纯注释和连续空行，不按答案 target 裁剪源码，不包含 PERF、reference、child、latency 或收益。

`output` 优先使用从当前输入版本到正式最佳版本、针对起始源码独立归一化并校验的统一策略：

```json
{"strategy":{"kinds":["..."],"evidence":["文件::symbol | 源码事实 | 静态成本公式"],"reasoning":["..."],"targets":["..."],"changes":["..."],"guards":["..."]}}
```

单轮策略原样使用。多轮策略禁止机械拼接：只有 `terminal_policy.json` 已基于起始源码、最终源码和全部有效步骤重新生成，并保证 evidence 可从起始源码定位、changes 表达 initial-to-final、reasoning 是重新生成的六段统一推理时才导出长链；否则降级为父子单轮有效策略。不复制中间源码，不保留中间参数或跨版本“现状”。

执行链内部的 `planning.json` 仅用于验证策略确由当前源码的计算模式、布局和完整静态工作向量推出，不复制进 input 或 output。

当前队列管理的算子只有进入终态后才能导出；不在当前目标队列中的历史算子按其持久有效链导出。连续链上的每个子版本必须精度 PASS 且相对此前历史最佳 Task Duration 严格下降超过 1%。正式训练集只包含非空有效策略；`strategy:null` 终态只写 audit，不进入 JSONL。性能无收益的失败策略不得作为答案。运行中、待重试和中间版本不导出；无效步骤及其后续不跨越拼接。latency、收益、源码指纹和排除原因只写 audit。

导出时重新运行现行 planning、strategy 和 implementation validator；非空策略必须减少静态工作，或在每输出热路径工作不增加时提高有效核覆盖；已有Cube的任务展开不得增加A/B热路径重载或格式转换。执行期 compile/precision repair、replan反馈和未接受child均不进入input/output。
