# Implementation 失败诊断知识

Implementation 使用本表反向加载同目录的 `action-api-knowledge.md` 和 `action-pattern-knowledge.md`。诊断只能修复现有 change 的落地，不得改写策略选择；初次实施后最多修复 3 次。

| 失败现象 | 优先根因 | 必查知识 | 修复边界 |
|---|---|---|---|
| API参数或重载编译失败 | tensor类型、字段类型、架构签名 | GM搬运、当前 SDK核验 | 适配当前签名，不换策略 |
| AICore禁止动态类型转换 | Scalar转换限制 | dtype、Cast与索引 | float累加器或 Vector Cast |
| 输出随机/大面积错 | 缺同步、UB复用、未初始化 | TBuf/TQue与流水线 | Queue/event/barrier及生命周期 |
| 仅尾块错 | valid/aligned混淆、padding/mask | 对齐、尾块 | 中性padding，仅写有效范围 |
| 错误集中在固定tile lane | Vector count、mask布局、写回粒度 | Compare对齐、Writeback | 对齐计算并批量搬出 |
| int64出现两个32位值打包 | 逐元素转换/store路径 | dtype、输出ABI、Writeback | UB组装ABI后MTE3写回 |
| 结果全0或未写 | 输出Queue位置、MTE3依赖 | Queue与流水线 | VECOUT EnQue/DeQue和CopyOut |
| 编译通过但设备异常 | 越界、UB地址不对齐、容量 | 搬运、容量、尾块 | 先做CopyIn→CopyOut最小验证 |
| 精度正确但明显变慢 | blockDim、Queue开销、tile过小 | Parallel、Batch transfer | 只能在原change范围调参数 |

每次修复：记录失败阶段和直接证据，只形成一个主要根因假设，做最小修改，然后从编译门禁重新验证。源码变化后旧精度和性能失效。发现 action 存在可前移的通用规则时，回灌知识文件和后续 Strategy 校验门禁；不得回写已经冻结的父版本 `strategy.json`。
