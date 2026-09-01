# Implementation 失败诊断知识

Implementation 使用本表定位失败，再只加载阶段输入指定的通用契约、架构覆盖层相关节。诊断只能修复冻结策略的落地，不得改写策略；初次实施后最多修复 3 次。

| 失败现象 | 优先根因 | 必查知识 | 修复边界 |
|---|---|---|---|
| `unsigned long/size_t` 窄化、结构体初始化失败 | `sizeof`/count/stride 表达式与字段类型不同 | `implementation-core:transfer,sdk-check` | 证明范围后显式构造目标类型 |
| API 参数、重载或符号作用域编译失败 | tensor/dtype、字段类型、命名空间、架构签名 | `implementation-core:sdk-check` 与当前 profile | 适配 header 真实声明，不换策略 |
| `TCubeTiling` 字段不可访问 | Host 误用私有字段 | 当前 SDK 生成的 tiling 类声明 | 改用真实 `get_*` 公开访问器 |
| `static assertion` / `if constexpr` 非常量 | 运行时 tiling 值被用作模板或编译期分支 | Cube/API 模板声明 | 保留编译期配置，运行时值只驱动普通分支 |
| Compare/Select 报 `__ubuf__ half*` 或 mask 类型不匹配 | 混用 dtype 或不同重载族的 mask 布局 | `implementation-core:vector` 与当前 Compare/Select headers | Compare 输出与 Select 输入使用同一重载族 |
| FATBIN 缺少 `*_mix_aic_0.o` | 更早的 kernel/code-channel 编译未生成目标对象 | 首个 OPC 编译诊断与 tiling key | 修复首个编译错误，不创建空对象或单独重试链接 |
| `BinaryGetFunctionByEntry` / `funcEntry=0` | kernel entry、tiling key、code channel 或 vendor metadata 不一致 | entry 声明、生成 binary metadata、Host 注册 | 归类为 build/entry 落地错误，不当作数值精度重选策略 |
| AICore 禁止整数/浮点转换 | Scalar 转换限制或 Vector API dtype 不匹配 | `implementation-core:vector,sdk-check` | 使用已核验 Vector Cast 链或可证明精度的累加表示 |
| `use of undeclared identifier` | 结构修改后声明/获取/使用不闭环 | `implementation-core:sdk-check` | 恢复原策略内的对象闭环，删除旧标识符引用 |
| 每次错误数量变化、输出随机/局部错 | 跨流水可见性、片上复用、未初始化 | `implementation-core:lifetime` 与当前 profile | 修复生产消费与复用边界，不用 PipeBarrier 替代跨流水 event |
| 仅尾块错 | valid/aligned 混淆、padding/mask | `implementation-core:transfer,vector` | 中性 padding，仅写有效范围 |
| 错误集中在固定 tile lane | Vector count、mask 布局、写回粒度 | `implementation-core:vector,abi` | 对齐计算空间并批量搬出 |
| int64 出现两个 32 位值打包 | 逐元素转换/store 路径 | `implementation-core:vector,abi` | 片上组装目标 ABI 后批量写回 |
| 结果全 0 或未写 | 输出 Buffer 位置、写回依赖 | `implementation-core:lifetime,abi` | 修复生产消费关系和 CopyOut |
| MTE/AIV/AIC 设备异常、地址未对齐或 Vector 异常 | GM/片上越界、参数不合法、容量 | `implementation-core:transfer,lifetime` 与当前 profile | 一次复核本轮全部搬运点；同 PC 重复则否定旧假设 |
| 精度正确但明显变慢 | blockDim、Queue 槽数、tile 过小 | `implementation-core:tiling,lifetime` | 比较单槽大 tile 与容量可行的交错，不预设 TBuf/TQue 更快 |

每次修复：记录失败阶段和直接证据，只形成一个主要根因假设，做最小修改，然后从编译门禁重新验证。MTE 假设的证据集必须覆盖本轮所有搬运点，但仍只修复一个共同根因。源码变化后旧精度和性能失效。发现策略存在可前移的通用规则时，回灌知识文件和后续 Strategy 校验门禁；不得回写已经冻结的父版本 `strategy.json`。
