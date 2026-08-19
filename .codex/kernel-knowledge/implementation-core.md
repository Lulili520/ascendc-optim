# AscendC 实施通用契约

只读取当前 actions 涉及的章节。这里仅保存跨架构实施不变量；具体对齐粒度、字段上限、dtype/API 支持和指令语义来自阶段输入指定的架构覆盖层与当前 SDK headers。

## transfer

- 对每个 CopyIn/CopyOut 分别列出方向、GM/片上 offset、有效长度、搬运长度、block/stride、dtype 字节数和 tail，并证明两端范围不越界。
- 分别确定 `valid_elems`、`compute_elems`、`storage_stride`、`allocated_elems` 和 `writeback_elems`；二维片上地址只用 `storage_stride` 解码，不能把非对齐的有效长度直接当行跨度。padding 必须保持数学语义，无效元素不得写回。
- 只有地址关系可由当前 API 精确表达时才合并搬运；热路径避免逐元素 GM 访问。
- 传入 API/结构体字段的字节数、count 和 stride 必须在容量证明后显式构造为目标字段类型。`sizeof` 参与表达式时不得依赖 `size_t` 向窄类型的隐式转换，例如先形成 `uint32_t copyBytes = elems * static_cast<uint32_t>(sizeof(T))`。

## vector

- 对每个 Vector API 从当前 headers 确认输入/输出 dtype、count/mask 单位、repeat/stride 范围、原地限制和临时空间。
- dtype 转换必须给出完整可编译链；索引表示必须证明精度范围，ArgMax/ArgMin 的 value/index 使用同一比较条件并保持首索引语义。

## lifetime

- 对每个片上对象标明 producer、producer pipeline、可见性边界、consumer pipeline、last-use 和复用点；跨流水数据必须选择 Queue，或显式写出正向可见性 event 和复用前的反向 event。
- `TBuf` 只分配计算临时内存，不提供 EnQue/DeQue 或自动跨流水同步。GM→片上→计算、计算→片上→GM 等阶段传递优先使用对应 `TQue`；只有完整证明当前架构 event 对时才使用 `TBuf` 手写流水。
- `PipeBarrier<PIPE_X>` 只解决同一 `PIPE_X` 内的数据依赖，不得代替 MTE2→Vector、Vector→Scalar、Scalar→MTE3 等跨流水 `SetFlag/WaitFlag`。
- Buffer 容量计入全部同时活跃 Queue 槽、临时对象、mask、padding 和 API workspace；流水槽数改变时同步重算容量和生命周期。
- 单槽 Queue 只提供所有权与同步；双槽只有容量容纳两份且至少两个 tile 能交错时才用于隐藏搬运。不得为了双缓冲缩小 tile 而忽略新增搬运、Queue 和循环开销。

## tiling

- Host 的 blockDim、task/tile 数和 Kernel 解码共用同一组 tiling 字段；证明任务全覆盖、无重复，空核显式返回。
- tile/chunk 由 shape、可靠硬件容量、全部活跃 Buffer、对齐和字段范围共同求解，不用经验常数或运行时试参替代证明。

## abi

- 保持公开输入输出、数学语义和工程已有 tiling 注册方式；字段变化同时更新定义、Host 赋值、序列化、Kernel 解析和必要 padding。
- 写回前在片上形成目标 dtype/layout；多核输出必须证明区间互斥，确有重叠时使用数学语义允许的归并方式。

## sdk-check

1. 激活仓库环境。
2. 修改源码前，从当前 SDK headers 定位 action 新增或改变的每个 API、字段和符号；逐项确认精确作用域/所属对象、参数顺序与类型、dtype、mask/count/stride 单位、原地限制和架构宏。包括 event 常量在内的符号不得凭记忆添加 `AscendC::` 或其他命名空间。
3. 实施后按修改文件检查局部对象的“声明/获取 → 使用 → last-use/释放”闭环；删除或移动 Buffer、Tensor 或临时量后，不得保留旧标识符引用。
4. 在 `implementation.json` 中记录已核对的 API/header/调用形式；编译器已否定的形式不得再次生成。
5. 用设备文件的可靠容量和拓扑求解参数。硬件 profile 与 headers 冲突时以 headers 为准；未知架构不得套用其他 profile。
