# Strategy 校验门禁：一次检查与至多一次修复

## 目标

将固定策略契约检查和 target/action 可实施性检查合并为一个校验门禁。在不改变固定策略选择的前提下，首次检查无问题时直接输出；发现问题时只允许统一修复一轮并复检一次。

## 检查输入

- 固定的 `strategy_key`、description 和有序 `change_key[]`；
- 已具体化的 target/action；
- 当前版本 `op_host/`、`op_kernel/`；
- `source /data/lu/activate_evokernel.sh` 后的当前 `$ASCEND_HOME_PATH` headers；
- performance 中已有硬件容量字段，未知值保持未知，不臆造。

## 统一首次检查

先检查固定契约，再逐项检查每个 change：

1. `strategy_key`、description 和有序 `change_key[]` 与固定推导器一致。
2. strategy JSON 契约、reasoning 引用和 change 覆盖完整。
3. target 文件、符号和修改范围真实存在。
4. action 指定或隐含的 API 在当前 SDK 存在，参数 tensor 类型与 dtype 路径有合法实现。
5. GM、片上缓冲、计算和输出的数据流完整；涉及不同流水线时写明 Queue 或受支持的同步关系。
6. 有效长度、硬件对齐长度、搬运方式和尾块 padding 明确，padding 不改变数学结果。
7. 所有缓冲及必要的单/双缓冲容量可满足；未知硬件容量不得伪装成已确认充足。
8. 索引范围、转换路径、tie-breaking、输出 shape/dtype/ABI 和边界写回明确。
9. 所有 changes 可以同时成立，不要求下游重新选择策略。

首次检查无 issue 时标记 `pass`，原样保留 target/action/reasoning，不进行“顺手改进”。

## 可修复问题

以下问题进入唯一修复轮，不得直接产生 `strategy=null`：

- API 名称、参数类型或 tensor 包装不完整；
- 可用 API 路径存在但 action 没有选择；
- dtype 转换可经受支持的中间类型完成；
- 缺少 TQue、EnQue/DeQue、event 或必要 barrier 描述；
- 有效长度与对齐长度混淆；
- DataCopy/DataCopyPad、mask、padding 或尾块写回未说明；
- UB 预算遗漏但能由当前 shape/tile 推导；
- 逐元素 GM 路径可改为 UB 组装与批量写回；
- target 符号不精确但能在允许源码中唯一解析；
- action 过于通用，缺少接口或数学语义约束。

唯一修复轮必须统一处理首次检查发现的全部结构和语义 issue。固定推导得到的 `strategy_key`、description、change_key 集合及顺序不可改写；只允许修正 JSON 表达、target、action 和对应 reasoning，不得引入第二策略。修复后只复检一次；复检用于形成最终输出，不得触发第二轮修复。

## 硬阻断

仅当事实证明不存在合法修复路径时使用规范硬阻断：

- `target不存在`：允许范围内没有任何对应目标；
- `越界修改`：完成 change 必然修改禁止区域；
- `API不支持`：当前 SDK/架构不存在任何等价 API 路径；
- `dtype不支持`：不存在保持语义的直接或中间 dtype 路径；
- `容量不足`：所有合理 tiling 均明确超过容量；
- `接口语义冲突`；
- `数学语义冲突`；
- `change冲突`。

复检仍发现普通问题时，不得谎报硬阻断或开始第二轮修复；输出唯一修复轮形成的 strategy，由 Implementation 的轻量硬错误检查、编译和精度门禁继续发现落地问题。本阶段不定义“strategy 生成失败”状态。确定性校验脚本若因 JSON 契约或固定映射报错，属于流程执行错误，不写入 strategy 训练数据，也不触发第二轮语义修复。

## Change 知识路由

| Change 特征 | 首次检查重点 |
|---|---|
| `stage_input` / `transfer` | GlobalTensor、DataCopyPad、32B 对齐、TQue、UB 容量 |
| `vector` / `reduce` | API/dtype 支持、Compare/Select 对齐、临时缓冲、tie-breaking |
| `handle_tail` | valid/aligned length、中性 padding、mask、有效写回范围 |
| `double_buffer` | Queue 数量、buffer 数、事件、容量重算、生产者消费者关系 |
| `cast` | 当前 SDK Cast 支持矩阵、中间 dtype、RoundMode、输出 ABI |
| `writeback` | VECOUT/MTE3、地址对齐、批量搬出、非对齐尾块 |
| `tiling` / `parallel` | 任务覆盖、block 映射、余数分配、无重复无遗漏 |

具体规则优先读取 `../../../kernel-knowledge/action-api-knowledge.md` 与 `../../../kernel-knowledge/action-pattern-knowledge.md`。需要确认具体签名时搜索当前 SDK headers；整理知识与当前 header 冲突时以当前 SDK 为准。
