# 单策略实施记录

策略被冻结为 `kinds/evidence/reasoning/targets/changes/guards`。Implementation 只把全部 changes 落到 targets，并保持 guards；原始源码证据由冻结策略保留，实施记录用 modified_files 和源码差异连接修复结果，不重复复制 evidence。

落地前先在当前上下文中完成但不额外导出的实施草图：每个 task 的输入/输出区间；每个 Buffer 的 dtype/allocated elements/producer/consumer/last-use；每次 DMA 的 GM/UB offset、valid/transfer/compute/writeback、block/stride、full/tail 和 API；每次 Vector/Cube 的实际子地址与 count/mask；全部同时活跃容量。按 `32/sizeof(dtype)` 对每条实际指令验证 `base_offset + dynamic_offset` 的范围和所需对齐；只验证 Buffer 基址、总容量或完整 tile 不算闭合，运行时 tail 和分核起点也必须闭合。实施后的循环次数和 API 覆盖范围必须与策略静态工作量一致。

```json
{
  "strategy_kinds": ["batch_transfer", "vectorize"],
  "summary": ["op_kernel/x.cpp::Process | 已将连续范围改为单次批量搬运并保留tail分支"],
  "modified_files": ["op_kernel/x.cpp"],
  "attempts": [{
    "attempt": 1,
    "kind": "initial",
    "trigger": null,
    "summary": "op_kernel/x.cpp::Process | 已将连续范围改为单次批量搬运并保留tail分支",
    "modified_files": ["op_kernel/x.cpp"]
  }]
}
```

`summary` 每条必须使用 `strategy target | 已实施的可核对事实`，并覆盖全部 targets。target 是必须直接发生源码变化的函数、类型或文件级常量；禁止为了让 target 显示变化而添加注释、等价重写或无语义改动。若冻结策略误把无需编辑的间接受影响符号列为 target，应让校验失败并交还控制器修正策略。初次实施生成 `kind=initial`、trigger 为 null；控制器提供真实编译证据后，才可在同一冻结 action 内追加最多 3 个 `kind=repair`。所有文本无首尾空白、无 Markdown、无模糊措辞。精度、性能和工作区状态只由各自报告记录，不在 implementation 中复制。局部编译落地错误由控制器进入 `compile_fix`；策略不可实施或修复用尽时才退回父版本重新规划。

首次实施必须以一次正式编译通过为目标。先从 input 的 `sdk_include_roots` 定点定位每个新增或改变符号的真实声明，再编辑源码；不得通过候选调用或反复编译探索 API。静态 validator 只固化已经由当前 SDK 明确否定且与参数无关的形式，其他重载、dtype 和架构能力仍逐项以 header 为准。

Prefix/Scan 实施必须逐级记录 scan 源、目标、shifted operand 和 offset Tensor 的实际基址；普通 Vector 二元指令不得使用未满足重载对齐要求的 `tensor[stride]`。策略冻结对齐 Gather 链时，offset 的单位、下界钳制、首 `stride` 元素的单位元、Gather 全部访问范围、ping-pong 交替方向和 tile carry 最后元素必须分别闭合。设备报告 UB Vector address unaligned 时，直接修复子视图布局，禁止当作环境错误重试。

Window/Pooling 实施必须逐 tap 核对输入索引、padding lane 和 Gather 最大 offset。`Select` 的 mask 按当前 header 定义的 bit 布局生成，禁止未经证明地用 `LocalTensor<uint8_t>` 逐元素写 `0/1`。首 tap 能直接写归约 dst 时不得额外使用 LocalTensor→LocalTensor `DataCopy`；保留该 copy 时必须用 Queue/event 证明 MTE→V RAW 和复用前反向依赖。

校验：

```bash
python .codex/skills/kernel-implementation/scripts/validate_implementation.py \
  --strategy <parent>/strategy/strategy.json \
  --implementation <child>/strategy/implementation.json \
  --parent <parent> --project-dir <child> --require-attempts
```
