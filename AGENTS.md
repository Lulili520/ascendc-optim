# AscendC KernelBench 910B 工作规范

本仓库用最短闭环优化 AscendC 算子并导出 OPD self-distill 数据：完整源码直接产生一个策略，实施后只用精度和 Task Duration 判断结果。

## 边界与版本

- 原始工程 `kernel/<KernelBench910B|Attention910B|MHC910B>/<OperatorName>` 和 PyTorch reference 禁止修改；原始工程不按 level 分目录。
- 工作版本为 `kernel_workspace/<Suite>/<OperatorName>_<version>`；普通优化只修改其 `op_host/`、`op_kernel/`，复用 `CppExtension`。三个 suite 的队列、锁、版本和数据彼此隔离，精度与性能仍共享唯一设备验证槽。
- 所有 Python、Torch-NPU 和 CANN 命令先执行 `source /data/lu/activate_evokernel.sh`。
- 不同算子使用独立 vendor 和独立 agent 上下文；规划和实施最多并发处理4个算子。所有精度与性能命令共用唯一设备验证槽，任一时刻只允许一个验证进程执行 NPU Kernel；精度和 Task Duration 禁止互相并发。
- 首次复制完整工程为 `_0`；已存在则复用，禁止覆盖。新版本从当前已接受版本依次复制为 `_1` 至 `_4`。
- child 不继承 precision、performance 和构建中间文件，初始状态为 `PREPARED`。Host/Kernel 变化立即使旧结果失效。
- `workspace.json` 记录来源、父版本、vendor、状态、源码指纹和结果路径。
- 状态主链：`PREPARED -> BUILDING -> PRECISION_PASS -> PERFORMANCE_DONE`；失败为 `BUILD_FAILED`、`PRECISION_FAILED`、`PERFORMANCE_FAILED`。

## 源码策略

先识别计算模式、独立输出、数学主体和跨 tile 状态，再由 shape、容量、字段范围与任务数求唯一合法布局；同时比较 GM 字节、DMA 事务、Vector/Cube 调用、Scalar 迭代、task 生命周期、状态初始化、同步与写回事务，禁止用局部下降掩盖热路径工作量的数量级增长。后续固定类别顺序只用于完整扫描，不作为参数求解顺序。

Window/Pooling 先确定 Vector 数学主体和中间 Buffer，再反推 tile；禁止用最大搬运 tile 占满 UB 并保留同数量级 Scalar 热路径。已有 Cube 必须从 Host/Kernel 恢复全部 MNK，对当前和目标完整计算 A/B/C 有效字节与 A/B 重载次数；任务展平或循环换序不得破坏外层驻留。

必须使用 `kernel-strategy` skill。完整扫描所有产生输出的 Host tiling、task 映射、Process、CopyIn、Compute、CopyOut 路径，找出全部有直接源码证据且能确定修复的问题，固定按“纯冗余→重复GM→碎片搬运→数学主体→已有Cube→work unit→任务所有权→流水”依次确定修改。每轮一次修复全部类别的全部确定问题，不在完成一个类别后提前结束；所有变化必须共享唯一且无冲突的任务、tile、容量、地址、dtype、对齐、tail、同步和 ABI 设计。不得静默遗漏有确定解的问题。收益通过后从 child 完整源码重新扫描全部类别；无问题或接受四个 child 后停止。若没有确定优化，输出 `{"strategy": null}`。

性能类别扫描前必须完成八项内部审计：数学与 dtype/特殊值、核到输出所有权及跨核合并、每条搬运和 Vector 子地址的对齐/tail、全部活跃 Buffer/Queue/workspace 的峰值容量与生命周期、Reduction/Prefix/Window 跨 tile 状态、API 指令与 workspace、跨流水同步、Host/Kernel ABI。每个发现内部判为 actionable、unresolved 或 clean；契约不足的问题不得冻结但也不得当作 clean，并须继续扫描其他问题。输出 kind 仍只保留既有七类，`strategy:null` 仅表示当前输入下没有确定可实施 action，不表示源码绝对没有瓶颈。

容量上界只证明方案可运行，不证明最大 tile/baseM/baseN/baseK 性能最优；增大工作块不得删除已有有效双缓冲、预取或库内部空间。逻辑 block 多于物理核不等于冗余，`SetDim(n)` 不等于必须启动 n 个 block。只减少 C++ 循环、`End`、逻辑 block、task 生命周期或状态初始化，而未减少完整 launch 的 GM/DMA、Vector/Cube工作、A/B聚合重载或有直接依赖证据的等待时，策略必须为空。任何并行度变化都要按全部新增task重算热路径重载、每核串行工作和负载尾差。

kinds 按直接变化量判定并去重：Cube 主路径的 MNK/K/L1/L0/FixPipe 数据流变化包含 `cube`，其他数学指令主体及不转移工作的确定性 Scalar 数学冗余消除包含 `vectorize`，GM 总字节减少包含 `reuse_onchip`，仅 GM 事务减少包含 `batch_transfer`，核间所有权改变包含 `parallelize`，仅单任务工作量改变包含 `resize_tile`，仅阶段依赖或重叠改变包含 `pipeline`。首次把 Scalar/Vector contraction 转为 Cube 仅在当前 SDK API、完整 MNK、格式、L1/L0、重载、输出所有权、tail、同步和 ABI 全部闭合时允许，否则记为 unresolved。辅助 tile、blockDim、搬运或 TQue/TBuf 修改不增加 kind。

```json
{"strategy":{"kinds":["..."],"evidence":["文件::symbol | 源码事实 | 静态成本公式"],"reasoning":["[任务] ...","[现状] ...","[问题] ...","[策略] ...","[推导] ...","[边界] ..."],"targets":["..."],"changes":["..."],"guards":["..."]}}
```

非空策略还必须生成内部 `strategy/planning.json`，固定计算模式、shape 模型、任务映射、参数、Buffer、搬运、计算、完整静态工作向量和证明，并绑定当前源码指纹。该文件只服务实施和校验，不进入训练数据。

## 实施与修复

必须使用 `kernel-implementation` skill。控制器创建干净 child 并冻结策略；Agent 只实施该策略，只修改 child 的 `op_host/`、`op_kernel/`，记录 `strategy/implementation.json`，不得重新诊断或扩展优化。

实施后先精度、后时延。控制器命令异常最多执行两次，仍失败则以对应阶段失败终止，禁止无限重试。构建或精度报告失败时，可依据直接错误证据在冻结策略范围内最小修复，最多三次；每次源码变化均重新构建和验证。性能失败不触发源码修改。

## 精度

必须使用 `kernel-precision` skill：

```bash
python .codex/skills/kernel-precision/scripts/validate_precision.py \
  <OperatorName> --project-dir <OperatorName_version>
```

- 同一原始输入执行 reference 与自定义算子，仅验证 manifest 的一个原始 shape。
- 队列中所有精度和性能命令串行；OPP 安装与 NPU Kernel 执行都不得与另一验证进程重叠。首次无 expected 缓存时先执行自定义 Kernel smoke，成功后才生成 reference expected。
- reference 源码、参数、Torch 和 seed 不变时，可在临时目录复用输入及 expected；Kernel 变化只重跑自定义算子。
- 输出转 FP32，以 `atol=1e-2,rtol=1e-2` 比较。
- 退出码为 0、打印 `precision=PASS` 且 `precision.json` 指纹匹配才通过。
- `precision.json` 只作正确性门禁和失败诊断，不进入训练数据。
- 环境失败识别必须同时读取 `precision.json` 与其指向的详细构建/运行日志；`No space left`、设备忙或工具缺失不得固化为 Kernel `BUILD_FAILED`，环境恢复后可用 `--recover-environment-failures` 精确重置这些项。

## 时延

必须使用 `kernel-performance` skill：

```bash
python .codex/skills/kernel-performance/scripts/collect_performance.py \
  <OperatorName> --device 0 --project-dir <OperatorName_version>
```

- 要求当前源码精度 PASS；不运行 reference。
- 在 `msprof` 外预热同一自定义算子一次。
- 启动一个 `msprof`，显式使用 `--export=on`，只采一次 `PipeUtilization`，latency 只取 `Task Duration(us)`。
- 原始证据保存为 `performance/op_summary_PipeUtilization.csv`，结果写入 `performance/latency.json`；不采其他指标、逐核 cycle 或性能瓶颈。
- 无论成功或失败都保留 `performance/msprof.log`，CSV 缺失时不得只记录泛化错误。
- child 必须相对此前所有有效版本的最低 Task Duration 严格下降超过 5%；否则保留历史最佳并停止该算子。

## 循环与队列

每个算子使用独立上下文，planning 与 implementation 分开 thread，算子之间不共享上下文。控制器最多并发推进4个算子，但 precision 与 performance 共用一个设备验证槽；其余槽位可继续运行 planning 或 implementation：

```text
baseline 精度/时延 -> 源码策略 -> child 实施 -> 精度 -> Task Duration
                         ^                         |
                         +---- 收益通过后继续 ----+
```

child 构建或精度失败时不建立独立 repair：归档失败 child，把直接错误证据反馈给父版本重新完整规划。同一算子最多进行 3 次跨 child replan，达到上限时置 `implementation_blocked`，禁止清空历史后无限重试。遇到 `strategy=null`、收益不超过 5% 或原始 `_0` 硬失败即停止；最多接受四个 child（到 `_4`），不创建 `_5`。批量失败记录阶段后继续下一算子。

## 文件与训练数据

```text
<OperatorName>_<version>/
├── op_host/              ├── op_kernel/
├── CppExtension/         ├── precision/
├── performance/          ├── strategy/（含内部 planning.json）
└── workspace.json
```

训练样本只保留 `system_prompt`、`input`、`output`、`ops`：input 是完整精简 Host/Kernel，output 是父版本唯一 strategy。对后续新优化，Task Duration、精度和 implementation 只用于确认 child 正确且收益超过 5%，不进入模型输入输出；既有训练数据不追溯重筛。

```text
parent source + strategy -> child implementation -> precision PASS
                         -> Task Duration 改善 > 5% -> 一条监督样本
```
