# AscendC KernelBench 910B 工作规范

本仓库用于初始化、验证、采集和优化 AscendC KernelBench 910B 算子。

## 边界

- 原始工程：`kernel/KernelBench910B/<level>/<OperatorName>`，禁止直接优化。
- PyTorch reference：`kernel/pytorch-references/KernelBench/<level>`，禁止修改。
- 工作版本：`kernel_workspace/KernelBench910B/<level>/<OperatorName>_<version>`。
- 普通优化只修改工作版本的 `op_host/`、`op_kernel/`；复用 `CppExtension`。
- 所有 Python、Torch-NPU 和 CANN 命令先执行 `source /data/lu/activate_evokernel.sh`。

## 工作区与版本

- 首次处理时复制完整可构建工程为 `_0`；复制不等于移动，不删除原始内容。
- `_0` 已存在时复用，禁止静默覆盖；内容冲突则报告并停止该算子。
- 新版本依次使用 `_1`、`_2`；从指定父版本复制，未指定则使用最新版本。
- 新版本不继承精度、性能和构建中间文件，初始状态为 `PREPARED`。
- `op_host/` 或 `op_kernel/` 变化后，旧精度和性能立即失效。
- 每版用 `workspace.json` 记录来源、父版本、vendor、状态、源码指纹及结果路径。

## Vendor 与并发

- 从原始 `manifest.json` 读取 reference、参数和独立 vendor；禁止修改该 manifest。
- 不同算子不得共享 vendor；部署的 vendor 必须对应当前源码版本。
- 单算子任务一次只处理一个版本。level 批量任务也必须串行安装 OPP、验证和 profiling。
- 批量时记录失败阶段并继续下一个算子；精度失败不得采集性能。

## 状态机

```text
DISCOVERED -> PREPARED -> BUILDING -> PRECISION_PASS -> PERFORMANCE_DONE
```

失败状态：`PREPARE_FAILED`、`BUILD_FAILED`、`PRECISION_FAILED`、`PERFORMANCE_FAILED`。
状态只能由本轮命令的退出码、结果文件和当前源码指纹共同决定。

## 精度

精度任务必须使用 `kernel-precision` skill，并以工作版本为 `--project-dir`：

```bash
python .codex/skills/kernel-precision/scripts/validate_precision.py \
  <OperatorName> --project-dir <OperatorName_version>
```

- 用同一组原始输入执行 reference 和自定义算子，只验证一个原始 shape。
- 输出转 FP32 统计误差，以 `atol=1e-2, rtol=1e-2` 比较。
- 仅当退出码为 0、打印 `precision=PASS`，且 `precision/precision.json` 的源码指纹匹配时，状态才是 `PRECISION_PASS`。
- 成功时保存 shape、dtype、`max_abs`、`mean_abs`、退出码、日志和源码指纹。失败时也生成 `precision/precision.json`：OPP/Kernel/Host 或扩展构建失败记录为 `BUILD_FAILED`，算子运行、reference 或数值比较失败记录为 `PRECISION_FAILED`，并保存粗粒度 `failure_stage`、详细 `stage`、退出码、原因、日志路径和源码指纹。
- `precision.json` 只用于正确性门禁和运行管理，不进入策略训练数据。
- 默认清理可再生成的构建文件，保留 `CppExtension/*.so`；排错时才保留中间文件。

## 性能

性能任务必须使用 `kernel-performance` skill。入口会强制检查当前源码对应的精度 PASS：

```bash
python .codex/skills/kernel-performance/scripts/collect_performance.py \
  <OperatorName> --device 0 --project-dir <OperatorName_version>
```

- 性能阶段不执行 reference，不计算 speedup。
- 确认 NPU 空闲、`msprof` 可用、vendor 和源码一致。
- 整轮先在 `msprof` 外预热一次。
- 七组指标和 sample-based 逐核 cycle 各用一个独立 `msprof` 进程，各执行一次算子。
- 七组指标为 `PipeUtilization`、`ArithmeticUtilization`、`Memory`、`MemoryL0`、`MemoryUB`、`L2Cache`、`ResourceConflictRatio`。
- latency 只取 `PipeUtilization` 的 `Task Duration(us)`；不使用 quick、compare 或 repeats。
- 七组 CSV、有效 latency 或逐核 cycle 任一缺失，均为 `PERFORMANCE_FAILED`。
- 采集前自动用 `npu-smi` 确认逻辑设备映射，用 CANN `PlatformAscendC` 取得核数、各级容量和 Byte/cycle，并从当前精确 SoC 配置读取额定 Cube 频率。所有可靠设备数值及推导公式保存到 `kernel_workspace/KernelBench910B/hardware/device_<id>.json`；无效或不可取得的数值为 `null`，身份或参数交叉校验不一致时停止。
- GM 单核峰值带宽只按 CANN HBM Byte/cycle 与额定 Cube MHz 推导，缺少可靠输入时保持 `null`，禁止按芯片名称猜测。频率和原始 Byte/cycle 只留在设备文件，不进入算子 `performance.json`。
- 采集脚本解析七组 CSV 和逐核 cycle 后，直接生成面向瓶颈分析的 `performance.json`，包含任务、流水线、算术、各级存储、L2、冲突、逐核统计和精简硬件配置；存储部分可包含客观的单核 GM 路径总带宽及峰值利用率。
- `performance.json` 不自动判定 Bound 或生成优化建议。
- 结果固定写入 `performance/`，重采时更新已知文件，不创建轮次目录。

## 输出与汇总

```text
<OperatorName>_<version>/
├── op_host/                 ├── precision/
├── op_kernel/               ├── performance/
├── CppExtension/            ├── bottleneck/
├── strategy/                └── workspace.json
```

level 汇总至少包含：算子、版本、精度误差、latency、活跃核数、负载偏差、状态和失败阶段。离线对比只读取已有报告，不能替代正式采集；瓶颈结论必须与采集数据分开呈现。

## 瓶颈分析

瓶颈请求必须使用 `kernel-bottleneck` skill。只分析状态为 `PERFORMANCE_DONE`、源码指纹匹配且报告完整的工作版本；不重复 profiling、不运行 reference、不修改源码。

客观输入直接使用 `performance/performance.json`，不再生成重复的 `bottleneck/evidence.json`。分析结论写入 `bottleneck/bottleneck.json`。每次只生成一个稳定 `bottleneck_key` 和一个中粒度 `cause_key`；证据不足时 `bottleneck=null`，不得强选。主 Bound 必须按 skill 的固定规则判定，并用相关指标和源码交叉验证。

`bottleneck.json` 只包含 `reasoning`、`evidence` 和唯一 `bottleneck`。每条 evidence 固定为 `evidence_key + source + observation`；`bottleneck` 只包含 `bottleneck_key`、`cause_key`。事实留在 evidence，推导留在 reasoning，不重复保存 description、evidence_keys 或 causal_explanation。缺少正式主 Bound 或直接 cause 证据时输出 `bottleneck=null`。

## 策略推导

策略请求必须使用 `kernel-strategy` skill，并以当前版本通过校验的 `bottleneck/bottleneck.json` 和当前 `op_host/op_kernel` 为输入。按固定的 `bottleneck_key + cause_key → strategy_key` 选择唯一策略，再根据当前源码生成最少且完整的 `actions`。

`strategy.json` 只包含 `reasoning` 和唯一 `strategy`；`strategy` 只包含 `strategy_key`、`actions`。每项 action 包含真实 `target`、由 strategy 固定的 `operation`、1–4 条当前源码 `edits` 和 1–4 条当前算子 `constraints`。每条 edit 固定写成“源码对象：明确变换；必要参数”。相同 target+operation 必须合并。不要在 edits 重复通用知识，边界放入 constraints。

## 策略实施

实施请求必须使用 `kernel-implementation` skill。以父版本通过校验的可执行 `strategy/strategy.json` 为不可改写输入，创建下一个未占用的新版本，只修改新版本的 `op_host/` 和 `op_kernel/`。下游执行 agent 按顺序实施全部 actions 中的 `target + operation + edits + constraints`，不得替换策略、遗漏 action 或混入其他优化。

下游只做轻量硬错误检查：target 不存在、源码与 action 明显冲突、API/dtype 明确不支持、容量明确不足或无法保持接口/数学语义时停止且不生成 implementation；不要因为尚未证明性能收益而拒绝实施。

具体落地记录写入新版本 `strategy/implementation.json`：保留 `strategy_key`、按 `action_index` 对应的修改摘要、修改文件和有序 `attempts`。`attempts` 记录初次实施及最多 3 次修复的触发证据、知识 key、涉及 action 序号、修改摘要、文件和各门禁状态；不保存知识来源路径。reasoning 必须引用全部 action，`modified_files` 必须覆盖 strategy 中所有 target 文件。

cause 到 strategy 是确定性转换；每个 action 的 target、operation、edits、constraints 是策略 Agent 针对当前源码生成的可执行方案，实施 Agent 忠实执行，不重新选择或扩展策略。

实施记录校验通过后，先使用 `kernel-precision` 验证新版本。构建或精度失败时停止本次验证，不采性能；依据直接证据可在原 action 范围内最小修复，最多修复 3 次，每次源码变化后重新校验 implementation 并从构建开始验证。修复用尽仍失败则停止该算子。精度通过后再使用 `kernel-performance` 采集性能；性能采集失败不触发源码修复。当前流程只保留精度和客观性能结果，不生成策略评价。

## 可追溯优化数据

一次完整优化保留以下可连接记录：

```text
performance/performance.json
  → bottleneck/bottleneck.json
  → strategy/strategy.json
  → strategy/implementation.json
  → performance/performance.json（新版本）
```

策略训练主链只使用 `performance.json → bottleneck.json → strategy.json`。`implementation.json` 服务执行模型；`precision.json` 只作运行门禁。训练 JSON 不保存 schema、状态、源码指纹等工作区管理字段，这些信息留在 `workspace.json`。
