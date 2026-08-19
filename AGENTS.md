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
- 新版本依次使用 `_1`、`_2`、`_3`、`_4`；从指定父版本复制，未指定则使用最新版本。
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
- 同一算子在 reference 源码、初始化参数、Torch 版本和固定 seed 未变化时，可在临时目录复用输入与 reference expected；Kernel 源码变化只重跑自定义算子。缓存不进入工作版本或训练数据，算子终止时清理。
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
- 当前版本先采正式 `PipeUtilization`；若相对此前完整 `PERFORMANCE_DONE` 最佳版本的 latency 提升不超过 1%，写 `performance/screening.json`、状态置为 `PERFORMANCE_SCREENED_NO_IMPROVEMENT` 并停止，不生成完整 `performance.json`。确有提升时再补齐其余六组与逐核 cycle。
- 除上述无提升早停外，七组 CSV、有效 latency 或逐核 cycle 任一缺失，均为 `PERFORMANCE_FAILED`。
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

输入只使用完整 Host/Kernel、`performance.json` 和阶段内临时 `source_facts`。按 skill 完整扫描所有执行阶段，输出有直接源码 evidence、按可消除热路径成本排序的最多 3 个确定 cause；性能只补充影响，不按 ratio/Bound 阈值产生问题。只写 `bottleneck.json`，不生成 coverage、source model 或候选处置文件。

## 策略推导

策略请求必须使用 `kernel-strategy` skill。按固定 `cause→strategy→operation` 覆盖全部 issues，不重新诊断。全轮最多 6 个 actions，可包含多个主体变化，但必须共享唯一且无冲突的任务、tiling、容量、地址、dtype、对齐、tail、同步和 ABI 设计；参数来自 shape、可靠硬件与静态约束，不生成候选或上板试参。

## 策略实施

实施请求必须使用 `kernel-implementation` skill。控制器创建干净 child 并冻结 strategy；实施只读取 actions 相关的通用契约和精确 SoC 架构覆盖层，未知架构只查当前 SDK headers。按序实施全部 actions，只修改 child 的 `op_host/`、`op_kernel/` 并记录 `implementation.json`；不得重选、遗漏或扩展策略。validator 最多执行两次。仅 target 不存在、API/dtype 明确禁止、容量无解或 ABI/数学语义必坏时阻断。

实施记录校验通过后，先使用 `kernel-precision` 验证新版本。构建或精度失败时停止本次验证，不采性能；依据直接证据可在原 action 范围内最小修复，最多修复 3 次，每次源码变化后重新校验 implementation 并从构建开始验证。修复用尽仍失败则停止该算子。精度通过后再使用 `kernel-performance` 采集性能；性能采集失败不触发源码修复。子版本相对此前所有已完成版本中的最低正式 latency 必须严格下降超过 1%；否则记录 `stopped_no_improvement`，保留历史最佳版本并停止该算子，禁止让退化版本进入下一轮。收益通过后把新版本视为全新 Kernel 重新运行完整 bottleneck 分析，不只复检父 cause。新版本的全部确定问题继续进入下一版本，直到 `issues=[]`、完成四轮优化或硬失败；达到 `_4` 仍有 issues 时记录为“四轮完成但残留瓶颈”，不得标记优化完成或创建 `_5`。

## 可追溯优化数据

一次完整优化保留以下可连接记录：

```text
performance/performance.json
  → bottleneck/bottleneck.json
  → strategy/strategy.json
  → strategy/implementation.json
  → performance/performance.json（新版本）
```

策略训练主链只使用 `performance.json → bottleneck.json → strategy.json`，导出时合并成唯一紧凑 `policy` 字段；单条 input 只保留精简 PERF 与完整 HOST/KERNEL，不重复固定知识、symbol 清单和输出模板。`implementation.json` 服务执行模型；`precision.json` 只作运行门禁。训练 JSON 不保存 schema、状态、源码指纹等工作区管理字段，这些信息留在 `workspace.json`。
