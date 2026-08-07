---
name: kernel-performance
description: 采集单个 AscendC KernelBench 910B 自定义算子的完整 NPU 性能指标并生成客观报告。用户要求 msprof 上板采集、算子硬件性能数据、流水线或存储指标、逐核负载时使用；不运行 PyTorch reference，不计算加速比，不在采集脚本中自动判定瓶颈。
---

# AscendC 单算子性能采集

只采集已通过精度验证且源码指纹未变化的工作区版本。性能阶段不执行 reference，不计算 speedup。

## 执行

确认 NPU 空闲、`msprof` 可用且当前独立 vendor 已安装，然后运行：

```bash
source /data/lu/activate_evokernel.sh
python .codex/skills/kernel-performance/scripts/collect_performance.py \
  <OperatorName> --device 0 --project-dir \
  kernel_workspace/KernelBench910B/<level>/<OperatorName>_<version>
```

入口先核验 `precision/precision.json` 的 PASS、退出码和源码指纹，然后自动调用 `scripts/probe_hardware.py`：用 `npu-smi` 确认逻辑设备映射，用 CANN `PlatformAscendC` 获取核数、各级容量和 Byte/cycle，并从当前精确 SoC 配置读取额定 Cube 频率。探测结果保存到 `kernel_workspace/KernelBench910B/hardware/device_<id>.json`；设备身份、核数或带宽参数交叉校验不一致时停止。需要使用显式配置时才传 `--hardware-config`。随后整轮在 `msprof` 外预热一次，并用八个独立进程各执行一次算子：

1. `PipeUtilization`
2. `ArithmeticUtilization`
3. `Memory`
4. `MemoryL0`
5. `MemoryUB`
6. `L2Cache`
7. `ResourceConflictRatio`
8. sample-based 逐核 cycle

不使用 quick、compare 或 repeats。失败重试只用于错误恢复。

## 成功条件

- 七组指标都包含目标 AI Core 记录；
- latency 为 `PipeUtilization` 的正数 `Task Duration(us)`；
- 从 `aicore.db` 和/或 `ai_vector_core.db` 得到非空逐核 cycle；
- 报告生成成功，`workspace.json` 状态为 `PERFORMANCE_DONE`。

任一条件不满足即为 `PERFORMANCE_FAILED`，不得用部分报告宣称完成。

## 输出

固定更新当前版本的 `performance/`：`perf_report.md`、`performance.json`、`summary.txt`、七组 `op_summary_*.csv` 和 `per_core_cycles.csv`。脚本解析 CSV 后直接生成供瓶颈分析使用的 `performance.json`，其中包含任务、各流水线、算术、存储层级、L2、冲突、逐核统计和硬件配置。任务数据同时保存 `aicore_time_us`、`aiv_time_us`，并按 `Task Duration - max(AIC time, AIV time)` 计算 `max_core_time_us`、`head_overhead_us` 和百分制 `head_overhead_ratio`；AIC/AIV core time 均不可用时本轮采集失败。它只保存客观数据，不输出 Bound、根因或建议；不再生成重复的 `bottleneck/evidence.json`。

设备文件按 `identity`、`compute`、`memory_capacity`、`memory_bandwidth`、`derived` 保存所有可靠数值；接口不支持或返回无效值时保存 `null`。GM 单核峰值按 CANN 的 HBM Byte/cycle 与额定 Cube MHz 推导，并保存公式；缺少任一可靠输入时为 `null`，禁止按芯片名称猜测。

每个算子的 `performance.json.hardware` 只复制瓶颈分析需要的 SoC、核数、每核 UB/L1/L0 容量、L2 容量和 `gm_peak_bandwidth_gbps_per_core`，不复制频率、原始 Byte/cycle、设备映射、采集时间或来源。`performance.memory` 额外保存 AIC/AIV 单核读写总带宽、两条路径的最大值及客观峰值利用率；缺值时对应字段为 `null`，不据此自动判定 Bound。可单独运行 `python .codex/skills/kernel-performance/scripts/probe_hardware.py --device 0` 刷新和查看设备文件。

默认清理 PROF、日志、SQLite 和 Python 缓存。仅排错时使用 `--keep-intermediates`。
