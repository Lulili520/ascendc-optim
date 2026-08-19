# Strategy Action 实施

- `strategy.json` 不可改写；按顺序展平全部 actions，并从 1 编排 `action_index`。
- 只修改 child 的 `op_host/`、`op_kernel/`；不得遗漏、替换或扩展策略。

`implementation.json` 保存有序 `strategy_keys`、reasoning、展平后的逐项 `action_index + implementation_summary`、`modified_files` 和 attempts。attempts 使用 `action_indices` 标识本次涉及的既有 action，并记录 build、precision、performance 门禁状态。

控制器预建的 initial attempt 固定为：

```json
{"attempt":1,"kind":"initial","trigger":null,"knowledge_keys":[],"action_indices":[1],"implementation_summary":"实际落地摘要","modified_files":["op_kernel/x.cpp"],"validation":{"build":"NOT_RUN","precision":"NOT_RUN","performance":"NOT_RUN"}}
```

implementation 阶段只替换占位摘要、知识键和实际文件。repair 保持相同字段，使用 `kind=repair`，并记录 `trigger={"stage":"build|precision","symptom":"直接症状","evidence":"直接证据"}`。

新增或改变非基础 API 时，在 reasoning 或对应 `implementation_summary` 中紧凑记录 `API + 当前 header 路径 + 已采用调用形式`；不得仅写“已核对 SDK”。

校验命令必须显式区分父版本与 child：

```bash
python .codex/skills/kernel-implementation/scripts/validate_implementation.py \
  --bottleneck <parent>/bottleneck/bottleneck.json \
  --strategy <parent>/strategy/strategy.json \
  --implementation <child>/strategy/implementation.json \
  --parent <parent> --project-dir <child> --require-attempts
```

校验已包含 action target 的真实 symbol body 变化检查，不再单独运行 source-effect validator。
