#!/usr/bin/env python3
"""Run the Level1 optimization queue with one fresh Codex context per operator."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/data/lu/ascendc-optim")
QUEUE = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_queue.json"
RUNS = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_runs"
LOG = ROOT / "kernel_workspace/KernelBench910B/level1_regeneration_progress.tsv"


def save(queue: dict) -> None:
    temporary = QUEUE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, QUEUE)


def main() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    queue = json.loads(QUEUE.read_text())
    for item in queue["items"]:
        if item["state"] in {"completed", "stopped_no_bottleneck", "failed"}:
            continue
        operator = item["operator"]
        item["state"] = "running"
        item["attempts"] += 1
        item["started_at"] = datetime.now(timezone.utc).isoformat()
        save(queue)
        prompt = f"""只处理 Level1 算子 {operator}，这是该算子的独立上下文。仓库绝对路径固定为 /data/lu/ascendc-optim；执行任何 shell 命令都必须显式设置 workdir=/data/lu/ascendc-optim，若使用 bash -lc 则命令开头必须先 cd /data/lu/ascendc-optim，禁止依赖登录 shell 的当前目录。严格遵守 /data/lu/ascendc-optim/AGENTS.md，并完整使用其中对应 skills。复用 {operator}_0 中源码指纹匹配的已有精度/正式性能；缺失时先补齐。随后从 _0 开始依次完成瓶颈分析、策略推导、策略实施、精度和正式性能，成功后继续第二轮到 _2。单算子完整结束后停止，不处理其他算子。没有真实瓶颈、strategy=null、修复用尽或门禁失败时如实记录并结束；禁止修改 kernel/ 原始工程和 reference。不要等待用户确认。最终说明该算子的终态。"""
        run_log = RUNS / f"{item['position']:03d}_{operator}.jsonl"
        final_message = RUNS / f"{item['position']:03d}_{operator}.final.txt"
        with run_log.open("w") as output:
            result = subprocess.run([
                "codex", "exec", "--json", "--ephemeral",
                "--dangerously-bypass-approvals-and-sandbox",
                "-C", str(ROOT), "-o", str(final_message), prompt,
            ], cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, text=True)
        item["finished_at"] = datetime.now(timezone.utc).isoformat()
        item["returncode"] = result.returncode
        item["state"] = "completed" if result.returncode == 0 else "failed"
        save(queue)
        with LOG.open("a") as progress:
            progress.write(f"{item['finished_at']}\t{item['position']}\t{operator}\t{item['state']}\t{result.returncode}\n")


if __name__ == "__main__":
    main()
