#!/usr/bin/env python3
"""Plot source, production-optimized, and Agent-evaluated Task Duration."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-report", type=Path, required=True)
    parser.add_argument("--policy-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def policy_latency_by_ops(audit: dict) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for item in audit["items"]:
        if item.get("eligible") and item.get("parent_latency_us") and item.get("child_latency_us"):
            result[item["ops"]] = (item["parent_latency_us"], item["child_latency_us"])
    return result


def blocked_bridge(
    steps: list[int], values: list[float], index: int, baseline: float
) -> tuple[list[int], list[float], float]:
    left = next((i for i in range(index - 1, -1, -1) if not math.isnan(values[i])), None)
    right = next((i for i in range(index + 1, len(values)) if not math.isnan(values[i])), None)
    if left is not None and right is not None:
        return [steps[left], steps[index], steps[right]], [values[left], baseline, values[right]], baseline
    if left is not None:
        return [steps[left], steps[index]], [values[left], baseline], baseline
    if right is not None:
        return [steps[index], steps[right]], [baseline, values[right]], baseline
    raise ValueError("Blocked point has no valid neighbor")


def plot_operator(items: list[dict], policy_latencies: dict[str, tuple[float, float]], output_dir: Path) -> None:
    items = sorted(items, key=lambda item: item["step"])
    ops = items[0]["ops"]
    if ops not in policy_latencies:
        raise KeyError(f"No production child latency in policy audit for {ops}")

    source_us, production_optimized_us = policy_latencies[ops]
    steps = [item["step"] for item in items]
    candidate_ms = [
        ((item.get("result") or {}).get("candidate_latency_us") / 1000.0)
        if (item.get("result") or {}).get("candidate_latency_us") is not None
        else math.nan
        for item in items
    ]
    source_ms = source_us / 1000.0
    production_ms = production_optimized_us / 1000.0
    short_name = ops.removesuffix("Custom_0")

    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    ax.plot(
        steps,
        [source_ms] * len(steps),
        linestyle="--",
        linewidth=2.2,
        color="#687386",
        label="Source baseline",
    )
    ax.plot(
        steps,
        [production_ms] * len(steps),
        linestyle="-.",
        linewidth=2.4,
        color="#389e0d",
        label="Production optimized (policy-data child)",
    )
    ax.plot(
        steps,
        candidate_ms,
        marker="o",
        markersize=7,
        linewidth=2.5,
        color="#1677ff",
        label="Agent evaluation",
    )

    for index, (item, value) in enumerate(zip(items, candidate_ms)):
        if math.isnan(value):
            bridge_steps, bridge_values, red_value = blocked_bridge(steps, candidate_ms, index, source_ms)
            ax.plot(
                bridge_steps,
                bridge_values,
                linestyle="--",
                linewidth=2.2,
                color="#d4380d",
                zorder=3,
                label="Not adopted / blocked" if index == next(i for i, v in enumerate(candidate_ms) if math.isnan(v)) else None,
            )
            ax.scatter(item["step"], red_value, s=72, color="#d4380d", edgecolor="white", linewidth=1.3, zorder=5)
            ax.annotate(
                "not adopted",
                (item["step"], red_value),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#a61d24",
            )
        else:
            ax.annotate(
                f"{value:,.1f}",
                (item["step"], value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
            )

    ax.annotate(
        f"production: {production_ms:,.1f} ms",
        (steps[-1], production_ms),
        xytext=(-4, -17),
        textcoords="offset points",
        ha="right",
        fontsize=8.5,
        color="#237804",
    )
    ax.set_title(f"Task Duration vs. Training Step\n{short_name}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Task Duration (ms, lower is better)")
    ax.set_xticks(steps)
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="best")

    stem = "task_duration_" + ("conv_standard3d" if "Standard3d" in ops else "conv_pointwise2d")
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    fig.savefig(output_dir / f"{stem}.svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    report = json.loads(args.campaign_report.read_text(encoding="utf-8"))
    audit = json.loads(args.policy_audit.read_text(encoding="utf-8"))
    latencies = policy_latency_by_ops(audit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for item in report["items"]:
        grouped.setdefault(item["ops"], []).append(item)
    for items in grouped.values():
        plot_operator(items, latencies, args.output_dir)


if __name__ == "__main__":
    main()
