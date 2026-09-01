#!/usr/bin/env python3
"""Render per-operator Task Duration charts from a finalized evaluation JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def render(operator: dict, output_dir: Path) -> Path:
    name = operator["operator"]
    points = sorted(operator["points"], key=lambda item: item["step"])
    steps = [item["step"] for item in points]
    durations = [item["task_duration_us"] for item in points]
    baseline = operator["baseline_task_duration_us"]
    data_best = operator.get("validated_data_best_us")

    fig, axis = plt.subplots(figsize=(10, 5.8))
    axis.plot(steps, durations, linewidth=2.2, color="#64748b", zorder=2)
    for index, item in enumerate(points):
        valid = item["status"] == "VALID"
        color = "#2563eb" if valid else "#dc2626"
        axis.scatter(item["step"], item["task_duration_us"], s=85, color=color,
                     edgecolor="white", linewidth=1.2, zorder=4)
        align = "left" if index == 0 else "right" if index == len(points) - 1 else "center"
        axis.annotate(f'{item["task_duration_us"]:.3f} us',
                      (item["step"], item["task_duration_us"]), xytext=(0, 11),
                      textcoords="offset points", ha=align, fontsize=9, color=color)

    baseline_line = axis.axhline(
        baseline, color="#f59e0b", linestyle="--", linewidth=1.8,
        label=f"Initial baseline: {baseline:.3f} us")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2563eb",
               markeredgecolor="white", markersize=9, label="Valid measured optimization"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#dc2626",
               markeredgecolor="white", markersize=9, label="Invalid operator optimization"),
        baseline_line,
    ]
    if data_best is not None:
        handles.append(axis.axhline(
            data_best, color="#16a34a", linestyle="-.", linewidth=1.8,
            label=f"Validated data best: {data_best:.3f} us"))

    x_margin = max(1, (max(steps) - min(steps)) * 0.05)
    lower_values = durations + ([data_best] if data_best is not None else [])
    axis.set_xlim(min(steps) - x_margin, max(steps) + x_margin)
    axis.set_ylim(min(lower_values) * 0.94, max(durations + [baseline]) * 1.075)
    axis.set_xticks(steps)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Task Duration (us)")
    axis.set_title(f"{name} Task Duration by Training Step")
    axis.grid(True, linestyle="--", alpha=0.28)
    axis.legend(handles=handles, loc="lower left")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{slug(name)}_task_duration.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = json.loads(args.results.read_text())
    output_dir = args.output_dir or args.results.parent / "plots"
    for operator in report["operators"]:
        print(render(operator, output_dir))


if __name__ == "__main__":
    main()
