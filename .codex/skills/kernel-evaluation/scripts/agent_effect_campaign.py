#!/usr/bin/env python3
"""Manage a serial all-snapshot Agent effect-evaluation campaign."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_effect_eval as effect  # noqa: E402


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def campaign_path(campaign_dir: Path) -> Path:
    return campaign_dir / "campaign.json"


def load(campaign_dir: Path) -> dict[str, Any]:
    return json.loads(campaign_path(campaign_dir).read_text(encoding="utf-8"))


def sync(campaign_dir: Path, campaign: dict[str, Any]) -> None:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for snapshot in campaign["snapshots"]:
        queue = effect.load_queue(Path(snapshot["run_dir"]))
        for item in queue["items"]:
            by_key[(snapshot["snapshot"], item["ops"])] = item
    for case in campaign["cases"]:
        item = by_key[(case["snapshot"], case["ops"])]
        for key in ("status", "failure_stage", "reason", "candidate_project", "result"):
            case[key] = item.get(key)
    campaign["updated_at"] = effect.now()
    dump(campaign_path(campaign_dir), campaign)


def active_cases(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for case in campaign["cases"] if case["status"] == "MATERIALIZED"]


def next_case(campaign: dict[str, Any]) -> dict[str, Any] | None:
    return next((case for case in campaign["cases"] if case["status"] not in effect.TERMINAL), None)


def init_campaign(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.resolve()
    if campaign_dir.exists():
        raise SystemExit(f"campaign-dir 已存在，禁止覆盖：{campaign_dir}")
    imported = json.loads(args.training_eval.read_text(encoding="utf-8"))
    snapshots = imported.get("snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise SystemExit("training_output_eval.json 缺少 snapshots")
    snapshots = sorted(snapshots, key=lambda item: (item["step"], item["epoch"]))
    evaluation_workspace = (
        args.evaluation_workspace.resolve() if args.evaluation_workspace
        else ROOT / "kernel_workspace_eval/KernelBench910B"
    )
    campaign_dir.mkdir(parents=True)
    campaign_snapshots: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    position = 0
    for snapshot in snapshots:
        name = snapshot["snapshot"]
        run_dir = campaign_dir / "runs" / name
        init_args = argparse.Namespace(
            dataset=Path(snapshot["dataset"]), predictions=Path(snapshot["predictions"]),
            run_dir=run_dir, evaluation_workspace=evaluation_workspace,
            template_root=args.template_root,
        )
        effect.init_run(init_args)
        queue = effect.load_queue(run_dir)
        campaign_snapshots.append({
            "snapshot": name, "epoch": snapshot["epoch"], "step": snapshot["step"],
            "run_dir": str(run_dir.resolve()), "offline_report": snapshot["offline_report"],
        })
        for item in queue["items"]:
            position += 1
            cases.append({
                "position": position, "snapshot": name, "epoch": snapshot["epoch"],
                "step": snapshot["step"], "ops": item["ops"], "run_dir": str(run_dir.resolve()),
                "status": item["status"], "failure_stage": item["failure_stage"],
                "reason": item["reason"], "candidate_project": None, "result": None,
            })
    campaign = {
        "created_at": effect.now(), "updated_at": effect.now(),
        "training_eval": str(args.training_eval.resolve()),
        "evaluation_workspace": str(evaluation_workspace),
        "ordering": "snapshot step ascending, then dataset eval_ops order",
        "snapshots": campaign_snapshots, "cases": cases,
    }
    dump(campaign_path(campaign_dir), campaign)
    print(f"campaign={campaign_dir} snapshots={len(campaign_snapshots)} cases={len(cases)}")
    return 0


def show_next(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.resolve()
    campaign = load(campaign_dir)
    sync(campaign_dir, campaign)
    current = next_case(campaign)
    if current is None:
        print("campaign=COMPLETE")
    else:
        print(json.dumps(current, ensure_ascii=False, indent=2))
    return 0


def materialize_next(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.resolve()
    campaign = load(campaign_dir)
    sync(campaign_dir, campaign)
    active = active_cases(campaign)
    if active:
        raise SystemExit(f"已有 MATERIALIZED case：{active[0]['snapshot']}::{active[0]['ops']}")
    current = next_case(campaign)
    if current is None:
        raise SystemExit("campaign 已完成")
    if current["status"] != "PREDICTED":
        raise SystemExit(f"下一 case 状态不可物化：{current['status']}")
    effect.materialize(argparse.Namespace(
        run_dir=Path(current["run_dir"]), ops=current["ops"],
        evaluation_workspace=Path(campaign["evaluation_workspace"]),
    ))
    sync(campaign_dir, campaign)
    current = next_case(campaign)
    print(f"current={current['snapshot']}::{current['ops']}\nstatus={current['status']}")
    return 0


def finalize_current(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.resolve()
    campaign = load(campaign_dir)
    sync(campaign_dir, campaign)
    active = active_cases(campaign)
    if len(active) != 1:
        raise SystemExit(f"需要且只能有一个 MATERIALIZED case，当前={len(active)}")
    current = active[0]
    effect.finalize(argparse.Namespace(run_dir=Path(current["run_dir"]), ops=current["ops"]))
    sync(campaign_dir, campaign)
    return show_next(argparse.Namespace(campaign_dir=campaign_dir))


def fail_current(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.resolve()
    campaign = load(campaign_dir)
    sync(campaign_dir, campaign)
    active = active_cases(campaign)
    if len(active) != 1:
        raise SystemExit(f"需要且只能有一个 MATERIALIZED case，当前={len(active)}")
    current = active[0]
    effect.mark_failed(argparse.Namespace(
        run_dir=Path(current["run_dir"]), ops=current["ops"],
        status=args.status, stage=args.stage, reason=args.reason,
    ))
    sync(campaign_dir, campaign)
    return show_next(argparse.Namespace(campaign_dir=campaign_dir))


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in items if item["status"] == "COMPLETED"]
    improved = [item for item in completed if item.get("result", {}).get("improved_over_1_percent")]
    precision_attempted = [item for item in items if item["status"] in {"FAILED_PRECISION", "FAILED_PERFORMANCE", "COMPLETED"}]
    precision_passed = [item for item in precision_attempted if item.get("result", {}).get("precision_pass") is True]
    reductions = [item["result"]["reduction_percent"] for item in completed]
    return {
        "cases": len(items),
        "status_counts": dict(sorted(collections.Counter(item["status"] for item in items).items())),
        "implementation_attempted_cases": sum(item["candidate_project"] is not None for item in items),
        "precision_attempted_cases": len(precision_attempted),
        "precision_passed_cases": len(precision_passed),
        "performance_completed_cases": len(completed),
        "improved_over_1_percent": len(improved),
        "improved_over_1_percent_rate": len(improved) / len(completed) if completed else None,
        "mean_reduction_percent": sum(reductions) / len(reductions) if reductions else None,
    }


def report(args: argparse.Namespace) -> int:
    campaign_dir = args.campaign_dir.resolve()
    campaign = load(campaign_dir)
    sync(campaign_dir, campaign)
    by_snapshot = {}
    for snapshot in campaign["snapshots"]:
        items = [item for item in campaign["cases"] if item["snapshot"] == snapshot["snapshot"]]
        by_snapshot[snapshot["snapshot"]] = {
            "epoch": snapshot["epoch"], "step": snapshot["step"], **summarize(items),
        }
    value = {"overall": summarize(campaign["cases"]), "by_snapshot": by_snapshot, "items": campaign["cases"]}
    output = args.output or campaign_dir / "campaign_report.json"
    dump(output, value)
    print(f"report={output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--training-eval", type=Path, required=True)
    init.add_argument("--campaign-dir", type=Path, required=True)
    init.add_argument("--evaluation-workspace", type=Path)
    init.add_argument("--template-root", type=Path, default=ROOT / "kernel/KernelBench910B")
    for name in ("next", "materialize-next", "finalize-current", "report"):
        command = sub.add_parser(name)
        command.add_argument("--campaign-dir", type=Path, required=True)
        if name == "report":
            command.add_argument("--output", type=Path)
    failed = sub.add_parser("fail-current")
    failed.add_argument("--campaign-dir", type=Path, required=True)
    failed.add_argument("--status", choices=["STOPPED_IMPLEMENTATION", "FAILED_PERFORMANCE"], required=True)
    failed.add_argument("--stage", required=True)
    failed.add_argument("--reason", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return {
        "init": init_campaign, "next": show_next, "materialize-next": materialize_next,
        "finalize-current": finalize_current, "fail-current": fail_current, "report": report,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
