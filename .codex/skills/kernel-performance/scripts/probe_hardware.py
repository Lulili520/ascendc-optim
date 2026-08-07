#!/usr/bin/env python3
"""Probe current Ascend hardware and write a performance hardware config."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[4]
SOURCE = Path(__file__).with_name("hardware_probe.cpp")
DEFAULT_OUTPUT_ROOT = WORKSPACE / "kernel_workspace/KernelBench910B/hardware"


def run(command: list[str]) -> str:
    completed = subprocess.run(
        command, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"命令失败({completed.returncode})：{' '.join(command)}\n{detail}")
    return completed.stdout


def resolve_cann() -> Path:
    value = os.environ.get("ASCEND_HOME_PATH")
    if not value:
        raise RuntimeError("ASCEND_HOME_PATH 未设置，请先 source activate_evokernel.sh")
    root = Path(value).resolve()
    if not root.is_dir():
        raise RuntimeError(f"ASCEND_HOME_PATH 不存在：{root}")
    return root


def device_identity(logical_device: int) -> dict:
    output = run(["npu-smi", "info", "-m"])
    pattern = re.compile(
        r"^\s*(?P<physical>\d+)\s+(?P<chip>\d+)\s+"
        r"(?P<logical>-|\d+)\s+(?P<name>Ascend\s+\S+|\S+)\s*$"
    )
    for line in output.splitlines():
        match = pattern.match(line)
        if match and match.group("logical") == str(logical_device):
            return {
                "logical_device": logical_device,
                "physical_npu": int(match.group("physical")),
                "chip_id": int(match.group("chip")),
                "chip_name": match.group("name"),
            }
    raise RuntimeError(f"npu-smi 映射中找不到逻辑 device {logical_device}")


def platform_values(cann: Path) -> dict:
    include = cann / "x86_64-linux/asc/include"
    public_include = cann / "x86_64-linux/include"
    library = cann / "x86_64-linux/lib64"
    if not include.is_dir() or not library.is_dir():
        raise RuntimeError("当前 CANN 缺少 PlatformAscendC host headers 或 libraries")
    with tempfile.TemporaryDirectory(prefix="ascend-hardware-probe-") as temp:
        executable = Path(temp) / "hardware_probe"
        run([
            "g++", "-std=c++17", str(SOURCE), f"-I{include}",
            f"-I{public_include}", f"-L{library}",
            "-ltiling_api", "-lplatform", "-lunified_dlog",
            f"-Wl,-rpath,{library}",
            "-o", str(executable),
        ])
        return json.loads(run([str(executable)]))


def platform_config(cann: Path, chip_name: str, values: dict) -> dict:
    soc = chip_name.replace(" ", "")
    candidates = list(cann.glob(f"*-linux/data/platform_config/{soc}.ini"))
    if len(candidates) != 1:
        raise RuntimeError(f"无法唯一定位当前 SoC 配置 {soc}.ini：{candidates}")
    parser = configparser.ConfigParser(strict=False)
    parser.read(candidates[0], encoding="utf-8")
    try:
        aic = parser.getint("SoCInfo", "ai_core_cnt")
        aiv = parser.getint("SoCInfo", "vector_core_cnt")
        frequency = parser.getint("AICoreSpec", "cube_freq")
        ddr_rate = parser.getint("AICoreMemoryRates", "ddr_rate")
    except (configparser.Error, ValueError) as error:
        raise RuntimeError(f"当前 SoC 配置缺少核数、频率或 ddr_rate：{error}") from error
    if aic != values.get("aic_core_count") or aiv != values.get("aiv_core_count"):
        raise RuntimeError("PlatformAscendC 核数与当前 SoC 配置不一致")
    if ddr_rate != values.get("hbm_bytes_per_cycle_per_core"):
        raise RuntimeError("PlatformAscendC HBM Byte/cycle 与当前 SoC 配置不一致")
    return {
        "soc": soc,
        "aic_core_count": aic,
        "aiv_core_count": aiv,
        "cube_frequency_mhz": frequency,
        "hbm_bytes_per_cycle_per_core": ddr_rate,
    }


def probe(device: int) -> dict:
    identity = device_identity(device)
    cann = resolve_cann()
    values = platform_values(cann)
    config = platform_config(cann, identity["chip_name"], values)
    peak_per_core = (
        config["hbm_bytes_per_cycle_per_core"]
        * config["cube_frequency_mhz"] / 1000.0
    )
    return {
        "identity": {
            **identity,
            "soc": config["soc"],
            "soc_version_enum": values.get("soc_version_enum"),
        },
        "compute": {
            "aic_core_count": config["aic_core_count"],
            "aiv_core_count": config["aiv_core_count"],
            "cube_frequency_mhz": config["cube_frequency_mhz"],
            "vector_frequency_mhz": None,
        },
        "memory_capacity": {
            "ub_bytes_per_core": values.get("ub_bytes_per_core"),
            "l1_bytes_per_core": values.get("l1_bytes_per_core"),
            "l0a_bytes_per_core": values.get("l0a_bytes_per_core"),
            "l0b_bytes_per_core": values.get("l0b_bytes_per_core"),
            "l0c_bytes_per_core": values.get("l0c_bytes_per_core"),
            "l2_bytes": values.get("l2_bytes"),
            # PlatformAscendC returns 0 when HBM capacity is unavailable on
            # this platform. Preserve unknown as null instead of a false size.
            "hbm_bytes": values.get("hbm_bytes") or None,
        },
        "memory_bandwidth": {
            "hbm_bytes_per_cycle_per_core": values.get("hbm_bytes_per_cycle_per_core"),
            "l2_bytes_per_cycle_per_core": values.get("l2_bytes_per_cycle_per_core"),
            "gm_peak_bandwidth_gbps_per_core": peak_per_core,
        },
        "derived": {
            "gm_peak_bandwidth_gbps_per_core": peak_per_core,
            "gm_peak_formula": "hbm_bytes_per_cycle_per_core * cube_frequency_mhz / 1000",
        },
        "sources": {
            "identity": "npu-smi info -m",
            "topology_capacity_and_rates": "CANN PlatformAscendC",
            "rated_frequency": "CANN platform_config",
            "gm_peak_bandwidth": "derived from CANN values",
        },
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUT_ROOT / f"device_{args.device}.json"
    result = probe(args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"hardware_config={output.resolve()}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
