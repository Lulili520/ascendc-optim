#!/usr/bin/env python3
"""Create buildable _0 projects for Attention910B and MHC910B."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".codex"))
from suite_config import SUITES, load_suite_manifest, suite_paths

TEMPLATE = ROOT / "kernel/KernelBench910B/ArgmaxOverADimensionCustom"
PYTORCH_HELPER = Path("/data/lu/cannbot-skills/plugins-community/collaborative-agent-kernel-evolution/skills/ascendc-evaluation/scripts/template/CppExtension/csrc/pytorch_npu_helper.hpp")
SCAFFOLD = ("CMakeLists.txt", "CMakePresets.json", "build.sh",
            "op_host/CMakeLists.txt", "op_kernel/CMakeLists.txt")


def fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for folder in ("op_host", "op_kernel")
                       for path in (project / folder).rglob("*") if path.is_file()):
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def setup_text(module: str) -> str:
    return f'''from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension
import torch_npu

op_plugin_include = Path(torch_npu.__file__).resolve().parent / "include" / "third_party" / "op-plugin"

setup(
    name="{module}",
    ext_modules=[NpuExtension(
        name="{module}", sources=["csrc/op.cpp"],
        include_dirs=[str(op_plugin_include)], extra_compile_args=["-O2"],
    )],
    cmdclass={{"build_ext": BuildExtension}},
)
'''


def prepare(suite: str) -> tuple[int, int]:
    source_root, workspace_root, _ = suite_paths(ROOT, suite)
    entries = load_suite_manifest(ROOT, suite)
    workspace_root.mkdir(parents=True, exist_ok=True)
    created = reused = 0
    for operator, entry in sorted(entries.items()):
        source, target = source_root / operator, workspace_root / f"{operator}_0"
        if target.exists():
            reused += 1
        else:
            shutil.copytree(source, target, symlinks=True)
            for relative in SCAFFOLD:
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(TEMPLATE / relative, destination)
            preset = json.loads((target / "CMakePresets.json").read_text(encoding="utf-8"))
            preset["configurePresets"][0]["cacheVariables"]["vendor_name"]["value"] = entry["vendor"]
            (target / "CMakePresets.json").write_text(
                json.dumps(preset, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
            (target / "CppExtension/setup.py").write_text(
                setup_text(entry["extension_module"]), encoding="utf-8")
            created += 1
        tiling_headers = sorted((target / "op_host").glob("*_tiling.h"))
        for source_file in (target / "op_host").glob("*.cpp"):
            text = source_file.read_text(encoding="utf-8", errors="replace")
            for included in re.findall(r'#include\s+"([^"]+_tiling\.h)"', text):
                alias = target / "op_host" / included
                if not alias.exists() and len(tiling_headers) == 1:
                    shutil.copy2(tiling_headers[0], alias)
        kernel_sources = sorted((target / "op_kernel").glob("*.cpp"))
        canonical_kernel = target / "op_kernel" / f"{entry['function']}.cpp"
        if not canonical_kernel.exists() and len(kernel_sources) == 1:
            kernel_sources[0].rename(canonical_kernel)
        extension_source = target / "CppExtension/csrc/op.cpp"
        if extension_source.is_file() and "pytorch_npu_helper.hpp" in extension_source.read_text(
                encoding="utf-8", errors="replace"):
            shutil.copy2(PYTORCH_HELPER, target / "CppExtension/csrc/pytorch_npu_helper.hpp")
        (target / "workspace.json").write_text(json.dumps({
            "operator": operator, "suite": suite, "version": 0,
            "parent_version": None, "source_project": str(source),
            "status": "PREPARED", "vendor": entry["vendor"],
            "source_fingerprint": fingerprint(target),
            "precision": None, "performance": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return created, reused


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[name for name in SUITES if name != "KernelBench910B"], required=True)
    args = parser.parse_args()
    created, reused = prepare(args.suite)
    print(f"suite={args.suite} created={created} reused={reused}")


if __name__ == "__main__":
    main()
