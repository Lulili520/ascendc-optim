"""Shared suite discovery for AscendC optimization tools."""

from __future__ import annotations

import json
import re
from pathlib import Path


SUITES = {
    "KernelBench910B": "KernelBench",
    "Attention910B": "Attention",
    "MHC910B": "MHC",
}


def normalize(value: str) -> str:
    value = re.sub(r"^\d+_", "", value)
    return re.sub(r"[^a-z0-9]", "", value.lower()).removesuffix("custom")


def suite_paths(root: Path, suite: str) -> tuple[Path, Path, Path]:
    if suite not in SUITES:
        raise ValueError(f"unsupported suite: {suite}")
    return (root / "kernel" / suite,
            root / "kernel_workspace" / suite,
            root / "kernel/pytorch-references" / SUITES[suite])


def _binding(project: Path) -> str:
    source = project / "CppExtension/csrc/op.cpp"
    match = re.search(r'm\.def\(\s*"([A-Za-z_]\w*)"',
                      source.read_text(encoding="utf-8", errors="replace"))
    if match is None:
        raise ValueError(f"missing pybind binding: {source}")
    return match.group(1)


def _reference_map(reference_root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    if not reference_root.is_dir():
        return result
    for path in sorted(reference_root.glob("*.py")):
        result.setdefault(normalize(path.stem), []).append(path.name)
    return result


def load_suite_manifest(root: Path, suite: str) -> dict[str, dict]:
    project_root, _, references = suite_paths(root, suite)
    if suite == "KernelBench910B":
        value = json.loads((project_root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("KernelBench910B manifest must be an object")
        return value

    reference_map = _reference_map(references)
    entries: dict[str, dict] = {}
    for project in sorted(path for path in project_root.iterdir() if path.is_dir()):
        descriptor_paths = [path for path in project.glob("*.json") if path.name != "result.json"]
        if len(descriptor_paths) != 1:
            raise ValueError(f"{project.name} must have exactly one operator descriptor")
        descriptor = json.loads(descriptor_paths[0].read_text(encoding="utf-8"))
        descriptor = descriptor[0] if isinstance(descriptor, list) and descriptor else descriptor
        if not isinstance(descriptor, dict):
            raise ValueError(f"invalid operator descriptor: {descriptor_paths[0]}")
        function = _binding(project)
        refs = reference_map.get(normalize(project.name), [])
        parameters = [item.get("name") for key in ("input_desc", "attr")
                      for item in descriptor.get(key, []) if isinstance(item, dict) and item.get("name")]
        entries[project.name] = {
            "definition": str(descriptor_paths[0].relative_to(project_root)),
            "extension_module": f"{function}_ext",
            "function": function,
            "parameters": parameters,
            "reference_candidates": refs,
            "project": project.name,
            "vendor": "evokernel_" + function.removesuffix("_custom"),
        }
    return entries
