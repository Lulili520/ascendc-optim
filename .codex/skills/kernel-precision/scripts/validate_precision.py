#!/usr/bin/env python3
"""编译 AscendC Host/Kernel，并与本地 PyTorch reference 做单输入精度验证。"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[4]
PROJECT_ROOT = WORKSPACE / "kernel/KernelBench910B"
MANIFEST = PROJECT_ROOT / "manifest.json"
REFERENCES = WORKSPACE / "kernel/pytorch-references/KernelBench"
ACTIVATE = Path("/data/lu/activate_evokernel.sh")
ATOL = RTOL = 1e-2
REFERENCE_CACHE = Path("/data/lu/ascendc-tmp/kernel_precision_reference_cache")
REFERENCE_SEED = 1234


def quote(value: Path | str) -> str:
    return shlex.quote(str(value))


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def resolve_operator(name: str, project_dir: str | None = None) -> tuple[dict, Path]:
    items = load_manifest()
    if name not in items:
        matches = [candidate for candidate in items if candidate.lower() == name.lower()]
        hint = f"；是否是 {matches[0]}" if matches else ""
        raise SystemExit(f"未知算子：{name}{hint}")
    item = items[name]
    project = Path(project_dir).resolve() if project_dir else PROJECT_ROOT / item["project"]
    if not project.is_dir():
        raise SystemExit(f"算子工程不存在：{project}")
    refs = item["reference_candidates"]
    if len(refs) != 1:
        raise SystemExit(f"{name} 必须唯一映射一个 reference，当前为：{refs}")
    return item, project


def run_bash(script: str, log_path: Path) -> None:
    log_path.parent.mkdir(exist_ok=True)
    completed = subprocess.run(
        ["bash", "-lc", script], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    log_path.write_text(output, encoding="utf-8")
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, completed.args, output=output)


def source_fingerprint(project: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        path for folder in ("op_host", "op_kernel")
        for path in (project / folder).rglob("*") if path.is_file()
    )
    if not files:
        raise RuntimeError("op_host/op_kernel 中没有源码文件")
    for path in files:
        digest.update(path.relative_to(project).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_result(project: Path, result: dict) -> None:
    result_dir = project / "precision"
    result_dir.mkdir(exist_ok=True)
    (result_dir / "precision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (result_dir / "precision.log").write_text(
        "\n".join(f"{key}={value}" for key, value in result.items()) + "\n",
        encoding="utf-8",
    )
    workspace_path = project / "workspace.json"
    workspace = json.loads(workspace_path.read_text()) if workspace_path.is_file() else {}
    workspace.update({
        "operator": result["operator"],
        "status": "PRECISION_PASS",
        "source_fingerprint": result["source_fingerprint"],
        "precision": {
            "status": "PASS", "exit_code": 0,
            "max_abs": result["max_abs"], "mean_abs": result["mean_abs"],
            "result": "precision/precision.json",
        },
        "performance": None,
        "updated_at": result["completed_at"],
    })
    workspace_path.write_text(
        json.dumps(workspace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_failure(
    project: Path, name: str, error: BaseException,
    failure_stage: str, stage: str, log_path: Path,
) -> None:
    completed_at = datetime.now(timezone.utc).isoformat()
    fingerprint = source_fingerprint(project)
    exit_code = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
    if not log_path.is_file():
        log_path.parent.mkdir(exist_ok=True)
        log_path.write_text(str(error) + "\n", encoding="utf-8")
    relative_log = log_path.relative_to(project).as_posix()
    result = {
        "operator": name,
        "status": "FAILED",
        "exit_code": exit_code,
        "failure_stage": failure_stage,
        "stage": stage,
        "reason": str(error),
        "log": relative_log,
        "source_fingerprint": fingerprint,
        "completed_at": completed_at,
    }
    result_dir = project / "precision"
    result_dir.mkdir(exist_ok=True)
    (result_dir / "precision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (result_dir / "precision.log").write_text(
        "\n".join(f"{key}={value}" for key, value in result.items()) + "\n",
        encoding="utf-8",
    )
    workspace_path = project / "workspace.json"
    workspace = json.loads(workspace_path.read_text()) if workspace_path.is_file() else {}
    workspace.update({
        "operator": name,
        "status": "BUILD_FAILED" if failure_stage == "build" else "PRECISION_FAILED",
        "source_fingerprint": fingerprint,
        "precision": {
            "status": "NOT_RUN" if failure_stage == "build" else "FAILED",
            "exit_code": exit_code,
            "failure_stage": failure_stage,
            "stage": stage,
            "reason": str(error),
            "result": "precision/precision.json",
            "log": relative_log,
        },
        "performance": None,
        "updated_at": completed_at,
    })
    workspace_path.write_text(
        json.dumps(workspace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def build_opp(name: str, item: dict, project: Path, log_path: Path) -> None:
    print(f"[1/3] 编译并部署 {name} 的 op_host/op_kernel", flush=True)
    run_bash(
        f"""
set -euo pipefail
source {quote(ACTIVATE)}
cd {quote(project)}
grep -q '"value": "ascend910b"' CMakePresets.json
bash build.sh
package=$(find build_out -maxdepth 1 -type f -name 'custom_opp_*.run' -print -quit)
test -n "$package"
bash "$package" --quiet --install-path="$ASCEND_HOME_PATH/opp"
""",
        log_path,
    )


def ensure_extension(name: str, item: dict, project: Path, log_path: Path) -> None:
    extension_dir = project / "CppExtension"
    module = item["extension_module"]
    if list(extension_dir.glob(f"{module}*.so")):
        print("[2/3] 复用已有 PyTorch 扩展", flush=True)
        return

    vendor_env = (
        Path("/usr/local/Ascend/cann-9.0.0/opp/vendors")
        / item["vendor"]
        / "bin/set_env.bash"
    )
    if not vendor_env.is_file():
        raise RuntimeError(f"OPP vendor 尚未安装，不能编译扩展：{vendor_env}")
    print("[2/3] 首次编译 PyTorch 扩展", flush=True)
    run_bash(
        f"""
set -euo pipefail
source {quote(ACTIVATE)}
set +u
source {quote(vendor_env)}
set -u
cd {quote(extension_dir)}
python setup.py build_ext --inplace --force
""",
        log_path,
    )


def run_precision_process(name: str, item: dict, project: Path, log_path: Path) -> None:
    vendor_env = (
        Path("/usr/local/Ascend/cann-9.0.0/opp/vendors")
        / item["vendor"]
        / "bin/set_env.bash"
    )
    if not vendor_env.is_file():
        raise RuntimeError(f"找不到算子 OPP 环境：{vendor_env}")
    print("[3/3] 与本地 PyTorch reference 比较精度", flush=True)
    run_bash(
        f"""
set -euo pipefail
source {quote(ACTIVATE)}
set +u
source {quote(vendor_env)}
set -u
export PYTHONPATH={quote(project / 'CppExtension')}${{PYTHONPATH:+:${{PYTHONPATH}}}}
exec python {quote(Path(__file__).resolve())} --internal-run {quote(name)} \
  --project-dir {quote(project)}
""",
        log_path,
    )


def cleanup_artifacts(item: dict, project: Path) -> None:
    """删除本轮可重新生成的文件，保留扩展 .so 和已安装 OPP。"""
    reference_path = REFERENCES / item["reference_candidates"][0]
    targets = [
        project / "build_out",
        project / "CppExtension/build",
        project / "CppExtension/__pycache__",
        reference_path.parent / "__pycache__",
        Path(__file__).resolve().parent / "__pycache__",
    ]
    allowed_roots = (project.resolve(), REFERENCES.resolve(), WORKSPACE.resolve())

    print("[清理] 删除精度验证中间文件", flush=True)
    for target in targets:
        resolved = target.resolve()
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            raise RuntimeError(f"拒绝清理工作区外路径：{resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
            print(f"[清理] removed {resolved}", flush=True)


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower()).removesuffix("opt")


def extension_function_name(function: str) -> str:
    """Manifest may store either a pybind name or a complete torch schema."""
    return function.split("(", 1)[0].strip()


def extension_parameters(project: Path, function: str) -> list[tuple[str, str]]:
    """Read the checked-in pybind wrapper signature without modifying it."""
    source = project / "CppExtension/csrc/op.cpp"
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    binding = re.search(
        rf'm\.def\(\s*"{re.escape(function)}"\s*,\s*&([A-Za-z_]\w*)', text, re.S
    )
    if binding is None:
        raise RuntimeError(f"CppExtension 中找不到 pybind 函数 {function!r}")
    implementation = binding.group(1)
    declaration = re.search(
        rf'\b{re.escape(implementation)}\s*\((.*?)\)\s*(?:\{{|;)', text, re.S
    )
    if declaration is None:
        raise RuntimeError(f"CppExtension 中找不到 {implementation!r} 的参数声明")
    parameter_text = declaration.group(1)
    result = []
    for raw in parameter_text.split(","):
        parameter = raw.strip()
        match = re.search(r"([A-Za-z_]\w*)\s*$", parameter)
        if match is None:
            raise RuntimeError(f"无法解析 C++ 参数：{parameter!r}")
        result.append((match.group(1), parameter[:match.start(1)].strip()))
    return result


def expected_tensor_shape(project: Path, parameter: str) -> tuple[int, ...] | None:
    text = (project / "CppExtension/csrc/op.cpp").read_text(encoding="utf-8")
    match = re.search(
        rf'"{re.escape(parameter)} must be \[([0-9, ]+)\]', text
    )
    if match:
        return tuple(int(value.strip()) for value in match.group(1).split(","))
    return None


def parameter_variants(parameter: str) -> list[str]:
    key = norm(parameter)
    variants = [key]
    replacements = {
        "convbiasopt": "convbias", "biasopt": "bias",
        "lingamma": "linearweight", "linbias": "linearbias",
        "gngamma": "groupnormweight", "gnbeta": "groupnormbias",
        "gamma": "groupnormweight", "beta": "groupnormbias",
        "scaling": "scalingfactor", "scale": "scalefactor",
        "sub": "subtractvalue", "mul": "multiplyvalue", "c": "constant",
    }
    if key in replacements:
        variants.append(replacements[key])
    if key == "w":
        variants.append("weight")
    elif key == "b":
        variants.append("bias")
    elif key.endswith("w"):
        variants.append(key[:-1] + "weight")
    elif key.endswith("b"):
        variants.append(key[:-1] + "bias")
    return list(dict.fromkeys(variants))


def packed_parameter(parameter: str, named: list[tuple[str, object]], inputs: list, project: Path, torch):
    key = norm(parameter)
    family = None
    if key in {"wpacked", "bpacked"}:
        family = "weight" if key.startswith("w") else "bias"
    elif key in {"wih", "whh", "bih", "bhh"}:
        family = {
            "wih": "weightih", "whh": "weighthh",
            "bih": "biasih", "bhh": "biashh",
        }[key]
    if family:
        tensors = [value for name, value in named if family in norm(name)]
        if not tensors:
            return None
        if key in {"wih", "whh"}:
            width = max(tensor.shape[1] for tensor in tensors)
            padded = []
            for tensor in tensors:
                if tensor.shape[1] < width:
                    pad = torch.zeros(
                        (tensor.shape[0], width - tensor.shape[1]),
                        dtype=tensor.dtype, device=tensor.device,
                    )
                    tensor = torch.cat((tensor, pad), dim=1)
                padded.append(tensor)
            return torch.cat(padded, dim=0).contiguous()
        return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0).contiguous()
    shape = expected_tensor_shape(project, parameter)
    if key.endswith("buf") and shape:
        template = next((value for value in inputs if isinstance(value, torch.Tensor)), None)
        if template is not None:
            return torch.empty(shape, dtype=template.dtype, device=template.device).contiguous()
    return None


def host_input_dtypes(project: Path) -> dict[str, str]:
    result = {}
    for source in (project / "op_host").rglob("*.cpp"):
        text = source.read_text(encoding="utf-8")
        for match in re.finditer(
            r'this->Input\("([^"]+)"\)(.*?)(?=this->(?:Input|Output)\(|\Z)', text, re.S
        ):
            dtype = re.search(r"\.DataType\(\{\s*ge::(DT_[A-Z0-9_]+)", match.group(2))
            if dtype:
                result[norm(match.group(1))] = dtype.group(1)
    return result


def adapt_argument(value, cpp_type: str, parameter: str, dtypes: dict, inputs: list, torch):
    is_tensor = "Tensor" in cpp_type
    if is_tensor and not isinstance(value, torch.Tensor):
        template = next((item for item in inputs if isinstance(item, torch.Tensor)), None)
        dtype = template.dtype if template is not None and template.is_floating_point() else torch.float32
        value = torch.tensor(value, dtype=dtype).npu().contiguous()
    elif not is_tensor and isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise RuntimeError(f"标量参数 {parameter!r} 收到非标量 Tensor")
        value = value.item()

    dtype_name = dtypes.get(norm(parameter))
    dtype_map = {
        "DT_FLOAT": torch.float32,
        "DT_FLOAT16": torch.float16,
        "DT_INT32": torch.int32,
        "DT_INT64": torch.int64,
        "DT_BOOL": torch.bool,
        "DT_UINT8": torch.uint8,
    }
    if isinstance(value, torch.Tensor) and dtype_name in dtype_map and value.dtype != dtype_map[dtype_name]:
        value = value.to(dtype_map[dtype_name]).contiguous()
    return value


def load_python_file(path: Path):
    spec = importlib.util.spec_from_file_location(f"reference_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"无法加载 reference：{path}")
    spec.loader.exec_module(module)
    return module


def to_npu(value, torch):
    if isinstance(value, torch.Tensor):
        return value.npu().contiguous()
    if isinstance(value, tuple):
        return tuple(to_npu(x, torch) for x in value)
    if isinstance(value, list):
        return [to_npu(x, torch) for x in value]
    return value


def to_cpu(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, tuple):
        return tuple(to_cpu(x, torch) for x in value)
    if isinstance(value, list):
        return [to_cpu(x, torch) for x in value]
    return value


def reference_cache_path(name: str, reference_path: Path, init_args: list, torch) -> Path:
    digest = hashlib.sha256()
    digest.update(reference_path.read_bytes())
    digest.update(repr(init_args).encode())
    digest.update(str(torch.__version__).encode())
    digest.update(str(REFERENCE_SEED).encode())
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return REFERENCE_CACHE / f"{safe_name}.{digest.hexdigest()[:20]}.pt"


def prepare_call(name: str, item: dict, project: Path, torch):
    reference_path = REFERENCES / item["reference_candidates"][0]
    if not reference_path.is_file():
        raise RuntimeError(f"本地 PyTorch reference 不存在：{reference_path}")
    reference = load_python_file(reference_path)
    init_args = reference.get_init_inputs() if hasattr(reference, "get_init_inputs") else []
    model = reference.Model(*init_args).eval().npu()
    cache_path = reference_cache_path(name, reference_path, init_args, torch)
    inputs = None
    expected = None
    if cache_path.is_file():
        print(f"REFERENCE_CACHE_HIT={cache_path}", flush=True)
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        inputs_cpu = cached["inputs"]
        expected_cpu = cached["expected"]
    else:
        print("INPUT_START", flush=True)
        inputs_cpu = reference.get_inputs()
        print("INPUT_DONE", flush=True)
        inputs_for_reference = [to_npu(value, torch) for value in inputs_cpu]
        print("REFERENCE_START", flush=True)
        with torch.inference_mode():
            expected = model(*inputs_for_reference)
        torch.npu.synchronize()
        print("REFERENCE_DONE", flush=True)
        if not isinstance(expected, torch.Tensor):
            raise RuntimeError("当前精度脚本要求 reference 返回单个 Tensor")
        expected_cpu = expected.detach().cpu().contiguous()
        inputs_cpu = to_cpu(inputs_cpu, torch)
        REFERENCE_CACHE.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".pt.tmp")
        torch.save({"inputs": inputs_cpu, "expected": expected_cpu}, temporary)
        temporary.replace(cache_path)
        print(f"REFERENCE_CACHE_SAVED={cache_path}", flush=True)
        inputs = inputs_for_reference
    if inputs is None:
        inputs = [to_npu(value, torch) for value in inputs_cpu]
    if expected is None:
        expected = expected_cpu.npu().contiguous()
    del inputs_cpu, expected_cpu

    forward_names = list(inspect.signature(model.forward).parameters)
    values = {norm(key): value for key, value in zip(forward_names, inputs)}
    init_names = [
        key for key in inspect.signature(reference.Model.__init__).parameters if key != "self"
    ]
    values.update({norm(key): value for key, value in zip(init_names, init_args)})
    for key in init_names:
        if norm(key) not in values and hasattr(model, key):
            values[norm(key)] = getattr(model, key)
    named_parameters = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    named_items = list(named_parameters.items()) + list(named_buffers.items())
    values.update({norm(key): value for key, value in named_parameters.items()})
    values.update({norm(key): value for key, value in named_buffers.items()})

    aliases = {
        "self": norm(forward_names[0]) if forward_names else "",
        "predict": "predictions",
        "target": "targets",
        "gamma": "gnweight",
        "beta": "gnbias",
        "wdepthwise": "depthwiseweight",
        "wpointwise": "pointwiseweight",
    }
    function = extension_function_name(item["function"])
    parameters = extension_parameters(project, function)
    dtypes = host_input_dtypes(project)
    custom_args = []
    used_named_values = set()
    for position, (parameter, cpp_type) in enumerate(parameters):
        key = norm(parameter)
        lookup = aliases.get(key, key)
        lookup = lookup.removesuffix("double").removesuffix("float")
        variants = parameter_variants(lookup)
        value = next((values[candidate] for candidate in variants if candidate in values), None)
        if value is None:
            ranked = []
            for order, (candidate_name, candidate_value) in enumerate(named_items):
                candidate = norm(candidate_name)
                if id(candidate_value) in used_named_values:
                    continue
                score = max(
                    (
                        100 if candidate == variant else
                        80 if candidate.endswith(variant) else
                        70 if variant.endswith(candidate) else
                        0
                    )
                    for variant in variants
                )
                if key.startswith("conv") and "conv" in candidate:
                    score += 20
                if key.startswith("gn") and any(token in candidate for token in ("gn", "groupnorm", "norm")):
                    score += 20
                if key.startswith("bn") and any(token in candidate for token in ("bn", "batchnorm")):
                    score += 20
                if score:
                    ranked.append((score, -order, candidate_value))
            if ranked:
                ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
                value = ranked[0][2]
                used_named_values.add(id(value))
        if value is None:
            value = packed_parameter(parameter, named_items, inputs, project, torch)
        if value is None and position < len(inputs):
            value = inputs[position]
        if value is None:
            raise RuntimeError(
                f"{name} 的 C++ 参数 {parameter!r} 无法唯一映射"
            )
        custom_args.append(adapt_argument(value, cpp_type, parameter, dtypes, inputs, torch))

    sys.path.insert(0, str(project / "CppExtension"))
    extension = importlib.import_module(item["extension_module"])
    custom = getattr(extension, function)
    return inputs, expected, custom, custom_args, reference_path


def precision(name: str, project_dir: str | None = None) -> None:
    import torch
    import torch_npu  # noqa: F401 - 注册 torch.npu 后端

    item, project = resolve_operator(name, project_dir)
    torch.manual_seed(REFERENCE_SEED)
    inputs, expected, custom, custom_args, reference_path = prepare_call(
        name, item, project, torch
    )
    print("CUSTOM_START", flush=True)
    with torch.inference_mode():
        actual = custom(*custom_args)
    torch.npu.synchronize()
    print("CUSTOM_DONE", flush=True)
    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise RuntimeError("当前精度脚本要求 reference 和自定义算子均返回单个 Tensor")
    if actual.shape != expected.shape:
        raise RuntimeError(f"输出 shape 不一致：actual={actual.shape}, expected={expected.shape}")

    print("COMPARE_START", flush=True)
    diff = (actual.float() - expected.float()).abs()
    print(f"operator={name}")
    print(f"reference={reference_path}")
    print(f"shape={tuple(actual.shape)}, dtype={actual.dtype}")
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={mean_abs:.8f}")
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    print("COMPARE_DONE", flush=True)
    result = {
        "operator": name, "status": "PASS", "exit_code": 0,
        "reference": str(reference_path), "output_shape": list(actual.shape),
        "output_dtype": str(actual.dtype), "atol": ATOL, "rtol": RTOL,
        "max_abs": max_abs, "mean_abs": mean_abs,
        "source_fingerprint": source_fingerprint(project),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_result(project, result)
    print(f"precision=PASS (atol={ATOL}, rtol={RTOL})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operator", nargs="?", help="manifest 中的算子工程名")
    parser.add_argument(
        "--project-dir", help="覆盖 manifest 中的工程目录，用于验证工作区版本"
    )
    parser.add_argument("--skip-build", action="store_true", help="复用已安装 OPP")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="保留 build_out、扩展构建目录和 Python 缓存，用于排查失败",
    )
    parser.add_argument("--internal-run", metavar="OPERATOR", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.internal_run:
        precision(args.internal_run, args.project_dir)
        return
    if not args.operator:
        parser.error("必须提供 OperatorName")

    item, project = resolve_operator(args.operator, args.project_dir)
    failure_stage = "build"
    stage = "opp_build"
    log_path = project / "precision/opp_build.log"
    try:
        try:
            if args.skip_build:
                print("[1/3] 跳过 OPP 构建，复用已安装版本", flush=True)
            else:
                build_opp(args.operator, item, project, log_path)
            stage = "extension_build"
            log_path = project / "precision/extension_build.log"
            ensure_extension(args.operator, item, project, log_path)
            failure_stage = "precision"
            stage = "precision_run"
            log_path = project / "precision/precision_run.log"
            run_precision_process(args.operator, item, project, log_path)
        except (Exception, SystemExit) as error:
            write_failure(
                project, args.operator, error,
                failure_stage=failure_stage, stage=stage, log_path=log_path,
            )
            raise
    finally:
        if args.keep_artifacts:
            print("[清理] 已按 --keep-artifacts 保留中间文件", flush=True)
        else:
            cleanup_artifacts(item, project)


if __name__ == "__main__":
    main()
