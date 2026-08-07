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


def prepare_call(name: str, item: dict, project: Path, torch):
    reference_path = REFERENCES / item["reference_candidates"][0]
    if not reference_path.is_file():
        raise RuntimeError(f"本地 PyTorch reference 不存在：{reference_path}")
    reference = load_python_file(reference_path)
    init_args = reference.get_init_inputs() if hasattr(reference, "get_init_inputs") else []
    model = reference.Model(*init_args).eval().npu()
    inputs = [to_npu(value, torch) for value in reference.get_inputs()]

    forward_names = list(inspect.signature(model.forward).parameters)
    tensor_values = {norm(key): value for key, value in zip(forward_names, inputs)}
    tensor_values.update({norm(key): value for key, value in model.named_parameters()})
    tensor_values.update({norm(key): value for key, value in model.named_buffers()})

    custom_args = []
    unused_inputs = iter(inputs)
    for parameter in item["parameters"]:
        key = norm(parameter)
        if key == "self" and inputs:
            custom_args.append(inputs[0])
            continue
        if key in tensor_values:
            custom_args.append(tensor_values[key])
            continue
        matches = {
            id(value): value
            for candidate, value in tensor_values.items()
            if candidate.endswith(key) or key.endswith(candidate)
        }
        if len(matches) == 1:
            custom_args.append(next(iter(matches.values())))
            continue
        try:
            custom_args.append(next(unused_inputs))
        except StopIteration as error:
            raise RuntimeError(
                f"{name} 的 C++ 参数 {parameter!r} 无法唯一映射，候选数={len(matches)}"
            ) from error

    sys.path.insert(0, str(project / "CppExtension"))
    extension = importlib.import_module(item["extension_module"])
    custom = getattr(extension, item["function"])
    return model, inputs, custom, custom_args, reference_path


def precision(name: str, project_dir: str | None = None) -> None:
    import torch
    import torch_npu  # noqa: F401 - 注册 torch.npu 后端

    item, project = resolve_operator(name, project_dir)
    torch.manual_seed(1234)
    model, inputs, custom, custom_args, reference_path = prepare_call(
        name, item, project, torch
    )
    with torch.inference_mode():
        expected = model(*inputs)
        actual = custom(*custom_args)
    torch.npu.synchronize()
    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise RuntimeError("当前精度脚本要求 reference 和自定义算子均返回单个 Tensor")
    if actual.shape != expected.shape:
        raise RuntimeError(f"输出 shape 不一致：actual={actual.shape}, expected={expected.shape}")

    diff = (actual.float() - expected.float()).abs()
    print(f"operator={name}")
    print(f"reference={reference_path}")
    print(f"shape={tuple(actual.shape)}, dtype={actual.dtype}")
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    print(f"max_abs={max_abs:.8f}")
    print(f"mean_abs={mean_abs:.8f}")
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
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
