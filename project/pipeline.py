"""End-to-end local training pipeline for RTX 40-series single-GPU setups.

Sequence:
  1) Environment checks
  2) Download datasets
  3) Preprocess datasets
  4) Smoke test training
  5) Full experiments (cross-domain / ablation / baselines / in-domain / single-task)
  6) Benchmark
  7) Paper tables and figures
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

from project.train import TrainConfig, train


T = TypeVar("T")


@dataclass
class PipelineProfile:
    num_epochs: int = 10
    batch_size: int = 32
    early_stop_patience: int = 4
    num_workers: int = 4
    benchmark_runs: int = 10
    benchmark_seq_lengths: tuple[int, ...] = (128, 256, 512, 1024, 4096)


PROFILES: dict[str, PipelineProfile] = {
    "local_4060_full": PipelineProfile(),
}

STAGE_ORDER: tuple[str, ...] = (
    "download",
    "preprocess",
    "smoke",
    "cross_domain",
    "ablation",
    "baselines",
    "in_domain",
    "single_task",
    "benchmark",
    "paper",
)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {device}. Use auto|cuda|cpu.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def _str2bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    text = value.strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _print_environment(device: str) -> None:
    resolved = _resolve_device(device)
    print(f"[env] torch={torch.__version__}")
    print(f"[env] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")
    print(f"[env] selected_device={resolved}")


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def _run_with_auto_batch(
    fn: Callable[[TrainConfig], T],
    cfg: TrainConfig,
    auto_batch: bool,
    min_batch_size: int,
    step_name: str,
) -> T:
    current_batch = cfg.batch_size
    while True:
        attempt_cfg = copy.deepcopy(cfg)
        attempt_cfg.batch_size = current_batch
        try:
            print(f"[{step_name}] running with batch_size={current_batch}")
            result = fn(attempt_cfg)
            print(f"[{step_name}] completed with batch_size={current_batch}")
            return result
        except RuntimeError as exc:
            if not auto_batch or not _is_oom(exc):
                raise
            next_batch = max(min_batch_size, current_batch // 2)
            if next_batch >= current_batch:
                raise
            print(
                f"[{step_name}] CUDA OOM at batch_size={current_batch}; "
                f"retrying with batch_size={next_batch}"
            )
            current_batch = next_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _run_benchmark_with_auto_batch(
    output_dir: str,
    input_dim: int,
    batch_size: int,
    n_runs: int,
    seq_lengths: tuple[int, ...],
    device: str,
    auto_batch: bool,
    min_batch_size: int,
) -> dict:
    from project.benchmark import run_benchmark

    current_batch = batch_size
    while True:
        try:
            print(f"[benchmark] running with batch_size={current_batch}")
            return run_benchmark(
                seq_lengths=seq_lengths,
                batch_size=current_batch,
                input_dim=input_dim,
                n_runs=n_runs,
                device_str=device,
                output_dir=output_dir,
            )
        except RuntimeError as exc:
            if not auto_batch or not _is_oom(exc):
                raise
            next_batch = max(min_batch_size, current_batch // 2)
            if next_batch >= current_batch:
                raise
            print(
                f"[benchmark] CUDA OOM at batch_size={current_batch}; "
                f"retrying with batch_size={next_batch}"
            )
            current_batch = next_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _generate_tables_and_figures(results_dir: Path, output_dir: Path, fmt: str = "markdown") -> None:
    from project.generate_paper_tables import (
        format_table_2,
        format_table_3,
        format_table_4,
        format_table_5,
        generate_ablation_figure,
        generate_efficiency_figure,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "table2": format_table_2(results_dir, fmt),
        "table3": format_table_3(results_dir, fmt),
        "table4": format_table_4(results_dir, fmt),
        "table5": format_table_5(results_dir, fmt),
    }

    combined = []
    for content in tables.values():
        combined.append(content)
        combined.append("")

    ext = "tex" if fmt == "latex" else "md"
    table_path = output_dir / f"all_tables.{ext}"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined))
    print(f"[paper] tables written to {table_path}")

    generate_ablation_figure(results_dir, output_dir / "figure2a_ablation.png")
    generate_efficiency_figure(results_dir, output_dir / "figure2d_efficiency.png")


def _make_base_config(
    profile: PipelineProfile,
    args: argparse.Namespace,
    resolved_device: str,
) -> TrainConfig:
    return TrainConfig(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        input_dim=args.input_dim,
        seq_len=args.seq_len,
        batch_size=args.batch_size or profile.batch_size,
        num_epochs=args.num_epochs or profile.num_epochs,
        lr=args.lr,
        early_stop_patience=args.early_stop_patience or profile.early_stop_patience,
        num_workers=args.num_workers if args.num_workers is not None else profile.num_workers,
        target_domain=args.target_domain,
        use_dummy_data=False,
        device=resolved_device,
        auto_batch=args.auto_batch,
        min_batch_size=args.min_batch_size,
        run_id=args.run_id,
        control_dir=args.control_dir,
        check_stop_every=args.check_stop_every,
        save_every_steps=args.save_every_steps,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        allow_tf32=bool(args.allow_tf32),
        cudnn_benchmark=bool(args.cudnn_benchmark),
        pin_memory=bool(args.pin_memory),
        persistent_workers=bool(args.persistent_workers),
        prefetch_factor=args.prefetch_factor,
    )


def _default_progress_path(output_dir: str) -> Path:
    return Path(output_dir) / "pipeline_state.json"


def _stop_file(control_dir: str, run_id: str) -> Path:
    return Path(control_dir) / f"{run_id}.stop"


def _stop_requested(control_dir: str, run_id: str) -> bool:
    return _stop_file(control_dir, run_id).exists()


def _ensure_control_ready(control_dir: str, run_id: str) -> None:
    Path(control_dir).mkdir(parents=True, exist_ok=True)
    marker = _stop_file(control_dir, run_id)
    if marker.exists():
        print(f"[control] removing stale stop marker: {marker}")
        marker.unlink(missing_ok=True)


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _normalize_state(state: dict[str, Any], run_id: str) -> dict[str, Any]:
    now = int(time.time())
    normalized = dict(state)
    normalized["run_id"] = normalized.get("run_id") or run_id
    normalized["status"] = normalized.get("status") or "new"
    completed = normalized.get("completed_stages")
    if not isinstance(completed, list):
        completed = []
    normalized["completed_stages"] = [s for s in completed if s in STAGE_ORDER]
    normalized["started_at"] = _coerce_int(normalized.get("started_at"), now)
    normalized["updated_at"] = _coerce_int(normalized.get("updated_at"), now)
    normalized["stages_total"] = len(STAGE_ORDER)
    normalized["stages_completed"] = _coerce_int(normalized.get("stages_completed"), 0)
    normalized["progress_percent"] = _coerce_float(normalized.get("progress_percent"), 0.0)
    normalized["eta_seconds"] = normalized.get("eta_seconds")
    normalized["last_error"] = normalized.get("last_error")
    current_stage = normalized.get("current_stage")
    if not isinstance(current_stage, dict):
        current_stage = {"name": "", "index": 0, "total": len(STAGE_ORDER)}
    current_stage["name"] = current_stage.get("name") or ""
    current_stage["index"] = _coerce_int(current_stage.get("index"), 0)
    current_stage["total"] = len(STAGE_ORDER)
    normalized["current_stage"] = current_stage
    current_substage = normalized.get("current_substage")
    if not isinstance(current_substage, dict):
        current_substage = {"name": "", "done": 0, "total": 1}
    current_substage["name"] = current_substage.get("name") or ""
    current_substage["done"] = max(0, _coerce_int(current_substage.get("done"), 0))
    current_substage["total"] = max(1, _coerce_int(current_substage.get("total"), 1))
    if current_substage["done"] > current_substage["total"]:
        current_substage["done"] = current_substage["total"]
    normalized["current_substage"] = current_substage
    return normalized


def _refresh_progress_metrics(state: dict[str, Any], disable_eta: bool) -> None:
    now = int(time.time())
    completed = [s for s in state.get("completed_stages", []) if s in STAGE_ORDER]
    completed_count = len(completed)
    current_stage_name = state.get("current_stage", {}).get("name", "")
    stage_fraction = 0.0
    if current_stage_name in STAGE_ORDER and current_stage_name not in completed:
        sub = state.get("current_substage", {})
        sub_total = max(1, _coerce_int(sub.get("total"), 1))
        sub_done = min(sub_total, max(0, _coerce_int(sub.get("done"), 0)))
        stage_fraction = float(sub_done) / float(sub_total)

    overall_fraction = (completed_count + stage_fraction) / float(len(STAGE_ORDER))
    if state.get("status") == "completed":
        overall_fraction = 1.0
    overall_fraction = max(0.0, min(1.0, overall_fraction))

    state["stages_total"] = len(STAGE_ORDER)
    state["stages_completed"] = completed_count
    state["progress_percent"] = round(overall_fraction * 100.0, 2)

    if disable_eta or state.get("status") != "running" or overall_fraction <= 0.0:
        state["eta_seconds"] = None
    else:
        started_at = _coerce_int(state.get("started_at"), now)
        elapsed = max(0, now - started_at)
        eta = int(elapsed * (1.0 / overall_fraction - 1.0))
        state["eta_seconds"] = max(0, eta)


def _set_current_stage(state: dict[str, Any], stage_name: str) -> None:
    stage_index = STAGE_ORDER.index(stage_name) + 1
    state["current_stage"] = {
        "name": stage_name,
        "index": stage_index,
        "total": len(STAGE_ORDER),
    }
    state["current_substage"] = {"name": "", "done": 0, "total": 1}


def _set_current_substage(state: dict[str, Any], sub_name: str, done: int, total: int) -> None:
    total_safe = max(1, int(total))
    done_safe = max(0, min(int(done), total_safe))
    state["current_substage"] = {
        "name": sub_name,
        "done": done_safe,
        "total": total_safe,
    }


def _mark_stage_completed(state: dict[str, Any], stage_name: str) -> None:
    completed = state.setdefault("completed_stages", [])
    if stage_name not in completed:
        completed.append(stage_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline for local RTX 4060 training")
    parser.add_argument("--profile", default="local_4060_full", choices=sorted(PROFILES.keys()))
    parser.add_argument("--device", default="auto", help="auto|cuda|cpu")
    parser.add_argument("--auto-batch", action="store_true", help="Halve batch size on CUDA OOM and retry")
    parser.add_argument("--min-batch-size", type=int, default=4)
    parser.add_argument("--progress-interval-sec", type=float, default=2.0)
    parser.add_argument("--disable-progress-eta", action="store_true")
    parser.add_argument("--progress-file", default=None)

    parser.add_argument("--run-id", default="default")
    parser.add_argument("--control-dir", default="outputs/control")
    parser.add_argument("--resume", action="store_true", help="Resume from pipeline state file")

    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--target-domain", default="daphnet")

    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--benchmark-runs", type=int, default=None)

    parser.add_argument("--use-amp", type=_str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--amp-dtype", default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--allow-tf32", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--cudnn-benchmark", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--pin-memory", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--persistent-workers", type=_str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)

    parser.add_argument("--check-stop-every", type=int, default=20)
    parser.add_argument("--save-every-steps", type=int, default=200)

    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-paper", action="store_true")
    args = parser.parse_args()

    profile = PROFILES[args.profile]
    resolved_device = _resolve_device(args.device)
    _print_environment(args.device)
    _ensure_control_ready(control_dir=args.control_dir, run_id=args.run_id)

    progress_path = Path(args.progress_file) if args.progress_file else _default_progress_path(args.output_dir)
    state = _load_state(progress_path) if args.resume else {}
    state = _normalize_state(state=state, run_id=args.run_id)
    state["status"] = "running"
    state["last_error"] = None
    if not args.resume:
        state["started_at"] = int(time.time())
        state["completed_stages"] = []
        state["current_stage"] = {"name": "", "index": 0, "total": len(STAGE_ORDER)}
        state["current_substage"] = {"name": "", "done": 0, "total": 1}
    elif state.get("run_id") != args.run_id:
        state["run_id"] = args.run_id

    print(f"[pipeline] progress file: {progress_path}")
    if args.resume:
        print(f"[pipeline] resume enabled, loaded state from {progress_path}")

    last_save_ts = 0.0
    progress_interval = max(0.0, float(args.progress_interval_sec))

    def persist_state(force: bool = False) -> None:
        nonlocal last_save_ts
        _refresh_progress_metrics(state, disable_eta=bool(args.disable_progress_eta))
        now = time.time()
        if not force and progress_interval > 0 and (now - last_save_ts) < progress_interval:
            return
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        state["updated_at"] = int(now)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        last_save_ts = now

    base_cfg = _make_base_config(profile=profile, args=args, resolved_device=resolved_device)

    def make_stage_progress_cb(stage_name: str) -> Callable[[dict[str, Any]], None]:
        def _cb(event: dict[str, Any]) -> None:
            if stage_name != event.get("stage_name", stage_name):
                return
            sub_name = str(event.get("sub_name", "") or "")
            sub_done = _coerce_int(event.get("sub_done"), state["current_substage"]["done"])
            sub_total = _coerce_int(event.get("sub_total"), state["current_substage"]["total"])
            _set_current_substage(state, sub_name=sub_name, done=sub_done, total=sub_total)
            persist_state(force=False)

        return _cb

    def run_stage(stage_name: str, fn: Callable[[], None]) -> bool:
        if stage_name in state.get("completed_stages", []):
            print(f"[pipeline] stage '{stage_name}' already completed, skipping")
            return True
        if _stop_requested(args.control_dir, args.run_id):
            state["status"] = "stopped"
            state["stopped_stage"] = stage_name
            _set_current_stage(state, stage_name)
            persist_state(force=True)
            print(f"[pipeline] stop requested before stage '{stage_name}', exiting.")
            return False

        _set_current_stage(state, stage_name)
        state["status"] = "running"
        state["last_error"] = None
        persist_state(force=True)
        print(f"[pipeline] running stage '{stage_name}'")

        try:
            fn()
        except Exception as exc:
            state["status"] = "failed"
            state["last_error"] = f"{type(exc).__name__}: {exc}"
            state["failed_stage"] = stage_name
            persist_state(force=True)
            print(f"[pipeline] stage '{stage_name}' failed: {exc}")
            raise

        _set_current_substage(state, sub_name=state["current_substage"]["name"], done=state["current_substage"]["total"], total=state["current_substage"]["total"])
        _mark_stage_completed(state, stage_name)
        state["status"] = "running"
        state.pop("failed_stage", None)
        persist_state(force=True)

        if _stop_requested(args.control_dir, args.run_id):
            state["status"] = "stopped"
            state["stopped_stage"] = stage_name
            persist_state(force=True)
            print(f"[pipeline] stop requested after stage '{stage_name}', exiting.")
            return False
        return True

    def _stage_download() -> None:
        if args.skip_download:
            print("[pipeline] download skipped")
            return
        from project.data.download import download_all_datasets

        download_all_datasets(raw_dir=args.raw_dir)

    def _stage_preprocess() -> None:
        if args.skip_preprocess:
            print("[pipeline] preprocess skipped")
            return
        from project.data.preprocess import preprocess_all_datasets

        preprocess_all_datasets(raw_dir=args.raw_dir, processed_dir=args.processed_dir)

    def _stage_smoke() -> None:
        if args.skip_smoke:
            print("[pipeline] smoke skipped")
            return
        smoke_cfg = copy.deepcopy(base_cfg)
        smoke_cfg.use_dummy_data = True
        smoke_cfg.num_epochs = 1
        smoke_cfg.early_stop_patience = 1
        smoke_cfg.enable_tsne = False
        smoke_cfg.output_dir = str(Path(args.output_dir) / "smoke_gpu")
        _run_with_auto_batch(
            fn=train,
            cfg=smoke_cfg,
            auto_batch=args.auto_batch,
            min_batch_size=args.min_batch_size,
            step_name="smoke",
        )

    def _stage_cross_domain() -> None:
        from project.run_experiments import run_cross_domain

        progress_cb = make_stage_progress_cb("cross_domain")
        _run_with_auto_batch(
            fn=lambda cfg: run_cross_domain(cfg, progress_cb=progress_cb),
            cfg=base_cfg,
            auto_batch=args.auto_batch,
            min_batch_size=args.min_batch_size,
            step_name="cross_domain",
        )

    def _stage_ablation() -> None:
        from project.run_experiments import run_ablation

        progress_cb = make_stage_progress_cb("ablation")
        _run_with_auto_batch(
            fn=lambda cfg: run_ablation(cfg, progress_cb=progress_cb),
            cfg=base_cfg,
            auto_batch=args.auto_batch,
            min_batch_size=args.min_batch_size,
            step_name="ablation",
        )

    def _stage_baselines() -> None:
        from project.run_experiments import run_baselines

        progress_cb = make_stage_progress_cb("baselines")
        _run_with_auto_batch(
            fn=lambda cfg: run_baselines(cfg, progress_cb=progress_cb),
            cfg=base_cfg,
            auto_batch=args.auto_batch,
            min_batch_size=args.min_batch_size,
            step_name="baselines",
        )

    def _stage_in_domain() -> None:
        from project.run_experiments import run_in_domain

        progress_cb = make_stage_progress_cb("in_domain")
        _run_with_auto_batch(
            fn=lambda cfg: run_in_domain(cfg, progress_cb=progress_cb),
            cfg=base_cfg,
            auto_batch=args.auto_batch,
            min_batch_size=args.min_batch_size,
            step_name="in_domain",
        )

    def _stage_single_task() -> None:
        from project.run_experiments import run_single_task

        progress_cb = make_stage_progress_cb("single_task")
        _run_with_auto_batch(
            fn=lambda cfg: run_single_task(cfg, progress_cb=progress_cb),
            cfg=base_cfg,
            auto_batch=args.auto_batch,
            min_batch_size=args.min_batch_size,
            step_name="single_task",
        )

    def _stage_benchmark() -> None:
        if args.skip_benchmark:
            print("[pipeline] benchmark skipped")
            return
        _run_benchmark_with_auto_batch(
            output_dir=str(Path(args.output_dir) / "benchmark"),
            input_dim=base_cfg.input_dim,
            batch_size=base_cfg.batch_size,
            n_runs=args.benchmark_runs or profile.benchmark_runs,
            seq_lengths=profile.benchmark_seq_lengths,
            device=resolved_device,
            auto_batch=args.auto_batch,
            min_batch_size=args.min_batch_size,
        )

    def _stage_paper() -> None:
        if args.skip_paper:
            print("[pipeline] paper artifacts skipped")
            return
        _generate_tables_and_figures(
            results_dir=Path(args.output_dir),
            output_dir=Path(args.output_dir) / "paper",
            fmt="markdown",
        )

    stages: list[tuple[str, Callable[[], None]]] = [
        ("download", _stage_download),
        ("preprocess", _stage_preprocess),
        ("smoke", _stage_smoke),
        ("cross_domain", _stage_cross_domain),
        ("ablation", _stage_ablation),
        ("baselines", _stage_baselines),
        ("in_domain", _stage_in_domain),
        ("single_task", _stage_single_task),
        ("benchmark", _stage_benchmark),
        ("paper", _stage_paper),
    ]

    for stage_name, stage_fn in stages:
        ok = run_stage(stage_name, stage_fn)
        if not ok:
            return

    state["status"] = "completed"
    state.pop("stopped_stage", None)
    persist_state(force=True)
    print("[pipeline] all stages completed")


if __name__ == "__main__":
    main()
