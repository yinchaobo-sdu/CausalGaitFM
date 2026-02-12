"""Benchmark script for Table 5: computational efficiency comparison.

Measures inference time, throughput, and (GPU) memory usage for different
sequence lengths across model architectures:
  - Transformer
  - CNN-LSTM
  - Mamba (vanilla backbone)
  - CausalGaitFM (full model)

Usage:
  python -m project.benchmark
  python -m project.benchmark --seq-lengths 128 256 512 1024 4096 --batch-size 32
  python -m project.benchmark --device cpu
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from project.model import CausalGaitModel
from project.baselines.transformer import TransformerModel
from project.baselines.cnn_lstm import CNNLSTMModel
from project.baselines.backbone_shared import SharedBackbone


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _warmup(model: torch.nn.Module, x: Tensor, n: int = 5) -> None:
    """Warmup runs to stabilize CUDA kernels."""
    model.eval()
    with torch.no_grad():
        for _ in range(n):
            model(x)


def measure_inference_time(
    model: torch.nn.Module,
    x: Tensor,
    n_runs: int = 50,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Measure mean/std inference time per batch (ms) and throughput (samples/sec)."""
    model.eval()
    times_ms = []

    with torch.no_grad():
        for _ in range(n_runs):
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if device is not None and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000)

    times = np.array(times_ms)
    batch_size = x.size(0)
    mean_ms = float(times.mean())
    std_ms = float(times.std())
    throughput = batch_size / (mean_ms / 1000.0)  # samples/sec

    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "throughput_samples_per_sec": round(throughput, 1),
    }


def measure_gpu_memory(
    model: torch.nn.Module,
    x: Tensor,
) -> float:
    """Measure peak GPU memory usage (MB) during inference. Returns 0 for CPU."""
    if x.device.type != "cuda":
        return 0.0
    torch.cuda.reset_peak_memory_stats(x.device)
    model.eval()
    with torch.no_grad():
        model(x)
    peak_mb = torch.cuda.max_memory_allocated(x.device) / (1024 ** 2)
    return round(peak_mb, 1)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_models(input_dim: int, device: torch.device) -> dict[str, torch.nn.Module]:
    """Build all models for comparison with matched hyperparameters."""
    models = {}

    models["Transformer"] = TransformerModel(
        input_dim=input_dim,
        num_classes=4,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.0,
    )

    models["CNN-LSTM"] = CNNLSTMModel(
        input_dim=input_dim,
        num_classes=4,
        cnn_channels=(64, 128, 128),
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.0,
    )

    models["Mamba"] = SharedBackbone(
        input_dim=input_dim,
        num_classes=4,
        d_model=128,
        d_state=16,
        n_layers=4,
        dropout=0.0,
    )

    models["CausalGaitFM"] = CausalGaitModel(
        input_dim=input_dim,
        seq_len=256,  # placeholder, auto-adjusts
        d_model=128,
        d_state=16,
        n_layers=4,
        causal_dim=32,
        domain_dim=16,
        num_disease_classes=4,
        num_fall_classes=3,
        num_frailty_classes=5,
        dropout=0.0,
    )

    for name, m in models.items():
        m.to(device).eval()

    return models


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    seq_lengths: Sequence[int] = (128, 256, 512, 1024, 4096),
    batch_size: int = 32,
    input_dim: int = 32,
    n_runs: int = 50,
    device_str: str = "auto",
    output_dir: str = "outputs/benchmark",
) -> dict:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Benchmark device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence lengths: {list(seq_lengths)}")
    print(f"N runs per measurement: {n_runs}")
    print()

    models = build_models(input_dim, device)

    # Print parameter counts
    print("Parameter counts:")
    for name, m in models.items():
        print(f"  {name}: {count_parameters(m):,}")
    print()

    results = {}

    for seq_len in seq_lengths:
        results[seq_len] = {}
        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        print(f"--- Sequence length: {seq_len} ---")

        for model_name, model in models.items():
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            try:
                _warmup(model, x, n=3)
                timing = measure_inference_time(model, x, n_runs=n_runs, device=device)
                mem_mb = measure_gpu_memory(model, x)

                results[seq_len][model_name] = {
                    **timing,
                    "gpu_memory_mb": mem_mb,
                    "params": count_parameters(model),
                }
                print(
                    f"  {model_name:15s} | {timing['mean_ms']:7.2f}ms +- {timing['std_ms']:5.2f}ms | "
                    f"throughput={timing['throughput_samples_per_sec']:8.1f} samples/s | "
                    f"mem={mem_mb:.1f}MB"
                )
            except Exception as e:
                print(f"  {model_name:15s} | FAILED: {e}")
                results[seq_len][model_name] = {"error": str(e)}

        print()

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "benchmark_results.json"
    # Convert int keys to string for JSON
    serializable = {str(k): v for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Results saved to {json_path}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark computational efficiency (Table 5)")
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[128, 256, 512, 1024, 4096])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/benchmark")
    args = parser.parse_args()

    run_benchmark(
        seq_lengths=args.seq_lengths,
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        n_runs=args.n_runs,
        device_str=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
