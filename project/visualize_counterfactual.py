"""Counterfactual signal visualization script for Figure 3.

Generates plots comparing:
  (a) Original gait signal
  (b) Counterfactual signal (domain-swapped reconstruction)
  (c) Difference signal (highlights what changed)
  (d) Gait feature comparison (step regularity metrics)

Usage:
  python -m project.visualize_counterfactual --checkpoint outputs/best_model.pth
  python -m project.visualize_counterfactual --use-dummy-data  # for debugging
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

try:
    from project.model import CausalGaitModel
    from project.train import TrainConfig, make_dummy_batch, set_seed
except ModuleNotFoundError:
    from model import CausalGaitModel
    from train import TrainConfig, make_dummy_batch, set_seed


# ---------------------------------------------------------------------------
# Gait feature extraction (biomechanical metrics)
# ---------------------------------------------------------------------------

def compute_gait_features(signal: np.ndarray) -> dict[str, float]:
    """Extract simple gait features from a 1D sensor signal.

    signal: [T, D] -- time x channels.
    Returns dict of biomechanical-inspired metrics.
    """
    # Use the magnitude of all channels
    magnitude = np.sqrt((signal ** 2).sum(axis=-1))  # [T]

    # Step regularity via autocorrelation
    mag_centered = magnitude - magnitude.mean()
    autocorr = np.correlate(mag_centered, mag_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    else:
        autocorr = np.zeros_like(autocorr)

    # Find first significant peak after lag 0 (step regularity)
    peaks = []
    for i in range(2, len(autocorr) - 1):
        if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
            peaks.append((i, autocorr[i]))
    step_regularity = peaks[0][1] if peaks else 0.0

    # Signal variability
    rms = float(np.sqrt(np.mean(magnitude ** 2)))
    signal_std = float(np.std(magnitude))

    # Approximate cadence (dominant frequency via FFT)
    fft_vals = np.abs(np.fft.rfft(magnitude))
    freqs = np.fft.rfftfreq(len(magnitude))
    if len(fft_vals) > 1:
        # Exclude DC
        dominant_freq_idx = np.argmax(fft_vals[1:]) + 1
        dominant_freq = float(freqs[dominant_freq_idx])
    else:
        dominant_freq = 0.0

    return {
        "step_regularity": round(step_regularity, 4),
        "rms_magnitude": round(rms, 4),
        "signal_std": round(signal_std, 4),
        "dominant_freq": round(dominant_freq, 6),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_counterfactual_comparison(
    original: np.ndarray,
    counterfactual: np.ndarray,
    channel_idx: int = 0,
    source_domain: str = "Source",
    target_domain: str = "Target",
    save_path: str | Path = "outputs/counterfactual_figure3.png",
) -> Path:
    """Generate Figure 3-style visualization.

    Args:
        original: [T, D] original signal.
        counterfactual: [T, D] counterfactual signal.
        channel_idx: which sensor channel to visualize.
        source_domain: label for source domain.
        target_domain: label for target domain.
        save_path: output path.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Visualization requires matplotlib.") from e

    T = original.shape[0]
    t_axis = np.arange(T)

    orig_ch = original[:, channel_idx]
    cf_ch = counterfactual[:, channel_idx]
    diff_ch = cf_ch - orig_ch

    # Compute gait features
    orig_features = compute_gait_features(original)
    cf_features = compute_gait_features(counterfactual)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # (a) Original signal
    axes[0, 0].plot(t_axis, orig_ch, color="steelblue", linewidth=1.0)
    axes[0, 0].set_title(f"(a) Original Signal ({source_domain})")
    axes[0, 0].set_xlabel("Time step")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].grid(alpha=0.3)

    # (b) Counterfactual signal
    axes[0, 1].plot(t_axis, cf_ch, color="coral", linewidth=1.0)
    axes[0, 1].set_title(f"(b) Counterfactual Signal ({target_domain})")
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Amplitude")
    axes[0, 1].grid(alpha=0.3)

    # (c) Difference signal
    axes[1, 0].fill_between(t_axis, diff_ch, alpha=0.4, color="mediumpurple")
    axes[1, 0].plot(t_axis, diff_ch, color="darkviolet", linewidth=0.8)
    axes[1, 0].axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    axes[1, 0].set_title("(c) Difference Signal (CF - Original)")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Amplitude Diff")
    axes[1, 0].grid(alpha=0.3)

    # (d) Gait feature comparison
    features = list(orig_features.keys())
    orig_vals = [orig_features[k] for k in features]
    cf_vals = [cf_features[k] for k in features]

    x_pos = np.arange(len(features))
    width = 0.35
    bars1 = axes[1, 1].bar(x_pos - width / 2, orig_vals, width, label="Original",
                            color="steelblue", alpha=0.8)
    bars2 = axes[1, 1].bar(x_pos + width / 2, cf_vals, width, label="Counterfactual",
                            color="coral", alpha=0.8)
    axes[1, 1].set_title("(d) Gait Feature Comparison")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 3: Counterfactual Gait Signal Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {output_path}")
    return output_path


def generate_counterfactual_from_model(
    model: CausalGaitModel,
    x: Tensor,
    source_domain_ids: Tensor,
    target_domain_id: int = 0,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate counterfactual signal using the model.

    Returns (original, counterfactual) both as numpy arrays [T, D].
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        out = model(x.to(device), sample_domain=False)
        z_c = out["z_c"]
        z_d = out["z_d"]

        # Swap domain: generate new domain latent for target domain
        target_ids = torch.full((z_c.size(0),), target_domain_id, dtype=torch.long, device=device)
        x_cf, _ = model.generate_counterfactuals(
            z_c=z_c, z_d=z_d, batch_domain_ids=target_ids,
        )

    # Take first sample
    original = x[0].cpu().numpy()
    counterfactual = x_cf[0].cpu().numpy()
    return original, counterfactual


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Counterfactual visualization (Figure 3)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--use-dummy-data", action="store_true", help="Use random data for demo")
    parser.add_argument("--output", default="outputs/counterfactual_figure3.png")
    parser.add_argument("--channel-idx", type=int, default=0, help="Channel to visualize")
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    cfg = TrainConfig(input_dim=args.input_dim, seq_len=args.seq_len)
    model = CausalGaitModel(
        input_dim=cfg.input_dim, seq_len=cfg.seq_len,
        d_model=cfg.d_model, d_state=cfg.d_state, n_layers=cfg.n_layers,
        scales=cfg.scales, bidirectional=cfg.bidirectional, backend=cfg.backend,
        causal_dim=cfg.causal_dim, domain_dim=cfg.domain_dim,
        num_disease_classes=cfg.num_disease_classes,
        num_fall_classes=cfg.num_fall_classes,
        num_frailty_classes=cfg.num_frailty_classes,
    ).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint from {args.checkpoint}")
    else:
        print("[WARN] No checkpoint loaded, using random model weights (for demo).")

    # Generate data
    if args.use_dummy_data:
        batch = make_dummy_batch(cfg, device)
    else:
        batch = make_dummy_batch(cfg, device)
        print("[INFO] No real data loader; using dummy data.")

    original, counterfactual = generate_counterfactual_from_model(
        model=model,
        x=batch["x"],
        source_domain_ids=batch["domain_id"],
        target_domain_id=0,
        device=device,
    )

    # Generate visualization
    plot_counterfactual_comparison(
        original=original,
        counterfactual=counterfactual,
        channel_idx=args.channel_idx,
        source_domain="Domain A",
        target_domain="Domain B",
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
