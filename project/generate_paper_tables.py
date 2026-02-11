"""Results aggregation and paper table/figure generation script.

Reads experiment results from JSON files and generates:
  - Table 2: Cross-domain generalization results
  - Table 3: In-domain classification results
  - Table 4: Multi-task vs single-task comparison
  - Table 5: Computational efficiency comparison
  - Figure 2: Ablation study plots

Usage:
  python -m project.generate_paper_tables --results-dir outputs
  python -m project.generate_paper_tables --results-dir outputs --format latex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    print(f"[WARN] Results file not found: {path}")
    return None


# ---------------------------------------------------------------------------
# Table formatters
# ---------------------------------------------------------------------------

def format_table_2(results_dir: Path, fmt: str = "markdown") -> str:
    """Table 2: Cross-domain generalization results."""
    cross_domain = _load_json(results_dir / "cross_domain" / "results.json")
    baselines = _load_json(results_dir / "baselines" / "results.json")

    if not cross_domain and not baselines:
        return "Table 2: No results available yet.\n"

    lines = []
    if fmt == "latex":
        lines.append(r"\begin{table}[h]")
        lines.append(r"\caption{Cross-Domain Generalization Results (Table 2)}")
        lines.append(r"\begin{tabular}{l|cccccc|c}")
        lines.append(r"\hline")
        lines.append(r"Method & Daphnet & UCI-HAR & PAMAP2 & mHealth & WISDM & Opportunity & Avg \\")
        lines.append(r"\hline")
    else:
        lines.append("## Table 2: Cross-Domain Generalization Results")
        lines.append("")
        lines.append("| Method | Daphnet | UCI-HAR | PAMAP2 | mHealth | WISDM | Opportunity | Avg |")
        lines.append("|--------|---------|---------|--------|---------|-------|-------------|-----|")

    domains = ["daphnet", "ucihar", "pamap2", "mhealth", "wisdm", "opportunity"]

    # Baselines
    if baselines:
        for method, metrics in baselines.items():
            acc = metrics.get("accuracy", 0) * 100
            if fmt == "latex":
                lines.append(f"{method} & \\multicolumn{{6}}{{c|}}{{--}} & {acc:.1f} \\\\")
            else:
                lines.append(f"| {method} | -- | -- | -- | -- | -- | -- | {acc:.1f}% |")

    # CausalGaitFM
    if cross_domain:
        vals = []
        for d in domains:
            if d in cross_domain:
                vals.append(cross_domain[d].get("disease_acc", 0) * 100)
            else:
                vals.append(0.0)
        avg = np.mean(vals)
        val_strs = [f"{v:.1f}" for v in vals]
        if fmt == "latex":
            lines.append(
                f"CausalGaitFM & {' & '.join(val_strs)} & \\textbf{{{avg:.1f}}} \\\\"
            )
            lines.append(r"\hline")
            lines.append(r"\end{tabular}")
            lines.append(r"\end{table}")
        else:
            val_strs_md = [f"{v:.1f}%" for v in vals]
            lines.append(
                f"| **CausalGaitFM** | {' | '.join(val_strs_md)} | **{avg:.1f}%** |"
            )

    lines.append("")
    return "\n".join(lines)


def format_table_3(results_dir: Path, fmt: str = "markdown") -> str:
    """Table 3: In-domain classification results."""
    in_domain = _load_json(results_dir / "in_domain" / "results.json")

    if not in_domain:
        return "Table 3: No in-domain results available yet.\n"

    lines = []
    lines.append("## Table 3: In-Domain Classification Results (5-fold CV)")
    lines.append("")
    lines.append("| Fold | Disease Acc | Disease F1 | Fall Acc | Fall F1 | Frailty MAE | Frailty OA |")
    lines.append("|------|------------|------------|----------|---------|-------------|------------|")

    for fold_name, m in in_domain.items():
        d_acc = m.get("disease_acc", 0) * 100
        d_f1 = m.get("disease_macro_f1", 0) * 100
        f_acc = m.get("fall_acc", 0) * 100
        f_f1 = m.get("fall_macro_f1", 0) * 100
        fr_mae = m.get("frailty_mae", 0)
        fr_oa = m.get("frailty_ordinal_acc", 0) * 100
        bold = "**" if fold_name == "average" else ""
        lines.append(
            f"| {bold}{fold_name}{bold} | {d_acc:.1f}% | {d_f1:.1f}% | "
            f"{f_acc:.1f}% | {f_f1:.1f}% | {fr_mae:.3f} | {fr_oa:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def format_table_4(results_dir: Path, fmt: str = "markdown") -> str:
    """Table 4: Single-task vs multi-task comparison."""
    single_task = _load_json(results_dir / "single_task" / "results.json")

    if not single_task:
        return "Table 4: No single-task results available yet.\n"

    lines = []
    lines.append("## Table 4: Multi-Task vs Single-Task Prediction")
    lines.append("")
    lines.append("| Mode | Disease Acc | Disease AUC | Fall Acc | Fall OA | Frailty MAE | Frailty OA |")
    lines.append("|------|------------|-------------|----------|---------|-------------|------------|")

    for mode, m in single_task.items():
        d_acc = m.get("disease_acc", 0) * 100
        d_auc = m.get("disease_auc", 0) * 100
        f_acc = m.get("fall_acc", 0) * 100
        f_oa = m.get("fall_ordinal_acc", 0) * 100
        fr_mae = m.get("frailty_mae", 0)
        fr_oa = m.get("frailty_ordinal_acc", 0) * 100
        lines.append(
            f"| {mode} | {d_acc:.1f}% | {d_auc:.1f}% | {f_acc:.1f}% | "
            f"{f_oa:.1f}% | {fr_mae:.3f} | {fr_oa:.1f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def format_table_5(results_dir: Path, fmt: str = "markdown") -> str:
    """Table 5: Computational efficiency comparison."""
    benchmark = _load_json(results_dir / "benchmark" / "benchmark_results.json")

    if not benchmark:
        return "Table 5: No benchmark results available yet.\n"

    lines = []
    lines.append("## Table 5: Computational Efficiency Comparison")
    lines.append("")

    # Collect all models from first seq_len
    first_key = list(benchmark.keys())[0]
    models = list(benchmark[first_key].keys())

    lines.append(f"| Seq Length | {'  |  '.join(models)} |")
    lines.append(f"|-----------|{'|'.join(['--------' for _ in models])}|")

    for seq_len, model_results in benchmark.items():
        vals = []
        for m in models:
            if m in model_results and "mean_ms" in model_results[m]:
                vals.append(f"{model_results[m]['mean_ms']:.1f}ms")
            else:
                vals.append("--")
        lines.append(f"| {seq_len} | {' | '.join(vals)} |")

    lines.append("")

    # Parameter counts
    lines.append("### Parameter Counts")
    lines.append("")
    for m in models:
        params = benchmark[first_key].get(m, {}).get("params", 0)
        lines.append(f"- {m}: {params:,}")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def generate_ablation_figure(results_dir: Path, output_path: Path) -> Path | None:
    """Figure 2a: Progressive ablation bar chart."""
    ablation = _load_json(results_dir / "ablation" / "results.json")

    if not ablation:
        print("No ablation results for figure generation.")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available for figure generation.")
        return None

    names = list(ablation.keys())
    accs = [ablation[n].get("disease_acc", 0) * 100 for n in names]
    f1s = [ablation[n].get("disease_macro_f1", 0) * 100 for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w / 2, accs, w, label="Accuracy (%)", color="steelblue", alpha=0.85)
    ax.bar(x + w / 2, f1s, w, label="Macro-F1 (%)", color="coral", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Figure 2(a): Progressive Ablation Study")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Ablation figure saved to {output_path}")
    return output_path


def generate_efficiency_figure(results_dir: Path, output_path: Path) -> Path | None:
    """Figure 2d / Table 5: Inference time vs sequence length."""
    benchmark = _load_json(results_dir / "benchmark" / "benchmark_results.json")

    if not benchmark:
        print("No benchmark results for figure generation.")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available for figure generation.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    first_key = list(benchmark.keys())[0]
    models = list(benchmark[first_key].keys())
    seq_lens = [int(k) for k in benchmark.keys()]

    colors = {"Transformer": "red", "CNN-LSTM": "orange",
              "Mamba": "green", "CausalGaitFM": "blue"}

    for model_name in models:
        times = []
        for sl in seq_lens:
            r = benchmark[str(sl)].get(model_name, {})
            times.append(r.get("mean_ms", float("nan")))
        color = colors.get(model_name, None)
        ax.plot(seq_lens, times, "o-", label=model_name, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Inference Time (ms)")
    ax.set_title("Figure 2(d): Computational Efficiency")
    ax.set_xscale("log", base=2)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Efficiency figure saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate paper tables and figures")
    parser.add_argument("--results-dir", default="outputs", help="Root results directory")
    parser.add_argument("--output-dir", default="outputs/paper", help="Output directory for tables/figures")
    parser.add_argument("--format", default="markdown", choices=["markdown", "latex"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate tables
    tables = {
        "table2": format_table_2(results_dir, args.format),
        "table3": format_table_3(results_dir, args.format),
        "table4": format_table_4(results_dir, args.format),
        "table5": format_table_5(results_dir, args.format),
    }

    # Write combined results file
    combined = []
    for name, content in tables.items():
        combined.append(content)
        combined.append("")

    ext = "tex" if args.format == "latex" else "md"
    table_path = output_dir / f"all_tables.{ext}"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("\n".join(combined))
    print(f"All tables written to {table_path}")

    # Generate figures
    generate_ablation_figure(results_dir, output_dir / "figure2a_ablation.png")
    generate_efficiency_figure(results_dir, output_dir / "figure2d_efficiency.png")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
