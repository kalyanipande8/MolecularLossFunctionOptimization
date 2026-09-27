"""
End-to-end orchestration: generate -> naive baseline -> weight optimization
-> optimally-weighted eval -> report + figures.

Run with: python3 run_all.py
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_compound_data import generate, train_val_test_split, PROPERTY_NAMES, SEED
from loss_functions import ALL_LOSS_NAMES
from train_multi_objective import search

OUT_DIR = "output"
FIG_DIR = os.path.join(OUT_DIR, "figures")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=== 1. Generating synthetic compound data ===")
    df = generate()
    df.to_csv(os.path.join(OUT_DIR, "compounds.csv"), index=False)
    train_df, val_df, test_df = train_val_test_split(df)
    print(f"{len(df)} compounds, {len(PROPERTY_NAMES)} properties, "
          f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("=== 2/3. Naive baseline + optimal weight search ===")
    naive_best, optimized_best, trajectory = search(
        (train_df, val_df, test_df), seed=SEED, n_naive=6, n_extra=30
    )

    naive_acc = naive_best["agg_test_accuracy"]
    opt_acc = optimized_best["agg_test_accuracy"]
    improvement_pct = (opt_acc - naive_acc) / naive_acc * 100.0

    print(f"Naive equal-weighting aggregate test accuracy:     {naive_acc:.4f}")
    print(f"Optimally-weighted aggregate test accuracy:        {opt_acc:.4f}")
    print(f"Improvement:                                       {improvement_pct:+.1f}%")

    print("=== 4. Writing report and figures ===")
    write_report(naive_best, optimized_best, improvement_pct, trajectory)
    plot_per_property(naive_best, optimized_best)
    plot_weight_distribution(optimized_best, trajectory)
    print("Done. See output/report.md and output/figures/*.png")


def write_report(naive_best, optimized_best, improvement_pct, trajectory):
    lines = []
    lines.append("# Multi-Objective Molecular Property Loss Optimization -- Results\n")
    lines.append(f"Aggregate test accuracy (mean of 12 normalized per-property scores):\n")
    lines.append(f"- Naive equal-weighting baseline: **{naive_best['agg_test_accuracy']:.4f}**")
    lines.append(f"- Optimally-weighted model:       **{optimized_best['agg_test_accuracy']:.4f}**")
    lines.append(f"- Improvement:                    **{improvement_pct:+.1f}%**\n")

    lines.append("## Per-property test accuracy (naive vs optimized)\n")
    lines.append("| Property | Naive | Optimized | Delta |")
    lines.append("|---|---|---|---|")
    for p in PROPERTY_NAMES:
        n = naive_best["test_acc_per_property"][p]
        o = optimized_best["test_acc_per_property"][p]
        lines.append(f"| {p} | {n:.3f} | {o:.3f} | {o - n:+.3f} |")

    lines.append("\n## Selected configurations\n")
    lines.append(f"- Naive config: `{naive_best['config']}`")
    lines.append(f"- Naive loss weights (all equal): `{naive_best['W']}`")
    lines.append(f"- Optimized config: `{optimized_best['config']}`")
    lines.append("- Optimized loss weights:")
    for ln in ALL_LOSS_NAMES:
        lines.append(f"  - {ln}: {optimized_best['W'][ln]:.3f}")

    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_per_property(naive_best, optimized_best):
    props = PROPERTY_NAMES
    naive_vals = [naive_best["test_acc_per_property"][p] for p in props]
    opt_vals = [optimized_best["test_acc_per_property"][p] for p in props]

    x = np.arange(len(props))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, naive_vals, width, label="Naive equal-weighting", color="#8896a8")
    ax.bar(x + width / 2, opt_vals, width, label="Optimally-weighted", color="#2f6fed")
    ax.set_xticks(x)
    ax.set_xticklabels(props, rotation=45, ha="right")
    ax.set_ylabel("Normalized test accuracy (0-1)")
    ax.set_title("Per-property accuracy: naive vs. optimally-weighted multi-objective loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "per_property_accuracy.png"), dpi=150)
    plt.close(fig)


def plot_weight_distribution(optimized_best, trajectory):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # (a) final optimized weight distribution across the 8 custom losses
    names = ALL_LOSS_NAMES
    weights = [optimized_best["W"][n] for n in names]
    axes[0].barh(names, weights, color="#2f6fed")
    axes[0].set_xlabel("Optimized weight")
    axes[0].set_title("Final loss-weight distribution (optimized run)")
    axes[0].invert_yaxis()

    # (b) weight-search trajectory: validation aggregate accuracy per
    # candidate evaluated during the search, naive vs optimized candidates
    naive_accs = [acc for is_naive, W, acc in trajectory if is_naive]
    opt_accs = [acc for is_naive, W, acc in trajectory if not is_naive]
    axes[1].scatter(range(len(naive_accs)), naive_accs, color="#8896a8",
                     label="naive-pool candidates (equal weights)")
    axes[1].scatter(range(len(naive_accs), len(naive_accs) + len(opt_accs)), opt_accs,
                     color="#2f6fed", label="weight-search candidates")
    axes[1].axhline(max(naive_accs), color="#8896a8", linestyle="--", linewidth=1)
    axes[1].axhline(max(opt_accs), color="#2f6fed", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Candidate index (search order)")
    axes[1].set_ylabel("Validation aggregate accuracy")
    axes[1].set_title("Weight-search trajectory")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "weight_search_trajectory.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
