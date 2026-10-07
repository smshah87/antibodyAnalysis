"""
Compares one channel's intensity (e.g. nuclear Channel 1) between cells called
positive vs. negative in another channel (e.g. Channel 2), pooled across every
position/image under an input directory. Runs a Mann-Whitney U test (rank-based,
robust to the right-skewed/non-normal intensity distributions typical here --
unlike a t-test, its p-value is unaffected by a log transform) and saves a violin
plot with the p-value annotated.

Also fits a blocked ANCOVA (OLS on log1p-transformed intensity, with position
dummy-coded as a fixed blocking factor and the channel call as the effect of
interest) -- this is the position-adjusted counterpart to the naive pooled
tests above, and matters because pooling across positions with different
baseline intensities can produce a spurious pooled trend that doesn't hold
within any individual position (a Simpson's-paradox-style artifact).

Expects *_cells_calls.csv files (call_and_plot_positivity.py output) somewhere
under --input_dir; searches recursively, so it works whether they're one level
down (per-position subfolders from run_full_pipeline.py) or in the root.

Optionally restrict to a subpopulation first with --filter_channel/--filter_call
(e.g. only Channel-1-positive cells) before splitting that subpopulation by
--group_channel -- e.g. comparing Channel 3 intensity between Channel-2-positive
and Channel-2-negative cells, but only among cells that are also Channel-1-positive.

Usage:
    python compare_intensity_by_call.py --input_dir raw/ \
        --value_channel 1 --value_compartment Nucleus --value_stat Mean \
        --group_channel 2

    # H3 (channel 3) intensity in DLL3+ (channel 2) vs DLL3- cells, restricted
    # to SOX9-positive (channel 1) cells only
    python compare_intensity_by_call.py --input_dir raw/ \
        --filter_channel 1 --filter_call positive \
        --value_channel 3 --value_compartment Nucleus --value_stat Mean \
        --group_channel 2
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import statsmodels.api as sm

# Validated categorical palette (dataviz skill reference palette.md, slots 1 & 2)
COLOR_NEG = "#2a78d6"  # blue
COLOR_POS = "#eb6834"  # orange


def run(input_dir, value_channel, value_compartment, value_stat, group_channel, output_dir=None,
        filter_channel=None, filter_call="positive"):
    value_col = f"{value_compartment}: Channel {value_channel}: {value_stat}"
    group_col = f"ch{group_channel}_call"
    output_dir = output_dir or input_dir
    filter_suffix = f"_filteredch{filter_channel}{filter_call}" if filter_channel is not None else ""
    filter_desc = f" (restricted to ch{filter_channel}_call={filter_call})" if filter_channel is not None else ""

    csv_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*_cells_calls.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No *_cells_calls.csv files found under {input_dir}")
    print(f"Found {len(csv_paths)} *_cells_calls.csv files")

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["_position"] = os.path.basename(os.path.dirname(p))
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    # Restrict to a subpopulation (e.g. Channel-1-positive) before splitting it
    # by group_col -- e.g. comparing H3 between DLL3+/- cells, but only among
    # cells that are also SOX9-positive.
    if filter_channel is not None:
        filter_col = f"ch{filter_channel}_call"
        if filter_col not in all_df.columns:
            raise KeyError(f"'{filter_col}' not found -- available call columns: "
                            f"{[c for c in all_df.columns if c.endswith('_call')]}")
        n_before = len(all_df)
        all_df = all_df[all_df[filter_col] == filter_call].reset_index(drop=True)
        print(f"Filtered to {filter_col}={filter_call}: {len(all_df)}/{n_before} cells")

    if group_col not in all_df.columns:
        raise KeyError(f"'{group_col}' not found -- available call columns: "
                        f"{[c for c in all_df.columns if c.endswith('_call')]}")
    if value_col not in all_df.columns:
        raise KeyError(f"'{value_col}' not found -- available columns include: "
                        f"{[c for c in all_df.columns if f'Channel {value_channel}' in c]}")

    sub = all_df[all_df[group_col].isin(["positive", "negative"])].copy()
    pos_vals = sub.loc[sub[group_col] == "positive", value_col].values
    neg_vals = sub.loc[sub[group_col] == "negative", value_col].values

    print(f"Total cells (all positions): {len(all_df)}")
    print(f"{group_col}=positive: n={len(pos_vals)}, {group_col}=negative: n={len(neg_vals)} "
          f"(ambiguous excluded)")
    print("\nPer-position breakdown:")
    print(sub.groupby("_position")[group_col].value_counts().unstack(fill_value=0))

    stat, p_value = mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
    print(f"\nMann-Whitney U = {stat:.1f}, p = {p_value:.3e}")
    print(f"Median {value_col}: positive={np.median(pos_vals):.1f}, negative={np.median(neg_vals):.1f}")

    # Welch's t-test (unequal variances assumed) on log1p-transformed values --
    # a parametric cross-check of the Mann-Whitney result above.
    t_stat, t_p_value = ttest_ind(np.log1p(pos_vals), np.log1p(neg_vals), equal_var=False)
    print(f"\nWelch's t-test on log1p({value_col}): t = {t_stat:.3f}, p = {t_p_value:.3e}")
    print(f"Mean log1p({value_col}): positive={np.mean(np.log1p(pos_vals)):.3f}, "
          f"negative={np.mean(np.log1p(neg_vals)):.3f}")

    # Blocked ANCOVA: position dummy-coded as a fixed blocking factor (each
    # position gets its own intercept), group tested after that's accounted
    # for -- unlike the two tests above, this isn't fooled by positions having
    # different baseline intensities (Simpson's-paradox-style pooling artifact).
    log_value = np.log1p(sub[value_col].values)
    group_bin = (sub[group_col].values == "positive").astype(float)
    position_dummies = pd.get_dummies(sub["_position"], prefix="pos", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat(
        [pd.Series(group_bin, name="group", index=sub.index), position_dummies], axis=1
    ))
    block_model = sm.OLS(log_value, design).fit()
    block_coef = block_model.params["group"]
    block_p = block_model.pvalues["group"]
    print(f"\nBlocked ANCOVA (position as fixed blocking factor), OLS on log1p({value_col}):")
    print(f"  group (positive vs negative) coefficient = {block_coef:+.4f}, p = {block_p:.3e}")
    block_se = block_model.bse["group"]

    # --- Forest plot: per-position effect (with 95% CI) + overall blocked estimate ---
    # A position needs at least 2 cells in EACH group to get a within-position
    # effect + SE at all (0 cells -> undefined mean, 1 cell -> undefined variance);
    # skipped positions still contribute to the blocked ANCOVA overall estimate
    # above, just not to this per-position breakdown.
    forest_rows = []
    n_skipped = 0
    for pos_name, pos_sub in sub.groupby("_position"):
        log_p = np.log1p(pos_sub.loc[pos_sub[group_col] == "positive", value_col].values)
        log_n = np.log1p(pos_sub.loc[pos_sub[group_col] == "negative", value_col].values)
        if len(log_p) < 2 or len(log_n) < 2:
            n_skipped += 1
            continue
        diff = log_p.mean() - log_n.mean()
        se = np.sqrt(log_p.var(ddof=1) / len(log_p) + log_n.var(ddof=1) / len(log_n))
        short_label = re.search(r"Pos\d+", pos_name)
        forest_rows.append({
            "label": short_label.group() if short_label else pos_name,
            "effect": diff, "se": se, "n": len(log_p) + len(log_n), "overall": False,
        })
    forest_rows.append({
        "label": "Overall (blocked ANCOVA)", "effect": block_coef, "se": block_se,
        "n": len(sub), "overall": True,
    })
    if n_skipped:
        print(f"\n{n_skipped} position(s) skipped in the forest plot (fewer than 2 cells in one "
              f"{group_col} group -- still included in the blocked ANCOVA overall estimate above)")
    forest_df = pd.DataFrame(forest_rows)

    fig2, ax2 = plt.subplots(figsize=(7.5, 0.7 * len(forest_df) + 1.5))
    y_pos = np.arange(len(forest_df))[::-1]
    for y, row in zip(y_pos, forest_df.itertuples()):
        color = COLOR_NEG if row.overall else "#52514e"
        ax2.errorbar(row.effect, y, xerr=1.96 * row.se, fmt="D" if row.overall else "o",
                     color=color, markersize=9 if row.overall else 6.5,
                     capsize=4, linewidth=1.6, zorder=3)

    ax2.axvline(0, color="#c3c2b7", linestyle="--", linewidth=1.2, zorder=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{r.label}  (n={r.n})" for r in forest_df.itertuples()])
    ax2.set_xlabel(f"log1p({value_col}) difference, positive - negative (+/- 95% CI)", fontsize=10)
    ax2.set_title(f"Per-position effect of {group_col} on {value_col}{filter_desc}\n"
                  f"vs. position-adjusted overall estimate",
                  fontsize=12)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)
    ax2.margins(y=0.15)

    plt.tight_layout()
    forest_out_name = (f"{value_compartment.lower()}_ch{value_channel}_{value_stat.lower()}"
                        f"_by_ch{group_channel}_call{filter_suffix}_forest.png")
    forest_out_path = os.path.join(output_dir, forest_out_name)
    plt.savefig(forest_out_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)
    print(f"-> saved {forest_out_path}")

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # log-scale display; clip the rare exact-zero values so log() is defined
    plot_neg = np.maximum(neg_vals, 1)
    plot_pos = np.maximum(pos_vals, 1)

    parts = ax.violinplot([plot_neg, plot_pos], positions=[1, 2], showmedians=True, widths=0.7)
    for body, color in zip(parts["bodies"], [COLOR_NEG, COLOR_POS]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_edgecolor("#52514e")
        parts[key].set_linewidth(1.2)

    ax.set_yscale("log")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{group_col}=negative (n={len(neg_vals)})",
                         f"{group_col}=positive (n={len(pos_vals)})"])
    ax.set_ylabel(f"{value_col} (log scale)")
    ax.set_title(f"{value_col} by {group_col}{filter_desc}\n(all positions pooled, ambiguous excluded)")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    p_str = f"p = {p_value:.2e}" if p_value >= 1e-300 else "p < 1e-300"
    ax.text(
        0.5, 0.98, f"Mann-Whitney U test: {p_str}",
        transform=ax.transAxes, ha="center", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#c3c2b7", alpha=0.95),
    )

    plt.tight_layout()
    out_name = (f"{value_compartment.lower()}_ch{value_channel}_{value_stat.lower()}"
                f"_by_ch{group_channel}_call{filter_suffix}_violin.png")
    out_path = os.path.join(output_dir, out_name)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"\n-> saved {out_path}")

    return sub, stat, p_value


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True,
                         help="Folder to search recursively for *_cells_calls.csv files")
    parser.add_argument("--value_channel", type=int, default=1,
                         help="Channel number (1-based) whose intensity is compared (default: 1)")
    parser.add_argument("--value_compartment", default="Nucleus", choices=["Nucleus", "Cytoplasm", "Cell"],
                         help="Compartment for the compared intensity (default: Nucleus)")
    parser.add_argument("--value_stat", default="Mean", choices=["Mean", "Median"],
                         help="Statistic for the compared intensity (default: Mean)")
    parser.add_argument("--group_channel", type=int, default=2,
                         help="Channel number (1-based) whose positive/negative call defines the two "
                              "groups (default: 2)")
    parser.add_argument("--filter_channel", type=int, default=None,
                         help="Channel number (1-based) to restrict to before splitting by --group_channel "
                              "(default: none, use all cells) -- e.g. --filter_channel 1 to only compare "
                              "among Channel-1-positive cells")
    parser.add_argument("--filter_call", default="positive", choices=["positive", "negative"],
                         help="Which call value of --filter_channel to keep (default: positive)")
    parser.add_argument("--output_dir", default=None,
                         help="Where to save the violin plot (default: --input_dir)")
    args = parser.parse_args()

    run(args.input_dir, args.value_channel, args.value_compartment, args.value_stat,
        args.group_channel, output_dir=args.output_dir,
        filter_channel=args.filter_channel, filter_call=args.filter_call)


if __name__ == "__main__":
    main()
