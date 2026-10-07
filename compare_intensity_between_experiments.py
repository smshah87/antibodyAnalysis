"""
Compares one channel's intensity (e.g. nuclear H3) between cells matching a
given marker-positivity signature (e.g. marker-positive AND DLL3-positive)
across TWO GROUPS of experiments -- e.g. non-transforming tumors vs
transforming tumors, where each group is made of multiple independent tumor
samples, and each sample is itself made of multiple imaged positions.

Three levels of clustering, cell < position < experiment < group:
  - cells within one position share that image's staining/imaging conditions
  - positions within one experiment (tumor sample) share that sample's biology
  - experiments are the real, independent unit of biological replication for
    a "transforming vs non-transforming" question -- two positions of the
    SAME tumor are not independent evidence about that tumor's transformation
    status, only two DIFFERENT tumors are.

Why this can't reuse compare_intensity_by_call.py's blocked ANCOVA: that
approach dummy-codes position as a blocking factor and estimates the group
effect from the WITHIN-position contrast (each position has both call values
present). Here, every position belongs entirely to one experiment, and every
experiment belongs entirely to one group -- there's no within-position (or
even within-experiment) contrast to anchor a fixed-effect blocking model on,
so it would be rank-deficient/unidentifiable. Instead this reports four
views, in order of how much they can be trusted:

1. Naive pooled cell-level Mann-Whitney U + Welch's t-test -- reference only,
   treats every cell as an independent sample (ignores both per-image and
   per-tumor clustering, so overstates the real sample size the most).
2. Position-level tests -- each position collapsed to its own mean first.
   Better than (1) but still pseudoreplicated with respect to experiment: a
   tumor sampled at many positions still only counts as one biological
   replicate for the group-level question.
3. Experiment-level tests -- each EXPERIMENT (tumor sample) collapsed to its
   own mean first, across all its positions and cells. n = number of tumors
   per group, which is the correct unit of replication for this design.
   **This is the primary test to trust.**
4. A nested mixed-effects model (position nested within experiment as random
   effects, group as the fixed effect of interest) -- uses all cell-level
   data while still respecting the full clustering structure; the most
   statistically powerful option, but check for a convergence warning before
   trusting its p-value (more likely with few experiments per group).

Expects *_cells_calls.csv files (call_and_plot_positivity.py output)
somewhere under each experiment directory; searches recursively.

All experiment directories (in both groups) are assumed to share the same
--filter_channels/--value_channel numbering convention (e.g. channel 1 is
always the primary marker, channel 2 always DLL3, regardless of which
specific antibody channel 1 happens to be in a given experiment). Verify
this holds for every directory you pass in -- see the README's note on not
assuming a channel layout carries over between datasets.

Usage:
    # H3 (channel 3, Nucleus, Mean) in marker+/DLL3+ cells, non-transforming
    # tumors (several samples) vs transforming tumors (several samples)
    python compare_intensity_between_experiments.py \
        --group_a_dirs "G:/tumor1_sox9/" "G:/tumor2_olig2/" --label_a NonTransforming \
        --group_b_dirs "G:/tumor3_sox9/" "G:/tumor4_olig2/" "G:/tumor5_olig2/" --label_b Transforming \
        --filter_channels 1 2 --filter_calls positive positive \
        --value_channel 3 --value_compartment Nucleus --value_stat Mean

    # give each experiment a readable label instead of its folder's basename
    python compare_intensity_between_experiments.py \
        --group_a_dirs raw/tumor1 raw/tumor2 --experiment_labels_a T1 T2 --label_a NonTransforming \
        --group_b_dirs raw/tumor3 raw/tumor4 --experiment_labels_b T3 T4 --label_b Transforming \
        --filter_channels 1 2 --value_channel 3

    # experiments imaged in separate acquisition sessions can differ in overall
    # brightness for reasons that have nothing to do with the biology being
    # compared (laser power, detector gain/calibration, staining-day variation) --
    # --normalize_by_experiment_median divides each cell's --value_channel reading
    # by that SAME experiment's own MEDIAN of that channel across every cell (not
    # just the --filter_channels-selected subpopulation), so a uniform per-session
    # brightness shift cancels out before the group comparison runs. Median rather
    # than mean because per-cell intensity is typically right-skewed (a handful of
    # very bright cells can drag a mean around; the median is robust to that).
    python compare_intensity_between_experiments.py \
        --group_a_dirs raw/tumor1 raw/tumor2 --label_a MT \
        --group_b_dirs raw/tumor3 raw/tumor4 --label_b noMT \
        --filter_channels 1 2 --value_channel 3 --normalize_by_experiment_median

--normalize_by_experiment_median: why this and not exposure-time or a per-cell
reference channel (e.g. DAPI). All three were tried in practice on a real
MT-vs-noMT H3ac/H4ac comparison where raw intensity looked dramatically
different between groups (median ~8.8x). Dividing by each experiment's
recorded exposure time (from its metadata.txt) left most of the gap intact --
exposure differences didn't explain it. Per-cell DAPI-ratio normalization
narrowed it and moved the experiment-level p-value from 0.24 toward
significance (0.077), but DAPI is a different dye/staining chemistry, so it
only partially tracks --value_channel's own session-to-session technical
variation. Directly measuring --value_channel's own background (median
intensity in non-cell pixels, same dye/same detection path) turned out to
explain the gap almost completely (experiment-level p=0.88, fully-overlapping
ranges) -- but that requires re-opening every raw image and a segmentation
mask, which this CSV-only script can't do. --normalize_by_experiment_median is
the best CSV-only approximation of that background measurement: every cell in
an experiment (not just the biological subpopulation of interest) shares that
experiment's session-wide brightness/gain, so the whole-experiment median is
dominated by the same technical factor as the true background is, without
needing the raw images. On the same real comparison it gave results close to
the true-background version, and unlike exposure-time alone, needs no
external metadata.

Important asymmetry to know about before trusting this flag: it does NOT
require --value_channel and --filter_channels to be different (e.g.
normalizing channel 3 by channel 3's own experiment median, as above, is
exactly the intended use), but it is NOT appropriate to normalize a channel by
*its own* --filter_channels-selected subpopulation median, and it becomes less
trustworthy the more the --filter_channels-selected subpopulation dominates
the experiment's total cell count (the whole-experiment median starts to just
echo the subpopulation instead of estimating an independent technical
baseline). Cross-check against the unnormalized run rather than trusting this
flag's output alone -- if normalizing reverses which group has the higher
value, that reversal (not either single number) is the finding worth
investigating.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, ttest_ind
import statsmodels.formula.api as smf

# Validated categorical palette (dataviz skill reference palette.md, slots 1 & 2)
COLOR_A = "#2a78d6"  # blue
COLOR_B = "#eb6834"  # orange


def _broadcast(values, n, arg_name):
    if len(values) == 1:
        return list(values) * n
    if len(values) != n:
        raise ValueError(f"--{arg_name} got {len(values)} value(s) for {n} channel(s) -- pass either "
                          f"one (used for all) or exactly one per channel")
    return list(values)


def _load_and_filter(input_dir, filter_channels, filter_calls, value_col=None,
                      normalize_by_experiment_median=False):
    """One experiment directory -> filtered per-cell rows tagged with _position.

    normalize_by_experiment_median: if True, value_col is divided by ITS OWN
    median across every cell in this experiment (all calls, computed before
    the filter_channels restriction below) -- see the module docstring for
    why median rather than mean (robust to right-skew/outlier cells). Returns
    the normalizing constant (None if not requested) alongside the usual
    outputs, so the caller can report it.
    """
    csv_paths = sorted(glob.glob(os.path.join(input_dir, "**", "*_cells_calls.csv"), recursive=True))
    if not csv_paths:
        raise FileNotFoundError(f"No *_cells_calls.csv files found under {input_dir}")

    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df["_position"] = os.path.basename(os.path.dirname(p))
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    n_before = len(all_df)

    norm_const = None
    if normalize_by_experiment_median:
        if value_col not in all_df.columns:
            raise KeyError(f"'{value_col}' not found under {input_dir} -- check "
                            f"--value_channel/--value_compartment/--value_stat")
        norm_const = all_df[value_col].median()
        all_df[value_col] = all_df[value_col] / norm_const

    mask = np.ones(len(all_df), dtype=bool)
    for ch, call in zip(filter_channels, filter_calls):
        col = f"ch{ch}_call"
        if col not in all_df.columns:
            raise KeyError(f"'{col}' not found under {input_dir} -- available call columns: "
                            f"{[c for c in all_df.columns if c.endswith('_call')]}")
        mask &= (all_df[col] == call)
    filtered = all_df[mask].reset_index(drop=True)
    return filtered, len(csv_paths), n_before, norm_const


def _load_group(dirs, labels, group_label, value_col, filter_channels, filter_calls,
                 normalize_by_experiment_median=False):
    """Every experiment directory in one group -> tagged, filtered, concatenated cell rows."""
    if labels is None:
        labels = [os.path.basename(os.path.normpath(d)) for d in dirs]
    elif len(labels) != len(dirs):
        raise ValueError(f"Got {len(labels)} experiment label(s) for {len(dirs)} director(ies) in "
                          f"group '{group_label}' -- pass exactly one label per directory")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Experiment labels for group '{group_label}' must be unique: {labels}")

    filter_desc = " & ".join(f"ch{c}={v}" for c, v in zip(filter_channels, filter_calls))
    frames = []
    for d, exp_label in zip(dirs, labels):
        filtered, n_positions, n_before, norm_const = _load_and_filter(
            d, filter_channels, filter_calls, value_col=value_col,
            normalize_by_experiment_median=normalize_by_experiment_median,
        )
        if value_col not in filtered.columns:
            raise KeyError(f"'{value_col}' not found for experiment '{exp_label}' ({d}) -- check "
                            f"--value_channel/--value_compartment/--value_stat")
        filtered["_experiment"] = exp_label
        filtered["_group"] = group_label
        filtered["_experiment_full"] = f"{group_label}::{exp_label}"
        filtered["_position_full"] = filtered["_experiment_full"] + "::" + filtered["_position"].astype(str)
        norm_desc = f", normalized by experiment median={norm_const:.2f}" if norm_const is not None else ""
        print(f"  {exp_label}: {n_positions} position file(s), {filter_desc}: "
              f"{len(filtered)}/{n_before} cells{norm_desc}")
        frames.append(filtered)
    return pd.concat(frames, ignore_index=True)


def run(group_a_dirs, label_a, group_b_dirs, label_b,
        filter_channels, filter_calls,
        value_channel, value_compartment, value_stat,
        experiment_labels_a=None, experiment_labels_b=None,
        output_dir=None, normalize_by_experiment_median=False):
    output_dir = output_dir or group_a_dirs[0]
    os.makedirs(output_dir, exist_ok=True)

    filter_calls = _broadcast(filter_calls, len(filter_channels), "filter_calls")
    value_col = f"{value_compartment}: Channel {value_channel}: {value_stat}"
    value_desc = value_col if not normalize_by_experiment_median else f"{value_col} / experiment median"

    print(f"{label_a}:")
    df_a = _load_group(group_a_dirs, experiment_labels_a, label_a, value_col, filter_channels, filter_calls,
                        normalize_by_experiment_median=normalize_by_experiment_median)
    print(f"{label_b}:")
    df_b = _load_group(group_b_dirs, experiment_labels_b, label_b, value_col, filter_channels, filter_calls,
                        normalize_by_experiment_median=normalize_by_experiment_median)

    df_a = df_a.rename(columns={value_col: "_value"})
    df_b = df_b.rename(columns={value_col: "_value"})
    keep_cols = ["_value", "_position", "_position_full", "_experiment", "_experiment_full", "_group"]
    sub = pd.concat([df_a[keep_cols], df_b[keep_cols]], ignore_index=True)
    sub["_log_value"] = np.log1p(sub["_value"])

    a_vals = sub.loc[sub["_group"] == label_a, "_value"].values
    b_vals = sub.loc[sub["_group"] == label_b, "_value"].values
    n_exp_a = sub.loc[sub["_group"] == label_a, "_experiment"].nunique()
    n_exp_b = sub.loc[sub["_group"] == label_b, "_experiment"].nunique()
    print(f"\n{label_a}: n={len(a_vals)} cells across {n_exp_a} experiment(s)")
    print(f"{label_b}: n={len(b_vals)} cells across {n_exp_b} experiment(s)")

    # --- 1. Naive pooled cell-level tests (reference only) ---
    stat, p_value = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
    t_stat, t_p = ttest_ind(np.log1p(a_vals), np.log1p(b_vals), equal_var=False)
    print(f"\n[1] Naive pooled cell-level tests (NOT corrected for per-image or per-tumor "
          f"clustering -- treats every cell as an independent sample):")
    print(f"  Mann-Whitney U = {stat:.1f}, p = {p_value:.3e}")
    print(f"  Welch's t on log1p: t = {t_stat:.3f}, p = {t_p:.3e}")
    print(f"  Median {value_desc}: {label_a}={np.median(a_vals):.1f}, {label_b}={np.median(b_vals):.1f}")

    # --- 2. Position-level tests (still pseudoreplicated w.r.t. experiment) ---
    pos_means = sub.groupby(["_position_full", "_group"])["_log_value"].mean().reset_index()
    pos_a = pos_means.loc[pos_means["_group"] == label_a, "_log_value"].values
    pos_b = pos_means.loc[pos_means["_group"] == label_b, "_log_value"].values
    print(f"\n[2] Position-level tests (position as the unit of replication -- still "
          f"pseudoreplicated w.r.t. experiment, n={len(pos_a)} {label_a} / {len(pos_b)} {label_b} "
          f"position(s)):")
    if len(pos_a) >= 2 and len(pos_b) >= 2:
        pos_stat, pos_p = mannwhitneyu(pos_a, pos_b, alternative="two-sided")
        pos_t, pos_t_p = ttest_ind(pos_a, pos_b, equal_var=False)
        print(f"  Mann-Whitney U = {pos_stat:.1f}, p = {pos_p:.3e}")
        print(f"  Welch's t: t = {pos_t:.3f}, p = {pos_t_p:.3e}")
    else:
        print("  Skipped -- need at least 2 positions per group")

    # --- 3. Experiment-level tests (primary -- the correct unit of replication) ---
    exp_means = sub.groupby(["_experiment_full", "_group"])["_log_value"].mean().reset_index()
    exp_a = exp_means.loc[exp_means["_group"] == label_a, "_log_value"].values
    exp_b = exp_means.loc[exp_means["_group"] == label_b, "_log_value"].values
    print(f"\n[3] Experiment-level tests (experiment/tumor as the unit of replication -- "
          f"n={len(exp_a)} {label_a} / {len(exp_b)} {label_b} experiment(s)) "
          f"-- ** the test to trust for this design **:")
    if len(exp_a) >= 2 and len(exp_b) >= 2:
        exp_stat, exp_p = mannwhitneyu(exp_a, exp_b, alternative="two-sided")
        exp_t, exp_t_p = ttest_ind(exp_a, exp_b, equal_var=False)
        print(f"  Mann-Whitney U = {exp_stat:.1f}, p = {exp_p:.3e}")
        print(f"  Welch's t: t = {exp_t:.3f}, p = {exp_t_p:.3e}")
    else:
        print(f"  Skipped -- need at least 2 experiments per group (got {len(exp_a)}/{len(exp_b)}); "
              f"add more tumor samples to each group, or rely on [1]/[2]/[4] with appropriate caution")

    # --- 4. Nested mixed-effects model: position nested within experiment, group = fixed effect ---
    print(f"\n[4] Nested mixed-effects model (log1p({value_desc}) ~ group, position nested "
          f"within experiment as random effects):")
    try:
        md = smf.mixedlm("_log_value ~ _group", sub, groups=sub["_experiment_full"],
                          vc_formula={"position": "0 + C(_position_full)"})
        mdf = md.fit()
        group_param = next(p for p in mdf.params.index if p.startswith("_group"))
        print(f"  {group_param} coefficient = {mdf.params[group_param]:+.4f}, "
              f"p = {mdf.pvalues[group_param]:.3e}")
    except Exception as e:
        print(f"  Failed to fit ({e}) -- rely on [1]/[2]/[3] instead")

    # --- Per-experiment mean plot: one point per experiment, colored by group ---
    # (the between-group equivalent of a forest plot -- there's no within-experiment
    # group contrast to plot since each experiment belongs to only one group, so this
    # shows each experiment's own summary instead of a per-experiment effect size)
    plot_rows = []
    for (exp_full, group, exp), exp_sub in sub.groupby(["_experiment_full", "_group", "_experiment"]):
        n = len(exp_sub)
        mean = exp_sub["_log_value"].mean()
        se = exp_sub["_log_value"].std(ddof=1) / np.sqrt(n) if n >= 2 else np.nan
        n_positions = exp_sub["_position"].nunique()
        plot_rows.append({"experiment_full": exp_full, "label": f"{group}: {exp}",
                           "group": group, "n": n, "n_positions": n_positions,
                           "mean": mean, "se": se})
    plot_df = pd.DataFrame(plot_rows).sort_values(["group", "experiment_full"])

    fig1, ax1 = plt.subplots(figsize=(8.5, 0.55 * len(plot_df) + 1.5))
    y_pos = np.arange(len(plot_df))[::-1]
    colors = {label_a: COLOR_A, label_b: COLOR_B}
    for y, row in zip(y_pos, plot_df.itertuples()):
        se = 0 if np.isnan(row.se) else row.se
        ax1.errorbar(row.mean, y, xerr=1.96 * se, fmt="o", color=colors[row.group],
                     markersize=7, capsize=4, linewidth=1.6, zorder=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{r.label}  (n={r.n} cells, {r.n_positions} pos)" for r in plot_df.itertuples()])
    ax1.set_xlabel(f"log1p({value_desc}) per-experiment mean (+/- 95% CI)", fontsize=10)
    ax1.set_title(f"Per-experiment {value_desc}\n{label_a} vs {label_b}", fontsize=12)
    handles = [plt.Line2D([0], [0], marker="o", color=colors[l], linestyle="", markersize=7, label=l)
               for l in (label_a, label_b)]
    ax1.legend(handles=handles, loc="best", framealpha=0.9)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax1.set_axisbelow(True)
    ax1.margins(y=0.1)
    plt.tight_layout()

    safe_val = f"{value_compartment.lower()}_ch{value_channel}_{value_stat.lower()}"
    if normalize_by_experiment_median:
        safe_val += "_expnorm"
    experiments_out_name = f"{safe_val}_{label_a}_vs_{label_b}_experiments.png"
    experiments_out_path = os.path.join(output_dir, experiments_out_name)
    plt.savefig(experiments_out_path, dpi=180, bbox_inches="tight")
    plt.close(fig1)
    print(f"\n-> saved {experiments_out_path}")

    # --- Violin plot: pooled cell-level distributions ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 6))
    plot_a = np.maximum(a_vals, 1)
    plot_b = np.maximum(b_vals, 1)
    parts = ax2.violinplot([plot_a, plot_b], positions=[1, 2], showmedians=True, widths=0.7)
    for body, color in zip(parts["bodies"], [COLOR_A, COLOR_B]):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_edgecolor("#52514e")
        parts[key].set_linewidth(1.2)

    ax2.set_yscale("log")
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f"{label_a} (n={len(a_vals)})", f"{label_b} (n={len(b_vals)})"])
    ax2.set_ylabel(f"{value_desc} (log scale)")
    ax2.set_title(f"{value_desc}: {label_a} vs {label_b}\n"
                  f"(all cells pooled -- see experiment-level test for the trustworthy p-value)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)

    p_str = f"p = {p_value:.2e}" if p_value >= 1e-300 else "p < 1e-300"
    ax2.text(
        0.5, 0.98, f"Mann-Whitney U test (naive pooled): {p_str}",
        transform=ax2.transAxes, ha="center", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#c3c2b7", alpha=0.95),
    )

    plt.tight_layout()
    violin_out_name = f"{safe_val}_{label_a}_vs_{label_b}_violin.png"
    violin_out_path = os.path.join(output_dir, violin_out_name)
    plt.savefig(violin_out_path, dpi=180)
    plt.close(fig2)
    print(f"-> saved {violin_out_path}")

    return sub


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group_a_dirs", nargs="+", required=True,
                         help="One or more experiment (tumor sample) folders making up group A, "
                              "e.g. --group_a_dirs tumor1/ tumor2/ tumor3/")
    parser.add_argument("--label_a", default="A", help="Short name for group A (e.g. NonTransforming) "
                                                         "-- used in print output, plot legends, and "
                                                         "output filenames")
    parser.add_argument("--experiment_labels_a", nargs="+", default=None,
                         help="One label per --group_a_dirs entry (default: each directory's own "
                              "basename)")
    parser.add_argument("--group_b_dirs", nargs="+", required=True,
                         help="One or more experiment folders making up group B, e.g. Transforming")
    parser.add_argument("--label_b", default="B", help="Short name for group B (e.g. Transforming)")
    parser.add_argument("--experiment_labels_b", nargs="+", default=None,
                         help="One label per --group_b_dirs entry (default: each directory's own "
                              "basename)")

    parser.add_argument("--filter_channels", type=int, nargs="+", required=True,
                         help="1-based channel numbers that must ALL match --filter_calls for a cell "
                              "to be included, e.g. --filter_channels 1 2 for marker-positive AND "
                              "DLL3-positive. Applied identically to every experiment directory in "
                              "both groups -- see the module docstring's channel-layout caveat")
    parser.add_argument("--filter_calls", default=["positive"], nargs="+",
                         help="One value applied to all --filter_channels, or one per channel "
                              "(default: positive)")

    parser.add_argument("--value_channel", type=int, default=1,
                         help="Channel number (1-based) whose intensity is compared, applied "
                              "identically to every experiment directory in both groups")
    parser.add_argument("--value_compartment", default="Nucleus", choices=["Nucleus", "Cytoplasm", "Cell"])
    parser.add_argument("--value_stat", default="Mean", choices=["Mean", "Median"])

    parser.add_argument("--normalize_by_experiment_median", action="store_true",
                         help="Divide each cell's --value_channel reading by that SAME experiment's "
                              "own MEDIAN of the channel across every cell (all calls, not just the "
                              "--filter_channels subpopulation), before comparing groups -- corrects "
                              "for a session-to-session technical brightness/gain difference between "
                              "separately-acquired experiments. Median rather than mean since per-cell "
                              "intensity is typically right-skewed. See the module docstring for why "
                              "this beat exposure-time and DAPI-ratio normalization in practice, and "
                              "for when NOT to trust it. Default: off (compare raw --value_stat)")
    parser.add_argument("--output_dir", default=None,
                         help="Where to save plots (default: first --group_a_dirs entry)")
    args = parser.parse_args()

    run(args.group_a_dirs, args.label_a, args.group_b_dirs, args.label_b,
        args.filter_channels, args.filter_calls,
        args.value_channel, args.value_compartment, args.value_stat,
        experiment_labels_a=args.experiment_labels_a, experiment_labels_b=args.experiment_labels_b,
        output_dir=args.output_dir,
        normalize_by_experiment_median=args.normalize_by_experiment_median)


if __name__ == "__main__":
    main()
