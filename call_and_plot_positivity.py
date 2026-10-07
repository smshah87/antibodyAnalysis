"""
Per-channel positivity calling + visual QC overlay.

Takes a per-cell CSV (from either the QuPath export script or cell_analysis.py)
and its source image, calls positive/negative/ambiguous cells in one or more
channels using a 2-component Gaussian Mixture Model on log intensity, and saves
an overlay image per channel so you can visually check whether the calls line
up with what's actually bright in the image.

If you pass --autofluor_channel, two things happen before thresholding:
  1. Spectral correction: a linear fit of each target channel against the
     autofluorescence channel (across all cells) estimates how much of that
     channel's signal tracks autofluorescence bleed-through, and subtracts it
     out. The fitted correlation (r) is printed -- if it's weak (this is
     common; autofluorescence bleed is often localized to a subset of
     punctate granules rather than a smooth whole-cell relationship), the
     correction will do very little, and that's reported rather than hidden.
  2. Granule burden flag: cells with unusually spiky/punctate autofluorescence
     signal (high Std.Dev./Median ratio in that channel) are flagged in a
     'high_granule_burden' column for manual review -- these are the cells
     most likely to have granule contamination that a smooth linear
     correction can't fully remove. Using --stat Median instead of Mean is
     the more effective fix for that specific problem (median is robust to
     the punctate spikes; see conversation history for why).

If you DON'T pass --autofluor_channel (no dedicated channel available), the
granule burden flag still runs, using a different signal: cell_analysis.py's
min_nucleus_area_um2 filter already finds small bright DAPI-channel objects
that are almost certainly debris rather than real nuclei, and reports how
much of each cell's footprint they cover (the 'Granule_Area' CSV column).
That gets reused here the same way -- no spectral correction in this case
(there's no channel to correct against), just the burden flag.

--compartment and --stat each accept either one value (applied to every
channel) or one value per --channels entry, matched positionally -- e.g.
--channels 1 2 --compartment Nucleus Cell --stat Mean Dot_Density calls
channel 1 on its nuclear Mean intensity and channel 2 on its whole-cell
Dot_Density (see cell_analysis.py's --puncta_channel). Dot_Count/Dot_Density
are only computed for the Cell compartment (puncta aren't confined to either
sub-region, so whole-cell is the only compartment that makes sense to count/
normalize them by) -- pass --compartment Cell for any channel using one of
those stats. --channels itself accepts any number of channels, and defaults
to every channel in the image except --dapi_channel and --autofluor_channel
(the usual case: every marker channel gets called).

Pixel size (microns/pixel) is never a user-supplied argument -- it's read back
automatically from --image via preprocess.get_pixel_size_um().

Usage:
    python call_and_plot_positivity.py \
        --csv cells.csv \
        --image image.tif \
        --compartment Nucleus Cytoplasm \
        --autofluor_channel 3 \
        --output_dir qc/

Output per channel:
    - <output_dir>/<name>_ch<N>_overlay.png   (image + positive/negative/ambiguous markers)
    - <output_dir>/<name>_ch<N>_histogram.png (intensity histogram with threshold)
    - <output_dir>/<name>_calls.csv           (original CSV + a call/probability column per channel,
                                                 a granule_burden/high_granule_burden flag, and
                                                 AF-corrected columns if --autofluor_channel was given)
"""

import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import preprocess
from sklearn.mixture import GaussianMixture

CALL_COLORS = {"positive": (0.2, 1.0, 0.2), "negative": (1.0, 0.2, 0.2), "ambiguous": (1.0, 1.0, 0.2)}


def call_channel_positivity(df, channel, compartment="Cytoplasm", stat="Mean",
                             pos_thresh=0.8, neg_thresh=0.2, column_override=None, gmm=None):
    """
    Fits a 2-component GMM on log(intensity) for one channel/compartment and
    returns (probability_positive, call) where call is one of
    'positive' / 'negative' / 'ambiguous' based on pos_thresh/neg_thresh.

    column_override: if given, threshold on this column instead of the default
    "{compartment}: Channel {channel}: {stat}" -- used to threshold on the
    autofluorescence-corrected column when --autofluor_channel is supplied.

    gmm: a pre-fitted sklearn GaussianMixture to use instead of fitting a new
    one on just this df -- see fit_pooled_gmms(). Pass this when df is a
    single position's cells but you want to apply a threshold learned from
    cells pooled across many positions (a single position's own cells often
    aren't enough to reveal a real bimodal split for a sparse marker; fitting
    independently per position in that case just carves an arbitrary split
    out of what's really a unimodal population -- see run_full_pipeline.py).

    A cell whose entire compartment (e.g. Nucleus) was excluded by granule
    masking (see segment_granule_blobs()) has 0 pixels to average over there,
    so scipy's ndimage.mean/median/etc for it is NaN (0/0), not 0 -- this
    shows up as scipy's own "invalid value encountered in divide" warning at
    segmentation time, not here. Those cells are excluded from GMM fitting
    (a NaN would otherwise crash sklearn outright) and always reported as
    'ambiguous' with NaN probability, since there's no signal to call them
    on either way.
    """
    col = column_override or f"{compartment}: Channel {channel}: {stat}"
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in CSV. Available columns include: "
                        f"{[c for c in df.columns if f'Channel {channel}' in c]}")

    x_raw = df[col].values.astype(float)
    valid = ~np.isnan(x_raw)
    if (~valid).any():
        print(f"  Note: {(~valid).sum()}/{len(df)} cell(s) have NaN in '{col}' (compartment fully "
              f"excluded by granule masking) -- excluded from GMM fitting, called 'ambiguous'")

    x = np.log1p(x_raw[valid]).reshape(-1, 1)
    if gmm is None:
        gmm = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(x)
    order = np.argsort(gmm.means_.flatten())
    neg_idx, pos_idx = order[0], order[1]

    proba_pos = np.full(len(df), np.nan)
    proba_pos[valid] = gmm.predict_proba(x)[:, pos_idx]

    m_neg, m_pos = gmm.means_[neg_idx, 0], gmm.means_[pos_idx, 0]
    s_neg, s_pos = np.sqrt(gmm.covariances_[neg_idx, 0, 0]), np.sqrt(gmm.covariances_[pos_idx, 0, 0])
    separation = abs(m_pos - m_neg) / (0.5 * (s_neg + s_pos))
    threshold_raw = float(np.expm1(0.5 * (m_neg + m_pos)))

    call = np.full(len(df), "ambiguous", dtype=object)
    call[valid] = np.select(
        [proba_pos[valid] > pos_thresh, proba_pos[valid] < neg_thresh],
        ["positive", "negative"],
        default="ambiguous",
    )

    stats = {"separation": separation, "threshold_raw": threshold_raw, "column_used": col}
    return proba_pos, call, stats


def correct_for_autofluorescence(df, channel, autofluor_channel, compartment, stat):
    """
    Fits target_channel ~ slope * autofluor_channel + intercept across all cells,
    then subtracts the slope*autofluor component (the part of the signal that
    tracks autofluorescence) to get a corrected intensity.

    Returns (corrected_values, corrected_col_name, diagnostics_dict). Diagnostics
    include the Pearson r so you can judge whether the correction is doing
    anything meaningful -- a weak r means autofluorescence bleed isn't linearly
    related to whole-cell mean/median intensity here (common when it's driven by
    a handful of punctate granules rather than a smooth per-cell offset), and
    the correction will barely change the values.
    """
    col = f"{compartment}: Channel {channel}: {stat}"
    af_col = f"{compartment}: Channel {autofluor_channel}: {stat}"
    x = df[af_col].values.astype(np.float64)
    y = df[col].values.astype(np.float64)

    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    corrected = np.clip(y - slope * x, a_min=0, a_max=None)
    corrected_col = f"{col}_AFcorrected"

    diagnostics = {"slope": slope, "intercept": intercept, "r": r, "af_col": af_col, "corrected_col": corrected_col}
    return corrected, corrected_col, diagnostics


def compute_granule_burden(df, autofluor_channel, compartment, flag_percentile=0.9):
    """
    Per-cell 'spikiness' of the autofluorescence channel: Std.Dev./Median ratio.
    Punctate autofluorescent granules inflate this a lot more than diffuse,
    uniform autofluorescence does. Cells above the given percentile are flagged
    for manual review, not automatically excluded.
    """
    med_col = f"{compartment}: Channel {autofluor_channel}: Median"
    std_col = f"{compartment}: Channel {autofluor_channel}: Std.Dev."
    med = df[med_col].clip(lower=1)
    burden = df[std_col] / med
    threshold = burden.quantile(flag_percentile)
    flag = burden > threshold
    return burden, flag, threshold


def compute_granule_burden_from_area_column(df, area_col, flag_percentile=0.9):
    """
    Fallback for when there's no dedicated --autofluor_channel: burden as the
    fraction of each cell's whole-cell footprint covered by some other
    per-cell area column from cell_analysis.py -- either 'Granule_Area_Blob'
    (solid debris blobs found by shape in a single channel, see
    segment_granule_blobs() -- preferred, detected directly rather than as a
    segmentation side effect) or 'Granule_Area' (small DAPI-channel objects
    dropped during segmentation, see filter_small_objects() -- last-resort
    fallback). Same idea as compute_granule_burden() above, just sourced from
    segmentation/spatial structure instead of a channel's intensity texture.
    """
    burden = df[area_col] / df["Cell: Area"].clip(lower=1)
    threshold = burden.quantile(flag_percentile)
    flag = burden > threshold
    return burden, flag, threshold


def plot_overlay(image, ch_index, df, call_col, title, out_path,
                  label_mask=None, id_col="Cell_ID", x_col=None, y_col=None):
    """
    image: full (C, Y, X) array. ch_index: 0-based channel to display.
    df must have call_col ('positive'/'negative'/'ambiguous').

    If label_mask is given (an integer array where each cell's pixels equal its
    id_col value, e.g. the *_cell_labels.tif saved by cell_analysis.py), each
    cell's actual segmented footprint is filled with its call color. Otherwise
    falls back to open circles at (x_col, y_col) -- used for CSVs (e.g. from
    QuPath) that don't have a corresponding Python-side label mask.
    """
    channel_img = image[ch_index].astype(np.float32)
    lo, hi = np.percentile(channel_img, [1, 99.5])
    disp = np.clip((channel_img - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(disp, cmap="gray")

    if label_mask is not None:
        overlay = np.zeros((*disp.shape, 4))
        for call_value, rgb in CALL_COLORS.items():
            ids = df.loc[df[call_col] == call_value, id_col].values
            if len(ids):
                overlay[np.isin(label_mask, ids)] = (*rgb, 0.5)
        ax.imshow(overlay)
        handles = [
            mpatches.Patch(color=rgb, label=f"{call_value.capitalize()} (n={(df[call_col] == call_value).sum()})")
            for call_value, rgb in CALL_COLORS.items()
        ]
        ax.legend(handles=handles, loc="upper right", framealpha=0.9)
    else:
        for call_value, rgb in CALL_COLORS.items():
            sub = df[df[call_col] == call_value]
            ax.scatter(sub[x_col], sub[y_col], s=25, facecolors="none", edgecolors=[rgb],
                       linewidths=1.2, label=f"{call_value.capitalize()} (n={len(sub)})")
        ax.legend(loc="upper right", framealpha=0.9)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_histogram(df, col, stats, call_col, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(df[col], bins=40, color="#4C72B0", alpha=0.8)
    ax.axvline(stats["threshold_raw"], color="red", linestyle="--",
               label=f"GMM midpoint = {stats['threshold_raw']:.0f}")
    n_pos = (df[call_col] == "positive").sum()
    ax.set_title(f"{title}\n{n_pos} positive")
    ax.set_xlabel(f"{col} (raw intensity)")
    ax.set_ylabel("Cell count")
    ax.legend(loc="upper right")

    sep = stats["separation"]
    quality = "well separated" if sep > 2 else ("moderate" if sep > 1 else "weak/overlapping")
    ax.text(
        0.02, 0.98, f"Separation: {sep:.2f}\n({quality})",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def resolve_channels_and_compartment_from_n_channels(n_channels, channels, compartment,
                                                       dapi_channel, autofluor_channel):
    """Pure (no I/O) core of resolve_channels_and_compartment() below -- split out so callers that
    already have the image loaded (run()) don't have to read it a second time just for this."""
    if dapi_channel is None:
        dapi_channel = n_channels  # DAPI is almost always acquired last
    if channels is None:
        channels = [c for c in range(1, n_channels + 1) if c != dapi_channel and c != autofluor_channel]
        print(f"--channels not given -- defaulting to every channel except DAPI ({dapi_channel}) and "
              f"autofluorescence ({autofluor_channel}): {channels}")

    if isinstance(compartment, str) or len(compartment) == 1:
        single = compartment if isinstance(compartment, str) else compartment[0]
        compartment_by_channel = {ch: single for ch in channels}
    elif len(compartment) == len(channels):
        compartment_by_channel = dict(zip(channels, compartment))
    else:
        raise ValueError(f"Got {len(compartment)} compartment(s) for {len(channels)} channel(s) -- "
                          "pass either one compartment (used for all channels) or exactly one per channel")

    return dapi_channel, channels, compartment_by_channel


def resolve_stat_by_channel(stat, channels):
    """
    Same "one value applied to every channel, or one value per channel,
    matched positionally" pattern as compartment (see
    resolve_channels_and_compartment_from_n_channels()) -- lets a channel
    that's genuinely punctate (e.g. DLL3) use Dot_Count while other channels
    keep using Mean/Median, all from one --stat flag.
    """
    if isinstance(stat, str) or len(stat) == 1:
        single = stat if isinstance(stat, str) else stat[0]
        return {ch: single for ch in channels}
    elif len(stat) == len(channels):
        return dict(zip(channels, stat))
    else:
        raise ValueError(f"Got {len(stat)} stat(s) for {len(channels)} channel(s) -- "
                          "pass either one stat (used for all channels) or exactly one per channel")


def resolve_channels_and_compartment(image_path, channels, compartment, dapi_channel, autofluor_channel):
    """
    Shared defaulting logic used by both run() (single position) and
    fit_pooled_gmms() (whole-folder pooling), so a batch's channel/compartment
    choice is resolved identically and only printed once per batch rather
    than once per position.

    Returns (dapi_channel, channels, compartment_by_channel) -- dapi_channel
    and channels are concrete (never None) after this; compartment_by_channel
    maps each channel to its own compartment string.
    """
    image = tifffile.imread(image_path)
    if image.ndim != 3:
        raise ValueError(f"Expected a (channel, y, x) image, got shape {image.shape}")
    return resolve_channels_and_compartment_from_n_channels(
        image.shape[0], channels, compartment, dapi_channel, autofluor_channel
    )


def fit_pooled_gmms(csv_paths, channels, compartment_by_channel, stat_by_channel):
    """
    Fits one shared 2-component GMM per channel, on cells pooled across every
    position's *_cells.csv in csv_paths, for run_full_pipeline.py to reuse
    across all of a folder's positions via call_channel_positivity()'s `gmm`
    param.

    Why: a 2-component GMM fit independently on just one position's own cells
    assumes that position has enough of its own cells to reveal a real
    bimodal positive/negative split. For a sparse marker, many individual
    positions won't -- the fit still succeeds, but on an actually-unimodal
    population it just carves an arbitrary split out of noise (near-zero
    separation, most cells landing in "ambiguous"). Pooling across every
    position in the folder borrows statistical power from the whole batch,
    where a true positive population -- rare in any one field of view -- is
    much more likely to be large enough to detect reliably. This applies
    just as much to Dot_Count (see cell_analysis.py's puncta counting) as it
    does to Mean/Median intensity -- a single position's raw Dot_Count
    distribution is often a smooth, unimodal, monotonically decaying count
    distribution with no real gap in it, so a per-position fit on it carves
    an arbitrary split exactly the same way.

    stat_by_channel: {channel: stat} -- see resolve_stat_by_channel(). Lets a
    genuinely punctate channel pool on Dot_Count while others pool on
    Mean/Median, in the same call.

    Uses each CSV's raw (pre-granule-drop) cells for simplicity; the granule
    burden filter typically drops well under 1% of cells (see
    call_and_plot_positivity.py's own per-position dropping), so this has a
    negligible effect on the pooled fit.

    Returns {channel: fitted sklearn GaussianMixture}.
    """
    frames = []
    for p in csv_paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"  Skipping {p} while pooling for GMM fitting: {e}")
    if not frames:
        raise ValueError("No readable *_cells.csv files to pool for GMM fitting")
    pooled = pd.concat(frames, ignore_index=True)

    gmms = {}
    for ch in channels:
        col = f"{compartment_by_channel[ch]}: Channel {ch}: {stat_by_channel[ch]}"
        if col not in pooled.columns:
            raise KeyError(f"'{col}' not found while pooling for channel {ch} -- available columns "
                            f"include: {[c for c in pooled.columns if f'Channel {ch}' in c]}")
        x_raw = pooled[col].values.astype(float)
        valid = ~np.isnan(x_raw)
        n_nan = (~valid).sum()
        x = np.log1p(x_raw[valid]).reshape(-1, 1)
        gmms[ch] = GaussianMixture(n_components=2, random_state=0, n_init=10).fit(x)
        nan_note = f", dropped {n_nan} NaN (fully debris-excluded compartment)" if n_nan else ""
        print(f"Pooled GMM for channel {ch} ({col}): fit on {len(x)} cells across "
              f"{len(csv_paths)} position(s){nan_note}")
    return gmms


def run(csv_path, image_path, compartment, stat,
        output_dir, pos_thresh, neg_thresh, channels=None, dapi_channel=None, autofluor_channel=None,
        granule_flag_percentile=0.9, granule_drop_threshold=0.5, label_mask_path=None, pooled_gmms=None):
    """
    channels: 1-based channel numbers to call positivity for; None = every
    channel in the image except dapi_channel and autofluor_channel (the usual
    case -- every marker channel gets called).

    compartment: either a single "Nucleus"/"Cytoplasm"/"Cell" applied to every
    entry in `channels`, or a list the same length as `channels` giving each
    channel its own compartment (matched positionally).

    stat: either a single stat ("Mean"/"Median"/"Dot_Count"/"Dot_Density") applied to every
    entry in `channels`, or a list the same length as `channels` giving each
    channel its own stat (matched positionally) -- see resolve_stat_by_channel().
    Lets a genuinely punctate channel (e.g. DLL3) use Dot_Count while others
    keep using Mean/Median.

    granule_drop_threshold: cells with granule_burden above this are dropped
    entirely (excluded from positivity calling, overlays, and the output
    CSV) rather than just flagged -- None disables dropping (flag only). Only
    a literal area fraction (0-1) when using the --autofluor_channel-free
    debris-mask fallback (see compute_granule_burden_from_debris_mask); with
    --autofluor_channel, granule_burden is a Std.Dev./Median ratio instead,
    so the same numeric cutoff means "spikiness above this", not "% of area".

    pooled_gmms: {channel: fitted GaussianMixture} from fit_pooled_gmms() --
    if a channel has an entry here, its threshold comes from that shared,
    pooled-across-positions model instead of being fit fresh on just this
    position's own cells. See fit_pooled_gmms() for why that matters.
    """
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)
    image = tifffile.imread(image_path)
    if image.ndim != 3:
        raise ValueError(f"Expected a (channel, y, x) image, got shape {image.shape}")
    dapi_channel, channels, compartment_by_channel = resolve_channels_and_compartment_from_n_channels(
        image.shape[0], channels, compartment, dapi_channel, autofluor_channel
    )
    stat_by_channel = resolve_stat_by_channel(stat, channels)
    label_mask = tifffile.imread(label_mask_path) if label_mask_path else None
    pixel_size_um = preprocess.get_pixel_size_um(image_path)

    df["_x_px"] = df["Centroid_X_um"] / pixel_size_um
    df["_y_px"] = df["Centroid_Y_um"] / pixel_size_um

    if autofluor_channel is not None:
        # The autofluorescence channel's own burden calc isn't tied to any one
        # target channel's compartment choice -- use the first channel's.
        af_compartment = next(iter(compartment_by_channel.values()))
        burden, flag, burden_thresh = compute_granule_burden(
            df, autofluor_channel, af_compartment, granule_flag_percentile
        )
        df["granule_burden"] = burden
        df["high_granule_burden"] = flag
        print(f"Autofluorescence channel {autofluor_channel}: flagged {flag.sum()}/{len(df)} cells "
              f"with high granule burden (top {int((1-granule_flag_percentile)*100)}%, "
              f"burden > {burden_thresh:.2f}) -- see 'high_granule_burden' column")
    elif "Granule_Area_Blob" in df.columns and df["Granule_Area_Blob"].sum() > 0:
        # No dedicated autofluorescence channel -- fall back to solid debris
        # blobs cell_analysis.py found by shape in a single channel (see
        # segment_granule_blobs()), preferred over the DAPI-debris proxy below
        # since it's detected directly rather than as a segmentation side effect.
        burden, flag, burden_thresh = compute_granule_burden_from_area_column(
            df, "Granule_Area_Blob", granule_flag_percentile
        )
        df["granule_burden"] = burden
        df["high_granule_burden"] = flag
        print(f"No --autofluor_channel given -- using detected debris blobs as a granule burden proxy: "
              f"flagged {flag.sum()}/{len(df)} cells "
              f"(top {int((1-granule_flag_percentile)*100)}%, burden > {burden_thresh:.3f}) -- "
              f"see 'high_granule_burden' column")
    elif "Granule_Area" in df.columns and df["Granule_Area"].sum() > 0:
        # Last resort: the small DAPI-channel objects cell_analysis.py dropped
        # during segmentation (see filter_small_objects()) as a granule/debris
        # burden proxy.
        burden, flag, burden_thresh = compute_granule_burden_from_area_column(
            df, "Granule_Area", granule_flag_percentile
        )
        df["granule_burden"] = burden
        df["high_granule_burden"] = flag
        print(f"No --autofluor_channel given -- using dropped small-DAPI-object debris as a granule "
              f"burden proxy instead: flagged {flag.sum()}/{len(df)} cells "
              f"(top {int((1-granule_flag_percentile)*100)}%, burden > {burden_thresh:.3f}) -- "
              f"see 'high_granule_burden' column")

    if granule_drop_threshold is not None and "granule_burden" in df.columns:
        n_before = len(df)
        df = df[df["granule_burden"] <= granule_drop_threshold].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"Dropped {n_dropped}/{n_before} cells with granule_burden > {granule_drop_threshold} "
                  f"-- excluded from positivity calling, overlays, and CSV output entirely")

    summary_rows = []
    for ch in channels:
        ch_compartment = compartment_by_channel[ch]
        print(f"\nChannel {ch} ({ch_compartment}):")
        proba_col = f"ch{ch}_positive_probability"
        call_col = f"ch{ch}_call"

        ch_stat = stat_by_channel[ch]
        column_override = None
        if autofluor_channel is not None and ch != autofluor_channel:
            corrected, corrected_col, diag = correct_for_autofluorescence(
                df, ch, autofluor_channel, ch_compartment, ch_stat
            )
            df[corrected_col] = corrected
            column_override = corrected_col
            print(f"  AF correction: {diag['af_col']} -> r={diag['r']:.2f}, slope={diag['slope']:.3f}"
                  f"  ({'meaningful relationship' if abs(diag['r']) > 0.3 else 'weak/negligible -- correction will barely change values'})")

        proba_pos, call, stats = call_channel_positivity(
            df, ch, compartment=ch_compartment, stat=ch_stat,
            pos_thresh=pos_thresh, neg_thresh=neg_thresh, column_override=column_override,
            gmm=(pooled_gmms.get(ch) if pooled_gmms else None),
        )
        df[proba_col] = proba_pos
        df[call_col] = call

        counts = pd.Series(call).value_counts()
        print(f"  Separation (GMM component separation, in log-intensity units; "
              f"roughly: >2 well-separated/bimodal, 1-2 moderate, <1 weak/overlapping): "
              f"{stats['separation']:.2f}")
        print(f"  {dict(counts)}  (threshold~{stats['threshold_raw']:.0f} on {stats['column_used']})")

        summary_rows.append({
            "channel": ch,
            "column_used": stats["column_used"],
            "separation": stats["separation"],
            "threshold_raw": stats["threshold_raw"],
            "n_positive": int(counts.get("positive", 0)),
            "n_ambiguous": int(counts.get("ambiguous", 0)),
            "n_negative": int(counts.get("negative", 0)),
            "n_total": len(df),
        })

        overlay_path = os.path.join(output_dir, f"{name}_ch{ch}_overlay.png")
        hist_path = os.path.join(output_dir, f"{name}_ch{ch}_histogram.png")

        plot_overlay(
            image, ch_index=ch - 1, df=df, call_col=call_col,
            title=f"Channel {ch} — positivity calls ({stats['column_used']})",
            out_path=overlay_path,
            label_mask=label_mask, x_col="_x_px", y_col="_y_px",
        )
        plot_histogram(
            df, col=stats["column_used"], stats=stats, call_col=call_col,
            title=f"Channel {ch} — {stats['column_used']}",
            out_path=hist_path,
        )
        print(f"  -> {overlay_path}")
        print(f"  -> {hist_path}")

    df = df.drop(columns=["_x_px", "_y_px"])
    out_csv = os.path.join(output_dir, f"{name}_calls.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n-> {out_csv}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, f"{name}_separation_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"-> {summary_csv}")
    print("\nSeparation summary:")
    print(summary_df[["channel", "column_used", "separation", "n_positive", "n_ambiguous", "n_negative"]]
          .to_string(index=False))

    return df, summary_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Per-cell CSV (QuPath or cell_analysis.py output)")
    parser.add_argument("--image", required=True, help="Source (channel, y, x) TIFF the CSV was derived from")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                         help="Channel numbers (1-based) to call positivity for, e.g. --channels 1 2 "
                              "(default: every channel except --dapi_channel and --autofluor_channel)")
    parser.add_argument("--dapi_channel", type=int, default=None,
                         help="1-based channel number of the DAPI/nucleus channel, excluded from the "
                              "default --channels list (default: last channel of the image)")
    parser.add_argument("--compartment", nargs="+", default=["Cytoplasm"],
                         choices=["Nucleus", "Cytoplasm", "Cell"],
                         help="Which compartment's intensity to threshold on -- one value applied to "
                              "every channel, or one value per --channels entry, matched positionally "
                              "(default: Cytoplasm)")
    parser.add_argument("--stat", nargs="+", default=["Mean"],
                         choices=["Mean", "Median", "Dot_Count", "Dot_Density"],
                         help="Which statistic to threshold on -- one value applied to every channel, "
                              "or one value per --channels entry, matched positionally (default: Mean; "
                              "use Median if the channel suffers from punctate autofluorescence "
                              "contamination; use Dot_Count/Dot_Density for a genuinely punctate marker "
                              "counted via cell_analysis.py's --puncta_channel -- Dot_Density (dots / "
                              "cell area) is preferred over raw Dot_Count, which is confounded by cell "
                              "size; only computed for the Cell compartment, so pass --compartment Cell "
                              "for that channel)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pos_thresh", type=float, default=0.8,
                         help="Probability above which a cell is called positive (default 0.8)")
    parser.add_argument("--neg_thresh", type=float, default=0.2,
                         help="Probability below which a cell is called negative (default 0.2)")
    parser.add_argument("--autofluor_channel", type=int, default=None,
                         help="Channel number (1-based) of the autofluorescence channel, if present. "
                              "When given, target channels are linearly corrected against it before "
                              "thresholding, and cells with punctate/spiky autofluorescence are flagged.")
    parser.add_argument("--granule_flag_percentile", type=float, default=0.9,
                         help="Percentile of autofluorescence 'burden' above which a cell is flagged "
                              "as granule-heavy (default 0.9, i.e. top 10%%)")
    parser.add_argument("--granule_drop_threshold", type=float, default=0.5,
                         help="Cells with granule_burden above this are dropped entirely (excluded from "
                              "calling, overlays, and CSV output), not just flagged. Only a literal area "
                              "fraction (0-1) with the --autofluor_channel-free debris-mask fallback; "
                              "with --autofluor_channel it's a Std.Dev./Median ratio instead (default 0.5)")
    parser.add_argument("--no_granule_drop", action="store_true",
                         help="Disable automatic dropping -- still compute and flag granule_burden/"
                              "high_granule_burden, just don't remove any cells")
    parser.add_argument("--label_mask", default=None,
                         help="Path to a label-ID TIFF (the *_cell_labels.tif saved by cell_analysis.py) "
                              "where each cell's pixels equal its Cell_ID. If given, overlay PNGs fill "
                              "each cell's actual segmented footprint with its call color instead of "
                              "drawing an open circle at its centroid.")
    args = parser.parse_args()

    run(args.csv, args.image, args.compartment, args.stat,
        args.output_dir, args.pos_thresh, args.neg_thresh,
        channels=args.channels, dapi_channel=args.dapi_channel,
        autofluor_channel=args.autofluor_channel,
        granule_flag_percentile=args.granule_flag_percentile,
        granule_drop_threshold=(None if args.no_granule_drop else args.granule_drop_threshold),
        label_mask_path=args.label_mask)


if __name__ == "__main__":
    main()
