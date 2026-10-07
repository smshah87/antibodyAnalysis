"""
Python replacement for the QuPath StarDist + cell-expansion + measurement workflow.

Takes the multi-channel background-subtracted MIP TIFFs produced by preprocess.py
(shape: channel, y, x -- typical channel order 1=Ab1, 2=Ab2, 3=autofluorescence,
4=DAPI, matching your acquisition's ChNames order; channels are always 1-based
here, same convention as --channels elsewhere in this pipeline) and produces a
per-cell CSV in the same column format used by the earlier QuPath-derived CSVs,
so downstream scripts (positivity_calling.py etc.) work unchanged.

Pipeline:
  1. Nucleus segmentation on the DAPI channel, via EITHER Cellpose ('nuclei'
     model, GPU if available) OR StarDist ('2D_versatile_fluo' pretrained
     model) -- pick with --segmentation_method.
  2. Cytoplasm ring built by expanding each nucleus outward by a fixed distance
     without crossing into a neighboring cell's territory (skimage's
     expand_labels -- the Python equivalent of QuPath's "cell expansion").
  3. Per-cell, per-channel, per-compartment (nucleus / cytoplasm / whole-cell)
     intensity statistics: mean, median, std, min, max.
  4. CSV export with the same column naming convention as the QuPath export
     script, so it's a drop-in replacement.

Requires (Cellpose): cellpose, torch
Requires (StarDist):  stardist, csbdeep, tensorflow
Either way also needs: scikit-image, scipy, pandas, tifffile. GPU strongly
recommended for either backend -- CPU works but is slow for large images/batches.

Pixel size (microns/pixel) is never a user-supplied argument -- it's read back
automatically from each input image via preprocess.get_pixel_size_um(), which
in turn requires that image to carry pixel-size metadata (either written by
preprocess.py itself, or OME-XML/sidecar metadata on a raw file).

Any number of channels is supported -- channel names are generated on the fly
("Channel 1", "Channel 2", ...) from the image's actual channel count.

Usage (single image):
    python cell_analysis.py --input image.tif --output cells.csv \
        --segmentation_method cellpose

Usage (batch over a folder):
    python cell_analysis.py --input_dir processed/ --output_dir results/ \
        --segmentation_method stardist
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import expand_labels, find_boundaries, relabel_sequential
from skimage.measure import regionprops_table, label as label_components
from skimage.filters import threshold_otsu
from skimage.morphology import disk, dilation
from skimage.feature import blob_log

import preprocess

DAPI_CHANNEL = None             # 1-based channel number (same convention as --channels); None =
                                 # last channel of the image, since DAPI is almost always acquired last
CELL_EXPANSION_PX = 10.0        # cytoplasm ring width, in pixels
MIN_NUCLEUS_AREA_UM2 = 15.0     # objects smaller than this are dropped as granules/debris, not nuclei
GRANULE_BLOB_MIN_DIAMETER_UM = 0.8  # objects smaller than this are punctate real signal, not debris blobs
                                     # -- catches more small blobs than 1.0um without crossing into the
                                     # real punctate-signal noise floor (~0.2-0.5um, where object count
                                     # explodes without much added area -- a sign of false positives)
GRANULE_BLOB_MAX_ECCENTRICITY = 1.0  # 0 = circle, 1 = a line; effectively disabled -- debris blobs are
                                      # often noticeably elongated/irregular, not perfect circles, and
                                      # capping this at 0.9 barely changed detection (75 vs 82 objects on
                                      # a test image), so eccentricity isn't a useful discriminator here
GRANULE_BLOB_MIN_SOLIDITY = 0.6  # area / convex_area; the main effective shape discriminator (unlike
                                  # max_eccentricity, which barely filters anything) -- 0.9 misses a lot
                                  # of real debris blobs, 0.3 barely differs from 0.6, so 0.6 is the
                                  # sweet spot: rejects fragmented/noisy regions without rejecting real
                                  # irregular debris
GRANULE_BLOB_MIN_INTENSITY_RATIO = 1.0  # an object's own mean intensity must be at least this many
                                         # times the channel's Otsu threshold to count as a blob; 1.0
                                         # (default) is the validated setting -- shape does the real
                                         # discriminating here, raising this rarely helps (see
                                         # segment_granule_blobs() docstring for why)
GRANULE_BLOB_MASK_EXPANSION_PX = 3.0  # after shape/intensity filtering, the kept blob mask is dilated
                                       # outward by this many pixels -- debris typically has a dim halo
                                       # of partially-thresholded pixels right at its edge that clearing
                                       # Otsu alone doesn't catch; 0 disables expansion
PUNCTA_MIN_DIAMETER_UM = 0.2   # objects smaller than this are single/few-pixel camera noise, not real
                                # puncta -- matches the noise floor referenced in
                                # GRANULE_BLOB_MIN_DIAMETER_UM's comment above
PUNCTA_MAX_DIAMETER_UM = GRANULE_BLOB_MIN_DIAMETER_UM  # objects at or above this size are debris blobs
                                                        # (segment_granule_blobs()'s territory), not
                                                        # puncta -- keeps the two detectors' size ranges
                                                        # complementary rather than overlapping
PUNCTA_LOG_THRESHOLD_REL = 0.02  # LoG blob-detection response threshold, relative to the image's
                                   # strongest response (skimage.feature.blob_log's threshold_rel) --
                                   # the main sensitivity knob; lower catches more (dimmer) puncta.
                                   # Picked empirically on real data by sweeping 0.05/0.02/0.01/0.005 for
                                   # raw detection quality: 0.05 visibly missed real dim puncta; 0.01
                                   # started flagging circles over empty background with no visible
                                   # signal at all (clear noise); 0.02 sits in between.
                                   #
                                   # A higher value (0.03-0.05) was tried briefly because it gave better
                                   # single-position GMM separation on one test FOV, but that comparison
                                   # turned out not to mean what it looked like: Dot_Count's raw
                                   # distribution on that FOV is a smooth, unimodal, monotonically
                                   # decaying count distribution with no real gap in it (confirmed via
                                   # histogram), so ANY single-position 2-component GMM fit on it --
                                   # regardless of which threshold_rel produced the counts -- is carving
                                   # an arbitrary split out of noise, the same failure mode already found
                                   # and fixed for Mean/Median intensity on sparse markers (see
                                   # fit_pooled_gmms() in call_and_plot_positivity.py). Fix the fitting
                                   # (pool across positions), don't chase a threshold_rel that happens to
                                   # flatter a broken single-position fit.


_MODEL_CACHE = {}  # keyed by (backend, ...params) -> loaded model object, reused across images in
                    # this process rather than rebuilt from disk on every single call -- rebuilding
                    # (StarDist2D.from_pretrained / CellposeModel) reloads and reconstructs the whole
                    # network from weights on disk each time, which dominates runtime on a batch of images


def segment_nuclei_cellpose(dapi_img, use_gpu=None, diameter=None):
    """
    Runs Cellpose-SAM segmentation on a single-channel image.
    Returns an integer label mask (0 = background, 1..N = nucleus IDs).
    """
    from cellpose import models
    import torch

    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    cache_key = ("cellpose", use_gpu)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = models.CellposeModel(gpu=use_gpu)
    model = _MODEL_CACHE[cache_key]

    masks, flows, styles = model.eval(
        dapi_img,
        diameter=diameter,     # None = let Cellpose estimate it automatically
    )
    return masks.astype(np.int32)


def segment_nuclei_stardist(dapi_img, prob_thresh=None, nms_thresh=None):
    """
    Runs StarDist ('2D_versatile_fluo' pretrained model) on a single-channel
    fluorescence image. Returns an integer label mask (0 = background, 1..N =
    nucleus IDs) -- same convention as segment_nuclei_cellpose.

    prob_thresh / nms_thresh: override StarDist's default detection / overlap
    thresholds if needed (None = use the model's own defaults).
    """
    from stardist.models import StarDist2D
    from csbdeep.utils import normalize

    cache_key = ("stardist", "2D_versatile_fluo")
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = StarDist2D.from_pretrained("2D_versatile_fluo")
    model = _MODEL_CACHE[cache_key]

    img_norm = normalize(dapi_img.astype(np.float32), 1, 99.8)
    labels, _details = model.predict_instances(
        img_norm, prob_thresh=prob_thresh, nms_thresh=nms_thresh
    )
    return labels.astype(np.int32)


def segment_nuclei(dapi_img, method="cellpose", use_gpu=None, diameter=None,
                    stardist_prob_thresh=None, stardist_nms_thresh=None):
    """
    Dispatches to the requested segmentation backend. Both return the same
    label-mask convention, so everything downstream (cytoplasm expansion,
    measurement) is identical regardless of which one you pick.
    """
    if method == "cellpose":
        return segment_nuclei_cellpose(dapi_img, use_gpu=use_gpu, diameter=diameter)
    elif method == "stardist":
        return segment_nuclei_stardist(
            dapi_img, prob_thresh=stardist_prob_thresh, nms_thresh=stardist_nms_thresh
        )
    else:
        raise ValueError(f"Unknown segmentation method '{method}'. Use 'cellpose' or 'stardist'.")


def build_cell_labels(nucleus_labels, expansion_px):
    """
    Expands each nucleus outward by expansion_px to approximate the cell/cytoplasm
    boundary, without crossing into a neighboring cell's territory -- the direct
    equivalent of QuPath's "cell expansion" setting.
    """
    return expand_labels(nucleus_labels, distance=expansion_px)


def remove_border_cells(nucleus_labels, cytoplasm_labels, cell_labels):
    """
    Drops any cell whose whole-cell footprint (cell_labels, i.e. nucleus +
    cell-expansion ring) touches the image edge. Those cells are physically
    cut off by the field of view, so every per-cell measurement for them
    (area, intensity stats, dot counts) would be based on an incomplete
    cell and isn't comparable to fully-imaged cells. Checked on cell_labels
    rather than just nucleus_labels, since a nucleus that's fully inside
    the frame can still have its cytoplasm ring clipped by the edge.

    Returns (nucleus_labels, cytoplasm_labels, cell_labels, n_removed) with
    dropped cells zeroed out and survivors relabeled to consecutive integers
    1..N (same mapping applied to all three masks, so a given label still
    refers to the same physical cell across them) -- keeps downstream
    label_ids conventions intact.
    """
    border_ids = np.unique(np.concatenate([
        cell_labels[0, :], cell_labels[-1, :],
        cell_labels[:, 0], cell_labels[:, -1],
    ]))
    border_ids = border_ids[border_ids != 0]
    if border_ids.size == 0:
        return nucleus_labels, cytoplasm_labels, cell_labels, 0

    drop_mask = np.isin(cell_labels, border_ids)
    nucleus_labels = np.where(np.isin(nucleus_labels, border_ids), 0, nucleus_labels)
    cytoplasm_labels = np.where(drop_mask, 0, cytoplasm_labels)
    cell_labels = np.where(drop_mask, 0, cell_labels)

    keep_ids = np.unique(cell_labels)
    keep_ids = keep_ids[keep_ids != 0]
    remap = np.zeros(int(keep_ids.max()) + 1 if keep_ids.size else 1, dtype=cell_labels.dtype)
    remap[keep_ids] = np.arange(1, keep_ids.size + 1)

    def apply_remap(labels):
        out = np.zeros_like(labels)
        mask = labels > 0
        out[mask] = remap[labels[mask]]
        return out

    return apply_remap(nucleus_labels), apply_remap(cytoplasm_labels), apply_remap(cell_labels), int(border_ids.size)


def filter_small_objects(label_img, min_area_px):
    """
    Drops labeled objects smaller than min_area_px (small bright granules/debris
    that segmentation backends sometimes mistake for nuclei), then relabels the
    survivors to consecutive integers 1..N so downstream code's label_ids
    convention still holds.

    Returns (filtered_label_img, n_dropped, granule_mask): granule_mask is a
    boolean array marking every pixel that belonged to a dropped object. This
    DAPI-channel debris doubles as a per-cell "granule burden" signal when no
    dedicated autofluorescence channel is available -- see
    call_and_plot_positivity.py's --autofluor_channel fallback, which uses it
    the same way it'd otherwise use a real autofluorescence channel's texture.
    """
    granule_mask = np.zeros(label_img.shape, dtype=bool)
    if min_area_px <= 0:
        return label_img, 0, granule_mask

    props = regionprops_table(label_img, properties=("label", "area"))
    small_labels = props["label"][props["area"] < min_area_px]
    if len(small_labels) == 0:
        return label_img, 0, granule_mask

    granule_mask = np.isin(label_img, small_labels)
    filtered = label_img.copy()
    filtered[granule_mask] = 0
    filtered, _, _ = relabel_sequential(filtered)
    return filtered, len(small_labels), granule_mask


def segment_granule_blobs(channel_img, min_area_px, max_eccentricity, min_solidity, min_intensity_ratio=1.0,
                           mask_expansion_px=3.0):
    """
    Finds solid debris blobs in a single channel, distinguishing them from
    real punctate signal primarily by shape: real staining in a channel like
    this typically looks grainy/punctate under the microscope (many small
    separate bright specks), while autofluorescent debris (lipofuscin, dust,
    etc.) tends to form one solid, filled-in, round mass. Otsu-thresholding
    the raw channel and connected-component labeling captures both kinds of
    region, but they separate cleanly on shape: punctate signal breaks into
    many small, low-solidity components (a mostly-hollow bounding hull, since
    the real gaps between puncta count against it), while a genuine debris
    blob stays one large, high-solidity, round region.

    A 3-class multi-Otsu variant (candidate pixels = only the brightest of
    three intensity tiers, on the theory that debris is a distinct, much
    brighter population than real signal) was tried and rejected: visual QC
    on real data showed it under-flagging -- plenty of genuine debris blobs
    are only moderately brighter than real signal, not in the extreme top
    tier, so restricting candidates to that tier drops real debris rather
    than sharpening the cut. This matches what pooling
    intensity_mean/threshold ratios of shape-passing candidates across many
    images had already shown: a smooth, unimodal, decaying tail with no gap
    to cut at anywhere -- brightness doesn't cleanly separate debris from
    shape-passing false positives at the object level, so shape alone is
    left to do the real discriminating, exactly as originally designed.

    Deliberately skips the white-tophat pre-filtering used elsewhere in this
    module for granule detection -- tophat is designed to isolate small
    bright spots by suppressing anything larger than its structuring element,
    which is exactly the large solid blob this function is trying to keep.

    min_area_px / max_eccentricity / min_solidity: an object must be at least
    this large, at least this round (0=circle, 1=a line), and at least this
    solid (area / convex_area; 1.0 = perfectly convex/filled) to count as a
    blob rather than noise or real punctate texture.

    min_intensity_ratio: an object's own mean raw intensity must be at least
    this many times the Otsu threshold that defined "foreground" in the first
    place to count as a blob -- 1.0 (default, and the validated setting)
    applies no extra brightness requirement beyond clearing Otsu. Raising it
    trades recall for precision on brightness alone; given the above, this
    rarely helps and isn't the recommended way to tune this function --
    prefer adjusting the shape cutoffs instead.

    mask_expansion_px: after the shape/intensity filtering above, the kept
    blob mask is dilated outward by this many pixels (disk footprint) --
    debris typically has a dim halo of partially-thresholded pixels right at
    its edge that clearing Otsu alone doesn't catch. 0 disables expansion.

    Returns a boolean mask, same (Y, X) shape as channel_img.
    """
    threshold = threshold_otsu(channel_img)
    candidates = label_components(channel_img > threshold)
    if candidates.max() == 0:
        return np.zeros(candidates.shape, dtype=bool)

    props = regionprops_table(
        candidates, intensity_image=channel_img,
        properties=("label", "area", "eccentricity", "solidity", "intensity_mean"),
    )
    keep = (
        (props["area"] >= min_area_px)
        & (props["eccentricity"] <= max_eccentricity)
        & (props["solidity"] >= min_solidity)
        & (props["intensity_mean"] >= min_intensity_ratio * threshold)
    )
    mask = np.isin(candidates, props["label"][keep])
    if mask_expansion_px > 0:
        mask = dilation(mask, footprint=disk(int(round(mask_expansion_px))))
    return mask


def detect_puncta_blobs(channel_img, min_diameter_px, max_diameter_px, threshold_rel,
                         granule_blob_mask=None, num_sigma=5, overlap=0.5):
    """
    Finds small, bright, punctate real signal (e.g. DLL3) via Laplacian-of-
    Gaussian (LoG) blob detection -- the inverse problem to
    segment_granule_blobs() above, which excludes large solid debris rather
    than finding real signal.

    A white-tophat-plus-threshold approach was tried first and rejected:
    with any threshold sensitive enough to catch real dim puncta (checked
    against real data), it also picked up narrow fragments of the fibrous
    tissue background, since a thin fiber (narrower than the tophat
    structuring element) partially survives tophat as a bright streak that
    breaks into many small "blob"-shaped fragments once thresholded. LoG
    detects blobs by their actual Gaussian-like shape at a given scale
    directly, rather than by brightness alone, and checked side by side on
    the same image it both catches far more of the real dim puncta AND
    avoids the fiber-fragment false positives.

    min_diameter_px / max_diameter_px set the expected real-punctum size
    range directly as the LoG scale-space search range (min_sigma/max_sigma
    = diameter / (2*sqrt(2)), the standard relationship between a blob's
    diameter and the Gaussian scale its LoG response peaks at) -- no
    separate post-hoc size filtering needed, unlike the tophat approach.
    threshold_rel: LoG response threshold, relative to the strongest
    response in the image (skimage.feature.blob_log's threshold_rel) -- the
    main sensitivity knob; lower catches more (dimmer) puncta.

    granule_blob_mask: if given (the same channel's debris mask from
    segment_granule_blobs()), any detected blob centered on a flagged debris
    pixel is dropped. Detection deliberately runs on the raw, un-masked
    image first rather than zeroing debris pixels before detection: zeroing
    creates a sharp artificial edge at the debris boundary, which LoG then
    misdetects as blobs of its own, clustering false positives right around
    every debris blob's rim -- confirmed empirically. Filtering by center
    location afterward gets the same exclusion without that artifact.

    Returns an (N, 3) array of (y, x, sigma) for each detected punctum,
    already filtered against granule_blob_mask if given.
    """
    min_sigma = min_diameter_px / (2 * np.sqrt(2))
    max_sigma = max_diameter_px / (2 * np.sqrt(2))
    blobs = blob_log(
        channel_img, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma,
        threshold=None, threshold_rel=threshold_rel, overlap=overlap,
    )
    if granule_blob_mask is not None and len(blobs) > 0:
        ys = np.clip(np.round(blobs[:, 0]).astype(int), 0, channel_img.shape[0] - 1)
        xs = np.clip(np.round(blobs[:, 1]).astype(int), 0, channel_img.shape[1] - 1)
        blobs = blobs[~granule_blob_mask[ys, xs]]
    return blobs


def count_puncta_per_cell(blobs, label_img, label_ids):
    """
    Counts how many detected puncta (see detect_puncta_blobs()) fall within
    each region of label_img (a Nucleus/Cytoplasm/Cell label mask, as used
    elsewhere in this module), assigning each punctum to whichever label its
    center lands on -- avoids double-counting a punctum that happens to
    straddle a region or cell boundary.

    label_ids must be a contiguous 1..N array (as produced elsewhere in this
    module) -- used directly as a bincount index.

    Returns an integer array of counts, one per label_ids entry (0 if none).
    """
    if len(blobs) == 0:
        return np.zeros(len(label_ids), dtype=int)

    ys = np.clip(np.round(blobs[:, 0]).astype(int), 0, label_img.shape[0] - 1)
    xs = np.clip(np.round(blobs[:, 1]).astype(int), 0, label_img.shape[1] - 1)
    cell_of_punctum = label_img[ys, xs]
    counts = np.bincount(cell_of_punctum[cell_of_punctum > 0], minlength=label_ids.max() + 1)
    return counts[label_ids]


def plot_puncta_overlay(channel_img, blobs, title, out_path):
    """
    Draws each detected punctum (see detect_puncta_blobs()) as an open red
    circle (radius = sigma*sqrt(2), the blob's actual detected size) on top
    of the channel it came from, for visually checking what's getting
    counted. An open circle rather than a filled mask, since LoG gives a
    center + scale, not a precise segmentation boundary.
    """
    lo, hi = np.percentile(channel_img, [1, 99.5])
    disp = np.clip((channel_img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(disp, cmap="gray")
    for y, x, sigma in blobs:
        ax.add_patch(plt.Circle((x, y), sigma * np.sqrt(2), color="red", fill=False, linewidth=0.8))
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_granule_overlay(channel_img, granule_mask, title, out_path):
    """
    Draws the granule/debris blob mask (see segment_granule_blobs()) in red
    on top of the channel it was detected in, for visually checking what's
    actually getting flagged as debris.
    """
    lo, hi = np.percentile(channel_img, [1, 99.5])
    disp = np.clip((channel_img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)

    overlay = np.zeros((*disp.shape, 4))
    overlay[granule_mask] = (1, 0, 0, 0.7)  # red

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(disp, cmap="gray")
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_segmentation_overlay(dapi_img, nucleus_labels, cell_labels, title, out_path):
    """
    Draws nucleus (cyan) and cell/cytoplasm (magenta) segmentation boundaries
    on top of the DAPI channel, for visually checking the segmentation quality.
    """
    lo, hi = np.percentile(dapi_img, [1, 99.5])
    disp = np.clip((dapi_img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)

    cell_boundaries = find_boundaries(cell_labels, mode="inner")
    nucleus_boundaries = find_boundaries(nucleus_labels, mode="inner")

    overlay = np.zeros((*disp.shape, 4))
    overlay[cell_boundaries] = (1, 0, 1, 0.6)     # magenta: cell/cytoplasm boundary
    overlay[nucleus_boundaries] = (0, 1, 1, 0.9)  # cyan: nucleus boundary (drawn on top)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(disp, cmap="gray")
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_compartment_stats(image_c, label_img, label_ids, compartment_name, channel_idx):
    """
    image_c: single-channel 2D image
    label_img: integer label mask for this compartment (nucleus, cytoplasm, or whole cell)
    label_ids: sorted array of the cell IDs to compute stats for (1..N, excluding 0)
    Returns a dict of column_name -> array (one value per cell, in label_ids order)
    """
    prefix = f"{compartment_name}: Channel {channel_idx + 1}"

    means = ndimage.mean(image_c, labels=label_img, index=label_ids)
    medians = ndimage.median(image_c, labels=label_img, index=label_ids)
    stds = ndimage.standard_deviation(image_c, labels=label_img, index=label_ids)
    mins = ndimage.minimum(image_c, labels=label_img, index=label_ids)
    maxs = ndimage.maximum(image_c, labels=label_img, index=label_ids)

    return {
        f"{prefix}: Mean": means,
        f"{prefix}: Median": medians,
        f"{prefix}: Std.Dev.": stds,
        f"{prefix}: Min": mins,
        f"{prefix}: Max": maxs,
    }


def analyze_image(image, pixel_size_um, cell_expansion_px=CELL_EXPANSION_PX,
                   dapi_channel=DAPI_CHANNEL, segmentation_method="cellpose",
                   use_gpu=None, diameter=None, stardist_prob_thresh=None, stardist_nms_thresh=None,
                   min_nucleus_area_um2=MIN_NUCLEUS_AREA_UM2,
                   granule_blob_channel=None,
                   granule_blob_min_diameter_um=GRANULE_BLOB_MIN_DIAMETER_UM,
                   granule_blob_max_eccentricity=GRANULE_BLOB_MAX_ECCENTRICITY,
                   granule_blob_min_solidity=GRANULE_BLOB_MIN_SOLIDITY,
                   granule_blob_min_intensity_ratio=GRANULE_BLOB_MIN_INTENSITY_RATIO,
                   granule_blob_mask_expansion_px=GRANULE_BLOB_MASK_EXPANSION_PX,
                   extra_rolling_ball_channel=None, extra_rolling_ball_radius=3,
                   puncta_channel=None,
                   puncta_min_diameter_um=PUNCTA_MIN_DIAMETER_UM,
                   puncta_max_diameter_um=PUNCTA_MAX_DIAMETER_UM,
                   puncta_log_threshold_rel=PUNCTA_LOG_THRESHOLD_REL):
    """
    image: (C, Y, X) array, background-subtracted MIP (output of preprocess.py)
    dapi_channel: 1-based channel number (same convention as --channels elsewhere);
    None = last channel of the image (DAPI is almost always acquired last).

    granule_blob_channel: 1-based channel number to detect solid debris blobs
    in (see segment_granule_blobs()) -- None disables this (default). Pick a
    channel where real signal is punctate/grainy and debris shows up as solid
    round masses, distinguishable by shape alone (no cross-channel comparison
    needed). Pixels flagged this way are excluded from every channel's
    intensity statistics (Mean/Median/etc, all compartments) below, not just
    reported as a per-cell area.
    granule_blob_min_diameter_um / granule_blob_max_eccentricity /
    granule_blob_min_solidity / granule_blob_min_intensity_ratio /
    granule_blob_mask_expansion_px: shape/size/brightness cutoffs and mask
    dilation for what counts as a debris blob rather than real punctate
    signal -- see segment_granule_blobs().

    extra_rolling_ball_channel / extra_rolling_ball_radius: 1-based channel
    number to run a second, small-radius rolling-ball pass on (see
    preprocess.imagej_rolling_ball_background()), for finer local background
    flattening than preprocess.py's main pass provides -- None (default)
    disables this. Deliberately applied here, AFTER granule blob detection
    above, not in preprocess.py before it: a small-radius rolling ball
    erodes/attenuates bright local peaks, and debris blobs are exactly that
    kind of peak, so running it before detection was suppressing the very
    signal segment_granule_blobs() needs to see, causing missed blobs. Doing
    it here instead only affects the intensity values used for this
    channel's Mean/Median/etc stats below -- granule detection above already
    ran on the un-eroded image, and blob-pixel exclusion from those stats is
    via the label masks (zeroed wherever granule_blob_mask is True),
    independent of which version of the pixel values gets measured.

    puncta_channel: 1-based channel number to count small, bright, punctate
    real signal in (see detect_puncta_blobs()) -- None (default) disables
    this. For a marker that's genuinely granular/punctate (e.g. DLL3) rather
    than diffusely distributed, a per-cell dot count can separate positive
    from negative cells better than Mean/Median intensity, which gets
    diluted by cell area and by background between puncta. Adds a
    `Dot_Count` column per compartment (Nucleus/Cytoplasm/Cell), alongside
    the existing intensity stats rather than replacing them, so both can be
    compared. Also adds a `Dot_Density` column (Dot_Count / Cell: Area,
    i.e. dots per um^2) for the Cell compartment only -- preferred over raw
    Dot_Count for positivity calling, since Dot_Count is confounded by cell
    size (bigger cells simply have more area to catch background speckle
    in, regardless of real signal; confirmed empirically: Dot_Count
    correlated with Cell: Area at r=0.68 on a real test FOV, Dot_Density
    only r=0.28). Counted on the same un-eroded image granule blob detection
    uses (before any extra_rolling_ball_channel correction). If
    granule_blob_channel == puncta_channel, detected puncta centered on the
    debris mask are dropped (see detect_puncta_blobs()); the per-cell counts
    also use the granule-exclusion-adjusted label masks regardless of
    channel match, so a punctum landing in a region already flagged as
    debris (in any channel) doesn't get counted.
    puncta_min_diameter_um / puncta_max_diameter_um / puncta_log_threshold_rel:
    expected size range and LoG detection sensitivity for what counts as a
    real punctum -- see detect_puncta_blobs().

    Any cell whose whole-cell footprint touches the image edge is dropped
    entirely (unconditionally, no flag) -- see remove_border_cells(). Those
    cells are physically clipped by the field of view, so their measurements
    would be based on an incomplete cell.

    Returns (df, nucleus_labels, cell_labels, granule_blob_mask, puncta_blobs):
    a pandas DataFrame (one row per detected cell), the integer label masks
    used to build it, the boolean debris-blob mask (None if
    granule_blob_channel wasn't given, or no cells were detected), and the
    (N, 3) detected-puncta (y, x, sigma) array (None if puncta_channel
    wasn't given, or no cells were detected).
    """
    n_channels = image.shape[0]
    if dapi_channel is None:
        dapi_channel = n_channels
    dapi_channel_index = dapi_channel - 1
    dapi = image[dapi_channel_index]

    nucleus_labels = segment_nuclei(
        dapi, method=segmentation_method, use_gpu=use_gpu, diameter=diameter,
        stardist_prob_thresh=stardist_prob_thresh, stardist_nms_thresh=stardist_nms_thresh,
    )
    n_raw = nucleus_labels.max()

    min_area_px = min_nucleus_area_um2 / (pixel_size_um ** 2)
    nucleus_labels, n_dropped, granule_mask = filter_small_objects(nucleus_labels, min_area_px)
    n_cells = nucleus_labels.max()
    if n_dropped:
        print(f"  Detected {n_raw} objects, dropped {n_dropped} smaller than "
              f"{min_nucleus_area_um2:.1f} um^2 (likely granules/debris) -> {n_cells} nuclei "
              f"(method={segmentation_method})")
    else:
        print(f"  Detected {n_cells} nuclei (method={segmentation_method})")
    if n_cells == 0:
        return pd.DataFrame(), nucleus_labels, nucleus_labels, None, None

    cell_labels = build_cell_labels(nucleus_labels, cell_expansion_px)
    # cytoplasm = whole-cell region minus the nucleus itself
    cytoplasm_labels = np.where(nucleus_labels == 0, cell_labels, 0)

    nucleus_labels, cytoplasm_labels, cell_labels, n_border = remove_border_cells(
        nucleus_labels, cytoplasm_labels, cell_labels
    )
    if n_border:
        n_cells = cell_labels.max()
        print(f"  Dropped {n_border} cell(s) touching the image border (incomplete) -> {n_cells} cells")
    if n_cells == 0:
        return pd.DataFrame(), nucleus_labels, nucleus_labels, None, None

    label_ids = np.arange(1, n_cells + 1)

    # Centroid + shape info from the whole-cell label mask (regionprops is
    # convenient here since we want it once, not per channel)
    props = regionprops_table(
        cell_labels, properties=("label", "centroid", "area", "equivalent_diameter_area")
    )
    props_df = pd.DataFrame(props).set_index("label").reindex(label_ids)

    # Area of dropped small (likely granule/debris) objects that fall within
    # each cell's whole-cell footprint -- see filter_small_objects().
    granule_area_px = ndimage.sum(granule_mask, labels=cell_labels, index=label_ids)

    data = {
        "Cell_ID": label_ids,
        "Centroid_X_um": props_df["centroid-1"].values * pixel_size_um,
        "Centroid_Y_um": props_df["centroid-0"].values * pixel_size_um,
        "Cell: Area": props_df["area"].values * (pixel_size_um ** 2),
        "Cell: Max diameter µm": props_df["equivalent_diameter_area"].values * pixel_size_um,
        "Granule_Area": granule_area_px * (pixel_size_um ** 2),
    }

    nucleus_labels_clean, cytoplasm_labels_clean, cell_labels_clean = nucleus_labels, cytoplasm_labels, cell_labels
    granule_blob_mask = None
    if granule_blob_channel is not None:
        blob_min_area_px = np.pi * (0.5 * granule_blob_min_diameter_um / pixel_size_um) ** 2
        granule_blob_mask = segment_granule_blobs(
            image[granule_blob_channel - 1], blob_min_area_px,
            granule_blob_max_eccentricity, granule_blob_min_solidity,
            min_intensity_ratio=granule_blob_min_intensity_ratio,
            mask_expansion_px=granule_blob_mask_expansion_px,
        )
        blob_area_px = ndimage.sum(granule_blob_mask, labels=cell_labels, index=label_ids)
        data["Granule_Area_Blob"] = blob_area_px * (pixel_size_um ** 2)
        nucleus_labels_clean = np.where(granule_blob_mask, 0, nucleus_labels)
        cytoplasm_labels_clean = np.where(granule_blob_mask, 0, cytoplasm_labels)
        cell_labels_clean = np.where(granule_blob_mask, 0, cell_labels)

    puncta_blobs = None
    if puncta_channel is not None:
        puncta_min_diameter_px = puncta_min_diameter_um / pixel_size_um
        puncta_max_diameter_px = puncta_max_diameter_um / pixel_size_um
        mask_for_puncta = granule_blob_mask if granule_blob_channel == puncta_channel else None
        puncta_blobs = detect_puncta_blobs(
            image[puncta_channel - 1], puncta_min_diameter_px, puncta_max_diameter_px,
            puncta_log_threshold_rel, granule_blob_mask=mask_for_puncta,
        )
        prefix = f"Channel {puncta_channel}"
        data[f"Nucleus: {prefix}: Dot_Count"] = count_puncta_per_cell(puncta_blobs, nucleus_labels_clean, label_ids)
        data[f"Cytoplasm: {prefix}: Dot_Count"] = count_puncta_per_cell(
            puncta_blobs, cytoplasm_labels_clean, label_ids
        )
        data[f"Cell: {prefix}: Dot_Count"] = count_puncta_per_cell(puncta_blobs, cell_labels_clean, label_ids)
        # Dot_Density (dots / um^2 of whole-cell area) rather than raw Dot_Count
        # -- raw count is confounded by cell size (bigger cells simply have more
        # area to catch background speckle in, regardless of real signal;
        # confirmed empirically: Dot_Count correlated with Cell: Area at r=0.68
        # on a real test FOV, Dot_Density only r=0.28). Only computed for the
        # Cell compartment, not Nucleus/Cytoplasm -- puncta aren't confined to
        # either sub-region, so whole-cell is the only compartment that makes
        # sense to normalize by for a dot-based measure.
        data[f"Cell: {prefix}: Dot_Density"] = data[f"Cell: {prefix}: Dot_Count"] / data["Cell: Area"]

    for c in range(n_channels):
        channel_img = image[c]
        if extra_rolling_ball_channel is not None and c == extra_rolling_ball_channel - 1:
            extra_background = preprocess.imagej_rolling_ball_background(channel_img, extra_rolling_ball_radius)
            channel_img = np.clip(channel_img - extra_background, 0, None)
        data.update(compute_compartment_stats(channel_img, nucleus_labels_clean, label_ids, "Nucleus", c))
        data.update(compute_compartment_stats(channel_img, cytoplasm_labels_clean, label_ids, "Cytoplasm", c))
        data.update(compute_compartment_stats(channel_img, cell_labels_clean, label_ids, "Cell", c))

    return pd.DataFrame(data), nucleus_labels, cell_labels, granule_blob_mask, puncta_blobs


def process_one_image(tif_path, output_dir, cell_expansion_px, use_gpu, diameter,
                       segmentation_method="cellpose", stardist_prob_thresh=None, stardist_nms_thresh=None,
                       dapi_channel=None, min_nucleus_area_um2=MIN_NUCLEUS_AREA_UM2,
                       granule_blob_channel=None,
                       granule_blob_min_diameter_um=GRANULE_BLOB_MIN_DIAMETER_UM,
                       granule_blob_max_eccentricity=GRANULE_BLOB_MAX_ECCENTRICITY,
                       granule_blob_min_solidity=GRANULE_BLOB_MIN_SOLIDITY,
                       granule_blob_min_intensity_ratio=GRANULE_BLOB_MIN_INTENSITY_RATIO,
                       granule_blob_mask_expansion_px=GRANULE_BLOB_MASK_EXPANSION_PX,
                       extra_rolling_ball_channel=None, extra_rolling_ball_radius=3,
                       puncta_channel=None,
                       puncta_min_diameter_um=PUNCTA_MIN_DIAMETER_UM,
                       puncta_max_diameter_um=PUNCTA_MAX_DIAMETER_UM,
                       puncta_log_threshold_rel=PUNCTA_LOG_THRESHOLD_REL):
    name = os.path.splitext(os.path.basename(tif_path))[0]
    print(f"Analyzing {name} ...")
    image = tifffile.imread(tif_path)
    if image.ndim != 3:
        raise ValueError(f"{tif_path}: expected a (channel, y, x) image, got shape {image.shape}")
    pixel_size_um = preprocess.get_pixel_size_um(tif_path)

    if dapi_channel is None:
        dapi_channel = image.shape[0]  # DAPI is almost always acquired last
    df, nucleus_labels, cell_labels, granule_blob_mask, puncta_blobs = analyze_image(
        image, pixel_size_um, cell_expansion_px, dapi_channel=dapi_channel,
        use_gpu=use_gpu, diameter=diameter,
        segmentation_method=segmentation_method,
        stardist_prob_thresh=stardist_prob_thresh, stardist_nms_thresh=stardist_nms_thresh,
        min_nucleus_area_um2=min_nucleus_area_um2,
        granule_blob_channel=granule_blob_channel,
        granule_blob_min_diameter_um=granule_blob_min_diameter_um,
        granule_blob_max_eccentricity=granule_blob_max_eccentricity,
        granule_blob_min_solidity=granule_blob_min_solidity,
        granule_blob_min_intensity_ratio=granule_blob_min_intensity_ratio,
        granule_blob_mask_expansion_px=granule_blob_mask_expansion_px,
        extra_rolling_ball_channel=extra_rolling_ball_channel,
        extra_rolling_ball_radius=extra_rolling_ball_radius,
        puncta_channel=puncta_channel,
        puncta_min_diameter_um=puncta_min_diameter_um,
        puncta_max_diameter_um=puncta_max_diameter_um,
        puncta_log_threshold_rel=puncta_log_threshold_rel,
    )
    out_path = os.path.join(output_dir, f"{name}_cells.csv")
    df.to_csv(out_path, index=False)
    print(f"  -> saved {out_path} ({len(df)} cells)")

    overlay_path = os.path.join(output_dir, f"{name}_segmentation_overlay.png")
    plot_segmentation_overlay(
        image[dapi_channel - 1], nucleus_labels, cell_labels,
        title=f"{name} — nucleus (cyan) + cell (magenta) segmentation",
        out_path=overlay_path,
    )
    print(f"  -> saved {overlay_path}")

    if granule_blob_mask is not None:
        granule_overlay_path = os.path.join(output_dir, f"{name}_granule_overlay.png")
        plot_granule_overlay(
            image[granule_blob_channel - 1], granule_blob_mask,
            title=f"{name} — debris blobs (red) on Channel {granule_blob_channel}",
            out_path=granule_overlay_path,
        )
        print(f"  -> saved {granule_overlay_path}")

    if puncta_blobs is not None:
        puncta_overlay_path = os.path.join(output_dir, f"{name}_puncta_overlay.png")
        plot_puncta_overlay(
            image[puncta_channel - 1], puncta_blobs,
            title=f"{name} — detected puncta (red) on Channel {puncta_channel}",
            out_path=puncta_overlay_path,
        )
        print(f"  -> saved {puncta_overlay_path}")

    # Cell_ID in the CSV matches the pixel value in this mask 1:1, so downstream
    # QC plotting (call_and_plot_positivity.py) can look up which pixels belong
    # to which cell without re-running segmentation.
    label_mask_path = os.path.join(output_dir, f"{name}_cell_labels.tif")
    tifffile.imwrite(label_mask_path, cell_labels.astype(np.uint16))
    print(f"  -> saved {label_mask_path}")

    return out_path, label_mask_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Single input image (channel, y, x) TIFF")
    parser.add_argument("--input_dir", help="Folder of input images to batch process")
    parser.add_argument("--output", help="Output CSV path (single-image mode)")
    parser.add_argument("--output_dir", help="Output folder for CSVs (batch mode)")
    parser.add_argument("--cell_expansion_px", type=float, default=CELL_EXPANSION_PX,
                         help="Cytoplasm ring width in pixels")
    parser.add_argument("--dapi_channel", type=int, default=DAPI_CHANNEL,
                         help="1-based channel number of the DAPI/nucleus channel (same convention "
                              "as --channels elsewhere in this pipeline; default: last channel of "
                              "the image, since DAPI is almost always acquired last)")
    parser.add_argument("--diameter", type=float, default=None,
                         help="Expected nucleus diameter in pixels (Cellpose only; None = "
                              "auto-estimate; auto-estimation is slower and less reliable, so "
                              "set this once you know your typical nucleus size)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if a GPU is available (Cellpose only)")
    parser.add_argument("--segmentation_method", default="cellpose", choices=["cellpose", "stardist"],
                         help="Which nucleus segmentation backend to use (default: cellpose)")
    parser.add_argument("--stardist_prob_thresh", type=float, default=None,
                         help="StarDist detection probability threshold (None = model default)")
    parser.add_argument("--stardist_nms_thresh", type=float, default=None,
                         help="StarDist non-max-suppression / overlap threshold (None = model default)")
    parser.add_argument("--min_nucleus_area_um2", type=float, default=MIN_NUCLEUS_AREA_UM2,
                         help="Detected objects smaller than this (in um^2) are dropped as granules/"
                              "debris rather than real nuclei (default %(default)s; set to 0 to disable)")
    parser.add_argument("--granule_blob_channel", type=int, default=None,
                         help="1-based channel number to detect solid debris blobs in, distinguishing "
                              "them from real punctate/grainy signal by shape alone (see "
                              "segment_granule_blobs()) -- default: none, disabled. Flagged pixels are "
                              "excluded from every channel's intensity stats, saved as *_granule_overlay.png")
    parser.add_argument("--granule_blob_min_diameter_um", type=float, default=GRANULE_BLOB_MIN_DIAMETER_UM,
                         help="Minimum equivalent-circular-diameter (in um) for an object in "
                              "--granule_blob_channel to count as a debris blob rather than real punctate "
                              "signal (default %(default)s)")
    parser.add_argument("--granule_blob_max_eccentricity", type=float, default=GRANULE_BLOB_MAX_ECCENTRICITY,
                         help="Maximum eccentricity (0=circle, 1=a line) for a debris blob (default %(default)s)")
    parser.add_argument("--granule_blob_min_solidity", type=float, default=GRANULE_BLOB_MIN_SOLIDITY,
                         help="Minimum solidity (area/convex_area) for a debris blob -- real punctate "
                              "signal breaks into many small, low-solidity components at this threshold, "
                              "while genuine debris stays one solid, filled-in region (default %(default)s)")
    parser.add_argument("--granule_blob_min_intensity_ratio", type=float,
                         default=GRANULE_BLOB_MIN_INTENSITY_RATIO,
                         help="An object's own mean intensity must be at least this many times the "
                              "channel's Otsu threshold to count as a debris blob -- 1.0 (default) is the "
                              "validated setting (no extra requirement beyond clearing Otsu); shape does the "
                              "real discriminating here and raising this rarely helps, see "
                              "segment_granule_blobs() (default %(default)s)")
    parser.add_argument("--granule_blob_mask_expansion_px", type=float,
                         default=GRANULE_BLOB_MASK_EXPANSION_PX,
                         help="After shape/intensity filtering, dilate the kept debris blob mask outward "
                              "by this many pixels -- debris typically has a dim halo of partially-"
                              "thresholded pixels right at its edge that clearing Otsu alone doesn't catch. "
                              "0 disables expansion (default %(default)s)")
    parser.add_argument("--extra_rolling_ball_channel", type=int, default=None,
                         help="1-based channel number to run a second, small-radius rolling-ball pass on, "
                              "for finer local background flattening than preprocess.py's main pass -- "
                              "applied here (after granule blob detection above, if enabled) rather than in "
                              "preprocess.py, since a small-radius ball erodes debris blobs too and would "
                              "suppress the signal segment_granule_blobs() needs to see. None (default) "
                              "disables this.")
    parser.add_argument("--extra_rolling_ball_radius", type=int, default=3,
                         help="Rolling-ball radius (pixels) for the second pass, applied only to "
                              "--extra_rolling_ball_channel (default %(default)s)")
    parser.add_argument("--puncta_channel", type=int, default=None,
                         help="1-based channel number to count small, bright, punctate real signal in "
                              "(see detect_puncta_blobs()) -- default: none, disabled. For a genuinely "
                              "granular/punctate marker (e.g. DLL3), a per-cell dot count can separate "
                              "positive/negative cells better than Mean/Median intensity. Adds a "
                              "Dot_Count column per compartment, plus a Dot_Density column (dots / cell "
                              "area, preferred for positivity calling) for the Cell compartment only, "
                              "alongside the existing intensity stats (doesn't replace them), saved as "
                              "*_puncta_overlay.png")
    parser.add_argument("--puncta_min_diameter_um", type=float, default=PUNCTA_MIN_DIAMETER_UM,
                         help="Minimum expected diameter (in um) of a real punctum in --puncta_channel "
                              "-- sets the lower end of the LoG blob-detection scale search range "
                              "(default %(default)s)")
    parser.add_argument("--puncta_max_diameter_um", type=float, default=PUNCTA_MAX_DIAMETER_UM,
                         help="Maximum expected diameter (in um) of a real punctum in --puncta_channel "
                              "rather than a debris blob -- sets the upper end of the LoG blob-detection "
                              "scale search range (default: same as --granule_blob_min_diameter_um's "
                              "default, %(default)s)")
    parser.add_argument("--puncta_log_threshold_rel", type=float, default=PUNCTA_LOG_THRESHOLD_REL,
                         help="LoG blob-detection response threshold, relative to the image's strongest "
                              "response -- the main sensitivity knob for --puncta_channel; lower catches "
                              "more (dimmer) puncta (default %(default)s)")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for batch mode")
    args = parser.parse_args()

    use_gpu = False if args.cpu else None  # None -> auto-detect inside segment_nuclei_cellpose

    if args.input:
        out_dir = os.path.dirname(args.output) or "."
        os.makedirs(out_dir, exist_ok=True)
        process_one_image(args.input, out_dir, args.cell_expansion_px,
                           use_gpu, args.diameter, segmentation_method=args.segmentation_method,
                           stardist_prob_thresh=args.stardist_prob_thresh,
                           stardist_nms_thresh=args.stardist_nms_thresh,
                           dapi_channel=args.dapi_channel,
                           min_nucleus_area_um2=args.min_nucleus_area_um2,
                           granule_blob_channel=args.granule_blob_channel,
                           granule_blob_min_diameter_um=args.granule_blob_min_diameter_um,
                           granule_blob_max_eccentricity=args.granule_blob_max_eccentricity,
                           granule_blob_min_solidity=args.granule_blob_min_solidity,
                           granule_blob_min_intensity_ratio=args.granule_blob_min_intensity_ratio,
                           granule_blob_mask_expansion_px=args.granule_blob_mask_expansion_px,
                           extra_rolling_ball_channel=args.extra_rolling_ball_channel,
                           extra_rolling_ball_radius=args.extra_rolling_ball_radius,
                           puncta_channel=args.puncta_channel,
                           puncta_min_diameter_um=args.puncta_min_diameter_um,
                           puncta_max_diameter_um=args.puncta_max_diameter_um,
                           puncta_log_threshold_rel=args.puncta_log_threshold_rel)
    elif args.input_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tif_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not tif_files:
            print(f"No files matching {args.pattern} found in {args.input_dir}")
            return
        for tif_path in tif_files:
            try:
                process_one_image(tif_path, args.output_dir,
                                   args.cell_expansion_px, use_gpu, args.diameter,
                                   segmentation_method=args.segmentation_method,
                                   stardist_prob_thresh=args.stardist_prob_thresh,
                                   stardist_nms_thresh=args.stardist_nms_thresh,
                                   dapi_channel=args.dapi_channel,
                                   min_nucleus_area_um2=args.min_nucleus_area_um2,
                                   granule_blob_channel=args.granule_blob_channel,
                                   granule_blob_min_diameter_um=args.granule_blob_min_diameter_um,
                                   granule_blob_max_eccentricity=args.granule_blob_max_eccentricity,
                                   granule_blob_min_solidity=args.granule_blob_min_solidity,
                                   granule_blob_min_intensity_ratio=args.granule_blob_min_intensity_ratio,
                                   granule_blob_mask_expansion_px=args.granule_blob_mask_expansion_px,
                                   extra_rolling_ball_channel=args.extra_rolling_ball_channel,
                                   extra_rolling_ball_radius=args.extra_rolling_ball_radius,
                                   puncta_channel=args.puncta_channel,
                                   puncta_min_diameter_um=args.puncta_min_diameter_um,
                                   puncta_max_diameter_um=args.puncta_max_diameter_um,
                                   puncta_log_threshold_rel=args.puncta_log_threshold_rel)
            except Exception as e:
                print(f"  FAILED on {tif_path}: {e}")
    else:
        parser.error("Provide either --input (single image) or --input_dir (batch mode)")


if __name__ == "__main__":
    main()
