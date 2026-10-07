"""
End-to-end test/orchestration script: runs the full pipeline (preprocess ->
cell_analysis -> call_and_plot_positivity) on every raw image in a folder, and
saves ALL outputs for each image into a single subfolder named after that
image -- so everything for one image lives in one place instead of being
scattered across three separate output directories.

Expects preprocess.py, cell_analysis.py, and call_and_plot_positivity.py to be
in the same folder as this script.

Output layout:
    input_dir/
        image1_raw.tif
        image2_raw.tif
        image1_raw/                              <- created by this script
            MAX_BGSUB_image1_raw.tif              (preprocess.py output)
            MAX_BGSUB_image1_raw_cells.csv         (cell_analysis.py output)
            MAX_BGSUB_image1_raw_segmentation_overlay.png  (nucleus + cell mask QC overlay)
            MAX_BGSUB_image1_raw_cells_calls.csv   (call_and_plot_positivity.py output)
            MAX_BGSUB_image1_raw_cells_separation_summary.csv
            MAX_BGSUB_image1_raw_cells_ch1_overlay.png
            MAX_BGSUB_image1_raw_cells_ch1_histogram.png
            ...
        image2_raw/
            ...

Pixel size (microns/pixel) is never a user-supplied argument -- every stage
reads it back automatically from the image it's given (see
preprocess.get_pixel_size_um()). Any number of channels is supported, and
--channels defaults to every channel except --dapi_channel and
--autofluor_channel (the usual case: every marker channel gets called).

Segmentation is skipped and its existing output reused automatically for any
image whose *_cells.csv and *_cell_labels.tif are already in its output
folder -- only images without those get (re-)segmented. Pass --resegment to
force a fresh segmentation run for every image regardless (needed after
changing a segmentation-affecting option like --cell_expansion_px or
--segmentation_method).

Runs in three stages, not one pass per image: (1) preprocess + segment every
image, (2) fit ONE positivity-calling model per channel on cells POOLED
across every position in the batch, (3) call positivity + save QC overlays
per image using that shared model. Positivity calling always reruns (no
caching there), so re-running this after only segmentation was cached picks
up right where it left off and redoes calling for everyone with the current
pooled model. Stage 2 exists because a 2-component GMM fit independently on
just one position's own cells assumes that position has enough of its own
cells to reveal a real positive/negative split -- for a sparse marker, many
individual fields of view won't, and an independent per-position fit in that
case just carves an arbitrary split out of what's actually a unimodal
population (near-zero separation, most cells "ambiguous"). Pooling across
the whole batch borrows statistical power from positions where the marker
happens to be more common. See call_and_plot_positivity.fit_pooled_gmms().

Usage:
    python run_full_pipeline.py --input_dir raw/

    # with autofluorescence correction (only if that image actually has an
    # autofluorescence channel -- see call_and_plot_positivity.py docstring)
    python run_full_pipeline.py --input_dir raw/ --autofluor_channel 3

    # force every image to be re-segmented (e.g. after changing --cell_expansion_px)
    python run_full_pipeline.py --input_dir raw/ --resegment --cell_expansion_px 15

    # process 4 images at a time in parallel (each image's segmentation is
    # independent of the others, so this can speed up a large batch on a
    # multi-core machine)
    python run_full_pipeline.py --input_dir raw/ --workers 4
"""

import argparse
import concurrent.futures
import glob
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import preprocess
import cell_analysis
import call_and_plot_positivity as positivity


def _limit_threads(n_threads):
    """
    ProcessPoolExecutor initializer: caps how many CPU threads this worker
    process's numeric libraries (TensorFlow/StarDist, PyTorch/Cellpose, numpy's
    BLAS backend) are allowed to use. Without this, every worker tries to grab
    every core for its own single-image inference, so running --workers>1 of
    them at once causes CPU oversubscription instead of a speedup. Must run
    before those libraries are imported, which is why this is the pool
    initializer rather than something called from inside segment_one_image --
    cell_analysis.py only imports tensorflow/torch lazily (inside
    segment_nuclei_stardist()/segment_nuclei_cellpose()), so setting these
    env vars as soon as the worker process starts is early enough.
    """
    n_threads = str(max(1, n_threads))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = n_threads


def _segment_one_image_worker(raw_tif_path, args):
    """Runs in a worker process; exceptions can't cross the process boundary
    as-is, so they're caught and returned as a plain string instead."""
    try:
        result = segment_one_image(raw_tif_path, args)
        return raw_tif_path, True, result
    except Exception:
        return raw_tif_path, False, traceback.format_exc()


def segment_one_image(raw_tif_path, args):
    """
    Preprocessing + segmentation only. Positivity calling is a separate,
    later stage (see call_positivity_one_image()) run only after EVERY
    image in the batch has been segmented, so a positivity-calling model can
    be fit on cells pooled across the whole batch first -- see
    call_and_plot_positivity.fit_pooled_gmms() for why that matters.

    Returns (cells_csv_path, label_mask_path, preprocessed_path).
    """
    name = os.path.splitext(os.path.basename(raw_tif_path))[0]
    out_folder = os.path.join(os.path.dirname(raw_tif_path), name)
    os.makedirs(out_folder, exist_ok=True)
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    print("[1/2] Preprocessing (max projection + background subtraction)...")
    preprocessed_path = preprocess.process_one_image(
        raw_tif_path, out_folder,
        rolling_ball_radius=args.rolling_ball_radius,
    )

    processed_name = os.path.splitext(os.path.basename(preprocessed_path))[0]
    cells_csv_path = os.path.join(out_folder, f"{processed_name}_cells.csv")
    label_mask_path = os.path.join(out_folder, f"{processed_name}_cell_labels.tif")

    if not args.resegment and os.path.exists(cells_csv_path) and os.path.exists(label_mask_path):
        print(f"\n[2/2] Cell segmentation + measurement... SKIPPED -- reusing existing "
              f"{os.path.basename(cells_csv_path)} and {os.path.basename(label_mask_path)} "
              f"(pass --resegment to force a fresh run)")
    else:
        print("\n[2/2] Cell segmentation + measurement...")
        cells_csv_path, label_mask_path = cell_analysis.process_one_image(
            preprocessed_path, out_folder,
            cell_expansion_px=args.cell_expansion_px,
            use_gpu=(False if args.cpu else None),
            diameter=args.diameter,
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
            puncta_log_threshold_rel=args.puncta_log_threshold_rel,
        )

    return cells_csv_path, label_mask_path, preprocessed_path


def _call_positivity_one_worker(cells_csv_path, label_mask_path, preprocessed_path, args,
                                 pooled_gmms, channels, compartment_by_channel, stat_by_channel):
    try:
        out_folder = call_positivity_one_image(
            cells_csv_path, label_mask_path, preprocessed_path, args,
            pooled_gmms, channels, compartment_by_channel, stat_by_channel,
        )
        return cells_csv_path, True, out_folder
    except Exception:
        return cells_csv_path, False, traceback.format_exc()


def call_positivity_one_image(cells_csv_path, label_mask_path, preprocessed_path, args,
                               pooled_gmms, channels, compartment_by_channel, stat_by_channel):
    out_folder = os.path.dirname(cells_csv_path)
    name = os.path.splitext(os.path.basename(preprocessed_path))[0]
    print(f"\n--- Positivity calling + QC overlays: {name} ---")
    positivity.run(
        csv_path=cells_csv_path,
        image_path=preprocessed_path,
        label_mask_path=label_mask_path,
        channels=channels,
        dapi_channel=args.dapi_channel,
        compartment=[compartment_by_channel[ch] for ch in channels],
        stat=[stat_by_channel[ch] for ch in channels],
        output_dir=out_folder,
        pos_thresh=args.pos_thresh,
        neg_thresh=args.neg_thresh,
        autofluor_channel=args.autofluor_channel,
        granule_flag_percentile=args.granule_flag_percentile,
        granule_drop_threshold=(None if args.no_granule_drop else args.granule_drop_threshold),
        pooled_gmms=pooled_gmms,
    )
    print(f"-> All outputs for {name} saved in: {out_folder}")
    return out_folder


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Folder of raw MMStack TIFF files")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for raw input files")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                         help="Channel numbers (1-based) to call positivity for, e.g. --channels 1 2 "
                              "(default: every channel except --dapi_channel and --autofluor_channel)")

    # preprocess.py options
    parser.add_argument("--rolling_ball_radius", type=int, default=50)

    # cell_analysis.py options
    parser.add_argument("--cell_expansion_px", type=float, default=cell_analysis.CELL_EXPANSION_PX,
                         help="Cytoplasm ring width in pixels (default %(default)s)")
    parser.add_argument("--diameter", type=float, default=None,
                         help="Expected nucleus diameter in pixels (Cellpose only; recommended once "
                              "known; auto-estimation is slower/less consistent)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU for segmentation")
    parser.add_argument("--segmentation_method", default="cellpose", choices=["cellpose", "stardist"],
                         help="Nucleus segmentation backend (default: cellpose)")
    parser.add_argument("--stardist_prob_thresh", type=float, default=None,
                         help="StarDist detection probability threshold (None = model default)")
    parser.add_argument("--stardist_nms_thresh", type=float, default=None,
                         help="StarDist non-max-suppression threshold (None = model default)")
    parser.add_argument("--dapi_channel", type=int, default=cell_analysis.DAPI_CHANNEL,
                         help="1-based channel number of the DAPI/nucleus channel in the preprocessed "
                              "stack (same convention as --channels; default: last channel of the "
                              "image, since DAPI is almost always acquired last)")
    parser.add_argument("--min_nucleus_area_um2", type=float, default=cell_analysis.MIN_NUCLEUS_AREA_UM2,
                         help="Detected objects smaller than this (in um^2) are dropped as granules/"
                              "debris rather than real nuclei (default %(default)s; set to 0 to disable)")
    parser.add_argument("--granule_blob_channel", type=int, default=None,
                         help="1-based channel number to detect solid debris blobs in, distinguishing "
                              "them from real punctate/grainy signal by shape alone (see "
                              "cell_analysis.segment_granule_blobs()) -- default: none, disabled. "
                              "Flagged pixels are excluded from every channel's intensity stats, saved "
                              "as *_granule_overlay.png")
    parser.add_argument("--granule_blob_min_diameter_um", type=float,
                         default=cell_analysis.GRANULE_BLOB_MIN_DIAMETER_UM,
                         help="Minimum equivalent-circular-diameter (in um) for an object in "
                              "--granule_blob_channel to count as a debris blob rather than real punctate "
                              "signal (default %(default)s)")
    parser.add_argument("--granule_blob_max_eccentricity", type=float,
                         default=cell_analysis.GRANULE_BLOB_MAX_ECCENTRICITY,
                         help="Maximum eccentricity (0=circle, 1=a line) for a debris blob (default %(default)s)")
    parser.add_argument("--granule_blob_min_solidity", type=float,
                         default=cell_analysis.GRANULE_BLOB_MIN_SOLIDITY,
                         help="Minimum solidity (area/convex_area) for a debris blob -- real punctate "
                              "signal breaks into many small, low-solidity components at this threshold, "
                              "while genuine debris stays one solid, filled-in region (default %(default)s)")
    parser.add_argument("--granule_blob_min_intensity_ratio", type=float,
                         default=cell_analysis.GRANULE_BLOB_MIN_INTENSITY_RATIO,
                         help="An object's own mean intensity must be at least this many times the "
                              "channel's Otsu threshold to count as a debris blob -- 1.0 (default) is the "
                              "validated setting (no extra requirement beyond clearing Otsu); shape does the "
                              "real discriminating here and raising this rarely helps, see "
                              "cell_analysis.segment_granule_blobs() (default %(default)s)")
    parser.add_argument("--granule_blob_mask_expansion_px", type=float,
                         default=cell_analysis.GRANULE_BLOB_MASK_EXPANSION_PX,
                         help="After shape/intensity filtering, dilate the kept debris blob mask outward "
                              "by this many pixels -- debris typically has a dim halo of partially-"
                              "thresholded pixels right at its edge that clearing Otsu alone doesn't catch. "
                              "0 disables expansion (default %(default)s)")
    parser.add_argument("--extra_rolling_ball_channel", type=int, default=None,
                         help="1-based channel number to run a second, small-radius rolling-ball pass on, "
                              "for finer local background flattening than preprocess.py's main "
                              "--rolling_ball_radius pass -- applied after granule blob detection above "
                              "(not in preprocess.py), since a small-radius ball erodes debris blobs too and "
                              "would suppress the signal segment_granule_blobs() needs to see. None (default) "
                              "disables this.")
    parser.add_argument("--extra_rolling_ball_radius", type=int, default=3,
                         help="Rolling-ball radius (pixels) for the second pass, applied only to "
                              "--extra_rolling_ball_channel (default %(default)s)")
    parser.add_argument("--puncta_channel", type=int, default=None,
                         help="1-based channel number to count small, bright, punctate real signal in "
                              "(see cell_analysis.detect_puncta_blobs()) -- default: none, disabled. For "
                              "a genuinely granular/punctate marker (e.g. DLL3), a per-cell dot count can "
                              "separate positive/negative cells better than Mean/Median intensity. Adds a "
                              "Dot_Count column per compartment, plus a Dot_Density column (dots / cell "
                              "area, preferred over raw Dot_Count -- see cell_analysis.py) for the Cell "
                              "compartment only, alongside the existing intensity stats.")
    parser.add_argument("--puncta_min_diameter_um", type=float,
                         default=cell_analysis.PUNCTA_MIN_DIAMETER_UM,
                         help="Minimum expected diameter (in um) of a real punctum in --puncta_channel "
                              "-- sets the lower end of the LoG blob-detection scale search range "
                              "(default %(default)s)")
    parser.add_argument("--puncta_max_diameter_um", type=float,
                         default=cell_analysis.PUNCTA_MAX_DIAMETER_UM,
                         help="Maximum expected diameter (in um) of a real punctum in --puncta_channel "
                              "rather than a debris blob -- sets the upper end of the LoG blob-detection "
                              "scale search range (default %(default)s)")
    parser.add_argument("--puncta_log_threshold_rel", type=float,
                         default=cell_analysis.PUNCTA_LOG_THRESHOLD_REL,
                         help="LoG blob-detection response threshold, relative to the image's strongest "
                              "response -- the main sensitivity knob for --puncta_channel; lower catches "
                              "more (dimmer) puncta (default %(default)s)")
    parser.add_argument("--resegment", action="store_true",
                         help="Force segmentation (the slow step) to re-run even if an image's "
                              "*_cells.csv and *_cell_labels.tif already exist in the output folder. "
                              "By default, existing segmentation results are reused automatically and "
                              "only images without them get segmented -- pass this after changing "
                              "--cell_expansion_px, --segmentation_method, --min_nucleus_area_um2, "
                              "--granule_blob_channel, --dapi_channel, --diameter, "
                              "--extra_rolling_ball_channel/--extra_rolling_ball_radius, or "
                              "--puncta_channel and its related flags, since those change "
                              "segmentation (or the preprocessed pixels segmentation runs on) but won't "
                              "otherwise trigger a re-run.")

    # call_and_plot_positivity.py options
    parser.add_argument("--compartment", nargs="+", default=["Cytoplasm"],
                         choices=["Nucleus", "Cytoplasm", "Cell"],
                         help="One value applied to every channel, or one value per --channels entry, "
                              "matched positionally (default: Cytoplasm)")
    parser.add_argument("--stat", nargs="+", default=["Mean"],
                         choices=["Mean", "Median", "Dot_Count", "Dot_Density"],
                         help="One value applied to every channel, or one value per --channels entry, "
                              "matched positionally -- use Dot_Count/Dot_Density for a channel counted "
                              "via --puncta_channel (Dot_Density preferred -- see cell_analysis.py; both "
                              "only computed for the Cell compartment, so pass --compartment Cell for "
                              "that channel) (default: Mean)")
    parser.add_argument("--pos_thresh", type=float, default=0.8)
    parser.add_argument("--neg_thresh", type=float, default=0.2)
    parser.add_argument("--autofluor_channel", type=int, default=None,
                         help="Only set this if the image genuinely has a dedicated "
                              "autofluorescence channel -- don't point it at DAPI")
    parser.add_argument("--granule_flag_percentile", type=float, default=0.9)
    parser.add_argument("--granule_drop_threshold", type=float, default=0.5,
                         help="Cells with granule_burden above this are dropped entirely (excluded from "
                              "calling, overlays, and CSV output), not just flagged. Only a literal area "
                              "fraction (0-1) with the --autofluor_channel-free debris-mask fallback; "
                              "with --autofluor_channel it's a Std.Dev./Median ratio instead (default 0.5)")
    parser.add_argument("--no_granule_drop", action="store_true",
                         help="Disable automatic dropping -- still compute and flag granule_burden/"
                              "high_granule_burden, just don't remove any cells")

    # parallelism options
    parser.add_argument("--workers", type=int, default=1,
                         help="Number of images to process in parallel (default: 1, sequential). Each "
                              "image is independent, so this can speed up a large batch on a multi-core "
                              "machine -- but Cellpose/StarDist otherwise try to use every core for a "
                              "single image, so with --workers>1 each worker's thread count is capped "
                              "automatically (see --threads_per_worker) to avoid CPU oversubscription.")
    parser.add_argument("--threads_per_worker", type=int, default=None,
                         help="CPU threads each worker process may use (default: cpu_count // --workers, "
                              "floor 1). Only relevant when --workers > 1; tune this down if workers are "
                              "still competing for CPU, or up if --workers is small relative to core count.")

    args = parser.parse_args()

    raw_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not raw_files:
        print(f"No files matching {args.pattern} found in {args.input_dir}")
        return

    print(f"Found {len(raw_files)} image(s) to process.")
    n_threads = args.threads_per_worker or max(1, (os.cpu_count() or args.workers) // args.workers)

    # --- Stage 1/3: preprocess + segment every image (parallelizable, cached per image) ---
    seg_results = {}  # raw_tif_path -> (cells_csv_path, label_mask_path, preprocessed_path)
    seg_succeeded, seg_failed = [], []

    if args.workers <= 1:
        for raw_tif_path in raw_files:
            try:
                seg_results[raw_tif_path] = segment_one_image(raw_tif_path, args)
                seg_succeeded.append(raw_tif_path)
            except Exception as e:
                print(f"\n  FAILED (segmentation) on {raw_tif_path}: {e}")
                traceback.print_exc()
                seg_failed.append(raw_tif_path)
    else:
        print(f"Segmenting with {args.workers} parallel workers ({n_threads} threads each)...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=_limit_threads, initargs=(n_threads,)
        ) as pool:
            futures = [pool.submit(_segment_one_image_worker, p, args) for p in raw_files]
            for future in concurrent.futures.as_completed(futures):
                raw_tif_path, ok, result = future.result()
                if ok:
                    seg_results[raw_tif_path] = result
                    seg_succeeded.append(raw_tif_path)
                else:
                    print(f"\n  FAILED (segmentation) on {raw_tif_path}: {result}")
                    seg_failed.append(raw_tif_path)

    if not seg_succeeded:
        print("\nNo images segmented successfully -- nothing to call positivity on.")
        return

    # --- Stage 2/3: fit one positivity-calling model per channel, pooled across every
    # successfully segmented position in this batch -- see fit_pooled_gmms() for why a
    # model fit independently per position can fail badly for a sparse marker. ---
    print(f"\n{'=' * 70}\nFitting pooled positivity-calling model across "
          f"{len(seg_succeeded)} position(s)...\n{'=' * 70}")
    sample_preprocessed_path = seg_results[seg_succeeded[0]][2]
    dapi_channel, channels, compartment_by_channel = positivity.resolve_channels_and_compartment(
        sample_preprocessed_path, args.channels, args.compartment, args.dapi_channel, args.autofluor_channel
    )
    stat_by_channel = positivity.resolve_stat_by_channel(args.stat, channels)
    if args.autofluor_channel is not None:
        print("--autofluor_channel given -- pooled positivity calling isn't supported together with "
              "spectral correction yet, so falling back to per-position GMM fitting for this run.")
        pooled_gmms = None
    else:
        cells_csv_paths = [seg_results[p][0] for p in seg_succeeded]
        pooled_gmms = positivity.fit_pooled_gmms(cells_csv_paths, channels, compartment_by_channel, stat_by_channel)

    # --- Stage 3/3: positivity calling + QC overlays per position (parallelizable) ---
    call_succeeded, call_failed = [], []
    call_inputs = [seg_results[p] for p in seg_succeeded]

    if args.workers <= 1:
        for cells_csv_path, label_mask_path, preprocessed_path in call_inputs:
            try:
                call_positivity_one_image(cells_csv_path, label_mask_path, preprocessed_path, args,
                                           pooled_gmms, channels, compartment_by_channel, stat_by_channel)
                call_succeeded.append(cells_csv_path)
            except Exception as e:
                print(f"\n  FAILED (positivity calling) on {cells_csv_path}: {e}")
                traceback.print_exc()
                call_failed.append(cells_csv_path)
    else:
        print(f"\nCalling positivity with {args.workers} parallel workers...")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, initializer=_limit_threads, initargs=(n_threads,)
        ) as pool:
            futures = [
                pool.submit(_call_positivity_one_worker, c, l, p, args, pooled_gmms, channels,
                            compartment_by_channel, stat_by_channel)
                for c, l, p in call_inputs
            ]
            for future in concurrent.futures.as_completed(futures):
                cells_csv_path, ok, result = future.result()
                if ok:
                    call_succeeded.append(cells_csv_path)
                else:
                    print(f"\n  FAILED (positivity calling) on {cells_csv_path}: {result}")
                    call_failed.append(cells_csv_path)

    print(f"\n{'=' * 70}\nDone. {len(call_succeeded)}/{len(raw_files)} images fully processed "
          f"({len(seg_succeeded)}/{len(raw_files)} segmented successfully).")
    if seg_failed:
        print("Failed during segmentation:")
        for f in seg_failed:
            print(f"  - {f}")
    if call_failed:
        print("Failed during positivity calling:")
        for f in call_failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
