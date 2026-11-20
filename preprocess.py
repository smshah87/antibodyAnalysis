"""
Batch preprocessing pipeline: raw multi-z, multi-channel Micro-Manager TIFF stacks
-> maximum-intensity Z-projection -> rolling-ball background subtraction -> saved
multi-channel 2D TIFF, ready for the downstream QuPath detection script and the
Python positivity-calling analysis.

Expected input: Micro-Manager MMStack OME-TIFF files (one per imaging position),
shape (channel, z, y, x). Pixel size (microns/pixel) is never a user-supplied
argument anywhere in this pipeline -- get_pixel_size_um() below auto-detects it
from whichever metadata source the file actually has, and every downstream
script (cell_analysis.py, call_and_plot_positivity.py) reads it back the same
way from the images it's given, rather than requiring it be passed around.

Usage:
    python preprocess.py --input_dir /path/to/raw_images --output_dir /path/to/processed
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import tifffile
from scipy import ndimage


def _pixel_size_from_shaped_metadata(tif_path):
    """
    Pixel size as embedded by this pipeline's own process_one_image() below
    (tifffile "shaped" metadata unit='um' + the XResolution tag) -- the source
    for any file that already went through preprocess.py.
    """
    try:
        with tifffile.TiffFile(tif_path) as tf:
            shaped = tf.shaped_metadata
            page = tf.pages[0]
            res_tag = page.tags.get("XResolution")
            if shaped and res_tag and shaped[0].get("unit") == "um":
                num, denom = res_tag.value
                if num:
                    return denom / num
    except Exception:
        pass
    return None


def _pixel_size_from_ome_metadata(tif_path):
    """
    Pixel size from the OME-XML embedded directly in Micro-Manager MMStack OME-TIFFs.

    When a single acquisition spans many positions/files, Micro-Manager writes
    the full OME-XML (with every Image/Pixels/PhysicalSizeX declaration) into
    only ONE file, and every other file just gets a tiny
    `<BinaryOnly MetadataFile="...">` stub pointing at it -- avoids duplicating
    a multi-megabyte XML blob into every single file. A stub has no
    PhysicalSizeX of its own, so this follows the reference and reads the
    real metadata from there instead.

    Checks the same directory first, then every sibling directory under this
    file's parent -- covers files reorganized into per-tumor/per-experiment
    subfolders (see split_files_by_tumor.py), where the referenced master
    file can end up in a *different* subfolder than this one (the master is
    just wherever the acquisition happened to write its one full-XML file
    first -- it isn't tied to any particular tumor/experiment).
    """
    try:
        with tifffile.TiffFile(tif_path) as tf:
            xml = tf.ome_metadata
            if not xml:
                return None
            binary_only = re.search(r'<BinaryOnly\s+MetadataFile="([^"]+)"', xml)
            if binary_only:
                master_name = binary_only.group(1)
                this_dir = os.path.dirname(tif_path)
                candidates = [os.path.join(this_dir, master_name)]
                parent_dir = os.path.dirname(this_dir)
                if os.path.isdir(parent_dir):
                    candidates += [
                        os.path.join(parent_dir, sub, master_name)
                        for sub in os.listdir(parent_dir)
                        if os.path.isdir(os.path.join(parent_dir, sub))
                    ]
                master_path = next((p for p in candidates if os.path.isfile(p)), None)
                if master_path is None:
                    return None
                with tifffile.TiffFile(master_path) as master_tf:
                    xml = master_tf.ome_metadata
                    if not xml:
                        return None
            match = re.search(r'PhysicalSizeX="([\d.]+)"', xml)
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return None


def _pixel_size_from_sidecar(tif_path):
    """Legacy fallback: a matching Micro-Manager *_metadata.txt sidecar file, if present."""
    base = tif_path.replace("_ome.tif", "").replace(".ome.tif", "").replace(".tif", "")
    meta_path = base + "_metadata.txt"
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        for key, val in meta.items():
            if key.startswith("FrameKey") and "PixelSizeUm" in val:
                return val["PixelSizeUm"]
    except Exception:
        pass
    return None


def get_pixel_size_um(tif_path):
    """
    Auto-detects pixel size (microns/pixel) for any TIFF in this pipeline,
    trying (in order) this pipeline's own embedded metadata, the raw file's
    OME-XML, and a legacy Micro-Manager sidecar file. Raises ValueError if
    none of them have it -- every micron-based measurement downstream depends
    on this, so a silent wrong guess would be worse than failing loudly.
    """
    for source in (_pixel_size_from_shaped_metadata, _pixel_size_from_ome_metadata,
                   _pixel_size_from_sidecar):
        value = source(tif_path)
        if value:
            return float(value)
    raise ValueError(
        f"{tif_path}: could not determine pixel size (microns/pixel) from embedded "
        "pipeline metadata, OME-XML, or a sidecar *_metadata.txt file."
    )


def max_intensity_projection(stack):
    """
    stack: array shaped (channel, z, y, x) -- collapse the z axis with a max projection,
    per channel independently.
    Returns: (channel, y, x)
    """
    return stack.max(axis=1)


def _imagej_ball_kernel(radius):
    """
    Builds the rolling-ball structuring element and shrink (downsample)
    factor exactly as ImageJ/Fiji's RollingBall class does
    (ij.plugin.filter.BackgroundSubtracter), including its 0-padded (not
    infinite/unconstrained) corners outside the ball's inscribed circle but
    still inside its square bounding patch.

    This 0-padding is the detail that actually matters: it was confirmed by
    testing side by side against real Fiji "Subtract Background" output that
    skimage.restoration.rolling_ball's default kernel -- which instead
    excludes those corner offsets entirely, treating them as unconstrained --
    behaves very differently (badly hollows out real puncta) at small radii,
    while this 0-padded version matches Fiji's actual behavior.

    shrink_factor/arc_trim_per come directly from ImageJ's own radius
    brackets (not a user-adjustable knob -- this IS the algorithm).
    """
    if radius <= 10:
        shrink_factor, arc_trim_per = 1, 24
    elif radius <= 30:
        shrink_factor, arc_trim_per = 2, 24
    elif radius <= 100:
        shrink_factor, arc_trim_per = 4, 32
    else:
        shrink_factor, arc_trim_per = 8, 40

    small_ball_radius = max(radius / shrink_factor, 1)
    rsquare = small_ball_radius ** 2
    xtrim = int(arc_trim_per * small_ball_radius) // 100
    half_width = int(round(small_ball_radius - xtrim))
    width = 2 * half_width + 1
    y, x = np.mgrid[0:width, 0:width]
    xval, yval = x - half_width, y - half_width
    temp = rsquare - xval ** 2 - yval ** 2
    ball = np.where(temp > 0, np.sqrt(np.clip(temp, 0, None)), 0.0).astype(np.float32)
    return ball, shrink_factor


def _imagej_shrink_min(img, shrink_factor):
    """
    Block-minimum downsample, matching ImageJ's shrinkImage() -- the ball
    needs to roll under the DARKEST pixel in each block, since a mean or a
    strided sample could skip over the true local background floor and bias
    the estimated background upward.
    """
    h, w = img.shape
    sh, sw = -(-h // shrink_factor), -(-w // shrink_factor)  # ceil division
    pad_h, pad_w = sh * shrink_factor - h, sw * shrink_factor - w
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode="edge")
    blocks = padded.reshape(sh, shrink_factor, sw, shrink_factor)
    return blocks.min(axis=(1, 3))


def imagej_rolling_ball_background(channel_img, radius, presmooth=True):
    """
    Faithful port of ImageJ/Fiji's "Subtract Background" rolling-ball
    algorithm (ij.plugin.filter.BackgroundSubtracter.rollingBallFloatBackground),
    not skimage.restoration.rolling_ball -- see _imagej_ball_kernel()'s
    docstring for why the two aren't interchangeable at small radii.

    presmooth: run a 3x3 mean filter on the copy of the image used ONLY to
    estimate the background (matches ImageJ's "Disable smoothing" checkbox
    being left unchecked, i.e. smoothing on, by default) -- the background is
    still subtracted from the original, unsmoothed pixels, so real detail in
    the output isn't blurred.

    Returns: the estimated background, same shape as channel_img.
    """
    img = channel_img.astype(np.float32)
    source = ndimage.uniform_filter(img, size=3) if presmooth else img
    ball, shrink_factor = _imagej_ball_kernel(radius)
    if shrink_factor > 1:
        small = _imagej_shrink_min(source, shrink_factor)
        small_background = ndimage.grey_opening(small, structure=ball, mode="nearest")
        background = ndimage.zoom(small_background, shrink_factor, order=1)
        # zoom can be off by a pixel or two vs the original due to rounding; pad/crop to match
        background = background[: img.shape[0], : img.shape[1]]
        if background.shape != img.shape:
            pad_y = img.shape[0] - background.shape[0]
            pad_x = img.shape[1] - background.shape[1]
            background = np.pad(background, ((0, max(pad_y, 0)), (0, max(pad_x, 0))), mode="edge")
    else:
        background = ndimage.grey_opening(source, structure=ball, mode="nearest")
    return background


def subtract_background(mip, radius=50):
    """
    mip: array shaped (channel, y, x), one already-max-projected image per channel.
    Runs ImageJ-faithful rolling-ball background subtraction (see
    imagej_rolling_ball_background()) independently per channel.

    Note: an optional second, small-radius rolling-ball pass on a specific
    channel (for finer local background flattening than this main pass
    provides) lives in cell_analysis.py instead of here (see
    analyze_image()'s extra_rolling_ball_channel/_radius) -- a small-radius
    ball erodes debris blobs the same way it erodes real local peaks, and
    running it here, before granule blob detection, was suppressing the
    signal that detection needs to see.

    Returns: background-subtracted array, same shape/dtype as input.
    """
    corrected = np.empty_like(mip, dtype=np.float32)
    for c in range(mip.shape[0]):
        channel_img = mip[c].astype(np.float32)
        background = imagej_rolling_ball_background(channel_img, radius)
        corrected[c] = channel_img - background
    corrected = np.clip(corrected, 0, None)
    return corrected


def process_one_image(tif_path, output_dir, rolling_ball_radius=50, keep_raw=False):
    """
    By default, deletes tif_path once the preprocessed output has been
    written successfully (to save disk space -- nothing downstream of this
    function ever reads the raw file again, see the module docstring). Pass
    keep_raw=True to disable this and preserve the raw file.

    Deleting is safe to rely on across repeated runs: if tif_path is already
    missing but its MAX_BGSUB_* output already exists (from an earlier run
    that deleted it), that output is reused instead of failing -- the same
    cache-skip pattern cell_analysis.py's segmentation uses, just one stage
    earlier. If tif_path is missing AND there's no existing output to fall
    back on, raises FileNotFoundError (something actually went wrong).
    """
    name = os.path.splitext(os.path.basename(tif_path))[0]
    out_path = os.path.join(output_dir, f"MAX_BGSUB_{name}.tif")

    if not os.path.exists(tif_path):
        if os.path.exists(out_path):
            print(f"Processing {name} ... SKIPPED -- raw file not found (likely already deleted by "
                  f"a prior run; pass --keep_raw to stop this), reusing existing "
                  f"{os.path.basename(out_path)}")
            return out_path
        raise FileNotFoundError(
            f"{tif_path} not found, and no existing preprocessed output at {out_path} to fall back on"
        )

    print(f"Processing {name} ...")

    # is_ome/is_mmstack disabled: otherwise tifffile stitches every sibling
    # "PosN.ome.tif" file in the folder into one array via the OME/MMStack
    # cross-file metadata, prepending a spurious position axis.
    stack = tifffile.imread(tif_path, is_ome=False, is_mmstack=False)  # expected shape (C, Z, Y, X)
    if stack.ndim != 4:
        raise ValueError(
            f"{tif_path}: expected a 4D (channel, z, y, x) stack, got shape {stack.shape}. "
            "Check axis order for this file."
        )

    mip = max_intensity_projection(stack)                       # (C, Y, X)
    corrected = subtract_background(mip, radius=rolling_ball_radius)  # (C, Y, X), float32

    # Save as 16-bit, clipping to the valid range, to stay compatible with the
    # downstream QuPath script (which expects GRAY16 fluorescence images).
    corrected_16 = np.clip(corrected, 0, 65535).astype(np.uint16)

    pixel_size_um = get_pixel_size_um(tif_path)

    px_per_um = 1.0 / pixel_size_um
    tifffile.imwrite(
        out_path,
        corrected_16,
        photometric="minisblack",
        resolution=(px_per_um, px_per_um),
        metadata={"axes": "CYX", "unit": "um"},
        ome=False,  # otherwise tifffile auto-writes OME-XML for this shape, which
                    # doesn't carry the resolution/unit through -- see _pixel_size_from_shaped_metadata
    )
    print(f"  -> saved {out_path}  shape={corrected_16.shape}  dtype={corrected_16.dtype}")

    if not keep_raw:
        try:
            os.remove(tif_path)
            print(f"  -> deleted raw file {tif_path} (pass --keep_raw to disable)")
        except OSError as e:
            print(f"  -> WARNING: could not delete raw file {tif_path}: {e}")

    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Folder containing raw MMStack *.tif files")
    parser.add_argument("--output_dir", required=True, help="Folder to save processed images into")
    parser.add_argument("--rolling_ball_radius", type=int, default=50, help="Rolling-ball kernel radius in pixels")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for input files (default: *.tif)")
    parser.add_argument("--keep_raw", action="store_true",
                         help="Keep each raw file after its preprocessed output has been saved. By "
                              "default (this flag off), raw files are DELETED once their MAX_BGSUB_*.tif "
                              "output is saved, to save disk space -- nothing downstream of preprocessing "
                              "ever reads the raw file again. Safe to delete across repeated runs: if a "
                              "raw file is already deleted, its existing output is reused instead of "
                              "failing. Pass this flag if you still need the raw files (e.g. archiving "
                              "them elsewhere, or still tuning --rolling_ball_radius).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tif_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not tif_files:
        print(f"No files matching {args.pattern} found in {args.input_dir}")
        return

    processed_paths = []
    for tif_path in tif_files:
        try:
            out_path = process_one_image(
                tif_path, args.output_dir, args.rolling_ball_radius,
                keep_raw=args.keep_raw,
            )
            processed_paths.append(out_path)
        except Exception as e:
            print(f"  FAILED on {tif_path}: {e}")

    print(f"\nDone. {len(processed_paths)}/{len(tif_files)} images processed successfully.")
    print("Next step: run cell detection on these saved images (QuPath script), then the")
    print("positivity-calling analysis on the resulting per-cell CSVs.")


if __name__ == "__main__":
    main()
