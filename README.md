# Fluorescence image analysis pipeline

Six scripts. `run_full_pipeline.py` chains the first three together and is
the easiest way to run everything on a folder of images;
`compare_intensity_by_call.py` and `compare_intensity_between_experiments.py`
are separate statistical analyses you run afterward on the results. Use the
individual scripts only if you want to run/debug one stage at a time.

| Script | Does |
|---|---|
| `preprocess.py` | Raw multi-z MMStack TIFF → max-intensity projection → background subtraction |
| `cell_analysis.py` | Nucleus segmentation (Cellpose or StarDist) → cell/cytoplasm expansion → per-cell intensity CSV |
| `call_and_plot_positivity.py` | GMM-based positive/negative/ambiguous calling per channel + QC overlay images |
| `run_full_pipeline.py` | Runs all three of the above on every image in a folder |
| `compare_intensity_by_call.py` | Cross-position stats **within one dataset**: is intensity in channel A different between cells called positive vs. negative in channel B? |
| `compare_intensity_between_experiments.py` | Stats **between two groups of experiments** (e.g. transforming vs non-transforming tumors, each group made of several tumor samples): is intensity in channel A (among cells matching some marker signature) different between the two groups? |

Nothing here needs `--pixel_size_um`, `--channels`, or `--dapi_channel` to be
told explicitly in the common case — they're auto-detected or default
sensibly (see the variable reference below). Segmentation (the slow step) is
also skipped and reused automatically for any image already segmented in a
previous run. The minimal way to run everything on a folder of raw images is:

```bash
python run_full_pipeline.py --input_dir raw/ --segmentation_method stardist
```

## Setup

**Use Python 3.13, not the latest Python.** TensorFlow (needed for StarDist)
does not yet publish wheels for the newest Python releases, so if your
default interpreter is newer than 3.13 you'll hit a hard `pip install
tensorflow` failure with no available version. Check what's on your machine
with `py -0` (Windows) and install 3.13 from python.org if it's missing.

```bash
py -3.13 -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate

pip install numpy tifffile scipy scikit-image pandas matplotlib scikit-learn statsmodels

# Cellpose segmentation backend (GPU strongly recommended -- see note below):
pip install cellpose torch

# StarDist segmentation backend (CPU-friendly alternative):
pip install stardist csbdeep tensorflow
```

You don't need both backends -- pick whichever `--segmentation_method` you
plan to use and only install its dependencies. `compare_intensity_by_call.py`
needs `statsmodels` in addition to the core list above; nothing else needs it.

### GPU reality check

Cellpose is a large model (Cellpose-SAM, ~1.15GB, downloaded automatically on
first use) and is slow on CPU. Before assuming a GPU will help, check its
compute capability -- PyTorch's CUDA builds require **compute capability
3.5+**; anything older (e.g. a Quadro K600, or most GPUs from before ~2012)
will silently fall back to CPU no matter what driver you install, because the
hardware itself is unsupported, not the driver. Run `nvidia-smi` to check
your card and compare against
[NVIDIA's compute capability table](https://developer.nvidia.com/cuda-gpus)
before buying or troubleshooting a GPU for this. If you're CPU-bound,
**StarDist is the better default** -- it's a much smaller model and runs at
reasonable speed on CPU, whereas Cellpose-SAM does not.

### Windows-specific StarDist gotchas

Both of these were hit and fixed during development on Windows; if you're on
Mac/Linux you likely won't see either.

1. **`pip install stardist` fails needing "Microsoft C++ Build Tools"** --
   stardist ships a C extension with no prebuilt wheel for newer Python
   versions on Windows, so pip tries to compile it from source. Install the
   [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the "Desktop development with C++" workload, then retry.
2. **First StarDist run fails with `WinError 1314: A required privilege is
   not held by the client`** while downloading the pretrained model --
   non-admin Windows accounts can't create symlinks by default, and
   `csbdeep`'s model loader tries to. The model has already been downloaded
   and extracted by the time this fails; work around it by creating a
   directory junction in place of the symlink it wanted:
   ```powershell
   New-Item -ItemType Junction -Path "<...>\StarDist2D\2D_versatile_fluo\2D_versatile_fluo" -Target "<...>\StarDist2D\2D_versatile_fluo\2D_versatile_fluo_extracted"
   ```
   (path printed in the error message). Only needed once.

## Quickest path: run_full_pipeline.py

Runs preprocessing → cell segmentation → positivity calling on every raw
image in a folder, and saves everything for each image into one subfolder
named after that image (inside the same input directory).

```bash
python run_full_pipeline.py --input_dir raw/ --segmentation_method stardist
```

By default this uses Cellpose for nucleus segmentation; add
`--segmentation_method stardist` to use StarDist instead (see the GPU note
above for which to pick). Both produce the same downstream CSV format, so
everything else in the pipeline works identically either way -- worth
comparing both on one image before committing to a batch run.

**Segmentation is reused automatically.** Before segmenting an image,
`run_full_pipeline.py` checks its output folder for an existing
`*_cells.csv` and `*_cell_labels.tif` and, if both are there, skips straight
to positivity calling with them instead of re-running Cellpose/StarDist.
This makes it cheap to iterate on positivity-calling options
(`--channels`, `--compartment`, `--pos_thresh`, `--autofluor_channel`, ...)
across a whole folder without re-paying the slow segmentation cost each
time -- only images that haven't been segmented yet actually get segmented.
Pass `--resegment` to force every image to be segmented fresh regardless
(needed after changing anything that affects segmentation itself:
`--cell_expansion_px`, `--segmentation_method`, `--min_nucleus_area_um2`,
`--granule_blob_channel` and its related flags, `--dapi_channel`, `--diameter`).

**Batches can run in parallel.** Each image's segmentation is independent of
every other image, so pass `--workers N` to process N images at once instead
of one at a time -- worthwhile on a multi-core machine once you're running a
folder of more than a couple of images. Cellpose/StarDist otherwise try to
use every CPU core for a single image, so with `--workers > 1` each worker's
thread count is capped automatically (`cpu_count // --workers`, or set
`--threads_per_worker` explicitly) to avoid the workers fighting each other
for CPU. Separately, the segmentation model itself (StarDist or Cellpose) is
loaded once per process and reused for every image that process handles,
rather than being rebuilt from disk for each image -- this happens
automatically, no flag needed, and matters most in sequential (`--workers 1`)
runs over many images.

```bash
python run_full_pipeline.py --input_dir raw/ --segmentation_method stardist --workers 4
```

A more fully-specified example, overriding several defaults:

```bash
python run_full_pipeline.py --input_dir raw/ \
    --segmentation_method stardist --resegment \
    --channels 1 2 --compartment Nucleus Cytoplasm \
    --dapi_channel 3 --autofluor_channel 4 \
    --cell_expansion_px 10 --min_nucleus_area_um2 15
```

Output layout:
```
raw/
    image1.tif
    image1/
        MAX_BGSUB_image1.tif                       (preprocess.py output)
        MAX_BGSUB_image1_cells.csv                 (cell_analysis.py output)
        MAX_BGSUB_image1_segmentation_overlay.png  (nucleus + cell mask QC image)
        MAX_BGSUB_image1_cell_labels.tif           (per-cell label mask, pixel value = Cell_ID)
        MAX_BGSUB_image1_cells_calls.csv           (call_and_plot_positivity.py output)
        MAX_BGSUB_image1_cells_separation_summary.csv
        MAX_BGSUB_image1_cells_ch1_overlay.png
        MAX_BGSUB_image1_cells_ch1_histogram.png
        ...
```

---

## Running stages individually

Each of the three underlying scripts can also be run and batched on its own
(useful for debugging one stage without rerunning the whole thing). All
three live in the same folder and import each other, so keep them together.

### 1. preprocess.py

```bash
python preprocess.py --input_dir raw/ --output_dir processed/ --rolling_ball_radius 50
```

### 2. cell_analysis.py

```bash
python cell_analysis.py --input_dir processed/ --output_dir results/ --segmentation_method stardist
```

### 3. call_and_plot_positivity.py

```bash
python call_and_plot_positivity.py \
    --csv results/image_cells.csv \
    --image processed/MAX_BGSUB_image.tif \
    --label_mask results/image_cell_labels.tif \
    --output_dir qc/
```

`--label_mask` is optional but recommended when the CSV came from
`cell_analysis.py` -- without it, overlay images mark cells with open circles
at their centroid instead of filling in their actual segmented shape.

### 4. compare_intensity_by_call.py

Run this after the other stages, on the folder they wrote their results
into. Answers "do cells called positive in channel B have different channel
A intensity than cells called negative in channel B?", across every position
under `--input_dir` at once, with three statistical views:

```bash
python compare_intensity_by_call.py --input_dir raw/ \
    --value_channel 1 --value_compartment Nucleus --value_stat Mean \
    --group_channel 2
```

- A **Mann-Whitney U test** (rank-based, robust to the skewed intensity
  distributions typical here) and a **Welch's t-test on log-transformed
  values**, both on the naive pooled data.
- A **blocked ANCOVA** (position included as a fixed blocking factor) --
  use this one to trust the result. Pooling raw values across positions with
  different baseline intensities can produce a trend that doesn't actually
  hold within any individual position (a Simpson's-paradox-style artifact);
  the blocked model controls for that. It also drives a **forest plot**
  (`..._forest.png`) showing each position's own effect size next to the
  position-adjusted overall estimate, so you can see at a glance whether the
  result is a consistent signal or driven by one outlier position.
- A **violin plot** (`..._violin.png`) of the two raw distributions with the
  Mann-Whitney p-value annotated.

---

## Variable reference

### Auto-detected -- you will rarely need to set these

| Variable | What it means | How it's determined |
|---|---|---|
| Pixel size (µm/pixel) | Physical size of one pixel, needed for every µm-based measurement (cell area, centroid position, min nucleus area filter) | Never a CLI argument anywhere. `preprocess.get_pixel_size_um()` tries, in order: this pipeline's own embedded metadata (files it already processed), the raw file's OME-XML (present natively in Micro-Manager MMStack files), then a legacy `*_metadata.txt` sidecar. Raises an error if none are found -- a silent wrong guess would be worse. |
| `--dapi_channel` | 1-based channel number of the DAPI/nucleus channel, used for segmentation and excluded from the default `--channels` list | Defaults to the **last** channel of the image (DAPI is almost always acquired last). Pass explicitly if that's not true for your acquisition. |
| `--channels` | Which channels get positivity-called | Defaults to **every channel except `--dapi_channel` and `--autofluor_channel`** -- the usual case is every marker channel gets called. Pass explicitly to call only a subset. |
| Segmentation (Cellpose/StarDist) | Whether an image gets (re-)segmented | Skipped and reused automatically (`run_full_pipeline.py` only) if that image's `*_cells.csv` and `*_cell_labels.tif` already exist in its output folder. Pass `--resegment` to force a fresh run for every image. |

Any number of channels is supported throughout -- there's no hardcoded
channel-count limit anywhere in the pipeline.

### run_full_pipeline.py only

| Flag | Meaning | Default |
|---|---|---|
| `--resegment` | Force segmentation to re-run for every image even if cached `*_cells.csv` / `*_cell_labels.tif` already exist (see auto-detected table above). Needed after changing `--cell_expansion_px`, `--segmentation_method`, `--min_nucleus_area_um2`, `--granule_blob_channel` and its related flags, `--dapi_channel`, or `--diameter` -- those change segmentation but a stale cache won't otherwise know to invalidate itself | off (reuse existing) |
| `--pattern` | Glob pattern for raw input files | `*.tif` |
| `--workers` | Number of images to process in parallel (see "Batches can run in parallel" above) | 1 (sequential) |
| `--threads_per_worker` | CPU threads each worker process may use when `--workers > 1` | `cpu_count // --workers` |

The remaining `run_full_pipeline.py` flags are the union of the three
individual scripts' flags below, applied per-image as it chains them
together.

### preprocess.py

| Flag | Meaning | Default |
|---|---|---|
| `--input_dir` / `--output_dir` | Raw MMStack folder / where to save processed TIFFs | required |
| `--rolling_ball_radius` | Rolling-ball background-subtraction radius, in pixels | 50 |
| `--pattern` | Glob pattern for input files | `*.tif` |

**The rolling-ball engine is a faithful port of ImageJ/Fiji's actual "Subtract
Background" algorithm** (`ij.plugin.filter.BackgroundSubtracter`), not
`skimage.restoration.rolling_ball` -- confirmed by direct comparison against
real Fiji output that the two are not interchangeable at small radii.
skimage's default ball kernel excludes the corners of its square bounding
patch outside the inscribed circle entirely (treats them as unconstrained),
while ImageJ's kernel pads those corners with height 0 instead -- at a small
radius (e.g. 3px, comparable to real punctate signal), that difference is
enough for skimage's version to hollow bright puncta into rings, while the
0-padded version matches Fiji's actual (non-hollowing) behavior. See
`_imagej_ball_kernel()` / `imagej_rolling_ball_background()` in
`preprocess.py`. Also ported: ImageJ's default 3x3-mean presmoothing (used
only to estimate the background, not applied to the output) and its
radius-dependent shrink/downsample factor (1/2/4/8, chosen automatically from
the radius, not a user-adjustable flag -- this is part of the algorithm
itself in ImageJ, not a separate speed knob).

### cell_analysis.py

| Flag | Meaning | Default |
|---|---|---|
| `--input` / `--output` | Single-image mode | -- |
| `--input_dir` / `--output_dir` | Batch mode | -- |
| `--segmentation_method` | `cellpose` or `stardist` -- see the GPU note above for which to pick | `cellpose` |
| `--dapi_channel` | See auto-detected table above | last channel |
| `--diameter` | Expected nucleus diameter in pixels (**Cellpose only**). Set this once you know your typical nucleus size -- auto-estimation is slower and less consistent | auto-estimate |
| `--cpu` | Force CPU even if a GPU is detected (Cellpose only) | off |
| `--stardist_prob_thresh` / `--stardist_nms_thresh` | Override StarDist's detection probability / non-max-suppression (overlap) thresholds if the defaults split or merge nuclei incorrectly | model default |
| `--cell_expansion_px` | Cytoplasm ring width, **in pixels** -- how far each nucleus is expanded outward to approximate the cell boundary (skimage `expand_labels`, capped at the midline between neighboring cells). This has a real, non-trivial effect on positivity calls: a wider ring dilutes the per-cell mean with more surrounding signal. Check a `_segmentation_overlay.png` against the raw image before trusting a value -- too wide and the magenta cell boundary visibly overshoots real cell edges | 10.0 |
| `--min_nucleus_area_um2` | Detected objects smaller than this (µm²) are dropped as granules/debris rather than real nuclei, before cytoplasm expansion. Segmentation backends (especially StarDist's generalist pretrained model) can pick up small bright puncta as if they were nuclei; check the raw vs. filtered object count printed per image, and inspect a couple of `_segmentation_overlay.png` files to confirm real nuclei aren't also being cut. Set to 0 to disable. **Not thrown away**: the dropped debris area overlapping each cell is saved as a `Granule_Area` column in the CSV -- the *last-resort* granule-burden signal `call_and_plot_positivity.py` falls back to (see below) | 15.0 |
| `--granule_blob_channel` | 1-based channel number to detect solid debris blobs in, by shape alone -- see "Granule/debris blob detection" below. `None` disables this entirely (no pixels excluded, no `*_granule_overlay.png`) | none (disabled) |
| `--granule_blob_min_diameter_um` | Minimum equivalent-circular-diameter for an object in `--granule_blob_channel` to count as a debris blob rather than real punctate signal | 0.8 |
| `--granule_blob_max_eccentricity` | Maximum eccentricity (0=circle, 1=a line) for a debris blob. In practice this barely discriminates anything -- real debris blobs are often noticeably elongated, not circular -- so the default is 1.0 (effectively off); `--granule_blob_min_solidity` does the real filtering work | 1.0 |
| `--granule_blob_min_solidity` | Minimum solidity (`area / convex_area`) for a debris blob. This is the shape check that actually matters: real punctate/grainy signal breaks into many small, low-solidity connected components once thresholded, while a genuine solid debris blob stays one large, filled-in region. Too high (e.g. 0.9) misses real irregular debris; too low barely differs from a modest floor | 0.6 |
| `--granule_blob_min_intensity_ratio` | An object's own mean raw intensity must be at least this many times the channel's Otsu threshold to count as a debris blob -- a second, independent check on top of the shape cutoffs above. 1.0 (default) is the validated setting (no extra requirement beyond clearing Otsu); shape already does the real discriminating (see below), so raising this rarely helps and isn't the recommended way to tune this | 1.0 |
| `--granule_blob_mask_expansion_px` | After the shape/intensity filtering above, dilate the kept debris blob mask outward by this many pixels (disk footprint) -- debris typically has a dim halo of partially-thresholded pixels right at its edge that clearing Otsu alone doesn't catch. 0 disables expansion | 3.0 |
| `--extra_rolling_ball_channel` | 1-based channel number to run a second, small-radius rolling-ball pass on, for finer local background flattening than `preprocess.py`'s main `--rolling_ball_radius` pass provides. Deliberately applied **here**, after granule blob detection above, rather than in `preprocess.py` before it -- a small-radius ball erodes/attenuates bright local peaks, and debris blobs are exactly that kind of peak, so running it before detection was suppressing the very signal `segment_granule_blobs()` needs to see, causing missed blobs. Only affects the intensity values used for this channel's Mean/Median/etc stats -- granule detection itself always runs on the un-eroded image. `None` (default) disables this | none (disabled) |
| `--extra_rolling_ball_radius` | Rolling-ball radius (pixels) for the second pass, applied only to `--extra_rolling_ball_channel` | 3 |
| `--puncta_channel` | 1-based channel number to count small, bright, punctate real signal in -- see "Puncta (dot) counting" below. `None` disables this entirely (no `Dot_Count` columns, no `*_puncta_overlay.png`) | none (disabled) |
| `--puncta_min_diameter_um` | Minimum expected diameter of a real punctum in `--puncta_channel` -- sets the lower end of the LoG blob-detection scale search range | 0.2 |
| `--puncta_max_diameter_um` | Maximum expected diameter of a real punctum in `--puncta_channel` rather than a debris blob (`segment_granule_blobs()`'s territory instead) -- sets the upper end of the LoG scale search range | 0.8 (same as `--granule_blob_min_diameter_um`'s default, so the two detectors' size ranges are complementary, not overlapping) |
| `--puncta_log_threshold_rel` | LoG blob-detection response threshold, relative to the image's strongest response -- the main sensitivity knob; lower catches more (dimmer) puncta | 0.02 |
| `--pattern` | Glob pattern for batch mode | `*.tif` |

#### Border-cell removal (no flag -- always on)

Any cell whose whole-cell footprint (post cell-expansion) touches the image
edge is dropped entirely, and the survivors relabeled to consecutive
`Cell_ID`s 1..N. Those cells are physically clipped by the field of view, so
their area/intensity/dot-count measurements would be based on an incomplete
cell and aren't comparable to fully-imaged ones. Checked on the expanded
cell mask (`cell_labels`), not just the raw nucleus, since a nucleus fully
inside the frame can still have its cytoplasm ring clipped by the edge --
see `remove_border_cells()`. The per-image log line reports how many were
dropped: `Dropped N cell(s) touching the image border (incomplete) -> M cells`.
Segmentation is cached (`*_cells.csv` / `*_cell_labels.tif`), so this only
takes effect on a fresh run or after `--resegment`.

#### Puncta (dot) counting (`--puncta_channel`)

For a marker that's genuinely granular/punctate (e.g. DLL3) rather than
diffusely distributed across the cell, a per-cell **count of discrete
puncta** can separate positive from negative cells better than Mean/Median
intensity does: mean intensity gets diluted by cell area and by background
between puncta, so a cell with several bright dots on an otherwise dark
background can read the same mean as a cell with dim diffuse signal, despite
being biologically very different. This adds a `Dot_Count` column per
compartment (`Nucleus: Channel N: Dot_Count`, `Cytoplasm: ...`, `Cell: ...`)
**alongside** the existing intensity stats, not in place of them, so both can
be compared before deciding which one actually separates your positive and
negative populations better for a given marker.

**Prefer `Dot_Density` (Cell compartment only) over raw `Dot_Count` for
positivity calling.** Raw count is confounded by cell size: a bigger cell has
more area to catch background speckle in regardless of real signal, so count
partly just tracks cell size rather than biology -- confirmed empirically on
a real test field of view, `Dot_Count` correlated with `Cell: Area` at
r=0.68, while `Cell: Channel N: Dot_Density` (= `Dot_Count` / `Cell: Area`,
dots per µm²) only correlated at r=0.28. `Dot_Density` is computed for the
Cell compartment only, not Nucleus/Cytoplasm -- puncta aren't confined to
either sub-region, so whole-cell is the only compartment that makes sense to
normalize by. Pass `--compartment Cell` for any channel using `Dot_Count` or
`Dot_Density`.

Puncta are found via **Laplacian-of-Gaussian (LoG) blob detection**
(`skimage.feature.blob_log`), the standard technique for exactly this kind
of small-round-spot-counting problem (the same approach RNA-FISH spot-
counting pipelines use). `--puncta_min_diameter_um` / `--puncta_max_diameter_um`
set the LoG scale search range directly (`min_sigma`/`max_sigma` = diameter
/ (2·√2)) -- no separate post-hoc size filtering needed, since detection is
constrained to that scale range by construction.

**A white-tophat-plus-threshold approach was tried first and rejected.**
With any threshold sensitive enough to catch real dim puncta, it also picked
up narrow fragments of the fibrous tissue background: a thin fiber narrower
than the tophat structuring element partially survives tophat as a bright
streak, which breaks into many small blob-shaped fragments once thresholded
-- false positives shape-alone thresholding can't tell apart from real
puncta. LoG detects blobs by their actual shape at a given scale rather than
brightness alone, and checked side by side on the same image it both catches
far more real dim puncta and avoids the fiber-fragment false positives.

`--puncta_log_threshold_rel` (default 0.02) was picked empirically by
sweeping 0.05 → 0.02 → 0.01 → 0.005 on real data: 0.05 visibly missed real
dim puncta; 0.01 started flagging circles over empty background with no
visible signal at all (clear noise); 0.02 sits in between.

A higher value (0.03-0.05) was tried briefly because it gave better
single-position GMM separation on one test field of view, but that turned
out to be a red herring: `Dot_Count`'s raw distribution on that field of
view is a smooth, unimodal, monotonically decaying count distribution with
no real gap in it (confirmed via histogram) -- so any single-position
2-component GMM fit on it, regardless of which `--puncta_log_threshold_rel`
produced the counts, is carving an arbitrary split out of noise. This is the
same failure mode already found and fixed for Mean/Median intensity on
sparse markers (see `fit_pooled_gmms()` below) -- the fix is pooling across
positions, not chasing a detection threshold that happens to flatter a
broken single-position fit.

If `--granule_blob_channel` is the same channel as `--puncta_channel`, any
detected punctum centered on a flagged debris pixel is dropped. Detection
deliberately runs on the raw, un-masked image first rather than zeroing
debris pixels before detection: zeroing creates a sharp artificial edge at
the debris boundary, which LoG then misdetects as blobs of its own,
clustering false positives right around every debris blob's rim -- confirmed
empirically. Filtering by center location after detection gets the same
exclusion without that artifact.

Each detected punctum is assigned to whichever cell's label region contains
its center (not by "any overlap", which could double-count a punctum
straddling a boundary), using the same granule-exclusion-adjusted label masks
as the intensity stats -- so a punctum landing in a region already flagged as
debris (in any channel) doesn't get counted. Counting always runs on the
un-eroded image (the same one granule blob detection uses), before any
`--extra_rolling_ball_channel` correction, for the same reason that
correction was moved out of `preprocess.py`: eroding real local peaks before
detection would suppress the very puncta this is trying to count.

A `*_puncta_overlay.png` QC image marks each detected punctum as an open red
circle (at its detected center and size) on top of `--puncta_channel`.

#### Granule/debris blob detection (`--granule_blob_channel`)

A single-channel alternative to picking debris out by cross-channel
coincidence: Otsu-threshold the chosen channel directly (no pre-filtering --
see the tophat note below), connected-component label the result, and keep
only objects that clear the size/eccentricity/solidity cutoffs above. The
underlying assumption is that **real antibody signal in that channel looks
punctate/grainy** (many small, separate, irregular bright specks) while
**autofluorescent debris looks like one solid, filled-in blob** -- shape
alone, at the same brightness threshold, tells them apart.

**A 3-class multi-Otsu variant was tried and rejected.** The idea: these
images seemingly have three intensity populations (background, real signal,
much-brighter debris), so take only the brightest tier as debris candidates
directly, instead of relying on shape alone within a single Otsu
foreground/background split. In practice this under-flagged: visual QC
against the original single-Otsu-plus-shape output showed real debris blobs
that are only moderately brighter than real signal, not in the extreme top
tier, silently dropping out. This matched what pooling `intensity_mean /
threshold` ratios of shape-passing blobs across many images had already
shown -- a smooth, unimodal, decaying tail with no gap to cut at anywhere --
so there's no brightness cutoff, fixed or per-image-adaptive, that cleanly
separates debris from shape-passing false positives at the object level.
Shape is left to do the real discriminating, which is what the default
`--granule_blob_min_intensity_ratio 1.0` (no brightness restriction beyond
clearing Otsu) reflects.

Flagged pixels are excluded from **every channel's** intensity statistics
(Mean/Median/Std.Dev./Min/Max, all compartments) for any cell they overlap,
not just the channel they were detected in -- the assumption being that
physical debris (dust, lipofuscin, tissue folds) contaminates whatever
wavelength happens to pass through it, regardless of which channel made it
easiest to spot by shape. **Check this assumption before trusting it**: if
one of your channels is a real, biologically meaningful marker (e.g. a
nuclear transcription factor used for a downstream comparison), verify that
channel's separation/calling quality doesn't degrade after turning on
`--granule_blob_channel` pointed at a *different* channel -- if it does,
either that channel's compartment choice needs revisiting (nuclear TFs
should usually be called on `Nucleus`, not `Cytoplasm`, which also happens
to reduce overlap with cytoplasmic debris blobs) or the debris genuinely
isn't as channel-agnostic as assumed for that dataset.

The excluded area per cell is saved as a `Granule_Area_Blob` column
(second-priority granule-burden signal for `call_and_plot_positivity.py`,
see below), and a `*_granule_overlay.png` QC image marks the flagged pixels
in red on top of `--granule_blob_channel`.

Deliberately **skips white-tophat pre-filtering** (used elsewhere in this
codebase for isolating small bright spots): tophat suppresses anything
*larger* than its structuring element, which is exactly the large solid
blob this is trying to find -- the opposite of what a small-punctum filter
is for.

An earlier approach required a granule-shaped object to appear
*simultaneously* in two or more channels (real signal should be
channel-specific, so coincidence across channels was treated as a debris
signature). It was abandoned: it couldn't reliably tell real coincidental
overlap from chance, and depended on which channels happened to be
"non-signal" in a given experiment -- fragile across datasets, whereas
single-channel shape (solidity in particular) turned out to be a much more
direct and portable signal.

### call_and_plot_positivity.py

| Flag | Meaning | Default |
|---|---|---|
| `--csv` / `--image` | Per-cell CSV + its source image | required |
| `--label_mask` | Path to a `*_cell_labels.tif` (from `cell_analysis.py`) -- if given, overlay PNGs fill each cell's actual segmented shape with its call color instead of drawing an open circle at its centroid | none (circles) |
| `--channels` | See auto-detected table above | all except DAPI/autofluor |
| `--dapi_channel` | See auto-detected table above | last channel |
| `--compartment` | `Nucleus`, `Cytoplasm`, or `Cell` -- which region's intensity to threshold on. Accepts **one value applied to every channel, or one value per `--channels` entry** matched positionally, e.g. `--channels 1 2 --compartment Nucleus Cytoplasm` calls channel 1 on nuclear intensity and channel 2 on cytoplasmic intensity | `Cytoplasm` |
| `--stat` | `Mean`/`Median` intensity within the compartment, or `Dot_Count` (from `cell_analysis.py`'s `--puncta_channel`) for a genuinely punctate marker. Accepts **one value applied to every channel, or one value per `--channels` entry** matched positionally, the same way `--compartment` does, e.g. `--channels 1 2 --stat Mean Dot_Count` calls channel 1 on Mean intensity and channel 2 on Dot_Count. Use `Median` (not `Dot_Count`) if a channel just has punctate autofluorescent-granule contamination rather than being a genuinely punctate marker -- median is robust to the spikes; Mean is pulled up by them | `Mean` |
| `--pos_thresh` / `--neg_thresh` | GMM positive-probability cutoffs: a cell is called positive above `pos_thresh`, negative below `neg_thresh`, and ambiguous in between | 0.8 / 0.2 |
| `--autofluor_channel` | 1-based channel number of a **genuine dedicated autofluorescence channel**, if present -- don't point this at DAPI. When set: (1) every other called channel is linearly corrected against it (fitted slope/intercept, correlation `r` printed so you can judge whether the correction did anything meaningful), and (2) cells with unusually spiky/punctate autofluorescence (high Std.Dev./Median ratio in that channel) are flagged. Without it, the granule-burden flag still runs using a different signal -- see below | none |
| `--granule_flag_percentile` | Percentile of granule burden above which a cell gets **flagged** in the `granule_burden` / `high_granule_burden` columns (manual review, not excluded) | 0.9 (top 10%) |
| `--granule_drop_threshold` | Cells with `granule_burden` **above** this are **dropped entirely** -- excluded from positivity calling, overlays, and the output CSV, not just flagged. Number dropped is printed. This is a separate, harsher cut than `--granule_flag_percentile` above (an absolute cutoff vs. a relative top-N%) | 0.5 |
| `--no_granule_drop` | Disable dropping -- still compute and flag `granule_burden`/`high_granule_burden`, just don't remove any cells | off |

**Granule burden flagging always runs**, but the signal it's based on
falls back through three tiers depending on what's available, printed so
you can see which one fired:
1. **`--autofluor_channel`, if given**: Std.Dev./Median ratio of that
   channel's intensity per cell (punctate signal inflates Std.Dev. much more
   than diffuse signal does) -- not bounded to 0-1, so
   `--granule_drop_threshold 0.5` here means "Std.Dev. is half of Median",
   not "50% of the cell". This is also the only tier that runs a spectral
   correction (fitting/subtracting the autofluorescence channel's signal out
   of the other called channels).
2. **`Granule_Area_Blob`, if `cell_analysis.py` was run with
   `--granule_blob_channel`**: `burden = Granule_Area_Blob / Cell: Area`, a
   real 0-1 fraction of each cell's footprint covered by a shape-detected
   solid debris blob (see the granule-blob section above) -- preferred over
   tier 3 since it's detected directly rather than as a side effect of
   nucleus segmentation.
3. **`Granule_Area`, otherwise**: `burden = Granule_Area / Cell: Area`, what
   fraction of each cell's footprint was small DAPI-channel debris dropped
   during segmentation (see `--min_nucleus_area_um2` above) -- last resort,
   since it only exists because the segmentation backend mistook debris for
   a nucleus in the first place, which isn't guaranteed to happen.

Tiers 2 and 3 are both real 0-1 area fractions (unlike tier 1), so
`--granule_drop_threshold 0.5` means "more than half the cell's footprint is
debris" for either of them, and neither runs a spectral correction (there's
no channel to correct target channels against).

Cells above `--granule_flag_percentile` are flagged (`high_granule_burden`)
for manual review either way. Cells above `--granule_drop_threshold` (default
0.5) go further and get **removed entirely** before positivity calling even
runs, so they never appear in any output. Pass `--no_granule_drop` to disable
that and keep everything (flagged or not).

**Separation** (printed per channel, and saved in `_separation_summary.csv`)
is the GMM's between-component separation in log-intensity units: roughly,
**>2 well-separated/bimodal, 1-2 moderate, <1 weak/overlapping**. There are
no negative/isotype control images available for these datasets, so any
threshold is a statistical split of that specific image's own intensity
distribution, not a validated biological cutoff -- always check the
separation value and the overlay/histogram images before trusting a result.

### compare_intensity_by_call.py

| Flag | Meaning | Default |
|---|---|---|
| `--input_dir` | Folder to search recursively for `*_cells_calls.csv` files | required |
| `--value_channel` / `--value_compartment` / `--value_stat` | Which intensity column is being compared (the dependent variable) | 1 / `Nucleus` / `Mean` |
| `--group_channel` | Which channel's positive/negative call defines the two groups being compared (the independent variable; ambiguous cells excluded) | 2 |
| `--filter_channel` | Optional: restrict to a subpopulation before splitting by `--group_channel`, e.g. `--filter_channel 1` to only compare among channel-1-positive cells | none (use all cells) |
| `--filter_call` | Which call value of `--filter_channel` to keep | `positive` |
| `--output_dir` | Where to save the violin and forest plots | `--input_dir` |

Example: compare H3 (channel 3) nuclear intensity between DLL3-positive and
DLL3-negative cells, but only among cells that are also SOX9-positive:

```bash
python compare_intensity_by_call.py --input_dir raw/ \
    --filter_channel 1 --filter_call positive \
    --value_channel 3 --value_compartment Nucleus --value_stat Mean \
    --group_channel 2
```

A position contributes to the forest plot only if it has at least 2 cells in
**both** groups (otherwise a within-position effect/CI is undefined); a
position with fewer still contributes to the pooled tests and the blocked
ANCOVA's overall estimate, just not its own forest-plot row. The script
prints how many positions were skipped this way.

### compare_intensity_between_experiments.py

Answers a different question from `compare_intensity_by_call.py` above:
**is intensity different between two GROUPS of experiments** (e.g.
non-transforming tumors vs transforming tumors, where each group is made of
several independent tumor samples, and each sample is itself made of
several imaged positions), rather than between two groups within one
dataset. Filters every experiment to cells matching a marker-positivity
signature (e.g. marker-positive AND DLL3-positive) before comparing.

Three levels of clustering matter here -- cell < position < experiment <
group -- because two positions of the *same* tumor aren't independent
evidence about that tumor's group membership, only two *different* tumors
are.

| Flag | Meaning | Default |
|---|---|---|
| `--group_a_dirs` / `--group_b_dirs` | One or more experiment (tumor sample) folders making up each group, e.g. `--group_a_dirs tumor1/ tumor2/ tumor3/` | required |
| `--label_a` / `--label_b` | Short names for each group (e.g. `NonTransforming` / `Transforming`), used in print output, plot legends, and output filenames | `A` / `B` |
| `--experiment_labels_a` / `--experiment_labels_b` | One label per directory in the corresponding `--group_*_dirs`, for readable per-experiment output | each directory's own basename |
| `--filter_channels` | 1-based channel numbers that must ALL match `--filter_calls` for a cell to be included, e.g. `1 2` for marker-positive AND DLL3-positive. Applied identically to **every** experiment directory in both groups -- see the channel-layout caveat below | required |
| `--filter_calls` | One value applied to all `--filter_channels`, or one per channel | `positive` |
| `--value_channel` / `--value_compartment` / `--value_stat` | Which intensity column is being compared, applied identically to every experiment directory in both groups | 1 / `Nucleus` / `Mean` |
| `--output_dir` | Where to save the plots | first `--group_a_dirs` entry |

All experiment directories (in both groups) are assumed to share the same
`--filter_channels`/`--value_channel` numbering convention (e.g. channel 1
is always the primary marker, channel 2 always DLL3, regardless of which
specific antibody channel 1 happens to be in a given experiment) -- verify
this holds for every directory before trusting the result (see the
development note on not assuming a channel layout carries over between
datasets).

```bash
# H3 (channel 3, Nucleus, Mean) in marker+/DLL3+ cells, non-transforming
# tumors (several samples) vs transforming tumors (several samples)
python compare_intensity_between_experiments.py \
    --group_a_dirs "tumor1_sox9/" "tumor2_olig2/" --label_a NonTransforming \
    --group_b_dirs "tumor3_sox9/" "tumor4_olig2/" "tumor5_olig2/" --label_b Transforming \
    --filter_channels 1 2 --filter_calls positive positive \
    --value_channel 3 --value_compartment Nucleus --value_stat Mean
```

**Why this can't just reuse `compare_intensity_by_call.py`'s blocked
ANCOVA**: that approach dummy-codes position as a blocking factor and
estimates the group effect from the *within-position* contrast (each
position has both call values present). Here, every position belongs
entirely to one experiment, and every experiment belongs entirely to one
group -- there's no within-position (or even within-experiment) contrast to
anchor a fixed-effect blocking model on, so it would be unidentifiable.
Instead this script reports four views, in order of how much to trust them:

1. **Naive pooled cell-level tests** (Mann-Whitney U + Welch's t) --
   reference only, treats every cell as independent (ignores both
   per-image and per-tumor clustering, so overstates the real sample size
   the most).
2. **Position-level tests** -- each position collapsed to its own mean
   first. Better than (1) but still pseudoreplicated with respect to
   experiment: a tumor imaged at many positions still only counts as one
   biological replicate for the group-level question.
3. **Experiment-level tests** -- each experiment (tumor sample) collapsed to
   its own mean first, across all its positions and cells. n = number of
   tumors per group, the correct unit of replication for this design.
   **This is the primary test to trust** -- needs at least 2 experiments
   per group to run at all; add more tumor samples to a group if it's
   skipped.
4. **A nested mixed-effects model** (position nested within experiment as
   random effects, group as the fixed effect of interest) -- uses all
   cell-level data while still respecting the full clustering structure;
   the most statistically powerful option, but check for a convergence
   warning in the output before trusting its p-value (more likely with few
   experiments per group -- the variance-component estimate can hit a
   boundary).

The per-experiment plot (`..._experiments.png`) shows each experiment's own
mean ± 95% CI, colored by group -- the between-group equivalent of the other
script's forest plot. There's no within-experiment group contrast to plot
here (each experiment belongs to only one group), so this shows each
experiment's own summary instead of a per-experiment difference.

## Notes from development

- Micro-Manager MMStack OME-TIFFs need `is_ome=False, is_mmstack=False` when
  read with `tifffile.imread` -- otherwise it silently stitches every
  sibling `PosN.ome.tif` file in the folder into one array via the OME/MMStack
  cross-file metadata, prepending a spurious position axis. `preprocess.py`
  handles this already; worth knowing if you read raw files elsewhere.
- `tifffile.imwrite` auto-detects certain array shapes as OME-worthy and
  silently writes OME-XML instead of the simpler "shaped" metadata format,
  which doesn't carry resolution/unit info through. `preprocess.py` passes
  `ome=False` explicitly to avoid this -- it's why pixel size can round-trip
  through a processed file at all.
- Cellpose 4.x (Cellpose-SAM) replaced the old `models.Cellpose(...)` /
  `model_type="nuclei"` API with `models.CellposeModel(...)`, and its `.eval()`
  now returns a 3-tuple instead of 4 (no more `diams`). If you're working
  from an older Cellpose example online, the API has moved.
- Local background subtraction (as opposed to autofluorescence-channel
  correction) was tried and abandoned for densely-packed tissue.
- No negative/isotype control images are available for these datasets -- see
  the separation-metric caveat above.
- `StarDist2D.from_pretrained(...)` and `models.CellposeModel(...)` reload
  and reconstruct the entire network from disk on every call -- expensive if
  called once per image in a batch. Both are now cached at module level in
  `cell_analysis.py` (loaded once per process, reused for every subsequent
  image that process handles).
- A nuclear transcription factor's positivity should be called on the
  `Nucleus` compartment, not `Cytoplasm` -- this isn't just more
  biologically correct, it measurably improved GMM separation in practice
  (observed: 1.77 → 2.14 on one image after switching), partly because it
  also naturally avoids overlap with debris blobs that tend to sit in
  cytoplasmic regions.
- **DLL3 (a genuinely punctate marker) is called on `Cell: Channel N:
  Dot_Density`, not `Dot_Count` or Mean/Median intensity, on this dataset --
  decided after comparing four approaches head to head on the same 24-image
  subset**: Cytoplasm Mean (separation 0.34, ~6.5% positive -- weak,
  unreliable), Cell Mean (separation 0.77, 2.9% positive -- biologically
  plausible rate but weak separation), Cell `Dot_Count` (separation 2.17,
  43% positive -- good separation but implausibly over-called), Cell
  `Dot_Density` (separation ~2.0-2.1, stable across `--puncta_log_threshold_rel`
  0.02 vs 0.03, 23-27% positive). `Dot_Count` over-calls because it's
  confounded by cell size (see "Prefer `Dot_Density`..." above); raw Mean
  intensity under-calls/has weak separation because punctate signal gets
  diluted by background between puncta. `Dot_Density` was the only option
  with both strong separation and a plausible call rate.
- **Update (2026-11-19): a newer/better DLL3 antibody calls cleanly on plain
  `Cytoplasm: Channel N: Mean` intensity -- this does NOT contradict the
  `Dot_Density` finding above.** `Dot_Density` was a fix for a specific
  antibody's weak, sparse, punctate signal, where mean intensity got diluted
  by background between dots. A stronger/cleaner antibody lot doesn't have
  that problem, so simple Mean intensity gives a real bimodal GMM split
  without the dot-density workaround. Don't default to either method
  blindly for a new DLL3 dataset: check the GMM separation for Mean
  intensity first (simpler), and only fall back to puncta-based
  `Dot_Density` calling if that separation is weak (<1) with a high
  ambiguous fraction, the same symptom that motivated it originally.
- **Don't assume a channel-to-antibody mapping carries over between
  datasets, even similar-looking ones, and don't assume either the folder
  name or the raw filename is automatically correct if they disagree.** One
  dataset's folder name and its raw file's own name implied different
  channel orders for the same two antibodies; only whoever ran the
  acquisition could say which one was actually right (in that case, the
  folder name was correct and the filename had the typo). The embedded
  Micro-Manager metadata (`ChNames`) only gives laser lines/wavelengths, not
  antibody identity, so it can't resolve this kind of disagreement on its
  own -- when channel order looks inconsistent, ask rather than guess.
- Windows multiprocessing (`ProcessPoolExecutor`, used by `--workers`) uses
  the `spawn` start method, which re-imports the whole module in each worker
  process without running anything inside `if __name__ == "__main__":` --
  the worker function and its initializer must be plain module-level
  functions (not closures/lambdas) so they can be pickled and reconstructed
  in the child process.
- A position-blocked ANCOVA (dummy-coding position as a fixed effect) only
  identifies a group effect if that group varies *within* at least some
  positions. If group is perfectly determined by which dataset a position
  came from (as when comparing separate experiments, each with its own
  positions), the design matrix becomes rank-deficient and the "group"
  coefficient isn't reliably estimable -- `compare_intensity_between_experiments.py`
  exists because of this; it treats the actual unit of biological
  replication (experiment/tumor, not position or cell) as the primary test,
  backed by a nested mixed-effects model, rather than reusing
  `compare_intensity_by_call.py`'s blocking approach with a relabeled group
  variable.
- On the one real between-experiment comparison run so far (before the
  script was generalized to multiple experiments per group), the naive
  pooled cell-level test found "significance" (p=0.0015) that the
  position-level test did not (p=0.106, n=5 vs n=7 positions) -- a concrete
  demonstration of the pseudoreplication risk, not just a theoretical
  concern, for datasets where cell counts vary a lot by position (some
  positions here contributed as few as 1-2 filtered cells). The same risk
  applies one level up once there are multiple tumors per group: pooling
  cells (or even positions) across tumors as if they were independent
  replicates of "transforming" or "non-transforming" overstates the real
  sample size, which is the number of *tumors*, not positions or cells --
  hence why the experiment-level test is the one to trust for that design.
