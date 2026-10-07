"""
Splits a folder of raw acquisition files into per-tumor subfolders, based on
a numeric prefix captured from each filename by --regex and looked up in
--mapping.

Example: files named like
    ..._MMStack_1-Pos000_002.ome.tif
    ..._MMStack_1-Pos000_002_metadata.txt
    ..._MMStack_2-Pos003_014.ome.tif
    ..._MMStack_3-Pos001_007.ome.tif
with --mapping 1=1973 2=2006 3=2471 get sorted into
    <output_dir>/1973/..._MMStack_1-Pos000_002.ome.tif
    <output_dir>/1973/..._MMStack_1-Pos000_002_metadata.txt
    <output_dir>/2006/..._MMStack_2-Pos003_014.ome.tif
    <output_dir>/2471/..._MMStack_3-Pos001_007.ome.tif

Matches both the raw TIFFs and their `*_metadata.txt` sidecars by default
(each raw file's Micro-Manager metadata sidecar shares its base name) --
override --pattern if your files differ.

Dry-run by default -- prints the planned split (counts per tumor, any
unmatched files) without touching anything. Pass --execute to actually
move/copy the files.

Usage:
    # preview only
    python split_files_by_tumor.py --input_dir "raw/" \
        --mapping 1=1973 2=2006 3=2471

    # actually move the files
    python split_files_by_tumor.py --input_dir "raw/" \
        --mapping 1=1973 2=2006 3=2471 --execute

    # copy instead of move (uses more disk, leaves originals in place)
    python split_files_by_tumor.py --input_dir "raw/" \
        --mapping 1=1973 2=2006 3=2471 --execute --copy
"""

import argparse
import glob
import os
import re
import shutil


def parse_mapping(pairs):
    mapping = {}
    for pair in pairs:
        key, sep, val = pair.partition("=")
        if not sep or not key or not val:
            raise ValueError(f"Bad --mapping entry '{pair}' -- expected KEY=LABEL, e.g. 1=1973")
        mapping[key] = val
    return mapping


def plan_split(input_dir, patterns, regex, mapping):
    """Returns (plan, unmatched): plan maps tumor label -> list of file paths;
    unmatched is the list of filenames that didn't match --regex or whose
    captured key isn't in --mapping. patterns can match the same file more
    than once (e.g. an overly broad pattern list) -- duplicates are collapsed
    so no file is planned to move twice."""
    files = sorted(set(
        f for pattern in patterns for f in glob.glob(os.path.join(input_dir, pattern))
    ))
    compiled = re.compile(regex)

    plan = {label: [] for label in mapping.values()}
    unmatched = []
    for f in files:
        name = os.path.basename(f)
        m = compiled.search(name)
        if not m or m.group(1) not in mapping:
            unmatched.append(name)
            continue
        plan[mapping[m.group(1)]].append(f)
    return plan, unmatched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", required=True, help="Folder containing the combined raw files")
    parser.add_argument("--output_dir", default=None,
                         help="Where the per-tumor subfolders get created (default: --input_dir, i.e. "
                              "subfolders alongside the original files)")
    parser.add_argument("--mapping", nargs="+", required=True,
                         help="KEY=LABEL pairs, e.g. --mapping 1=1973 2=2006 3=2471 -- KEY is looked up "
                              "against the first group captured by --regex in each filename")
    parser.add_argument("--pattern", nargs="+", default=["*.tif", "*_metadata.txt"],
                         help="Glob pattern(s) for files to consider -- default matches both the raw "
                              "TIFFs and their Micro-Manager metadata sidecars")
    parser.add_argument("--regex", default=r"_MMStack_(\d+)-Pos",
                         help="Regex whose first capture group is the --mapping key (default matches "
                              "the digit right after '_MMStack_' and before '-Pos')")
    parser.add_argument("--copy", action="store_true",
                         help="Copy instead of move -- leaves originals in place, uses more disk")
    parser.add_argument("--execute", action="store_true",
                         help="Actually move/copy files. Without this flag, only prints the plan.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    mapping = parse_mapping(args.mapping)
    plan, unmatched = plan_split(args.input_dir, args.pattern, args.regex, mapping)

    total = sum(len(v) for v in plan.values())
    print(f"Found {total + len(unmatched)} file(s) matching {args.pattern} in {args.input_dir}")
    for label, group_files in plan.items():
        print(f"  {label}: {len(group_files)} file(s)")
    if unmatched:
        print(f"  UNMATCHED (won't be moved -- regex didn't match, or matched key isn't in --mapping): "
              f"{len(unmatched)} file(s)")
        for n in unmatched[:10]:
            print(f"    - {n}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")

    if not args.execute:
        print("\nDry run only -- pass --execute to actually move/copy these files.")
        return

    action = shutil.copy2 if args.copy else shutil.move
    verb = "Copied" if args.copy else "Moved"
    for label, group_files in plan.items():
        if not group_files:
            continue
        dest_dir = os.path.join(output_dir, label)
        os.makedirs(dest_dir, exist_ok=True)
        for f in group_files:
            action(f, os.path.join(dest_dir, os.path.basename(f)))
        print(f"{verb} {len(group_files)} file(s) to {dest_dir}")


if __name__ == "__main__":
    main()
