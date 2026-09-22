"""
Run this FIRST on any freshly unzipped dataset, before writing/trusting an
importer. Kaggle/GitHub datasets change their internal folder layout
between versions, so this prints what's actually inside your download
(directory tree + a peek at the first few rows of any CSV, and the shape/
keys of any .npy/.npz/.json landmark file) so you can confirm the import_*
scripts' assumptions match your specific copy — and fix them fast if not.

Usage:
    python data/inspect_dataset.py --path data/raw/include_raw
    python data/inspect_dataset.py --path data/raw/isl_translate_raw
    python data/inspect_dataset.py --path data/raw/isl_csltr_raw
"""
import argparse
import json
from pathlib import Path

import numpy as np


def tree(path: Path, max_depth=3, max_items=15, prefix=""):
    if max_depth < 0:
        return
    try:
        items = sorted(path.iterdir())
    except (PermissionError, NotADirectoryError):
        return
    for i, item in enumerate(items[:max_items]):
        print(f"{prefix}{item.name}{'/' if item.is_dir() else ''}")
        if item.is_dir():
            tree(item, max_depth - 1, max_items, prefix + "  ")
    if len(items) > max_items:
        print(f"{prefix}... ({len(items) - max_items} more)")


def peek_csv(path: Path, n_rows=3):
    import csv
    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        print(f"  columns: {header}")
        for i, row in enumerate(reader):
            if i >= n_rows:
                break
            print(f"  row {i}: {row}")


def peek_npy(path: Path):
    try:
        arr = np.load(path, allow_pickle=True)
        print(f"  shape={getattr(arr, 'shape', '?')} dtype={getattr(arr, 'dtype', '?')}")
    except Exception as e:
        print(f"  could not load: {e}")


def peek_json(path: Path):
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            print(f"  top-level keys: {list(data.keys())[:20]}")
        elif isinstance(data, list):
            print(f"  list of {len(data)} items, first item: {str(data[0])[:200]}")
    except Exception as e:
        print(f"  could not load: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Folder you unzipped the dataset into")
    parser.add_argument("--depth", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        print(f"{root} does not exist. Unzip your dataset there first.")
        return

    print(f"=== Directory tree: {root} ===")
    tree(root, max_depth=args.depth)

    print(f"\n=== Sample file contents ===")
    seen = 0
    for ext, peeker in [("*.csv", peek_csv), ("*.npy", peek_npy), ("*.json", peek_json)]:
        for f in root.rglob(ext):
            print(f"\n[{f.relative_to(root)}]")
            peeker(f)
            seen += 1
            if seen >= 6:
                break
        if seen >= 6:
            break

    if seen == 0:
        print("No .csv/.npy/.json files found at top levels — check for .tar.gz/.zip "
              "archives that still need extracting inside this folder.")


if __name__ == "__main__":
    main()
