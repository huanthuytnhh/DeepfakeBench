#!/usr/bin/env python3
"""Download LCC-FASD (Kaggle: faber24/lcc-fasd, ~5GB) via kagglehub and print its layout.

Kaggle credentials are required (one of):
  - ~/.kaggle/kaggle.json   (chmod 600), or
  - env KAGGLE_USERNAME + KAGGLE_KEY
Get a token at https://www.kaggle.com/settings  -> "Create New API Token".

Prints the resolved LCC_FASD root path on the LAST stdout line (so run_fas.sh can capture it).
Use --quiet to suppress the tree summary.
"""
import os
import sys
import glob
import collections


def _find_lcc_root(base):
    """Find the directory that directly contains the *_training/_development/_evaluation splits."""
    for d, subs, _ in os.walk(base):
        names = [s.lower() for s in subs]
        if any("train" in n for n in names) and any(("eval" in n or "test" in n) for n in names):
            return d
    return base


def _summary(root):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import list_split, _infer_label  # noqa
    print(f"LCC-FASD root: {root}", file=sys.stderr)
    for split in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(split):
            continue
        items = list_split(split)
        c = collections.Counter(l for _, l in items)
        print(f"  {os.path.basename(split):24s} images={len(items):6d}  "
              f"live(0)={c.get(0,0)}  spoof(1)={c.get(1,0)}", file=sys.stderr)


def main():
    quiet = "--quiet" in sys.argv
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed  ->  pip install kagglehub", file=sys.stderr)
        sys.exit(2)
    try:
        path = kagglehub.dataset_download("faber24/lcc-fasd")
    except Exception as e:
        print(f"ERROR: kagglehub download failed ({e}).\n"
              f"  Set Kaggle creds (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY) and retry,\n"
              f"  or download manually: https://www.kaggle.com/datasets/faber24/lcc-fasd",
              file=sys.stderr)
        sys.exit(3)
    root = _find_lcc_root(path)
    if not quiet:
        _summary(root)
    print(root)                                   # LAST line = the path (captured by run_fas.sh)


if __name__ == "__main__":
    main()
