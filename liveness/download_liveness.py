#!/usr/bin/env python3
"""download_liveness.py — tải LCC-FASD (Kaggle: faber24/lcc-fasd, ~5GB) qua kagglehub.

Cần Kaggle creds: ~/.kaggle/kaggle.json (chmod 600) HOẶC env KAGGLE_USERNAME + KAGGLE_KEY.
Token: https://www.kaggle.com/settings -> "Create New API Token".
In ra đường dẫn LCC_FASD ở DÒNG CUỐI (để run_liveness.sh bắt). --quiet để bỏ tóm tắt.
"""
import os
import sys
import glob
import collections


def _find_lcc_root(base):
    for d, subs, _ in os.walk(base):
        names = [s.lower() for s in subs]
        if any("train" in n for n in names) and any(("eval" in n or "test" in n) for n in names):
            return d
    return base


def main():
    quiet = "--quiet" in sys.argv
    try:
        import kagglehub
    except ImportError:
        print("ERROR: pip install kagglehub", file=sys.stderr); sys.exit(2)
    try:
        path = kagglehub.dataset_download("faber24/lcc-fasd")
    except Exception as e:
        print(f"ERROR: download failed ({e}). Set Kaggle creds and retry.", file=sys.stderr); sys.exit(3)
    root = _find_lcc_root(path)
    if not quiet:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from dataset_liveness import list_split_liveness
        print(f"LCC-FASD root: {root}", file=sys.stderr)
        for split in sorted(glob.glob(os.path.join(root, "*"))):
            if os.path.isdir(split):
                c = collections.Counter(l for _, l in list_split_liveness(split))
                print(f"  {os.path.basename(split):22s} live(0)={c.get(0,0)}  spoof(1)={c.get(1,0)}", file=sys.stderr)
    print(root)


if __name__ == "__main__":
    main()
