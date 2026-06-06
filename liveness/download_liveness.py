#!/usr/bin/env python3
"""download_liveness.py — lấy LCC-FASD theo thứ tự ưu tiên:
  1. env LCC_FASD_DIR (đường dẫn đã có) -> dùng luôn.
  2. HF dataset zip (huanthuytnhh/deepfake-data/lcc-fasd.zip) -> tải + giải nén  [KHÔNG cần Kaggle creds].
  3. kagglehub (faber24/lcc-fasd)                                                [cần Kaggle creds].
In ra đường dẫn LCC_FASD root ở DÒNG CUỐI (để run_liveness.sh bắt). --quiet để bỏ tóm tắt.

HF cần token: export HF_TOKEN=hf_...  (hoặc env HUGGINGFACE_HUB_TOKEN).
"""
import os
import sys
import glob
import zipfile
import collections

HF_REPO = os.environ.get("LCC_HF_REPO", "huanthuytnhh/deepfake-data")
HF_FILE = "lcc-fasd.zip"
UNZIP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data")


def _find_lcc_root(base):
    for d, subs, _ in os.walk(base):
        names = [s.lower() for s in subs]
        if any("train" in n for n in names) and any(("eval" in n or "test" in n) for n in names):
            return d
    return base


def _from_hf():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    from huggingface_hub import hf_hub_download
    print(f"[hf] tải {HF_REPO}/{HF_FILE} ...", file=sys.stderr)
    zp = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset", token=tok)
    os.makedirs(UNZIP_DIR, exist_ok=True)
    if not glob.glob(os.path.join(UNZIP_DIR, "**", "*_training"), recursive=True):
        print("[hf] giải nén ...", file=sys.stderr)
        with zipfile.ZipFile(zp) as z:
            z.extractall(UNZIP_DIR)
    return _find_lcc_root(UNZIP_DIR)


def _from_kaggle():
    import kagglehub
    return _find_lcc_root(kagglehub.dataset_download("faber24/lcc-fasd"))


def main():
    quiet = "--quiet" in sys.argv
    env = os.environ.get("LCC_FASD_DIR")
    root = None
    if env and os.path.isdir(env):
        root = _find_lcc_root(env)
    if root is None:
        try:
            root = _from_hf()
        except Exception as e:
            print(f"[hf] thất bại ({e}); thử kagglehub ...", file=sys.stderr)
            try:
                root = _from_kaggle()
            except Exception as e2:
                print(f"ERROR: cả HF lẫn Kaggle đều lỗi ({e2})", file=sys.stderr); sys.exit(3)
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
