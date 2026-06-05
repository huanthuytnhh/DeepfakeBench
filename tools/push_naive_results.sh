#!/usr/bin/env bash
# push_naive_results.sh — rescue the NAIVE run (B4 baseline + naive SFDCT): build figures + push
# model(.pth)+figures to HF, with the CORRECT ckpt per detector (fixes the efficientnetb4_* glob that
# greedily matched efficientnetb4_sfdct_*). Run from the box that trained naive:
#   export HF_TOKEN=hf_...; bash tools/push_naive_results.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python; command -v python >/dev/null 2>&1 || PY=python3

# _2* anchors the timestamp so efficientnetb4_2* does NOT match efficientnetb4_sfdct_2*
B4=$(ls -t logs/training/efficientnetb4_2*/test/Celeb-DF-v2/ckpt_best.pth 2>/dev/null | head -1)
SF=$(ls -t logs/training/efficientnetb4_sfdct_2*/test/Celeb-DF-v2/ckpt_best.pth 2>/dev/null | head -1)
echo "B4    ckpt = ${B4:-<none>}"
echo "SFDCT ckpt = ${SF:-<none>}"
[ -z "$B4" ] && [ -z "$SF" ] && { echo "!! no ckpts found under logs/training/ — nothing to push"; exit 1; }

CB4=training/config/detector/efficientnetb4.yaml
CSF=training/config/detector/efficientnetb4_sfdct.yaml

if [ -n "$B4" ]; then
  echo "== figures: B4 =="
  "$PY" training/eval_and_viz.py --detector_path "$CB4" --weights_path "$B4" \
      --test_dataset FaceForensics++ Celeb-DF-v2 --out viz_out/b4 || echo "(B4 viz failed, continue)"
fi
if [ -n "$SF" ]; then
  echo "== figures: naive SFDCT =="
  "$PY" training/eval_and_viz.py --detector_path "$CSF" --weights_path "$SF" \
      --test_dataset FaceForensics++ Celeb-DF-v2 --out viz_out/sfdct_naive || echo "(SFDCT viz failed, continue)"
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! HF_TOKEN not set -> figures are in viz_out/, .pth in logs/training/. export HF_TOKEN and re-run to push."
  exit 0
fi

echo "== push to HF =="
B4="$B4" SF="$SF" "$PY" - <<'PY'
import os, time, os.path as osp
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ.get("HF_REPO", "huanthuytnhh/deepfake")
ts = "naive-" + time.strftime("%Y%m%d-%H%M%S")
api.create_repo(repo, repo_type="model", exist_ok=True)
b4, sf = os.environ.get("B4", ""), os.environ.get("SF", "")
if b4:
    api.upload_folder(folder_path=osp.dirname(b4), repo_id=repo, repo_type="model",
                      path_in_repo=f"runs/{ts}/b4/ckpt", allow_patterns=["*.pth", "*.pickle"])
if sf:
    api.upload_folder(folder_path=osp.dirname(sf), repo_id=repo, repo_type="model",
                      path_in_repo=f"runs/{ts}/sfdct_naive/ckpt", allow_patterns=["*.pth", "*.pickle"])
for d, tag in (("viz_out/b4", "b4"), ("viz_out/sfdct_naive", "sfdct_naive")):
    if osp.isdir(d):
        api.upload_folder(folder_path=d, repo_id=repo, repo_type="model", path_in_repo=f"runs/{ts}/{tag}/viz")
print("DONE ->", f"https://huggingface.co/{repo}/tree/main/runs/{ts}")
PY
