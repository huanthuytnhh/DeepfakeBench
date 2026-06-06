#!/usr/bin/env bash
# run_fas.sh — Liveness (FAS) SECONDARY module: reuse the deepfake B4 vs B4+block-DCT (SFDCT) on
# LCC-FASD (2 runs, intra-dataset), compare, then commit results to the current (FAS) branch and
# optionally push checkpoints to Hugging Face.
#
#   SMOKE=1 ./tools/run_fas.sh                  # 1 epoch, 200 imgs/split — smoke-test FIRST (the rule)
#   BATCH=16 EPOCHS=10 ./tools/run_fas.sh       # full run on RTX 3050Ti 4GB (AMP on)
#   BATCH=8  ./tools/run_fas.sh                 # if CUDA OOM, drop the batch
#   PUSH=0   ./tools/run_fas.sh                 # don't git-push results
#   HF_TOKEN=hf_... ./tools/run_fas.sh          # also push .pth checkpoints to HF
set -uo pipefail
cd "$(dirname "$0")/.."                                       # DeepfakeBench repo root
PY=python; command -v python >/dev/null 2>&1 || PY=python3
BATCH=${BATCH:-16}; EPOCHS=${EPOCHS:-10}; PUSH=${PUSH:-1}
EXTRA=""; [ "${SMOKE:-0}" = "1" ] && EXTRA="--epochs 1 --max_per_split 200"

echo "== [1/4] download LCC-FASD (~5GB, Kaggle) =="
$PY -c "import kagglehub" 2>/dev/null || pip install -q kagglehub
DATA=$($PY tools/fas/download.py | tail -1)
[ -d "$DATA" ] || { echo "!! download failed — set Kaggle creds (~/.kaggle/kaggle.json) and retry"; exit 1; }
echo "DATA=$DATA"

# warm-start from the existing deepfake checkpoints if present (transfer deepfake -> liveness)
B4_CK=$(ls -t logs/training/efficientnetb4_2*/test/Celeb-DF-v2/ckpt_best.pth 2>/dev/null | head -1 || true)
SF_CK=$(ls -t logs/training/efficientnetb4_sfdct_2*/test/Celeb-DF-v2/ckpt_best.pth 2>/dev/null | head -1 || true)
[ -n "$B4_CK" ] && echo "warm-start B4    <- $B4_CK"
[ -n "$SF_CK" ] && echo "warm-start SFDCT <- $SF_CK"

echo "== [2/4] RUN 1/2 — B4 baseline =="
$PY tools/train_fas.py --config training/config/detector/efficientnetb4.yaml \
    --data_root "$DATA" --out logs/fas/b4 --batch "$BATCH" --epochs "$EPOCHS" \
    ${B4_CK:+--init_ckpt "$B4_CK"} $EXTRA || { echo "!! B4 run failed (OOM? try BATCH=8)"; exit 1; }

echo "== [2/4] RUN 2/2 — B4 + block-DCT (SFDCT, S1-S5 OFF) =="
$PY tools/train_fas.py --config training/config/detector/efficientnetb4_sfdct.yaml \
    --data_root "$DATA" --out logs/fas/sfdct --batch "$BATCH" --epochs "$EPOCHS" \
    ${SF_CK:+--init_ckpt "$SF_CK"} $EXTRA || { echo "!! SFDCT run failed (OOM? try BATCH=8)"; exit 1; }

echo "== [3/4] compare =="
$PY tools/fas/compare.py logs/fas/b4/metrics.json logs/fas/sfdct/metrics.json | tee logs/fas/comparison.txt

if [ "$PUSH" = "1" ]; then
  echo "== [4/4] push results to current branch =="
  BR=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
  # small json/txt -> git (force past logs/ gitignore); big .pth stays out of git
  git add tools/fas tools/train_fas.py 2>/dev/null || true
  git add -f logs/fas/b4/metrics.json logs/fas/sfdct/metrics.json logs/fas/comparison.txt 2>/dev/null || true
  if git commit -m "results(fas): LCC-FASD B4 vs B4+block-DCT (liveness secondary module)"; then
    git push origin "$BR" && echo "pushed results -> $BR"
  else
    echo "(nothing new to commit)"
  fi
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "-- push checkpoints to HF --"
    HF_TOKEN="$HF_TOKEN" $PY - <<'PYEOF'
import os, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "huanthuytnhh/deepfake"
ts = "fas-" + time.strftime("%Y%m%d-%H%M%S")
api.create_repo(repo, repo_type="model", exist_ok=True)
for tag in ("b4", "sfdct"):
    d = f"logs/fas/{tag}"
    if os.path.isdir(d):
        api.upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                          path_in_repo=f"runs/{ts}/{tag}", allow_patterns=["*.pth", "*.json", "*.txt"])
print("HF ->", f"https://huggingface.co/{repo}/tree/main/runs/{ts}")
PYEOF
  fi
fi
echo "== DONE — results in logs/fas/ (comparison.txt) =="
