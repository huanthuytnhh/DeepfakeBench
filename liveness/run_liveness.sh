#!/usr/bin/env bash
# run_liveness.sh — module LIVENESS độc lập: tải LCC-FASD -> train B4 baseline (≈ B4 detector deepfake)
# -> push model+metrics lên HF (để kéo về). KHÔNG đụng code deepfake.
#   Trên vast 5090 (sau khi fix1/fix2 xong):
#     export HF_TOKEN=hf_...; BATCH=32 ./liveness/run_liveness.sh
#   Smoke trước (rule):  SMOKE=1 ./liveness/run_liveness.sh
#   GPU 4GB local:       BATCH=16 ./liveness/run_liveness.sh
set -uo pipefail
cd "$(dirname "$0")/.."                                  # repo root
PY=python; command -v python >/dev/null 2>&1 || PY=python3
BATCH=${BATCH:-32}; EPOCHS=${EPOCHS:-15}                 # batch 32 = khớp protocol deepfake
REPO_HF="${HF_REPO:-huanthuytnhh/deepfake}"
TS="liveness-$(date +%Y%m%d-%H%M%S)"
EXTRA=""; [ "${SMOKE:-0}" = "1" ] && EXTRA="--epochs 1 --max_per_split 200"

echo "== [1/3] tải LCC-FASD (~5GB, Kaggle) =="
$PY -c "import kagglehub" 2>/dev/null || pip install -q kagglehub
DATA=$($PY liveness/download_liveness.py | tail -1)
[ -d "$DATA" ] || { echo "!! tải lỗi — đặt Kaggle creds ~/.kaggle/kaggle.json trên box rồi chạy lại"; exit 1; }
echo "DATA=$DATA"

echo "== [2/3] train B4 baseline liveness (batch $BATCH) =="
$PY liveness/train_liveness.py --data_root "$DATA" --out liveness/out/b4 \
    --batch "$BATCH" --epochs "$EPOCHS" $EXTRA \
    || { echo "!! train lỗi (OOM? BATCH=16/8)"; exit 1; }
cat liveness/out/b4/metrics_liveness.json 2>/dev/null

echo "== [3/3] push model + metrics -> HF runs/$TS/b4 =="
if [ -n "${HF_TOKEN:-}" ]; then
  TS="$TS" REPO="$REPO_HF" "$PY" - <<'PY' || echo "(HF push lỗi)"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["REPO"]; ts = os.environ["TS"]
api.create_repo(repo, repo_type="model", exist_ok=True)
if os.path.isdir("liveness/out/b4"):
    api.upload_folder(folder_path="liveness/out/b4", repo_id=repo, repo_type="model",
                      path_in_repo=f"runs/{ts}/b4", allow_patterns=["*.pth", "*.json"])
print(f"PUSHED -> https://huggingface.co/{repo}/tree/main/runs/{ts}/b4")
print(f"KÉO VỀ local: huggingface-cli download {repo} --include 'runs/{ts}/b4/*' --local-dir ./liveness_pulled")
PY
else
  echo "  HF_TOKEN chưa set -> model ở liveness/out/b4/ (chưa push). export HF_TOKEN rồi chạy lại bước push."
fi
echo "== DONE ($TS) =="
