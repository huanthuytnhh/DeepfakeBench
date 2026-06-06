#!/usr/bin/env bash
# run_liveness.sh — module LIVENESS độc lập: tải LCC-FASD -> train B4 baseline -> báo cáo ACER/AUC.
# KHÔNG đụng code deepfake. Chạy local trên GPU 4GB (AMP).
#   SMOKE=1 ./liveness/run_liveness.sh         # 1 epoch, 200 ảnh/split (smoke trước - rule)
#   BATCH=32 ./liveness/run_liveness.sh        # full (5090/đủ VRAM)
#   BATCH=16 ./liveness/run_liveness.sh        # nếu OOM 4GB
set -uo pipefail
cd "$(dirname "$0")/.."                                  # repo root
PY=python; command -v python >/dev/null 2>&1 || PY=python3
BATCH=${BATCH:-16}; EPOCHS=${EPOCHS:-15}
EXTRA=""; [ "${SMOKE:-0}" = "1" ] && EXTRA="--epochs 1 --max_per_split 200"

echo "== [1/2] tải LCC-FASD (~5GB, Kaggle) =="
$PY -c "import kagglehub" 2>/dev/null || pip install -q kagglehub
DATA=$($PY liveness/download_liveness.py | tail -1)
[ -d "$DATA" ] || { echo "!! tải lỗi — đặt Kaggle creds ~/.kaggle/kaggle.json rồi chạy lại"; exit 1; }
echo "DATA=$DATA"

echo "== [2/2] train B4 baseline liveness =="
$PY liveness/train_liveness.py --data_root "$DATA" --out liveness/out/b4 \
    --batch "$BATCH" --epochs "$EPOCHS" $EXTRA \
    || { echo "!! train lỗi (OOM? thử BATCH=16/8)"; exit 1; }

echo "== DONE — kết quả: liveness/out/b4/metrics_liveness.json =="
cat liveness/out/b4/metrics_liveness.json 2>/dev/null
