#!/usr/bin/env bash
# run_liveness.sh — module LIVENESS độc lập: ablation B4 vs B4+block-DCT trên LCC-FASD
# (mỗi model TƯƠNG ĐƯƠNG detector deepfake tương ứng) -> compare -> push HF. KHÔNG đụng code deepfake.
#   Trên vast 5090 (sau khi fix1/fix2 xong):
#     export HF_TOKEN=hf_...; BATCH=32 ./liveness/run_liveness.sh
#   Smoke trước:  SMOKE=1 ./liveness/run_liveness.sh
#   GPU 4GB:      BATCH=16 ./liveness/run_liveness.sh
#   Chỉ B4:       ONLY=b4 ./liveness/run_liveness.sh   (hoặc ONLY=b4dct)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=python; command -v python >/dev/null 2>&1 || PY=python3
BATCH=${BATCH:-32}; EPOCHS=${EPOCHS:-15}; ONLY=${ONLY:-both}
REPO_HF="${HF_REPO:-huanthuytnhh/deepfake}"
TS="liveness-$(date +%Y%m%d-%H%M%S)"
EXTRA=""; [ "${SMOKE:-0}" = "1" ] && EXTRA="--epochs 1 --max_per_split 200"

echo "== [1/4] lấy LCC-FASD (HF zip -> unzip, fallback Kaggle) =="
DATA=$($PY liveness/download_liveness.py | tail -1)
[ -d "$DATA" ] || { echo "!! lấy data lỗi"; exit 1; }
echo "DATA=$DATA"

run_one () {  # $1 = b4 | b4dct
  local m="$1"
  echo "== train [$m] (batch $BATCH) =="
  $PY liveness/train_liveness.py --data_root "$DATA" --model "$m" --out "liveness/out/$m" \
      --batch "$BATCH" --epochs "$EPOCHS" $EXTRA || { echo "!! [$m] lỗi (OOM? BATCH=16/8)"; return 1; }
}

echo "== [2/4] train =="
[ "$ONLY" = "both" ] || [ "$ONLY" = "b4" ]    && run_one b4
[ "$ONLY" = "both" ] || [ "$ONLY" = "b4dct" ] && run_one b4dct

echo "== [3/4] compare B4 vs B4+DCT =="
$PY liveness/compare_liveness.py liveness/out/b4/metrics_liveness.json liveness/out/b4dct/metrics_liveness.json \
    2>/dev/null | tee liveness/out/compare_liveness.txt || true

echo "== [4/4] push -> HF runs/$TS/ =="
if [ -n "${HF_TOKEN:-}" ]; then
  TS="$TS" REPO="$REPO_HF" "$PY" - <<'PY' || echo "(HF push lỗi)"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["REPO"]; ts = os.environ["TS"]
api.create_repo(repo, repo_type="model", exist_ok=True)
for m in ("b4", "b4dct"):
    d = f"liveness/out/{m}"
    if os.path.isdir(d):
        api.upload_folder(folder_path=d, repo_id=repo, repo_type="model",
                          path_in_repo=f"runs/{ts}/{m}", allow_patterns=["*.pth", "*.json"])
if os.path.isfile("liveness/out/compare_liveness.txt"):
    api.upload_file(path_or_fileobj="liveness/out/compare_liveness.txt",
                    path_in_repo=f"runs/{ts}/compare_liveness.txt", repo_id=repo, repo_type="model")
print(f"PUSHED -> https://huggingface.co/{repo}/tree/main/runs/{ts}")
print(f"KÉO VỀ: huggingface-cli download {repo} --include 'runs/{ts}/*' --local-dir ./liveness_pulled")
PY
else
  echo "  HF_TOKEN chưa set -> model ở liveness/out/. export HF_TOKEN rồi chạy lại."
fi
echo "== DONE ($TS) =="
