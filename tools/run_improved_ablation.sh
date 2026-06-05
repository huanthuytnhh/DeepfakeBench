#!/usr/bin/env bash
# run_improved_ablation.sh — ONE command to measure the thing that settles the goal:
#   Celeb-DF-v2 cross-test AUC of SFDCT-improved (DCT-Fo-Mixup + consistency) vs same-pipeline B4,
#   over N seeds, train FF++ c23 -> test CDFv2. Run this on a FREE GPU box (current B4 run must be done,
#   or use a 2nd GPU via CUDA_VISIBLE_DEVICES). Real numbers come ONLY from this full pipeline (not local smoke).
#
# Usage:   bash tools/run_improved_ablation.sh "1024 2025 7 42 123"
#          CUDA_VISIBLE_DEVICES=1 bash tools/run_improved_ablation.sh "1024 2025 7"   # 2nd GPU
set -euo pipefail
cd "$(dirname "$0")/.."

# Default = 1 seed => the "2 new runs": Row1 + Row2. Pass more seeds to expand (e.g. "1024 2025 7").
# B4 baseline is REUSED from the existing run by default; set RUN_B4=1 to (re)train it here too.
SEEDS="${1:-1024}"
B4="training/config/detector/efficientnetb4.yaml"
SFDCT="training/config/detector/efficientnetb4_sfdct.yaml"
OUT="logs/ablation_cdfv2"; mkdir -p "$OUT" /tmp/abl

run_one () {  # $1=base_yaml  $2=tag  $3=seed  $4=extra_sed(optional)
  local base="$1" tag="$2" seed="$3" extra="${4:-}"
  local y="/tmp/abl/${tag}_s${seed}.yaml"
  cp "$base" "$y"
  sed -i "s/^manualSeed:.*/manualSeed: ${seed}/" "$y"
  [ -n "$extra" ] && sed -i "$extra" "$y"
  echo ">>> [$tag seed=$seed] train FF++ -> test CDFv2 ..."
  python training/train.py --detector_path "$y" \
      --train_dataset FaceForensics++ --test_dataset Celeb-DF-v2 --no-save_feat \
      > "$OUT/${tag}_s${seed}.log" 2>&1 || echo "!! ${tag} seed=${seed} FAILED (see log)"
}

for s in $SEEDS; do
  # optional: re-train B4 baseline here (default OFF -> reuse the existing B4 run)
  [ "${RUN_B4:-0}" = "1" ] && run_one "$B4" "b4" "$s"
  # ROW 1 = hand-designed frequency: S1 DCT-sign + S2 SRM-residual + S3 Fo-Mixup + drop-low-bands (0-param branch)
  run_one "$SFDCT" "row1_sfdct" "$s" \
    "s/^use_dct_fomixup:.*/use_dct_fomixup: true/; s/^dct_drop_low_bands:.*/dct_drop_low_bands: 3/; s/^dct_use_sign:.*/dct_use_sign: true/; s/^dct_srm_residual:.*/dct_srm_residual: true/"
  # ROW 2 = learned frequency: S4 FcaNet learnable DCT attention + S5 single-center loss + S3 Fo-Mixup
  run_one "$SFDCT" "row2_sfdct_v2" "$s" \
    "s/^use_dct_fomixup:.*/use_dct_fomixup: true/; s/^dct_drop_low_bands:.*/dct_drop_low_bands: 3/; s/^dct_fca_attention:.*/dct_fca_attention: true/; s/^use_single_center_loss:.*/use_single_center_loss: true/"
done

echo; echo "================ Celeb-DF-v2 cross-test AUC ================"
for f in "$OUT"/*.log; do
  auc=$(grep -oE "Celeb-DF-v2:[[:space:]]+auc=[0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+$")
  printf "%-28s CDFv2 AUC = %s\n" "$(basename "$f" .log)" "${auc:-<none>}"
done
echo "==========================================================="
echo "Compare mean(sfdct_improved) vs mean(b4). A real win needs mean higher AND CI lower bound > 0."
