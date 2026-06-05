#!/usr/bin/env bash
# run_improved_ablation.sh — train the 2 NEW runs (Row1 + Row2), build figures, and PUSH model+figures to HF
# per-row (so neither row is lost). Measures Celeb-DF-v2 cross-test AUC vs the (reused) B4 baseline.
# Run on a FREE GPU box AFTER `./start.sh all` (deps+data+smoke verified).
#
# Usage:   export HF_TOKEN=hf_...; bash tools/run_improved_ablation.sh "1024"
#          RUN_B4=1 bash tools/run_improved_ablation.sh "1024 2025 7"   # also (re)train B4 baseline
# Real numbers ONLY from this full pipeline (not local smoke). HF push needs HF_TOKEN (else zips to /workspace).
set -uo pipefail            # NOT -e: one failed run must not kill the rest
cd "$(dirname "$0")/.."

SEEDS="${1:-1024}"          # default = 1 seed -> the "2 new runs" (Row1 + Row2)
SFDCT="training/config/detector/efficientnetb4_sfdct.yaml"
B4="training/config/detector/efficientnetb4.yaml"
REPO="${HF_REPO:-huanthuytnhh/deepfake}"
TS="$(date +%Y%m%d-%H%M%S)"
OUT="logs/ablation_cdfv2"; mkdir -p "$OUT" /tmp/abl viz_out
PYBIN="${PYBIN:-python}"; command -v "$PYBIN" >/dev/null 2>&1 || PYBIN=python3

push_hf () {  # $1=tag  $2=detector_yaml  $3=run_dir
  local tag="$1" y="$2" rundir="$3"
  local ck="$rundir/test/Celeb-DF-v2/ckpt_best.pth"
  [ -s "$ck" ] || { echo "  !! no ckpt for $tag ($rundir) -> skip viz/push"; return 0; }
  echo "  [$tag] building figures ..."
  "$PYBIN" training/eval_and_viz.py --detector_path "$y" --weights_path "$ck" \
      --test_dataset FaceForensics++ Celeb-DF-v2 --out "viz_out/$tag" >> "$OUT/$tag.viz.log" 2>&1 \
      || echo "  (viz failed for $tag, continuing)"
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "  [$tag] pushing model+figures -> HF $REPO/runs/$TS/$tag ..."
    RUNDIR="$rundir" TAG="$tag" TS="$TS" REPO="$REPO" "$PYBIN" - <<'PY' 2>>"$OUT/$tag.viz.log" || echo "  (HF push failed for $tag, see log)"
import os, os.path as osp
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["REPO"]
api.create_repo(repo, repo_type="model", exist_ok=True)
tag, ts, rundir = os.environ["TAG"], os.environ["TS"], os.environ["RUNDIR"]
api.upload_folder(folder_path=rundir, repo_id=repo, repo_type="model",
                  path_in_repo=f"runs/{ts}/{tag}/ckpt", allow_patterns=["*.pth", "*.pickle"])   # model + metrics
if osp.isdir(f"viz_out/{tag}"):
    api.upload_folder(folder_path=f"viz_out/{tag}", repo_id=repo, repo_type="model",
                      path_in_repo=f"runs/{ts}/{tag}/viz")                                       # figures
for lg in (f"logs/ablation_cdfv2/{tag}.train.log", f"logs/ablation_cdfv2/{tag}.viz.log"):        # logs
    if osp.isfile(lg):
        api.upload_file(path_or_fileobj=lg, path_in_repo=f"runs/{ts}/{tag}/{osp.basename(lg)}",
                        repo_id=repo, repo_type="model")
print(f"  pushed {tag} (model+figures+logs) -> https://huggingface.co/{repo}/tree/main/runs/{ts}/{tag}")
PY
  else
    echo "  [$tag] HF_TOKEN missing -> zipping locally"
    zip -qr "/workspace/${tag}_${TS}.zip" "$rundir" "viz_out/$tag" 2>/dev/null \
      && echo "  zipped /workspace/${tag}_${TS}.zip" || true
  fi
}

run_row () {  # $1=tag  $2=base_yaml  $3=seed  $4=sed_expr
  local tag="$1" base="$2" seed="$3" sed_expr="$4"
  local y="/tmp/abl/${tag}.yaml"
  cp "$base" "$y"; sed -i "s/^manualSeed:.*/manualSeed: ${seed}/" "$y"
  [ -n "$sed_expr" ] && sed -i "$sed_expr" "$y"
  echo ">>> [$tag] train FF++ c23 -> test CDFv2 (seed $seed) ..."
  "$PYBIN" training/train.py --detector_path "$y" \
      --train_dataset FaceForensics++ --test_dataset Celeb-DF-v2 \
      > "$OUT/$tag.train.log" 2>&1 || { echo "  !! $tag FAILED (see $OUT/$tag.train.log)"; return 0; }
  local rundir
  if [ "$base" = "$B4" ]; then
    rundir="$(ls -dt logs/training/efficientnetb4_*/ 2>/dev/null | grep -v sfdct | head -1)"
  else
    rundir="$(ls -dt logs/training/efficientnetb4_sfdct_*/ 2>/dev/null | head -1)"   # newest = this row
  fi
  push_hf "$tag" "$y" "${rundir%/}"
}

for s in $SEEDS; do
  [ "${RUN_B4:-0}" = "1" ] && run_row "b4_s${s}" "$B4" "$s" ""
  # ROW 1 = hand-designed frequency: S1 DCT-sign + S2 SRM-residual + S3 Fo-Mixup + drop-low-bands (0-param branch)
  run_row "row1_sfdct_s${s}" "$SFDCT" "$s" \
    "s/^use_dct_fomixup:.*/use_dct_fomixup: true/; s/^dct_drop_low_bands:.*/dct_drop_low_bands: 3/; s/^dct_use_sign:.*/dct_use_sign: true/; s/^dct_srm_residual:.*/dct_srm_residual: true/"
  # ROW 2 = learned frequency: S4 FcaNet learnable DCT attention + S5 single-center loss + S3 Fo-Mixup
  run_row "row2_sfdct_v2_s${s}" "$SFDCT" "$s" \
    "s/^use_dct_fomixup:.*/use_dct_fomixup: true/; s/^dct_drop_low_bands:.*/dct_drop_low_bands: 3/; s/^dct_fca_attention:.*/dct_fca_attention: true/; s/^use_single_center_loss:.*/use_single_center_loss: true/"
done

echo
SUMMARY="$OUT/cdfv2_summary_${TS}.txt"
{
  echo "============== Celeb-DF-v2 cross-test AUC (run $TS) =============="
  for f in "$OUT"/*.train.log; do
    [ -e "$f" ] || continue
    auc=$(grep -oE "Celeb-DF-v2:[[:space:]]+auc=[0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+$")
    printf "%-30s CDFv2 AUC = %s\n" "$(basename "$f" .train.log)" "${auc:-<none>}"
  done
  echo "================================================================"
  echo "B4 baseline (reused) bar ~0.7487. A real win = a Row's AUC strictly above your own B4 number."
} | tee "$SUMMARY"

# push the run-level summary too (so EVERYTHING is on HF: model + figures + logs + summary)
if [ -n "${HF_TOKEN:-}" ]; then
  TS="$TS" REPO="$REPO" SUMMARY="$SUMMARY" "$PYBIN" - <<'PY' 2>/dev/null || echo "(summary push failed)"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=os.environ["SUMMARY"],
                path_in_repo=f"runs/{os.environ['TS']}/cdfv2_summary.txt",
                repo_id=os.environ["REPO"], repo_type="model")
print(f"  summary pushed -> https://huggingface.co/{os.environ['REPO']}/blob/main/runs/{os.environ['TS']}/cdfv2_summary.txt")
PY
  echo "ALL results (model+figures+logs+summary) on HF: https://huggingface.co/$REPO/tree/main/runs/$TS"
fi
