#!/usr/bin/env bash
# run_fix_ablation.sh — REDESIGNED improved rows after Row1/Row2 (stacked levers) underperformed naive.
#
# Diagnosis of the old rows (both < naive SFDCT 0.7572):
#   Row1 = Fo-Mixup + drop_low_bands:3 + sign + SRM-residual  -> 0.7332  (SRM REPLACED the winning YCbCr
#          block-DCT input; Fo-Mixup over-regularized; drop:3 too aggressive; 4 levers stacked/confounded)
#   Row2 = Fo-Mixup + drop_low_bands:3 + FcaNet + single-center -> ~low   (extra params overfit FF++; +Fo-Mixup)
#
# Fix strategy: KEEP the winning naive config (YCbCr block-DCT + zero-init gated fusion, all levers OFF),
# DROP the culprits (Fo-Mixup, SRM, aggressive drop, stacking), add ONE gentle cue per row so any gain is
# cleanly attributable. NOT guaranteed to beat naive — naive 0.7572 (> B4 0.7497) is the safety-net method.
#
# Usage:   export HF_TOKEN=hf_...; bash tools/run_fix_ablation.sh "1024"
# Pushes model+figures+config+logs per row to HF runs/fix-<ts>/<tag>/ (same layout as the improved run).
set -uo pipefail            # NOT -e: one failed run must not kill the rest
cd "$(dirname "$0")/.."

SEEDS="${1:-1024}"
SFDCT="training/config/detector/efficientnetb4_sfdct.yaml"
REPO="${HF_REPO:-huanthuytnhh/deepfake}"
TS="fix-$(date +%Y%m%d-%H%M%S)"        # -> runs/fix-<ts>/ : separate from naive (runs/<ts>/) and improved (runs/improved-<ts>/)
OUT="logs/ablation_cdfv2"; mkdir -p "$OUT" /tmp/abl viz_out
PYBIN="${PYBIN:-python}"; command -v "$PYBIN" >/dev/null 2>&1 || PYBIN=python3

push_hf () {  # $1=tag  $2=detector_yaml  $3=run_dir
  local tag="$1" y="$2" rundir="$3"
  local ck="$rundir/test/Celeb-DF-v2/ckpt_best.pth"
  [ -s "$ck" ] || { echo "  !! no ckpt for $tag ($rundir) -> skip viz/push"; return 0; }
  echo "  [$tag] building figures ..."
  mkdir -p "viz_out/$tag"
  "$PYBIN" training/eval_and_viz.py --detector_path "$y" --weights_path "$ck" \
      --test_dataset FaceForensics++ Celeb-DF-v2 --out "viz_out/$tag" >> "$OUT/$tag.viz.log" 2>&1 \
      || echo "  (viz failed for $tag, continuing)"
  "$PYBIN" tools/plot_training_curve.py --log "$OUT/$tag.train.log" \
      --out "viz_out/$tag/training_curve.png" --title "$tag" >> "$OUT/$tag.viz.log" 2>&1 || true
  if [ -n "${HF_TOKEN:-}" ]; then
    echo "  [$tag] pushing model+figures -> HF $REPO/runs/$TS/$tag ..."
    RUNDIR="$rundir" TAG="$tag" TS="$TS" REPO="$REPO" CFGYAML="$y" "$PYBIN" - <<'PY' 2>>"$OUT/$tag.viz.log" || echo "  (HF push failed for $tag, see log)"
import os, os.path as osp
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ["REPO"]
api.create_repo(repo, repo_type="model", exist_ok=True)
tag, ts, rundir = os.environ["TAG"], os.environ["TS"], os.environ["RUNDIR"]
api.upload_folder(folder_path=rundir, repo_id=repo, repo_type="model",
                  path_in_repo=f"runs/{ts}/{tag}/ckpt", allow_patterns=["*.pth", "*.pickle"])
if osp.isdir(f"viz_out/{tag}"):
    api.upload_folder(folder_path=f"viz_out/{tag}", repo_id=repo, repo_type="model",
                      path_in_repo=f"runs/{ts}/{tag}/viz")
cfg = os.environ.get("CFGYAML", "")
if cfg and osp.isfile(cfg):
    api.upload_file(path_or_fileobj=cfg, path_in_repo=f"runs/{ts}/{tag}/config.yaml",
                    repo_id=repo, repo_type="model")
for lg in (f"logs/ablation_cdfv2/{tag}.train.log", f"logs/ablation_cdfv2/{tag}.viz.log"):
    if osp.isfile(lg):
        api.upload_file(path_or_fileobj=lg, path_in_repo=f"runs/{ts}/{tag}/{osp.basename(lg)}",
                        repo_id=repo, repo_type="model")
print(f"  pushed {tag} -> https://huggingface.co/{repo}/tree/main/runs/{ts}/{tag}")
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
  local rundir="$(ls -dt logs/training/efficientnetb4_sfdct_*/ 2>/dev/null | head -1)"   # newest = this row
  push_hf "$tag" "$y" "${rundir%/}"
}

for s in $SEEDS; do
  # FIX-1 (0-param, gentle): naive + DCT-sign (SPSL phase) + drop ONLY the DC band (drop_low_bands:1).
  #   Keeps the winning YCbCr block-DCT input; NO Fo-Mixup, NO SRM-residual, NO aggressive drop.
  run_row "fix1_sign_droplow1_s${s}" "$SFDCT" "$s" \
    "s/^dct_use_sign:.*/dct_use_sign: true/; s/^dct_drop_low_bands:.*/dct_drop_low_bands: 1/"
  # FIX-2 (single learnable): naive + FcaNet multi-spectral attention ONLY.
  #   NO Fo-Mixup, NO single-center loss -> isolate whether learnable DCT attention alone helps.
  run_row "fix2_fca_s${s}" "$SFDCT" "$s" \
    "s/^dct_fca_attention:.*/dct_fca_attention: true/"
done

echo
SUMMARY="$OUT/cdfv2_summary_${TS}.txt"
{
  echo "============== Celeb-DF-v2 cross-test AUC (run $TS) =============="
  for f in "$OUT"/fix*.train.log; do
    [ -e "$f" ] || continue
    auc=$(grep -oE "Celeb-DF-v2:[[:space:]]+auc=[0-9.]+" "$f" | tail -1 | grep -oE "[0-9.]+$")
    printf "%-34s CDFv2 AUC = %s\n" "$(basename "$f" .train.log)" "${auc:-<none>}"
  done
  echo "----------------------------------------------------------------"
  echo "Bars to beat:  B4 = 0.7497   |   naive SFDCT = 0.7572 (current best/safety-net method)"
  echo "A real win = a fix row's AUC STRICTLY above 0.7572."
  echo "================================================================"
} | tee "$SUMMARY"

if [ -n "${HF_TOKEN:-}" ]; then
  TS="$TS" REPO="$REPO" SUMMARY="$SUMMARY" "$PYBIN" - <<'PY' 2>/dev/null || echo "(summary push failed)"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=os.environ["SUMMARY"],
                path_in_repo=f"runs/{os.environ['TS']}/cdfv2_summary.txt",
                repo_id=os.environ["REPO"], repo_type="model")
print(f"  summary -> https://huggingface.co/{os.environ['REPO']}/blob/main/runs/{os.environ['TS']}/cdfv2_summary.txt")
PY
  echo "ALL fix results on HF: https://huggingface.co/$REPO/tree/main/runs/$TS"
fi
