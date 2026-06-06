#!/usr/bin/env bash
# start.sh — one-command setup + train for the Hybrid Spatial-Frequency (block-DCT) thesis on a fresh GPU box.
# Run from the repo root (where this file lives), e.g. on vast:  cd /workspace/DeepfakeBench && ./start.sh all
#
#   ./start.sh setup   # deps + B4 weight + JSONs + patch train_config + verify   (fast, safe, idempotent)
#   ./start.sh data    # download + extract FF++ & Celeb-DF-v2 (heavy; skips if already present)
#   ./start.sh verify  # a frame resolves on disk + the 2 needed detectors are registered
#   ./start.sh smoke   # 1-epoch end-to-end smoke (catch bugs before the paid full run)
#   ./start.sh train   # THIS BATCH: 2 runs in tmux (baseline EffB4 -> method SFDCT) -> viz -> auto-push results
#   ./start.sh viz     # full figure set (ROC/PR/radar/AP-bar/heatmap/t-SNE/frequency/Grad-CAM/gate) per run
#   ./start.sh results # collect LIGHT results (figures+metrics+logs) and push to git (token or SSH; no .pth/.npz)
#   ./start.sh model   # upload checkpoints (.pth) + scores + figures to Hugging Face Hub (needs HF_TOKEN)
#   ./start.sh all     # setup -> data -> verify -> smoke -> train (-> viz -> push git + upload .pth to HF)
set -euo pipefail
cd "$(dirname "$0")"; ROOT="$(pwd)"
PYBIN="$(command -v python || command -v python3)"

# --- editable: Google Drive IDs (override with env vars if needed) + paths ---
WEIGHT_URL="${WEIGHT_URL:-https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth}"
JSON_FFPP_ID="${JSON_FFPP_ID:-11BxHUbcYl10SctvS-BWaSnPtMIQTT6AY}"
JSON_CDF_ID="${JSON_CDF_ID:-1CEr_vuI8UuJkD6oAExl6_Hf6cZmYMgpm}"
DATA_FFPP_ID="${DATA_FFPP_ID:-1Qolh4nuuBNzu3XpoHx2l4nO4fsALNB0h}"   # FF++ data (zip) — Google-Drive fallback
DATA_CDF_ID="${DATA_CDF_ID:-1oSihXtB0caSGAX0Tt3MxgFbsuY46ecml}"
DATA_HF_REPO="${DATA_HF_REPO:-huanthuytnhh/deepfake-data}"          # HF dataset (fast CDN) — PREFERRED source
DATAROOT="$ROOT/datasets"
JSONDIR="$ROOT/preprocessing/dataset_json"
REPRO="./training/config/detector/efficientnetb4_repro.yaml"
SFDCT="./training/config/detector/efficientnetb4_sfdct.yaml"

log(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

cmd_setup(){
  log "deps (image's CUDA torch + the rest; uses uv = fast parallel resolver, falls back to pip)"
  # NOTE: do NOT install imgaug — it uses np.sctypes (removed in NumPy 2.0, which torch 2.11 ships) and crashes.
  # albumentations 1.3.1 wraps the imgaug import in try/except -> falls back to stubs; our aug uses no imgaug transform.
  # ALL versions PINNED to the locally smoke-validated stack (2026-06-05) so a fresh box reproduces it exactly,
  # no version drift. numpy/torch are intentionally NOT pinned here: they come from the box's CUDA template
  # (torch arch-coupled). These pinned app-libs work with BOTH numpy 1.26 (Ada/cu124) and numpy 2.0 (Blackwell/cu128).
  PKGS="gdown==6.0.0 tensorboard==2.15.2 lmdb==2.2.0 efficientnet_pytorch==0.7.1 albumentations==1.3.1 \
opencv-python-headless==4.11.0.86 scikit-image==0.25.2 scikit-learn==1.6.1 pandas==2.1.4 tqdm==4.67.1 \
pyyaml==6.0.2 imageio==2.37.3 einops==0.8.1 kornia==0.8.2 timm==1.0.27 huggingface_hub==1.17.0 hf_transfer==0.1.9"
  pip install -q -U uv >/dev/null 2>&1 || true
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PYBIN" $PKGS || pip install -q $PKGS    # uv: seconds, not minutes
  else
    pip install -q $PKGS || true
  fi
  # ---- HARDEN against the exact library errors we hit (so a fresh box never crashes mid-train) ----
  # 1) imgaug: uses np.sctypes (removed in NumPy 2.0) -> AttributeError on import. If the box's template
  #    pre-installed it, albumentations 1.3.1 will try to import it and crash. Remove it so alb falls back to stubs.
  pip uninstall -y imgaug >/dev/null 2>&1 || true
  # 2) albumentations MUST be exactly 1.3.1 (1.4+ -> ZeroDivisionError in A.OneOf([IsotropicResize],p=1);
  #    also changed ImageCompression(quality_lower/upper) API). Force it if a different version slipped in.
  AVER="$("$PYBIN" -c 'import albumentations as A;print(A.__version__)' 2>/dev/null || echo none)"
  [ "$AVER" = "1.3.1" ] || { echo "   albumentations=$AVER -> forcing 1.3.1"; pip install -q --force-reinstall --no-deps albumentations==1.3.1 || true; }
  # 3) FAIL-FAST: the real aug API path must import + run now, or stop here (NOT 2h into training).
  "$PYBIN" - <<'PY' || { echo '!! LIBRARY CHECK FAILED — fix deps before training (see error above)'; exit 1; }
import numpy as np, albumentations as A, cv2, torch, lmdb, skimage, sklearn  # all train-critical imports
img=(np.random.rand(256,256,3)*255).astype('uint8')
t=A.Compose([A.OneOf([A.GaussianBlur(p=1),A.GaussNoise(p=1)],p=1),                 # OneOf p=1 -> 1.4+ ZeroDivisionError
             A.ImageCompression(quality_lower=40,quality_upper=100,p=1)])          # quality_lower/upper API
_=t(image=img)['image']
print('   lib check OK: numpy',np.__version__,'| albumentations',A.__version__,'| imgaug absent | aug runs')
PY
  "$PYBIN" -c "import torch; assert torch.cuda.is_available(),'no CUDA'; x=torch.randn(64,64,device='cuda'); _=(x@x).sum().item(); print('torch',torch.__version__,'| cuda',torch.version.cuda,'|',torch.cuda.get_device_name(0),'-> CUDA op OK')" \
    || { echo '!! torch cannot run on this GPU (RTX 50-series/Blackwell sm_120 needs torch>=2.6 + cu126/cu128).'; \
         echo '   Fix: pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu126'; exit 1; }
  log "pretrained B4 weight"
  mkdir -p training/pretrained
  [ -s training/pretrained/efficientnet-b4-6ed6700e.pth ] || wget -q -O training/pretrained/efficientnet-b4-6ed6700e.pth "$WEIGHT_URL"
  ls -lh training/pretrained/efficientnet-b4-6ed6700e.pth
  log "dataset JSONs (FULL FF++ = 719 vids/manip)"
  mkdir -p "$JSONDIR"
  [ -s "$JSONDIR/FaceForensics++.json" ] || gdown "$JSON_FFPP_ID" -O "$JSONDIR/FaceForensics++.json"
  [ -s "$JSONDIR/Celeb-DF-v2.json" ]     || gdown "$JSON_CDF_ID"  -O "$JSONDIR/Celeb-DF-v2.json"
  "$PYBIN" - <<PY
import json;d=json.load(open("$JSONDIR/FaceForensics++.json"))["FaceForensics++"]
print("FF++ train videos/manip:",{k:len(v["train"]["c23"]) for k,v in d.items()})
PY
  log "patch train_config.yaml + test_config.yaml (these OVERRIDE the detector yaml for paths)"
  for c in training/config/train_config.yaml training/config/test_config.yaml; do
    sed -i "s#^rgb_dir:.*#rgb_dir: $DATAROOT#" "$c"
    sed -i "s#^dataset_json_folder:.*#dataset_json_folder: ./preprocessing/dataset_json#" "$c"
    sed -i "s#^lmdb:.*#lmdb: False#" "$c"        # raw frames; test_config ships lmdb:True -> would break eval/viz
    echo "-- $c --"; grep -nE "rgb_dir|dataset_json_folder|^log_dir|lmdb:" "$c"
  done
  cmd_verify
}

cmd_data(){
  log "datasets — PREFER Hugging Face $DATA_HF_REPO (fast CDN), fall back to Google Drive (slow, throttled). Run in tmux."
  command -v unzip >/dev/null 2>&1 || { echo "installing unzip..."; apt-get update -qq && apt-get install -y -qq unzip; } || true
  mkdir -p "$DATAROOT"
  if [ -d "$DATAROOT/FaceForensics++" ] && [ -d "$DATAROOT/Celeb-DF-v2" ]; then echo "data already present"; ls "$DATAROOT"; return 0; fi
  local got=0
  # 1) try the HF dataset (zips) — fast, no Drive throttle
  if [ -n "${HF_TOKEN:-}" ] && "$PYBIN" -c "import os,sys;from huggingface_hub import HfApi;sys.exit(0 if any(f.endswith('.zip') for f in HfApi(token=os.environ['HF_TOKEN']).list_repo_files('$DATA_HF_REPO',repo_type='dataset')) else 1)" 2>/dev/null; then
    echo "==> downloading data zips from HF $DATA_HF_REPO (hf_transfer, fast)"
    HF_HUB_ENABLE_HF_TRANSFER=1 "$PYBIN" - <<PY && got=1
import os
from huggingface_hub import snapshot_download
snapshot_download("$DATA_HF_REPO", repo_type="dataset", local_dir="$DATAROOT", allow_patterns=["*.zip"], token=os.environ["HF_TOKEN"])
PY
  fi
  # 2) fall back to Google Drive
  if [ "$got" = 0 ]; then
    echo "==> HF dataset not available -> Google Drive (slow). Tip: after this, run ./start.sh data-to-hf to make next time fast."
    ( cd "$DATAROOT"
      [ -d "$DATAROOT/FaceForensics++" ] || gdown "$DATA_FFPP_ID"
      [ -d "$DATAROOT/Celeb-DF-v2" ]     || gdown "$DATA_CDF_ID" ) || true
  fi
  ( cd "$DATAROOT"; shopt -s nullglob
    for f in *.zip; do
      name="${f%.zip}"                                   # Celeb-DF-v2 | FaceForensics++
      first="$(unzip -Z1 "$f" 2>/dev/null | head -1)"    # first entry in the archive
      case "$first" in
        "$name"/*) echo "unzip $f (has $name/ wrapper) -> flat";  unzip -qn "$f" ;;
        *)         echo "unzip $f (no wrapper) -> datasets/$name/"; mkdir -p "$name"; unzip -qn "$f" -d "$name" ;;
      esac
    done
    for f in *.tar *.tar.gz *.tgz; do echo "untar $f"; tar xf "$f"; done )
  echo "== datasets/ after extract =="; ls -la "$DATAROOT"
  for n in "FaceForensics++" "Celeb-DF-v2"; do
    [ -d "$DATAROOT/$n" ] && echo "$n/ OK" || echo "⚠️ no $n/ — zip layout wrong"
  done
}

cmd_data_to_hf(){
  log "ONE-TIME: upload the downloaded data ZIPs to HF dataset $DATA_HF_REPO so future rents download fast (CDN)"
  [ -z "${HF_TOKEN:-}" ] && { echo "set HF_TOKEN first (export HF_TOKEN=hf_xxx with write access)"; return 1; }
  pip install -q -U hf_transfer huggingface_hub >/dev/null 2>&1 || true
  HF_HUB_ENABLE_HF_TRANSFER=1 "$PYBIN" - <<PY
import os, glob
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = "$DATA_HF_REPO"
api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
zips = sorted(glob.glob("$DATAROOT/*.zip"))
if not zips:
    raise SystemExit("no *.zip in $DATAROOT (extracted zips may have been deleted). Keep the .zip next to the extracted folders to upload.")
for z in zips:
    print(f"uploading {os.path.basename(z)} ({os.path.getsize(z)/1e9:.1f} GB) ...")
    api.upload_file(path_or_fileobj=z, path_in_repo=os.path.basename(z), repo_id=repo, repo_type="dataset")
    print("  done:", os.path.basename(z))
print("DONE -> https://huggingface.co/datasets/" + repo + "  (next rent: ./start.sh data pulls from here, fast)")
PY
}

cmd_verify(){
  log "verify: a frame resolves + the 2 needed detectors register"
  "$PYBIN" - <<PY
import json,os
fp=list(json.load(open("$JSONDIR/FaceForensics++.json"))["FaceForensics++"]["FF-real"]["train"]["c23"].values())[0]["frames"][0]
full=os.path.join("$DATAROOT",fp)
print("sample frame:",full,"->","EXISTS" if os.path.exists(full) else "MISSING (run ./start.sh data, check extraction layout)")
PY
  "$PYBIN" - <<PY
import sys,warnings;warnings.filterwarnings("ignore");sys.path.insert(0,"training")
from detectors import DETECTOR
need=["efficientnetb4","efficientnetb4_sfdct"]
print("detectors registered:",len(DETECTOR.data),"| needed:",{k:(k in DETECTOR.data) for k in need})
PY
  log "preflight: upload credentials (fail fast BEFORE the paid 2h run)"
  if [ -n "${HF_TOKEN:-}" ]; then
    "$PYBIN" - <<'PY' || echo "  !! HF_TOKEN is SET but INVALID -> .pth upload would FAIL. Fix the token before ./start.sh train."
import os
from huggingface_hub import HfApi
u=HfApi(token=os.environ["HF_TOKEN"]).whoami()["name"]
print(f"  HF token OK -> user '{u}'. .pth will auto-upload to {os.environ.get('HF_REPO','huanthuytnhh/deepfake')}")
PY
  else
    echo "  WARNING: HF_TOKEN not set -> .pth will NOT auto-upload. Do: export HF_TOKEN=hf_xxx  (BEFORE ./start.sh train)"
  fi
  [ -n "${GH_TOKEN:-}" ] && echo "  GH_TOKEN set (figures -> git)." || echo "  (GH_TOKEN not set -> figures stay local + zipped; optional.)"
}

cmd_hftest(){
  log "HF push self-test — upload a marker to ${HF_REPO:-huanthuytnhh/deepfake} then delete it (proves the post-train push works)"
  if [ -z "${HF_TOKEN:-}" ]; then echo "  HF_TOKEN not set -> SKIP (set it so train can auto-push .pth); export HF_TOKEN=hf_xxx"; return 0; fi
  "$PYBIN" - <<'PY'
import os, io, time
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ.get("HF_REPO") or "huanthuytnhh/deepfake"
p = f"_smoketest/{time.strftime('%Y%m%d-%H%M%S')}.txt"
api.upload_file(path_or_fileobj=io.BytesIO(b"hf push pipeline ok"), path_in_repo=p, repo_id=repo, repo_type="model")
ok = p in api.list_repo_files(repo, repo_type="model")
api.delete_file(path_in_repo=p, repo_id=repo, repo_type="model")
print(f"  HF push self-test {'PASSED' if ok else 'FAILED'} -> can upload+delete on {repo} (token has write access)")
import sys; sys.exit(0 if ok else 1)
PY
}

cmd_smoke(){
  log "smoke — 1 epoch, 4 frames/vid (fast). train.py has NO --nEpochs, so override via a temp yaml copy."
  cp "$SFDCT" /tmp/sfdct_smoke.yaml
  sed -i "s/^nEpochs:.*/nEpochs: 1/; s/^frame_num:.*/frame_num: {'train': 4, 'test': 4}/" /tmp/sfdct_smoke.yaml
  "$PYBIN" training/train.py --detector_path /tmp/sfdct_smoke.yaml \
    --train_dataset FaceForensics++ --test_dataset Celeb-DF-v2 2>&1 | tee "$ROOT/smoke.log"
  cmd_hftest        # also test the HF push pipeline so an upload/token problem fails NOW, not after the 3h train
}

cmd_train(){
  log "THIS BATCH = 2 runs SEQUENTIALLY (baseline EffB4 -> method SFDCT) -> viz -> auto-push results, in one tmux"
  tmux new -d -s thesis "cd $ROOT && \
    echo '== RUN 1/2: baseline EffB4 =='; $PYBIN training/train.py --detector_path $REPRO 2>&1 | tee $ROOT/repro.log; \
    echo '== RUN 2/2: method SFDCT =='; $PYBIN training/train.py --detector_path $SFDCT 2>&1 | tee $ROOT/sfdct.log; \
    echo '== VIZ =='; ./start.sh viz; \
    echo '== PUSH LIGHT RESULTS (figures+metrics -> git) =='; ./start.sh results; \
    echo '== UPLOAD CHECKPOINTS (.pth + scores -> Hugging Face) =='; ./start.sh model; \
    echo '== ALL DONE =='"
  echo "launched tmux 'thesis' (train x2 -> viz -> push git + upload .pth to HF). watch:  tmux attach -t thesis"
  echo "ckpt at: logs/training/efficientnetb4_<ts>/test/Celeb-DF-v2/ckpt_best.pth (baseline)"
  echo "         logs/training/efficientnetb4_sfdct_<ts>/test/Celeb-DF-v2/ckpt_best.pth (method)"
  echo "~1 h/run on a 4090 -> ~2 h total, then figures->git and .pth->Hugging Face automatically."
  echo "NOTE: set creds on THIS box BEFORE training -> export GH_TOKEN=ghp_... (figures->git) and"
  echo "      export HF_TOKEN=hf_... (.pth->HF). Without them, results stay local (git commit + zip to /workspace)."
}

cmd_results(){
  log "collect + push LIGHT results (figures + metrics + logs; .pth/.npz EXCLUDED — too big for git)"
  TS="$(date +%Y%m%d-%H%M%S)"; DEST="results/$TS"; mkdir -p "$DEST/figures"
  for tag in repro sfdct; do
    if [ -d "viz_out/$tag" ]; then mkdir -p "$DEST/figures/$tag"
      cp viz_out/$tag/*.png        "$DEST/figures/$tag/" 2>/dev/null || true
      cp viz_out/$tag/results.json "$DEST/figures/$tag/" 2>/dev/null || true
    fi
  done
  cp ./*.log "$DEST/" 2>/dev/null || true
  find logs/training -name metric_dict_best.pickle -exec cp --parents {} "$DEST/" \; 2>/dev/null || true
  find logs/training -name training.log            -exec cp --parents {} "$DEST/" \; 2>/dev/null || true
  echo "results size:"; du -sh "$DEST" 2>/dev/null || true
  git add -f "$DEST" >/dev/null 2>&1 || true
  git -c user.email="vast@local" -c user.name="vast-runner" commit -q -m "results: vast run $TS (figures + metrics, light)" \
    || { echo "nothing new to commit"; return 0; }
  if [ -n "${GH_TOKEN:-}" ]; then
    git push "https://x-access-token:${GH_TOKEN}@github.com/huanthuytnhh/DeepfakeBench.git" HEAD:main && echo "results pushed to main (token)"
  elif [ -f "$HOME/.ssh/id_ed25519_github" ]; then
    GIT_SSH_COMMAND="ssh -i $HOME/.ssh/id_ed25519_github -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
      git push git@github.com:huanthuytnhh/DeepfakeBench.git HEAD:main && echo "results pushed to main (ssh)"
  else
    echo "No GH creds on this box -> results committed locally only."
    zip -qr "/workspace/results_$TS.zip" "$DEST" 2>/dev/null && echo "zipped: /workspace/results_$TS.zip (rsync it to your laptop)"
  fi
}

cmd_viz(){
  log "result figures from the newest checkpoints"
  for cfg in "$REPRO:repro" "$SFDCT:sfdct"; do
    y="${cfg%%:*}"; tag="${cfg##*:}"; mdl=$(grep -E '^model_name:' "$y" | awk '{print $2}')
    ck=$(ls -t logs/training/${mdl}_2*/test/Celeb-DF-v2/ckpt_best.pth 2>/dev/null | head -1)   # _2* = timestamp-anchored: efficientnetb4_2* won't match efficientnetb4_sfdct_*
    [ -z "$ck" ] && { echo "no ckpt for $tag ($mdl) yet"; continue; }
    "$PYBIN" training/eval_and_viz.py --detector_path "$y" --weights_path "$ck" \
      --test_dataset FaceForensics++ Celeb-DF-v2 --out "./viz_out/$tag"
  done
}

cmd_model(){
  log "upload checkpoints (.pth) + scores + figures to Hugging Face Hub (private model repo)"
  pip install -q -U huggingface_hub
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "Set a WRITE token first (headless, no OAuth):"
    echo "  export HF_TOKEN=hf_xxx     # create at https://huggingface.co/settings/tokens  (role: Write)"
    echo "  optional: export HF_REPO=<user>/<repo>   (default: huanthuytnhh/deepfake)"
    echo "Then re-run: ./start.sh model"
    return 1
  fi
  "$PYBIN" - <<'PY'
import os, glob, time, os.path as osp
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("HF_REPO") or "huanthuytnhh/deepfake"
api.create_repo(repo, repo_type="model", exist_ok=True)  # no-op if it exists; keeps its current visibility
ts = time.strftime("%Y%m%d-%H%M%S"); up = 0
for mdl in ("efficientnetb4", "efficientnetb4_sfdct"):
    cks = sorted(glob.glob(f"logs/training/{mdl}_2*/test/Celeb-DF-v2/ckpt_best.pth"), key=osp.getmtime)  # _2* anchors timestamp: efficientnetb4_2* excludes efficientnetb4_sfdct_*
    if not cks:
        print("no ckpt for", mdl); continue
    api.upload_folder(folder_path=osp.dirname(cks[-1]), repo_id=repo, repo_type="model",
                      path_in_repo=f"runs/{ts}/ckpt/{mdl}", allow_patterns=["*.pth", "*.pickle"])
    print("uploaded ckpt:", mdl); up += 1
for tag in ("repro", "sfdct"):
    if osp.isdir(f"viz_out/{tag}"):
        api.upload_folder(folder_path=f"viz_out/{tag}", repo_id=repo, repo_type="model",
                          path_in_repo=f"runs/{ts}/viz/{tag}")
        print("uploaded viz:", tag)
print(f"DONE -> https://huggingface.co/{repo}/tree/main/runs/{ts}" if up else "no checkpoints found")
PY
}

cmd_all(){
  # ONE command = WHOLE NEW pipeline, runs in tmux so it survives SSH disconnect:
  #   setup(pinned+fail-fast) -> data -> verify -> smoke(+hftest) -> Row1 + Row2 -> figures -> push EVERYTHING to HF.
  # B4 is NOT retrained (reused). Fail-fast: '&&' stops if setup/data/verify/smoke fail (before the ~9h train).
  # SEEDS env overrides seeds (default 1024). RUN_B4=1 ./start.sh all  -> also (re)train B4 here.
  command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux >/dev/null 2>&1; } || true
  local seeds="${SEEDS:-1024}"
  local CHAIN="cd $ROOT && export HF_TOKEN='${HF_TOKEN:-}' GH_TOKEN='${GH_TOKEN:-}' HF_REPO='${HF_REPO:-}' RUN_B4='${RUN_B4:-0}' && ./start.sh setup && ./start.sh data && ./start.sh verify && ./start.sh smoke && echo '== FULL TRAIN: Row1 + Row2 -> figures -> HF (B4 reused) ==' && bash tools/run_improved_ablation.sh '$seeds'"
  [ -z "${HF_TOKEN:-}" ] && echo "  !! HF_TOKEN not set -> results will ZIP to /workspace instead of HF. export it BEFORE ./start.sh all."
  if command -v tmux >/dev/null 2>&1; then
    tmux new -d -s thesis "$CHAIN 2>&1 | tee $ROOT/pipeline.log; echo '== PIPELINE DONE =='"
    sleep 1
    if tmux has-session -t thesis 2>/dev/null; then     # VERIFY it actually started (don't lie)
      echo "launched tmux 'thesis' = FULL pipeline (setup -> data -> verify -> smoke -> Row1+Row2 -> figures -> HF push). ~9h."
      echo "  watch:  tmux capture-pane -t thesis -p | tail -40     (attach: tmux attach -t thesis ; detach: Ctrl-b d)"
      return 0
    fi
  fi
  # tmux missing or failed -> nohup fallback (also survives SSH disconnect)
  echo "tmux unavailable -> running via nohup (survives disconnect)."
  nohup bash -c "$CHAIN" > "$ROOT/pipeline.log" 2>&1 &
  echo "  started PID $! = FULL pipeline. ~9h. watch:  tail -f $ROOT/pipeline.log"
}

case "${1:-setup}" in
  setup)  cmd_setup ;;
  data)       cmd_data ;;
  data-to-hf) cmd_data_to_hf ;;
  verify)     cmd_verify ;;
  smoke)   cmd_smoke ;;
  hftest)  cmd_hftest ;;
  train)   cmd_train ;;          # OLD flow (B4 + naive SFDCT). Kept for reference.
  ablation) bash "$ROOT/tools/run_improved_ablation.sh" "${2:-1024}" ;;  # Row1 + Row2 (B4 reused). RUN_B4=1 ./start.sh ablation -> also B4.
  viz)     cmd_viz ;;
  results) cmd_results ;;
  model)   cmd_model ;;
  all)     cmd_all ;;            # NEW full pipeline (Row1 + Row2 -> figures -> HF), in tmux/nohup
  all-old) cmd_setup; cmd_data; cmd_verify; cmd_smoke; cmd_train ;;
  *) echo "usage: ./start.sh [setup|data|verify|smoke|ablation|train|viz|results|model|all]"; exit 1 ;;
esac
log "done: ${1:-setup}"
