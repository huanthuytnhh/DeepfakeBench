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
DATA_FFPP_ID="${DATA_FFPP_ID:-1Qolh4nuuBNzu3XpoHx2l4nO4fsALNB0h}"   # FF++ data (zip) — updated link
DATA_CDF_ID="${DATA_CDF_ID:-1oSihXtB0caSGAX0Tt3MxgFbsuY46ecml}"
DATAROOT="$ROOT/datasets"
JSONDIR="$ROOT/preprocessing/dataset_json"
REPRO="./training/config/detector/efficientnetb4_repro.yaml"
SFDCT="./training/config/detector/efficientnetb4_sfdct.yaml"

log(){ printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

cmd_setup(){
  log "deps (uses the image's CUDA torch; installs the rest)"
  pip install -q -U gdown
  pip install -q efficientnet_pytorch albumentations opencv-python-headless imgaug \
                 scikit-image scikit-learn pandas tqdm pyyaml imageio einops kornia timm huggingface_hub || true
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
  log "datasets (heavy; bottleneck is Google-Drive throttle, not your link). Run inside tmux."
  mkdir -p "$DATAROOT"; ( cd "$DATAROOT"
    [ -d "$DATAROOT/FaceForensics++" ] || gdown "$DATA_FFPP_ID"
    [ -d "$DATAROOT/Celeb-DF-v2" ]     || gdown "$DATA_CDF_ID"
    shopt -s nullglob
    for f in *.zip;             do echo "unzip $f"; unzip -qn "$f"; done
    for f in *.tar *.tar.gz *.tgz; do echo "untar $f"; tar xf "$f"; done )
  ls "$DATAROOT"
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

cmd_smoke(){
  log "smoke (1 epoch, sfdct) — must finish without import/shape/NaN error and print an AUC"
  "$PYBIN" training/train.py --detector_path "$SFDCT" \
    --train_dataset FaceForensics++ --test_dataset Celeb-DF-v2 --nEpochs 1 2>&1 | tee "$ROOT/smoke.log"
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
    ck=$(ls -t logs/training/${mdl}_*/test/Celeb-DF-v2/ckpt_best.pth 2>/dev/null | head -1)
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
    cks = sorted(glob.glob(f"logs/training/{mdl}_*/test/Celeb-DF-v2/ckpt_best.pth"), key=osp.getmtime)
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

case "${1:-setup}" in
  setup)  cmd_setup ;;
  data)   cmd_data ;;
  verify) cmd_verify ;;
  smoke)   cmd_smoke ;;
  train)   cmd_train ;;
  viz)     cmd_viz ;;
  results) cmd_results ;;
  model)   cmd_model ;;
  all)     cmd_setup; cmd_data; cmd_verify; cmd_smoke; cmd_train ;;
  *) echo "usage: ./start.sh [setup|data|verify|smoke|train|viz|results|model|all]"; exit 1 ;;
esac
log "done: ${1:-setup}"
