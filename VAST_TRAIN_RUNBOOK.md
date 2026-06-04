# VAST_TRAIN_RUNBOOK — clone → data → train (corrected)

Supersedes the rough draft. Fixes the real landmines: **code must be pushed first**, vast paths
(`/workspace`, not `~/Desktop`), JSON/data locations the config actually expects, real deps
(repo `requirements.txt` is empty), gdown large-file handling, and a smoke gate before the paid full run.

Confirmed protocol (matches official SCLBD/DeepfakeBench `efficientnetb4.yaml`):
train **full FF++ c23** (719 vids/manip, ~115k frames), batch 32, frame_num 32/32, workers 8, 10 epochs,
adam lr 2e-4, seed 1024. Configs `efficientnetb4_repro.yaml` (baseline R) + `efficientnetb4_sfdct.yaml` (method P).

---

## STEP 0 — Push the thesis code to the fork (BLOCKER — do this on the LOCAL machine first)

A fresh `git clone` only gets what is **pushed**. Right now the SFDCT code is uncommitted, so the clone
would have NO `efficientnetb4_sfdct`. From the local repo:

```bash
cd ~/Desktop/thanhln/datn/DeepfakeBench
git add training/detectors/sfdct_core.py \
        training/detectors/efficientnetb4_sfdct_detector.py \
        training/detectors/__init__.py \
        training/config/detector/efficientnetb4_sfdct.yaml \
        training/config/detector/efficientnetb4_repro.yaml \
        training/dataset/abstract_dataset.py \
        training/train.py training/eval_and_viz.py \
        tools/smoke_sfdct.py tools/bench_gpu_sfdct.py tools/viz_dct.py tools/viz_gate.py
git commit -m "feat: hybrid spatial-frequency (block-DCT) detector + matched protocol configs + result-viz"
git push origin main
```
- **Do NOT commit** `efficientnetb4.yaml` / `train_config.yaml` / `test_config.yaml` — those were shrunk to the
  25-video smoke subset locally; leaving them out keeps the fork on the FULL official protocol.
- `abstract_dataset.py` is included because it carries the **critical Linux path fix** (`os.path.join` instead of
  the Windows `\\`) — without it, image reads fail on vast.

---

## STEP 1 — Rent + SSH (vast.ai)
- Instance: **RTX 4090 (24 GB)**, image **PyTorch 2.x / CUDA 12.x**, disk **≥ 150 GB** (FF++ preprocessed is large).
- `ssh -p <PORT> root@<VAST_IP>`  ← fill per instance (also update CLAUDE.md “Remote” block).

## STEP 2 — Clone + environment + deps (on vast)
```bash
cd /workspace
git clone https://github.com/huanthuytnhh/DeepfakeBench.git
cd DeepfakeBench

# Use the image's existing CUDA-enabled torch. Verify it sees the GPU:
python -c "import torch;print('cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0))"

# Install the deps the repo actually needs (requirements.txt is empty). torch-dct is NOT needed
# (sfdct_core is pure torch). opencv-headless avoids libGL errors on headless servers.
pip install -U gdown
pip install efficientnet_pytorch albumentations opencv-python-headless imgaug \
            scikit-image scikit-learn pandas tqdm pyyaml imageio einops kornia
```
> If you prefer an isolated env, `uv venv --python 3.10 && source .venv/bin/activate` THEN install
> CUDA torch explicitly: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
> (a bare `uv venv` will otherwise give you CPU-only torch and training will crawl).

## STEP 3 — Pretrained B4 weight (config points at it)
```bash
mkdir -p /workspace/DeepfakeBench/training/pretrained
wget -O /workspace/DeepfakeBench/training/pretrained/efficientnet-b4-6ed6700e.pth \
  https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth
```

## STEP 4 — JSONs into the folder the config expects
The detector configs read `./preprocessing/dataset_json`. Put the FULL jsons there (NOT a top-level `dataset_json`):
```bash
cd /workspace/DeepfakeBench/preprocessing/dataset_json
gdown 11BxHUbcYl10SctvS-BWaSnPtMIQTT6AY -O FaceForensics++.json   # full FF++ (719 vids/manip)
gdown 1CEr_vuI8UuJkD6oAExl6_Hf6cZmYMgpm -O Celeb-DF-v2.json
# sanity: must print 719 per manip
python - <<'PY'
import json;d=json.load(open("FaceForensics++.json"))["FaceForensics++"]
print({k:len(v["train"]["c23"]) for k,v in d.items()})
PY
```

## STEP 5 — Data (heavy; run in tmux, then EXTRACT)
JSON frame paths look like `FaceForensics++/original_sequences/youtube/c23/frames/001/000.png`, so the
extracted folders must sit directly under one data root. We use `/workspace/DeepfakeBench/datasets`.
```bash
tmux new -s dl
cd /workspace/DeepfakeBench/datasets        # create if missing: mkdir -p
gdown 1mZ9NNtgW_4oo9S996uQh9-SmRYaLxPnb     # FF++  (large)
gdown 1oSihXtB0caSGAX0Tt3MxgFbsuY46ecml     # Celeb-DF-v2
# Extract (archives, not raw folders). Adjust to the actual file names:
for f in *.zip; do unzip -q "$f"; done ;  for f in *.tar *.tar.gz; do tar xf "$f"; done
ls            # EXPECT: FaceForensics++/  Celeb-DF-v2/
# detach: Ctrl-b then d   (gdown on huge files can hit Google quota — if so, retry later / use a mirror)
```

## STEP 6 — Set data paths in `train_config.yaml` (it OVERRIDES the detector yaml for paths!)
> `train.py` does `config.update(train_config)`, so for `rgb_dir` / `dataset_json_folder` / `log_dir` / `lmdb`
> the **train_config.yaml value wins** — editing the detector yaml's path keys has NO effect. Set them here:
```bash
cd /workspace/DeepfakeBench
sed -i 's#^rgb_dir:.*#rgb_dir: /workspace/DeepfakeBench/datasets#' training/config/train_config.yaml
sed -i 's#^dataset_json_folder:.*#dataset_json_folder: ./preprocessing/dataset_json#' training/config/train_config.yaml  # FULL, not the 25-vid subset
grep -nE "rgb_dir|dataset_json_folder|^log_dir|lmdb:" training/config/train_config.yaml   # ensure lmdb: False, paths correct
# verify a real frame resolves:
python - <<'PY'
import json,os
d=json.load(open("preprocessing/dataset_json/FaceForensics++.json"))["FaceForensics++"]
fp=list(d["FF-real"]["train"]["c23"].values())[0]["frames"][0]
full=os.path.join("/workspace/DeepfakeBench/datasets",fp)
print(full, "EXISTS" if os.path.exists(full) else "❌ MISSING — fix extraction layout")
PY
```

> ⚠ **Run from the REPO ROOT** (`/workspace/DeepfakeBench`), exactly like the official README
> (`python training/train.py --detector_path ./training/config/detector/...`). Do NOT `cd training` —
> the code hardcodes `./training/config/test_config.yaml` and resolves `./datasets` / `./preprocessing`
> relative to the root.

## STEP 7 — SMOKE on vast BEFORE the full run (mandatory)
Tiny, fast — proves the whole pipeline (data → model → loss) end to end:
```bash
cd /workspace/DeepfakeBench
python training/train.py --detector_path ./training/config/detector/efficientnetb4_sfdct.yaml \
  --train_dataset FaceForensics++ --test_dataset Celeb-DF-v2 \
  --nEpochs 1 2>&1 | tee /workspace/smoke.log
# (optionally edit the yaml frame_num to {train:4,test:4} for an even faster smoke)
```
Pass = loads data, 1 epoch runs, no NaN/shape/import error, prints an AUC.

## STEP 8 — FULL runs in tmux (baseline R, then method P)
```bash
cd /workspace/DeepfakeBench
# (R) reproduce the leaderboard baseline — expect within-avg ≈ 0.9389, CDFv2 ≈ 0.7487 (±0.02)
tmux new -d -s repro 'cd /workspace/DeepfakeBench && python training/train.py --detector_path ./training/config/detector/efficientnetb4_repro.yaml 2>&1 | tee /workspace/repro.log'
# (P) the thesis method — same protocol, only the SFDCT module differs
tmux new -d -s sfdct 'cd /workspace/DeepfakeBench && python training/train.py --detector_path ./training/config/detector/efficientnetb4_sfdct.yaml 2>&1 | tee /workspace/sfdct.log'
# watch:  tmux attach -t sfdct      results land in:  logs/evaluations/effnb4_*/<model>_<timestamp>/
```
**Where the model lands** (`--save_ckpt` default True; base = train_config `log_dir: ./logs/training/`, which
overrides the detector yaml). The ckpt is saved for the best **cross-dataset** epoch (FFpp_pool excludes FF++):
```
logs/training/efficientnetb4_sfdct_<timestamp>/test/Celeb-DF-v2/ckpt_best.pth     ← the model
logs/training/efficientnetb4_sfdct_<timestamp>/test/<dataset>/metric_dict_best.pickle   ← AUC/EER
logs/training/efficientnetb4_sfdct_<timestamp>/training.log  +  .../metric_board/ (TensorBoard)
```
⚠ Keep a non-FF++ set (Celeb-DF-v2) in `test_dataset` — if `test_dataset` is FF++ only, **no checkpoint is saved**.
Time on a 4090 (measured-anchored): **~5 min/epoch, ~1 h per 10-epoch run.** Run baseline first.

## STEP 8.5 — Generate result figures from the trained checkpoint (visualization)
```bash
cd /workspace/DeepfakeBench
CK=$(ls -t logs/training/efficientnetb4_sfdct_*/test/Celeb-DF-v2/ckpt_best.pth | head -1)   # newest sfdct ckpt
python training/eval_and_viz.py \
  --detector_path ./training/config/detector/efficientnetb4_sfdct.yaml \
  --weights_path "$CK" --test_dataset FaceForensics++ Celeb-DF-v2 --out ./viz_out/sfdct
# → roc_ekyc.png (FPR≤5% line + TPR@FPR), auc_bar.png, tsne.png, gate_alpha.png (the zero-init gate after training),
#   results.json (auc/eer/ap/video-auc/TPR@FPR), scores_*.npz (raw prob/label/feat for any later plot)
# B1 alone, anytime, from just the .pth (no data/deps):  python tools/viz_gate.py --ckpt "$CK"
# Repeat for the baseline ckpt with efficientnetb4_repro.yaml → ./viz_out/repro (baseline-vs-method side by side).
```

## STEP 9 — PERSIST before you destroy the instance (vast is ephemeral!)
> A **destroyed** vast instance wipes the disk. Pull EVERYTHING back (or push to durable storage) **before** destroying.
```bash
# from the LOCAL machine — pull checkpoints, metrics, figures, logs:
mkdir -p ~/Desktop/thanhln/datn/vast_results
rsync -avz -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/DeepfakeBench/logs       ~/Desktop/thanhln/datn/vast_results/
rsync -avz -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/DeepfakeBench/viz_out    ~/Desktop/thanhln/datn/vast_results/
rsync -avz -e "ssh -p <PORT>" root@<VAST_IP>:/workspace/*.log                    ~/Desktop/thanhln/datn/vast_results/
```
Optional durable backup straight from vast (survives instance destroy): zip + upload to Google Drive
```bash
# on vast:
cd /workspace/DeepfakeBench && zip -r /workspace/run_artifacts.zip logs viz_out ../*.log
pip install gdown            # upload needs a configured rclone OR: download the zip via the vast file browser
# simplest: from LOCAL, the rsync above already saved everything — verify the .pth files arrived:
find ~/Desktop/thanhln/datn/vast_results -name "ckpt_best.pth" -o -name "*.png" | head

---

### Quick failure map
| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: efficientnetb4_sfdct` | code not pushed (STEP 0) | commit+push, re-clone |
| `... does not exist` on a frame | rgb_dir / extraction layout | STEP 6 verify; fix folder names to match JSON |
| `libGL.so.1` error | opencv-python (not headless) | `pip install opencv-python-headless` |
| training crawls (CPU) | CPU-only torch in fresh venv | install cu121 torch (STEP 2 note) |
| gdown “quota exceeded” | Google Drive cap on big file | retry later, or mirror the archive |
| baseline AUC ≪ 0.93 | trained on subset / wrong data | confirm 719 vids/manip (STEP 4) + lmdb:False |
