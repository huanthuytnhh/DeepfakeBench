# Liveness / Face Anti-Spoofing (FAS) — secondary module

Reuses the **existing deepfake detectors** (B4 baseline & SFDCT = B4 + block-wise DCT) for liveness,
to show the block-DCT branch helps **beyond** deepfake. Cheap: 1 small dataset, 2 runs, intra-dataset,
runs **locally on a 4 GB GPU** (AMP), ~1–2 h, $0 cloud.

## What it does
- **Dataset:** LCC-FASD (Kaggle `faber24/lcc-fasd`, ~5 GB, 18.8k images, ready `training/development/evaluation` splits). Label: **live=0, spoof=1**.
- **Models:** `efficientnetb4.yaml` (baseline) vs `efficientnetb4_sfdct.yaml` (B4+block-DCT, levers S1–S5 OFF). Same architecture/code as the deepfake runs — only the data + binary head are reused; optionally **warm-started from the trained deepfake checkpoints**.
- **Metrics:** APCER / BPCER / ACER / EER / AUC. Decision threshold fixed at **EER on dev** (no test peeking).

## Run
```bash
cd DeepfakeBench
# Kaggle creds once: put kaggle.json at ~/.kaggle/kaggle.json (chmod 600)  — https://www.kaggle.com/settings

SMOKE=1 ./tools/run_fas.sh        # 1) smoke FIRST (1 epoch, 200 imgs/split) — the rule
BATCH=16 ./tools/run_fas.sh       # 2) full: download -> B4 -> B4+DCT -> compare -> push to branch
BATCH=8  ./tools/run_fas.sh       # if CUDA OOM on 4GB
```
Results -> `logs/fas/{b4,sfdct}/metrics.json` + `logs/fas/comparison.txt`. `PUSH=1` (default) commits the
small metrics to the current branch; set `HF_TOKEN=hf_...` to also push checkpoints to HF.

## Expected ballpark (LCC-FASD, simple CNN — from verified literature)
AUC ~0.88–0.92, ACER ~16–21%, EER ~15–19% (cf. kprokofi MobileNetV3 ACER 16.33%/AUC 0.921).
**Much higher (e.g. AUC ~0.99) usually means an identity/video leak** — check the split.

## Files
| File | Role |
|---|---|
| `train_fas.py` | build B4/SFDCT, train (AMP + early-stop), eval, save `metrics.json` |
| `fas/dataset.py` | LCC-FASD loader (exact DeepfakeBench preprocessing, label from folder) |
| `fas/metrics.py` | APCER/BPCER/ACER/EER/AUC, threshold @ dev-EER |
| `fas/download.py` | kagglehub download + split summary |
| `fas/compare.py` | B4 vs B4+DCT table + Δ |
| `run_fas.sh` | one-command: download → 2 runs → compare → push to branch (+HF) |
