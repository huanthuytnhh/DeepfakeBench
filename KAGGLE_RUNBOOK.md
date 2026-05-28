# DeepfakeBench EfficientNet-B4 — Kaggle Runbook

Train EfficientNet-B4 on FaceForensics++ c23, cross-test on Celeb-DF-v2. Targets Kaggle T4 16GB.

> **TL;DR**: Open [`DeepfakeBench_Kaggle.ipynb`](./DeepfakeBench_Kaggle.ipynb) directly in Kaggle (Notebook → Import). It contains all cells described below.

---

## Data sources (concrete Kaggle datasets)

DeepfakeBench requires raw videos so you can extract face crops at 256×256. These public Kaggle Datasets work (add via **+ Add Data → Search**):

| Dataset | Slug after attach | What you get |
|---|---|---|
| [FaceForensics++ (C23)](https://www.kaggle.com/datasets/xdxd003/ff-c23) | `/kaggle/input/ff-c23` | FF++ c23 videos (4 fakes + real) |
| [FaceForensics extracted frames C23](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-extracted-dataset-c23) | `/kaggle/input/faceforensics-extracted-dataset-c23` | Already-extracted frames (smaller, faster start) |
| [FaceForensics (raw)](https://www.kaggle.com/datasets/greatgamedota/faceforensics) | `/kaggle/input/faceforensics` | Bigger raw FF++ |
| [Celeb-DF-v2 (videos)](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) | `/kaggle/input/celeb-df-v2` | Celeb-DF-v2 raw videos |
| [CelebsV2 Faces 224](https://www.kaggle.com/datasets/shreyanshmanavshukla/celebsv2-faces-224) | `/kaggle/input/celebsv2-faces-224` | Pre-cropped Celeb-DF-v2 at 224×224 (re-resize to 256) |
| [FF++ & Celeb-DF combined (1000 split)](https://www.kaggle.com/datasets/nanduncs/1000-videos-split) | combined | Small combined sample, good for smoke test |

> **Note**: the layout inside each Kaggle Dataset varies. Always run `!find /kaggle/input -maxdepth 4 -type d` first to see the real folder names. Then edit `FFPP_VIDEO_ROOT` / `CDF_VIDEO_ROOT` in the notebook accordingly. The extraction script expects:
>
> - **FF++ root** must contain subfolders named: `youtube`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`. If they're nested differently, point `--video_root` to the parent.
> - **Celeb-DF-v2 root** must contain: `Celeb-real`, `Celeb-synthesis`, `YouTube-real`, and `List_of_testing_videos.txt`.

If your selected Kaggle Dataset has a different layout, either:
1. Symlink/`cp -r` to fix the layout before running extraction, or
2. Edit `FF_SUBSETS` / `CDF_SUBSETS` constants at the top of `tools/extract_faces_from_videos.py`.

---

## Pipeline overview

```
Kaggle Dataset (videos)
        │
        ▼  tools/extract_faces_from_videos.py    (MTCNN on GPU, ~3-6h FF++)
/kaggle/working/processed/
  FaceForensics++/original_sequences/youtube/c23/frames/<vid>/000.png
  FaceForensics++/manipulated_sequences/{Deepfakes,Face2Face,FaceSwap,NeuralTextures}/c23/frames/<vid>/...
  Celeb-DF-v2/{Celeb-real,Celeb-synthesis,YouTube-real}/frames/<vid>/...
        │
        ▼  tools/build_deepfakebench_json.py    (auto-downloads FF++ splits from GitHub)
preprocessing/dataset_json_v3/
  FaceForensics++.json
  FF-DF.json  FF-F2F.json  FF-FS.json  FF-NT.json
  Celeb-DF-v2.json
        │
        ▼  training/train.py
logs/evaluations/effnb4/<run>/ckpt_best.pth + metric_log.json
```

---

## What this repo provides

| File | Purpose |
|---|---|
| [`DeepfakeBench_Kaggle.ipynb`](./DeepfakeBench_Kaggle.ipynb) | One-click Kaggle notebook: clone → install → extract → JSON → train → resume → test |
| [`tools/extract_faces_from_videos.py`](./tools/extract_faces_from_videos.py) | MTCNN-based face crop extraction (Linux paths, GPU-accelerated) |
| [`tools/build_deepfakebench_json.py`](./tools/build_deepfakebench_json.py) | Build `FaceForensics++.json` + `Celeb-DF-v2.json` from cropped frames; auto-fetches official FF++ train/val/test splits |
| [`training/config/detector/efficientnetb4.yaml`](./training/config/detector/efficientnetb4.yaml) | Pre-fixed: `train_dataset=[FaceForensics++]`, `test_dataset=[FaceForensics++, Celeb-DF-v2]`, batch 16, no duplicate `save_ckpt`. Reproduces the baseline row in the paper benchmark. |

---

## Reproducibility checklist (commit before training)

1. Pin the DeepfakeBench commit: in Kaggle, run `!git rev-parse HEAD` and save the hash.
2. Save the YAML used (the notebook does this in Step 4).
3. Save full `train.log` (notebook does `tee /kaggle/working/train.log`).
4. Save final `metric_log.json` from the run dir.
5. When the next detector (DCT/frequency) is trained, keep these identical: `train_dataset`, `frame_num`, `train_batchSize`, `manualSeed`, `nEpochs`, `lr`. Only swap detector — that's clean ablation.

---

## Realistic timelines (Kaggle T4)

| Stage | Time |
|---|---|
| Clone + install | ~5 min |
| FF++ face extraction (full, 5000 videos × 32 frames) | **3-6 h** |
| Celeb-DF-v2 face extraction (~6000 videos × 32 frames) | 1-2 h |
| JSON manifest build | <2 min |
| Smoke test (1 epoch × 8 frames × batch 4) | ~10 min |
| Full train (10 epochs × 32 frames × batch 16) | **8-15 h** |
| Cross-test on Celeb-DF-v2 | 20-40 min |

→ Expect **2-3 Kaggle sessions** end-to-end. The resume cell in the notebook handles this.

## Suggested two-session split

1. **Session A** (extract + JSON + smoke). Save `/kaggle/working/processed/` and `preprocessing/dataset_json_v3/` as a **new Kaggle Dataset** of yours (Output → Save as Dataset). This avoids re-extracting next time.
2. **Session B** (full train): attach the dataset from Session A as input, change `PROCESSED_ROOT` to the attached path, skip extraction cells, run training.
3. **Session C** (if needed) — resume.

---

## Expected results

| Train | Test (within) | Test (cross) |
|---|---|---|
| FF++ c23 (EfficientNet-B4, 10 epoch) | AUC ~0.95 (FF++) | AUC ~0.74-0.80 (Celeb-DF-v2) |

This matches the published baseline row in the DeepfakeBench paper. If your cross-test AUC < 0.65 → manifest / label mismatch; not a model bug.

---

## Pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `dataset X not exist!` | Missing `dataset_json_v3/X.json` | Re-run Step 3 (notebook) |
| `KeyError: 'CelebDFv2_fake'` | Label naming in JSON doesn't match `label_dict` | Inspect the JSON keys, edit `train_config.yaml::label_dict` accordingly |
| `FileNotFoundError: train.json` | FF++ official splits not downloaded | `build_deepfakebench_json.py` auto-downloads them; check Internet is ON on Kaggle |
| OOM at batch 16 | T4 VRAM tight | Lower `train_batchSize` to 8; record it for ablation parity |
| Loss NaN at iter 100 | lr too high | Lower to `0.0001` |
| Mid-training Kaggle 12h timeout | Expected | Use the resume cell in Step 6 |
| `FF++ frames dir empty` after extraction | Kaggle Dataset layout differs from `FF_SUBSETS` | Inspect with `find`, adjust `tools/extract_faces_from_videos.py::FF_SUBSETS` |

---

## Where to go next

- Train a **frequency-domain detector** (F3Net / SPSL / SRM) on the same data using the same config style — only change `--detector_path`.
- Use the trained ckpt to evaluate on **more cross-domain datasets** (DFDC, DeeperForensics) once you have those Kaggle Datasets attached.
- Compare your numbers to the [DeepfakeBench paper Table 3](https://arxiv.org/abs/2307.01426) (EfficientNet-B4 row).
