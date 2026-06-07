#!/usr/bin/env python3
"""MT-4b · Bảng cross-dataset chính thức (mở rộng) + heatmap AUC train×test.
(← novelty #4 cross-dataset của Trí, nhưng theo GIAO THỨC CHUẨN deepfake: frame/video-AUC)

PHẦN BẢNG TỪ results.json: CHẠY ĐƯỢC NGAY (hiện chỉ Celeb-DF-v2).
PHẦN EVAL THÊM (DFDC/DeeperForensics): CODE-FIRST — chạy eval_and_viz trên test-set mới rồi
để vào viz_out/<model>/, script này tự gom.

Chạy mai để thêm bộ test mới:
  python training/eval_and_viz.py --detector_path <cfg> --weights_path <ckpt> \
      --test_dataset DFDC --out viz_out/<model>
  → rồi chạy lại: python report_prepare/mt04_cross_dataset.py

Output: outputs/mt04_cross_dataset.{csv,md}, outputs/mt04_cross_heatmap.png
"""
import numpy as np
import pandas as pd

import common as C

# dataset coi là "within-domain" (train trên đó) để phân biệt cross
WITHIN = {"FaceForensics++", "FF++"}


def main():
    plt = C.setup_mpl()
    models = C.list_models()
    rows = []
    all_ds = []
    for m in models:
        for ds, r in C.load_results(m).items():
            all_ds.append(ds)
            rows.append({"model": C.model_label(m), "test_set": ds,
                         "kind": "within" if ds in WITHIN else "cross",
                         "frame_auc": round(r.get("frame_auc", float("nan")), 4),
                         "video_auc": round(r.get("video_auc", float("nan")), 4),
                         "eer": round(r.get("eer", float("nan")), 4),
                         "tpr@fpr5%": round(r.get("tpr@fpr=5%", float("nan")), 4)})
    df = pd.DataFrame(rows)
    C.save_table(df, "mt04_cross_dataset.csv")

    # Heatmap AUC: model × test_set (tự mở rộng khi thêm DFDC...)
    datasets = sorted(set(all_ds))
    if datasets:
        M = np.full((len(models), len(datasets)), np.nan)
        for i, m in enumerate(models):
            res = C.load_results(m)
            for j, ds in enumerate(datasets):
                if ds in res:
                    M[i, j] = res[ds].get("frame_auc", np.nan)
        fig, ax = plt.subplots(figsize=(2 + 1.8 * len(datasets), 1 + 0.7 * len(models)))
        im = ax.imshow(M, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=20, ha="right")
        ax.set_yticks(range(len(models))); ax.set_yticklabels([C.model_label(m) for m in models])
        for i in range(len(models)):
            for j in range(len(datasets)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center",
                            color="white" if M[i, j] < 0.8 else "black")
        ax.set_title("MT-4 · Cross-dataset frame-AUC (model × test-set)")
        fig.colorbar(im, ax=ax, fraction=0.03, label="AUC")
        C.save_fig(fig, "mt04_cross_heatmap.png")
    n_cross = sum(1 for d in set(all_ds) if d not in WITHIN)
    print(f"[MT-4b] {len(set(all_ds))} test-set ({n_cross} cross). Thêm DFDC/DeeperForensics bằng eval_and_viz để mở rộng.")


if __name__ == "__main__":
    main()
