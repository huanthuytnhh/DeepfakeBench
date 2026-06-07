#!/usr/bin/env python3
"""MT-1 · Ma trận đánh giá đa-cấu hình + heatmap tổng hợp.
(← novelty #2 ma trận 16-config + #1 mode-aware của Trí)

Đọc viz_out/*/results.json (mọi model × mọi test-set) → bảng tổng + heatmap AUC.
Khi thêm cross-dataset (DFDC/DeeperForensics) hoặc model mới, heatmap tự mở rộng.

Trạng thái: CHẠY ĐƯỢC NGAY (data sẵn trong viz_out/).
Output: outputs/mt01_eval_matrix.{csv,md}, outputs/mt01_heatmap_auc.png
"""
import numpy as np
import pandas as pd

import common as C


def main():
    plt = C.setup_mpl()
    models = C.list_models()
    if not models:
        print("[MT-1] Chưa có model nào trong viz_out/."); return

    rows, datasets = [], []
    for m in models:
        res = C.load_results(m)
        for ds, r in res.items():
            datasets.append(ds)
            rows.append({
                "model": C.model_label(m), "dataset": ds,
                "frame_auc": r.get("frame_auc"), "video_auc": r.get("video_auc"),
                "ap": r.get("ap"), "eer": r.get("eer"),
                "tpr@fpr5%": r.get("tpr@fpr=5%"), "tpr@fpr1%": r.get("tpr@fpr=1%"),
                "n": r.get("n"),
            })
    df = pd.DataFrame(rows)
    C.save_table(df.round(4), "mt01_eval_matrix.csv")

    # Heatmap: hàng = model×dataset, cột = metric (mỗi cột chuẩn hoá min-max để tô màu, annotate giá trị thật)
    metric_cols = ["frame_auc", "video_auc", "ap", "tpr@fpr5%", "tpr@fpr1%"]
    H = df.set_index([df["model"] + " · " + df["dataset"]])[metric_cols].astype(float)
    norm = (H - H.min()) / (H.max() - H.min()).replace(0, 1)
    fig, ax = plt.subplots(figsize=(1.6 * len(metric_cols) + 3, 0.7 * len(H) + 2))
    im = ax.imshow(norm.values, cmap="crest" if _has_crest() else "viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_cols))); ax.set_xticklabels(metric_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index)
    for i in range(len(H)):
        for j in range(len(metric_cols)):
            v = H.values[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.3f}", ha="center", va="center",
                    color="white" if norm.values[i, j] > 0.55 else "black", fontsize=9)
    ax.set_title("MT-1 · Ma trận metric (màu = chuẩn hoá theo cột; số = giá trị thật)")
    fig.colorbar(im, ax=ax, fraction=0.025, label="normalized")
    C.save_fig(fig, "mt01_heatmap_auc.png")

    print(f"[MT-1] {len(models)} model × {len(set(datasets))} dataset.")
    print("       Mở rộng: thêm cross-dataset (DFDC...) bằng cách eval thêm rồi để vào viz_out/<model>/.")


def _has_crest():
    try:
        import seaborn  # noqa
        import matplotlib.pyplot as plt
        return "crest" in plt.colormaps()
    except Exception:
        return False


if __name__ == "__main__":
    main()
