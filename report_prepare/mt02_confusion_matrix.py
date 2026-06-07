#!/usr/bin/env python3
"""MT-2b · Confusion matrix THẬT (real/fake) tại ngưỡng vận hành τ + classification_report.
(← novelty #1; SỬA lỗi báo cáo: 'Hình 3.9 confusion matrix' hiện đang là heatmap metric×dataset)

Với mỗi model: τ = ngưỡng @FPR≤5% (toàn tập, làm điểm vận hành) → CM 2×2 normalize='true'
+ bảng Precision/Recall/F1 per-class (REAL/FAKE).

Trạng thái: CHẠY ĐƯỢC NGAY.
Output: outputs/mt02_confusion_<model>.png, outputs/mt02_per_class.{csv,md}
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import common as C

LABELS = ["REAL", "FAKE"]


def main():
    plt = C.setup_mpl()
    rows = []
    for m in C.list_models():
        for ds in C.datasets_of(m):
            prob, label, _ = C.load_scores(m, ds)
            thr, _, _ = C.threshold_at_fpr(label, prob, C.TT17_FPR)
            pred = (prob >= thr).astype(int)

            cm = confusion_matrix(label, pred, labels=[0, 1], normalize="true")
            fig, ax = plt.subplots(figsize=(4.6, 4))
            ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", values_format=".3f", colorbar=False)
            ax.set_title(f"MT-2 · Confusion (norm='true') @τ={thr:.3f}\n{C.model_label(m)} · {ds}")
            C.save_fig(fig, f"mt02_confusion_{m}.png")

            rep = classification_report(label, pred, labels=[0, 1], target_names=LABELS,
                                        output_dict=True, zero_division=0)
            for cls in LABELS:
                rows.append({
                    "model": C.model_label(m), "dataset": ds, "class": cls, "tau": round(thr, 4),
                    "precision": round(rep[cls]["precision"], 4),
                    "recall": round(rep[cls]["recall"], 4),
                    "f1": round(rep[cls]["f1-score"], 4),
                    "support": int(rep[cls]["support"]),
                })
    C.save_table(pd.DataFrame(rows), "mt02_per_class.csv")
    print("[MT-2b] Confusion matrix THẬT + per-class xong. → thay 'Hình 3.9' trong báo cáo.")


if __name__ == "__main__":
    main()
