#!/usr/bin/env python3
"""MT-2 · Calibrate ngưỡng vận hành τ theo eKYC (Thông tư 17/2024: FPR ≤ 5%).
(← novelty #1 mode-aware → "threshold-aware" cho deepfake)

Với mỗi model: chọn τ@FPR≤5% trên tập CALIB (một nửa giữ lại) → đo FPR/TPR/F1/ACC trên tập EVAL.
Kèm histogram phân bố score real-vs-fake + vạch τ. Cũng báo τ@FPR≤1% (chặt hơn).

Trạng thái: CHẠY ĐƯỢC NGAY (data sẵn trong viz_out/*.npz).
Output: outputs/mt02_threshold_table.{csv,md}, outputs/mt02_score_dist_<model>.png

Lưu ý trung thực: .npz không có video-id nên split calib/eval là theo frame (có thể leak
khung cùng video). Số τ vẫn hợp lệ làm "điểm vận hành"; khi có video-id nên split theo video.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

import common as C

CALIB_FRAC = 0.5
SEED = 42


def split_calib_eval(prob, label):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(label))
    k = int(len(idx) * CALIB_FRAC)
    return idx[:k], idx[k:]


def main():
    plt = C.setup_mpl()
    models = C.list_models()
    rows = []
    for m in models:
        for ds in C.datasets_of(m):
            prob, label, _ = C.load_scores(m, ds)
            ci, ei = split_calib_eval(prob, label)
            for fpr_target in (0.05, 0.01):
                thr, _, _ = C.threshold_at_fpr(label[ci], prob[ci], fpr_target)
                pred = (prob[ei] >= thr).astype(int)
                real = label[ei] == 0
                fpr_eval = float(pred[real].mean()) if real.any() else float("nan")
                tpr_eval = float(pred[label[ei] == 1].mean())
                rows.append({
                    "model": C.model_label(m), "dataset": ds,
                    "target_FPR": f"{fpr_target:.0%}", "tau": round(thr, 4),
                    "FPR_eval": round(fpr_eval, 4), "TPR_eval": round(tpr_eval, 4),
                    "F1": round(f1_score(label[ei], pred, zero_division=0), 4),
                    "ACC": round(accuracy_score(label[ei], pred), 4),
                })
            # Histogram score real vs fake + vạch τ@FPR5%
            thr5, _, _ = C.threshold_at_fpr(label[ci], prob[ci], 0.05)
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.hist(prob[label == 0], bins=50, alpha=0.65, label="REAL", color="tab:green", density=True)
            ax.hist(prob[label == 1], bins=50, alpha=0.65, label="FAKE", color="tab:red", density=True)
            ax.axvline(thr5, color="black", ls="--", lw=1.5, label=f"τ@FPR≤5% = {thr5:.3f}")
            ax.set_xlabel("prob(fake)"); ax.set_ylabel("mật độ")
            ax.set_title(f"MT-2 · Phân bố score — {C.model_label(m)} ({ds})")
            ax.legend()
            C.save_fig(fig, f"mt02_score_dist_{m}.png")
    df = pd.DataFrame(rows)
    C.save_table(df, "mt02_threshold_table.csv")
    print("[MT-2] Bảng τ eKYC + histogram score xong. Neo Thông tư 17 (FPR≤5%).")


if __name__ == "__main__":
    main()
