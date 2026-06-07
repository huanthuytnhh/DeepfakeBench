#!/usr/bin/env python3
"""MT-9b · Export CSV raw prob/label để AUDIT (← Trí có CSV dự đoán từng file).

Mỗi model×dataset → CSV: idx, prob_fake, label, pred@τ(FPR5%). Cho phép hội đồng/kiểm toán
truy ngược từng mẫu. Cũng là nguồn để vẽ lại mọi đường cong sau này.

Trạng thái: CHẠY ĐƯỢC NGAY.
Output: outputs/raw_scores/<model>__<dataset>.csv
"""
import numpy as np
import pandas as pd

import common as C


def main():
    for m in C.list_models():
        for ds in C.datasets_of(m):
            prob, label, _ = C.load_scores(m, ds)
            thr, _, _ = C.threshold_at_fpr(label, prob, C.TT17_FPR)
            df = pd.DataFrame({
                "idx": np.arange(len(prob)),
                "prob_fake": np.round(prob, 6),
                "label": label,
                "label_name": np.where(label == 1, "FAKE", "REAL"),
                "pred@FPR5%": (prob >= thr).astype(int),
            })
            out = C.OUT / "raw_scores" / f"{m}__{ds}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"[saved] {out.relative_to(C.REPO)}  ({len(df)} dòng, τ={thr:.4f})")
    print("[MT-9b] Export CSV raw xong.")


if __name__ == "__main__":
    main()
