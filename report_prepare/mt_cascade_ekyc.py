#!/usr/bin/env python3
"""G3.2 · Thí nghiệm CASCADE liveness→deepfake: "lọc qua liveness có tăng bắt deepfake @FPR≤5% không?"
(câu hỏi của bạn). So 3 cấu hình ở CÙNG ngân sách FPR≤5%:
  (a) deepfake-only · (b) liveness-only · (c) OR-combine (lọc liveness HOẶC deepfake).

Cần 1 file npz HỢP NHẤT trên CÙNG mẫu (lý tưởng = bộ VN sau G2, mỗi mặt có cả 2 điểm):
  keys: prob_live (P[spoof]), prob_fake (P[deepfake]), label (1=ATTACK gồm spoof|deepfake, 0=bonafide).
→ `python3 report_prepare/mt_cascade_ekyc.py --npz <combined.npz>`
Chưa có file → in hướng dẫn + chạy DEMO tổng hợp để minh hoạ phương pháp.
Output: outputs/mt_cascade.{csv,md}
"""
import argparse
import numpy as np
import pandas as pd

import common as C

FPR_BUDGET = 0.05


def tpr_at_fpr_single(prob, label, budget=FPR_BUDGET):
    """1 detector: chọn τ sao cho FPR(bonafide→attack)≤budget → trả (TPR, τ, FPR_đạt)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(label, prob, pos_label=1)
    ok = np.where(fpr <= budget)[0]
    i = ok[-1] if len(ok) else 0
    return float(tpr[i]), float(thr[i]), float(fpr[i])


def tpr_or_combine(prob_live, prob_fake, label, budget=FPR_BUDGET, grid=60):
    """OR-combine: ATTACK nếu prob_live≥τl HOẶC prob_fake≥τf. Quét lưới (τl,τf), giữ tổ hợp có
    FPR tổng ≤ budget và TPR lớn nhất. Trả (TPR, τl, τf, FPR_đạt)."""
    real = label == 0; atk = label == 1
    ql = np.quantile(prob_live, np.linspace(0, 1, grid))
    qf = np.quantile(prob_fake, np.linspace(0, 1, grid))
    best = (0.0, None, None, None)
    for tl in ql:
        flag_l = prob_live >= tl
        for tf in qf:
            pred = flag_l | (prob_fake >= tf)
            fpr = float(pred[real].mean()) if real.any() else 1.0
            if fpr <= budget:
                tpr = float(pred[atk].mean()) if atk.any() else 0.0
                if tpr > best[0]:
                    best = (tpr, float(tl), float(tf), fpr)
    return best


def run(prob_live, prob_fake, label, src):
    rows = []
    tpr_d, td, fd = tpr_at_fpr_single(prob_fake, label)
    rows.append({"cấu hình": "(a) deepfake-only", "TPR@FPR≤5%": round(tpr_d, 4), "ghi chú": f"τ_fake={td:.3f}, FPR={fd:.3f}"})
    tpr_l, tl, fl = tpr_at_fpr_single(prob_live, label)
    rows.append({"cấu hình": "(b) liveness-only", "TPR@FPR≤5%": round(tpr_l, 4), "ghi chú": f"τ_live={tl:.3f}, FPR={fl:.3f}"})
    tpr_c, tlc, tfc, fc = tpr_or_combine(prob_live, prob_fake, label)
    rows.append({"cấu hình": "(c) liveness OR deepfake", "TPR@FPR≤5%": round(tpr_c, 4),
                 "ghi chú": f"τl={tlc:.3f},τf={tfc:.3f}, FPR={fc:.3f}"})
    df = pd.DataFrame(rows)
    C.save_table(df, "mt_cascade.csv")
    print(f"[G3.2] nguồn={src}")
    print(df.to_string(index=False))
    gain = tpr_c - tpr_d
    print(f"\n[KẾT LUẬN] OR-combine vs deepfake-only @FPR≤5%: ΔTPR = {gain:+.4f} "
          f"→ {'liveness GIÚP tăng bắt deepfake' if gain > 0.02 else 'liveness KHÔNG tăng đáng kể (báo honest)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", help="combined npz: prob_live, prob_fake, label(1=attack)")
    a = ap.parse_args()
    if a.npz:
        z = np.load(a.npz, allow_pickle=True)
        run(z["prob_live"].astype(float), z["prob_fake"].astype(float), z["label"].astype(int), a.npz)
        return
    print("[G3.2] Chưa có npz hợp nhất. Cần (sau G2 VN): mỗi mặt có prob_live (liveness) + prob_fake (deepfake) + label(1=attack).")
    print("       Chạy DEMO tổng hợp (minh hoạ phương pháp — KHÔNG phải số thật):")
    rng = np.random.default_rng(0); n = 2000
    label = (rng.random(n) < 0.6).astype(int)
    prob_fake = np.clip(0.35 * label + rng.normal(0.3, 0.2, n), 0, 1)
    prob_live = np.clip(0.30 * label + rng.normal(0.3, 0.25, n), 0, 1)
    run(prob_live, prob_fake, label, "DEMO-synthetic")


if __name__ == "__main__":
    main()
