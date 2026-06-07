#!/usr/bin/env python3
"""§8 #3 · Hiệu chuẩn xác suất (calibration) — KHÔNG GPU, không train kiến trúc.
Từ scores .npz: chia 50/50 (calib/eval theo seed), fit **temperature scaling** + **Platt scaling** trên calib,
báo **ECE trước/sau** trên eval + reliability. Mục tiêu: ECE của naive/row1 đang xấu (0.20/0.31) → giảm?
*(AUC KHÔNG đổi khi calibrate — chỉ xác suất đáng tin hơn + τ ổn định hơn.)*

CHẠY: python3 report_prepare/mt_calibrate.py
Output: outputs/mt_calibration.{csv,md}, outputs/mt_reliability_cal_<model>.png
Caveat: chia theo INDEX (chưa có video-id); sau G1.2 nên chia theo VIDEO.
"""
import numpy as np
import pandas as pd

import common as C
SEED = 42


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(logit, y):
    from scipy.optimize import minimize_scalar
    def nll(T):
        p = 1 / (1 + np.exp(-logit / T))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    r = minimize_scalar(nll, bounds=(0.05, 20), method="bounded")
    return float(r.x)


def main():
    plt = C.setup_mpl()
    rng = np.random.default_rng(SEED)
    rows = []
    for m in C.list_models():
        ds = C.datasets_of(m)[0]
        prob, label, _ = C.load_scores(m, ds)
        idx = rng.permutation(len(label)); k = len(idx) // 2
        ci, ei = idx[:k], idx[k:]
        z = _logit(prob)
        # temperature
        T = fit_temperature(z[ci], label[ci])
        p_temp = 1 / (1 + np.exp(-z[ei] / T))
        # platt
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[ci].reshape(-1, 1), label[ci])
        p_platt = lr.predict_proba(z[ei].reshape(-1, 1))[:, 1]

        ece0 = C.expected_calibration_error(label[ei], prob[ei])[0]
        eceT = C.expected_calibration_error(label[ei], p_temp)[0]
        eceP = C.expected_calibration_error(label[ei], p_platt)[0]
        auc0 = C.compute_auc(label[ei], prob[ei])  # không đổi sau calibrate (đơn điệu)
        rows.append({"model": C.model_label(m), "T": round(T, 3),
                     "ECE_raw": round(ece0, 4), "ECE_temp": round(eceT, 4), "ECE_platt": round(eceP, 4),
                     "AUC(eval)": round(auc0, 4)})

        # reliability raw vs temp
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="hoàn hảo")
        for p_, lab_, col in [(prob[ei], f"raw ECE={ece0:.3f}", "#dc2626"), (p_temp, f"temp ECE={eceT:.3f}", "#2563eb")]:
            _, cen, accs, confs, cnt = C.expected_calibration_error(label[ei], p_)
            v = cnt > 0
            ax.plot(confs[v], accs[v], "o-", label=lab_, color=col)
        ax.set_xlabel("độ tự tin"); ax.set_ylabel("tỉ lệ fake thực"); ax.set_title(f"Reliability — {C.model_label(m)}")
        ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        C.save_fig(fig, f"mt_reliability_cal_{m}.png")
    df = pd.DataFrame(rows)
    C.save_table(df, "mt_calibration.csv")
    print(df.to_string(index=False))
    print("\n[§8#3] ECE giảm sau temp/platt ⇒ calibration GIÚP (xác suất đáng tin hơn, AUC giữ nguyên).")


if __name__ == "__main__":
    main()
