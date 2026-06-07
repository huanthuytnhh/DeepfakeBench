#!/usr/bin/env python3
"""G1.1 · Metric eKYC bổ sung từ scores .npz SẴN (KHÔNG GPU, không train):
  - DET curve overlay (FNR-FPR thang probit) cho mọi model.
  - Bảng BPCER@APCER∈{1%,5%,10%} (định nghĩa forgery: APCER = fake lọt = nhận nhầm thật).
  - ECE + reliability diagram mỗi model.
Tái dùng common.py (det_points, expected_calibration_error). Nguồn: viz_out/*/scores_*.npz.

CHẠY: python3 report_prepare/mt_ekyc_metrics.py
Output: outputs/mt_det_overlay.png, mt_bpcer_apcer.{csv,md}, mt_reliability_<model>.png
"""
import numpy as np
import pandas as pd

import common as C


def bpcer_at_apcer(prob, label, targets=(0.01, 0.05, 0.10)):
    """positive=fake(1). APCER = tỉ lệ fake bị cho qua (pred real); BPCER = tỉ lệ real bị chặn.
    Với mỗi APCER target, quét ngưỡng tìm BPCER nhỏ nhất thoả APCER≤target."""
    thr = np.unique(prob)
    out = {}
    for t in targets:
        best = np.nan
        for s in thr:
            pred_fake = prob >= s
            apcer = float((~pred_fake[label == 1]).mean()) if (label == 1).any() else np.nan
            bpcer = float((pred_fake[label == 0]).mean()) if (label == 0).any() else np.nan
            if apcer <= t and (np.isnan(best) or bpcer < best):
                best = bpcer
        out[t] = best
    return out


def main():
    plt = C.setup_mpl()
    from scipy.stats import norm
    models = C.list_models()

    # (1) DET overlay
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    for i, m in enumerate(models):
        ds = C.datasets_of(m)[0]
        prob, label, _ = C.load_scores(m, ds)
        fp, fn = C.det_points(label, prob)
        ax.plot(norm.ppf(fp), norm.ppf(fn), lw=2, label=C.model_label(m), color=["#2563eb", "#dc2626", "#059669"][i % 3])
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks)); ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlabel("FPR (APCER — fake lọt qua)"); ax.set_ylabel("FNR (bỏ sót fake)")
    ax.set_title("DET curve — Celeb-DF-v2 (thấp-trái = tốt)"); ax.legend(); ax.grid(alpha=.3)
    C.save_fig(fig, "mt_det_overlay.png")

    # (2) BPCER@APCER table
    rows = []
    for m in models:
        ds = C.datasets_of(m)[0]
        prob, label, _ = C.load_scores(m, ds)
        bp = bpcer_at_apcer(prob, label)
        rows.append({"model": C.model_label(m),
                     "BPCER@APCER≤1%": round(bp[0.01], 4), "BPCER@APCER≤5%": round(bp[0.05], 4),
                     "BPCER@APCER≤10%": round(bp[0.10], 4)})
    C.save_table(pd.DataFrame(rows), "mt_bpcer_apcer.csv")

    # (3) ECE + reliability mỗi model
    ece_rows = []
    for m in models:
        ds = C.datasets_of(m)[0]
        prob, label, _ = C.load_scores(m, ds)
        ece, centers, accs, confs, counts = C.expected_calibration_error(label, prob, n_bins=15)
        ece_rows.append({"model": C.model_label(m), "ECE": round(ece, 4)})
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="hiệu chuẩn hoàn hảo")
        valid = counts > 0
        ax.plot(confs[valid], accs[valid], "o-", color="#2563eb", label=f"thực tế (ECE={ece:.3f})")
        ax.set_xlabel("Độ tự tin TB (prob fake)"); ax.set_ylabel("Tỉ lệ fake thực tế")
        ax.set_title(f"Reliability — {C.model_label(m)}"); ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        C.save_fig(fig, f"mt_reliability_{m}.png")
    C.save_table(pd.DataFrame(ece_rows), "mt_ece.csv")
    print("[G1.1] DET + BPCER@APCER + ECE/reliability xong (no-GPU, từ .npz sẵn).")


if __name__ == "__main__":
    main()
