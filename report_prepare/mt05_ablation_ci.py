#!/usr/bin/env python3
"""MT-5 · Ablation B4 vs B4+block-DCT có Ý NGHĨA THỐNG KÊ (bootstrap CI).
(← novelty #5 dám báo negative result + #7 đối chứng baseline của Trí)

Từ viz_out/*/scores_*.npz: AUC mỗi model + KTC bootstrap 95%, và ΔAUC paired (naive − b4)
với KTC + p hai phía. Làm NGAY (resample trên scores, KHÔNG cần train lại).
Kèm bar chart AUC + errorbar.

Trạng thái: CHẠY ĐƯỢC NGAY.
Output: outputs/mt05_auc_ci.{csv,md}, outputs/mt05_paired_diff.{csv,md}, outputs/mt05_ablation_bar.png
"""
import numpy as np
import pandas as pd

import common as C

N_BOOT = 2000
BASELINE = "b4_local"   # mốc để so ΔAUC


def main():
    plt = C.setup_mpl()
    models = C.list_models()
    # gom theo dataset chung
    datasets = sorted(set(ds for m in models for ds in C.datasets_of(m)))

    ci_rows, diff_rows = [], []
    for ds in datasets:
        present = [m for m in models if (C.VIZ / m / f"scores_{ds}.npz").exists()]
        cache = {m: C.load_scores(m, ds) for m in present}
        for m in present:
            prob, label, _ = cache[m]
            auc, lo, hi = C.bootstrap_auc_ci(label, prob, N_BOOT)
            ci_rows.append({"model": C.model_label(m), "dataset": ds,
                            "AUC": round(auc, 4), "CI95_lo": round(lo, 4), "CI95_hi": round(hi, 4),
                            "n": len(label)})
        if BASELINE in cache:
            prob_b, lab_b, _ = cache[BASELINE]
            for m in present:
                if m == BASELINE:
                    continue
                prob_m, label_m, _ = cache[m]
                # Các .npz đánh giá riêng → KHÔNG đảm bảo cùng thứ tự mẫu. Chỉ paired khi nhãn trùng khít;
                # nếu không → unpaired (resample độc lập) để tránh ghép sai (bug ΔAUC âm giả).
                aligned = (len(label_m) == len(lab_b)) and bool(np.array_equal(label_m, lab_b))
                if aligned:
                    d, lo, hi, p = C.paired_bootstrap_diff(lab_b, prob_b, prob_m, N_BOOT)
                    method = "paired"
                else:
                    d, lo, hi, p = C.unpaired_bootstrap_diff(lab_b, prob_b, label_m, prob_m, N_BOOT)
                    method = "unpaired"
                sig = "có ý nghĩa" if (lo > 0 or hi < 0) else "trong nhiễu (không có ý nghĩa)"
                diff_rows.append({"dataset": ds, "so_sanh": f"{C.model_label(m)} − {C.model_label(BASELINE)}",
                                  "method": method, "dAUC": round(d, 4), "CI95_lo": round(lo, 4),
                                  "CI95_hi": round(hi, 4), "p_two_sided": round(p, 3), "ket_luan": sig})

    df_ci = pd.DataFrame(ci_rows)
    C.save_table(df_ci, "mt05_auc_ci.csv")
    if diff_rows:
        C.save_table(pd.DataFrame(diff_rows), "mt05_paired_diff.csv")

    # Bar AUC + errorbar (CI) cho dataset đầu tiên
    if not df_ci.empty:
        ds0 = datasets[0]
        sub = df_ci[df_ci["dataset"] == ds0]
        fig, ax = plt.subplots(figsize=(1.6 * len(sub) + 2, 4))
        x = np.arange(len(sub))
        err = [sub["AUC"] - sub["CI95_lo"], sub["CI95_hi"] - sub["AUC"]]
        ax.bar(x, sub["AUC"], yerr=err, capsize=6, color="tab:blue", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(sub["model"], rotation=15, ha="right")
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("frame-AUC"); ax.set_title(f"MT-5 · AUC ± KTC95 bootstrap ({ds0})")
        for xi, (a, l) in enumerate(zip(sub["AUC"], sub["CI95_lo"])):
            ax.text(xi, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        C.save_fig(fig, "mt05_ablation_bar.png")
    print("[MT-5] AUC±CI + paired ΔAUC xong. Trình bày TRUNG THỰC (kể cả Δ trong nhiễu).")
    print("       Multi-seed train (≥3 seed) → dùng mt05_fill_results_csv.py sau khi có thêm log.")


if __name__ == "__main__":
    main()
