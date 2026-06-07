#!/usr/bin/env python3
"""P0-① · "Block-DCT sửa được bao nhiêu lỗi của B4 spatial-only?" (phép thử RẺ, không train).
(← khung mode-aware của Trí: tín hiệu thứ 2 sửa lỗi có cấu trúc, chứng minh lift)

So B4 (spatial) vs naive SFDCT (B4+DCT) trên CÙNG tập CDFv2. Nếu 2 .npz cùng thứ tự mẫu
(per-sample) → tính recovery/regression chính xác; nếu KHÔNG → chỉ kết luận mức phân phối +
in cảnh báo cần re-eval chung loader để có per-sample.

CHẠY: python3 report_prepare/mt_dct_corrects_spatial.py
Output: outputs/p0_dct_corrects_spatial.{csv,md}
"""
import numpy as np
import pandas as pd

import common as C

BASE, DCT = "b4_local", "naive_local"


def main():
    ds = C.datasets_of(BASE)[0]
    pb, lb, _ = C.load_scores(BASE, ds)
    pd_, ld, _ = C.load_scores(DCT, ds)
    aligned = (len(lb) == len(ld)) and bool(np.array_equal(lb, ld))

    tb, _, _ = C.threshold_at_fpr(lb, pb, C.TT17_FPR)
    td, _, _ = C.threshold_at_fpr(ld, pd_, C.TT17_FPR)

    rows = [{"chỉ số": "AUC B4", "giá trị": round(C.compute_auc(lb, pb), 4)},
            {"chỉ số": "AUC naive(+DCT)", "giá trị": round(C.compute_auc(ld, pd_), 4)},
            {"chỉ số": "ΔAUC (DCT−B4)", "giá trị": round(C.compute_auc(ld, pd_) - C.compute_auc(lb, pb), 4)},
            {"chỉ số": "scores per-sample aligned?", "giá trị": "CÓ" if aligned else "KHÔNG"}]

    if aligned:
        pred_b = (pb >= tb).astype(int)
        pred_d = (pd_ >= td).astype(int)
        b_wrong = pred_b != lb
        d_right = pred_d == ld
        n_bw = int(b_wrong.sum())
        recovered = int((b_wrong & d_right).sum())          # B4 sai → DCT đúng
        b_right = ~b_wrong
        regressed = int((b_right & (pred_d != ld)).sum())    # B4 đúng → DCT sai
        rows += [
            {"chỉ số": "B4 sai (tại τ)", "giá trị": n_bw},
            {"chỉ số": "DCT THU HỒI (B4 sai→DCT đúng)", "giá trị": f"{recovered} ({100*recovered/max(n_bw,1):.1f}%)"},
            {"chỉ số": "DCT LÀM HỎNG (B4 đúng→DCT sai)", "giá trị": regressed},
            {"chỉ số": "Net (thu hồi − làm hỏng)", "giá trị": recovered - regressed},
        ]
        verdict = ("block-DCT sửa lỗi spatial RÕ → đáng đầu tư FSBI" if recovered - regressed > 0.02 * len(lb)
                   else "net ~0 → naive DCT chưa sửa được lỗi đáng kể; cần FSBI hoặc reframe")
    else:
        verdict = ("⚠️ 2 .npz KHÔNG cùng thứ tự mẫu (AUC(B4-label, DCT-prob)≈0.5) → KHÔNG tính được "
                   "recovery per-sample. Cần RE-EVAL cả 2 model trên CÙNG loader (lưu image-path) để "
                   "ghép theo mẫu. Hiện chỉ kết luận mức phân phối: ΔAUC=+0.0075 nằm trong nhiễu (xem MT-5).")
    df = pd.DataFrame(rows)
    C.save_table(df, "p0_dct_corrects_spatial.csv")
    print(df.to_string(index=False))
    print("\n[P0-①] KẾT LUẬN:", verdict)
    (C.OUT / "p0_dct_corrects_spatial.md").write_text(
        df.to_markdown(index=False) + f"\n\n**Kết luận:** {verdict}\n")


if __name__ == "__main__":
    main()
