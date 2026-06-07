#!/usr/bin/env python3
"""MT-8 · Module liveness chuẩn FAS (ISO/IEC 30107-3) + DET + BPCER@APCER≤5%.
(← novelty #9 MIR metrics của Trí → metric chuẩn ngành cho liveness; neo TT17 APCER≤5%)

PHẦN VẼ ĐÃ SẴN SÀNG. Chỉ thiếu dữ liệu: cần chạy train liveness trước (1-2h local, $0).
Script đọc liveness/out/metrics_liveness.json (do liveness/train_liveness.py xuất) hoặc một
file scores liveness (.npz prob/label) → ROC + DET + bar ACER/APCER/BPCER + histogram
genuine-vs-spoof + BPCER tại APCER cố định (1%/5%/10%).

Trạng thái: VẼ sẵn — CHẠY TRAIN trước (xem lệnh in ra nếu thiếu data).
Output: outputs/mt08_liveness_*.png, outputs/mt08_liveness_metrics.{csv,md}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

LIVE_OUT = C.LIVENESS / "out"
RUN_HINT = (
    "Chưa có kết quả liveness. Chạy (local RTX 3060, ~1-2h, $0):\n"
    "  cd {repo}\n"
    "  export HF_TOKEN=...   # để kéo LCC-FASD từ HF\n"
    "  AUG=fas BATCH=32 ./liveness/run_liveness.sh\n"
    "→ sinh liveness/out/metrics_liveness.json + scores_*.npz, rồi chạy lại script này."
).format(repo=C.REPO)


def bpcer_at_apcer(label, score, apcer_targets=(0.01, 0.05, 0.10)):
    """label: 1=spoof(attack/fake), 0=genuine(real). score cao = nghi spoof.
    APCER = tỉ lệ attack lọt (bị nhận genuine); BPCER = tỉ lệ genuine bị từ chối.
    Quét ngưỡng, với mỗi APCER target tìm BPCER nhỏ nhất thoả APCER≤target."""
    thr = np.unique(score)
    out = {}
    for t in apcer_targets:
        best = np.nan
        for s in thr:
            pred_spoof = score >= s
            apcer = float((~pred_spoof[label == 1]).mean()) if (label == 1).any() else np.nan  # attack bị cho qua
            bpcer = float((pred_spoof[label == 0]).mean()) if (label == 0).any() else np.nan   # genuine bị chặn
            if apcer <= t and (np.isnan(best) or bpcer < best):
                best = bpcer
        out[t] = best
    return out


def plot_from_scores(prob, label, tag="liveness"):
    plt = C.setup_mpl()
    from sklearn.metrics import roc_curve
    # ROC
    fpr, tpr, _ = roc_curve(label, prob, pos_label=1)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={C.compute_auc(label, prob):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.axvline(0.05, color="red", ls=":", label="APCER≤5% (TT17)")
    ax.set_xlabel("APCER (attack lọt)"); ax.set_ylabel("1-BPCER"); ax.set_title(f"MT-8 · ROC liveness — {tag}"); ax.legend()
    C.save_fig(fig, f"mt08_liveness_roc_{tag}.png")
    # DET
    fp, fn = C.det_points(label, prob)
    from scipy.stats import norm
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(norm.ppf(fp), norm.ppf(fn), lw=2)
    ticks = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks)); ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlabel("APCER"); ax.set_ylabel("BPCER"); ax.set_title(f"MT-8 · DET liveness — {tag}")
    C.save_fig(fig, f"mt08_liveness_det_{tag}.png")
    # Histogram genuine vs spoof
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(prob[label == 0], bins=50, alpha=0.65, color="tab:green", label="genuine", density=True)
    ax.hist(prob[label == 1], bins=50, alpha=0.65, color="tab:red", label="spoof", density=True)
    ax.set_xlabel("prob(spoof)"); ax.set_title(f"MT-8 · Phân bố score genuine-vs-spoof — {tag}"); ax.legend()
    C.save_fig(fig, f"mt08_liveness_scoredist_{tag}.png")
    # BPCER@APCER
    return bpcer_at_apcer(label, prob)


def main():
    rows = []
    metrics_json = LIVE_OUT / "metrics_liveness.json"
    npzs = sorted(LIVE_OUT.glob("scores_*.npz")) if LIVE_OUT.exists() else []
    if metrics_json.exists():
        d = json.loads(metrics_json.read_text())
        print("[MT-8] Đọc metrics_liveness.json:", json.dumps(d, indent=1)[:600])
        # d kỳ vọng dạng {model:{AUC,ACER,APCER,BPCER,EER}}
        for model, mm in (d.items() if isinstance(d, dict) else []):
            if isinstance(mm, dict):
                rows.append({"model": model, **{k: mm.get(k) for k in ("AUC", "ACER", "APCER", "BPCER", "EER")}})
    if npzs:
        for p in npzs:
            z = np.load(p, allow_pickle=True)
            bp = plot_from_scores(z["prob"].astype(float), z["label"].astype(int), tag=p.stem.replace("scores_", ""))
            rows.append({"model": p.stem, **{f"BPCER@APCER{int(k*100)}%": v for k, v in bp.items()}})
    if not rows:
        print(RUN_HINT); return
    df = pd.DataFrame(rows)
    C.save_table(df, "mt08_liveness_metrics.csv")
    # Bar ACER/APCER/BPCER nếu có
    plt = C.setup_mpl()
    cols = [c for c in ("ACER", "APCER", "BPCER") if c in df.columns]
    if cols and df[cols].notna().any().any():
        fig, ax = plt.subplots(figsize=(2 + 1.5 * len(df), 4))
        x = np.arange(len(df)); w = 0.25
        for i, c in enumerate(cols):
            ax.bar(x + (i - 1) * w, df[c].fillna(0), w, label=c)
        ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=15, ha="right")
        ax.axhline(0.05, color="red", ls="--", label="5% (TT17)")
        ax.set_ylabel("error rate"); ax.set_title("MT-8 · ACER/APCER/BPCER liveness"); ax.legend()
        C.save_fig(fig, "mt08_liveness_error_bar.png")
    print("[MT-8] Báo cáo liveness xong (neo TT17 APCER≤5%).")


if __name__ == "__main__":
    main()
