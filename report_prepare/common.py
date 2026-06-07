"""common.py — tiện ích dùng chung cho report_prepare/ (DeepfakeBench → báo cáo DATN).

Mọi script MT-xx import từ đây để: tải scores (.npz) / results.json / log training,
tính metric (AUC/EER/AP/TPR@FPR/threshold@FPR/DET/bootstrap-CI/ECE) và style matplotlib thống nhất.

Quy ước: positive = FAKE (label 1 = fake, 0 = real). `prob` = P(fake).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

# ── Đường dẫn chuẩn của repo (report_prepare/ nằm trong DeepfakeBench/) ──
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
VIZ = REPO / "viz_out"
LOGS = REPO / "logs" / "training"
EXPERIMENTS = REPO / "experiments"
LIVENESS = REPO / "liveness"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# ── Nhãn model hiển thị (fallback = tên thư mục) ──
MODEL_LABELS = {
    "b4_local": "B4 (baseline)",
    "naive_local": "B4+block-DCT (naive SFDCT)",
    "row1_local": "B4+block-DCT (hf, row1)",
}
MODEL_ORDER = ["b4_local", "naive_local", "row1_local"]
TT17_FPR = 0.05  # Thông tư 17/2024/TT-NHNN: FPR (real bị nhận nhầm fake) ≤ 5%


def model_label(key: str) -> str:
    return MODEL_LABELS.get(key, key)


def list_models() -> list[str]:
    """Các thư mục model trong viz_out/ có results.json — giữ thứ tự MODEL_ORDER, phần dư xếp sau."""
    found = [p.name for p in VIZ.iterdir() if p.is_dir() and (p / "results.json").exists()] if VIZ.exists() else []
    ordered = [m for m in MODEL_ORDER if m in found] + sorted(m for m in found if m not in MODEL_ORDER)
    return ordered


def load_results(model: str) -> dict:
    p = VIZ / model / "results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def datasets_of(model: str) -> list[str]:
    return list(load_results(model).keys())


def load_scores(model: str, dataset: str):
    """Trả (prob, label, feat) từ scores_<dataset>.npz; feat có thể None."""
    p = VIZ / model / f"scores_{dataset}.npz"
    if not p.exists():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=True)
    feat = z["feat"] if "feat" in z.files else None
    return z["prob"].astype(np.float64), z["label"].astype(int), feat


# ── Metrics ──
def compute_auc(label, prob) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(label, prob))


def compute_ap(label, prob) -> float:
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(label, prob))


def compute_eer(label, prob) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(label, prob, pos_label=1)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def tpr_at_fpr(label, prob, target_fpr=TT17_FPR) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(label, prob, pos_label=1)
    return float(np.interp(target_fpr, fpr, tpr))


def threshold_at_fpr(label, prob, target_fpr=TT17_FPR):
    """Ngưỡng τ sao cho FPR(real→fake) ≤ target. Trả (thr, fpr_đạt, tpr_đạt)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(label, prob, pos_label=1)
    ok = np.where(fpr <= target_fpr)[0]
    i = ok[-1] if len(ok) else 0
    return float(thr[i]), float(fpr[i]), float(tpr[i])


def det_points(label, prob):
    """DET: trả (fpr, fnr) đã lọc >0 để vẽ trên thang probit."""
    from sklearn.metrics import det_curve
    fpr, fnr, _ = det_curve(label, prob, pos_label=1)
    m = (fpr > 0) & (fnr > 0)
    return fpr[m], fnr[m]


def bootstrap_auc_ci(label, prob, n_boot=2000, seed=42, alpha=0.05):
    """AUC + khoảng tin cậy bootstrap (percentile). Trả (auc, lo, hi)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    label = np.asarray(label); prob = np.asarray(prob)
    N = len(label)
    base = float(roc_auc_score(label, prob))
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if label[idx].min() == label[idx].max():
            continue
        aucs.append(roc_auc_score(label[idx], prob[idx]))
    lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return base, float(lo), float(hi)


def paired_bootstrap_diff(label, prob_a, prob_b, n_boot=2000, seed=42, alpha=0.05):
    """ΔAUC = AUC(b) − AUC(a) trên cùng resample. CHỈ hợp lệ khi prob_a/prob_b cùng thứ tự mẫu.
    Trả (diff, lo, hi, p_two_sided)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    label = np.asarray(label); a = np.asarray(prob_a); b = np.asarray(prob_b)
    N = len(label)
    base = float(roc_auc_score(label, b) - roc_auc_score(label, a))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        if label[idx].min() == label[idx].max():
            continue
        diffs.append(roc_auc_score(label[idx], b[idx]) - roc_auc_score(label[idx], a[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return base, float(lo), float(hi), float(min(p, 1.0))


def unpaired_bootstrap_diff(label_a, prob_a, label_b, prob_b, n_boot=2000, seed=42, alpha=0.05):
    """ΔAUC = AUC(b) − AUC(a) khi 2 model đánh giá trên 2 mảng KHÔNG cùng thứ tự mẫu
    (resample ĐỘC LẬP mỗi model). Hơi bảo thủ nhưng luôn hợp lệ. Trả (diff, lo, hi, p_two_sided)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    la, pa = np.asarray(label_a), np.asarray(prob_a)
    lb, pb = np.asarray(label_b), np.asarray(prob_b)
    base = float(roc_auc_score(lb, pb) - roc_auc_score(la, pa))
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(la), len(la)); ib = rng.integers(0, len(lb), len(lb))
        if la[ia].min() == la[ia].max() or lb[ib].min() == lb[ib].max():
            continue
        diffs.append(roc_auc_score(lb[ib], pb[ib]) - roc_auc_score(la[ia], pa[ia]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return base, float(lo), float(hi), float(min(p, 1.0))


def expected_calibration_error(label, prob, n_bins=15):
    """ECE + dữ liệu reliability diagram. Trả (ece, bin_centers, bin_acc, bin_conf, bin_count)."""
    label = np.asarray(label); prob = np.asarray(prob)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs, counts = [], [], [], []
    ece = 0.0
    N = len(label)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        c = int(m.sum())
        centers.append((lo + hi) / 2)
        if c:
            acc = float((label[m] == 1).mean())   # tỉ lệ thực sự fake trong bin
            conf = float(prob[m].mean())
            ece += c / N * abs(acc - conf)
            accs.append(acc); confs.append(conf); counts.append(c)
        else:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
    return float(ece), np.array(centers), np.array(accs), np.array(confs), np.array(counts)


# ── Matplotlib style thống nhất ──
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def save_fig(fig, name: str):
    fig.tight_layout()
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[saved] {p.relative_to(REPO)}")
    return p


def save_table(df, name: str):
    """Lưu cả .csv lẫn .md (markdown để dán thẳng vào báo cáo)."""
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    try:
        (p.with_suffix(".md")).write_text(df.to_markdown(index=False))
    except Exception:
        pass
    print(f"[saved] {p.relative_to(REPO)}  (+ .md)")
    return p
