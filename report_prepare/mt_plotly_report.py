#!/usr/bin/env python3
"""Đồ thị TƯƠNG TÁC bằng Plotly (giống Trí xuất .html interactive) — đẹp + hover ra số.
Nguồn: viz_out/*/results.json + scores_*.npz (chạy ngay, không GPU).

Sinh: outputs/html/plotly_roc.html, plotly_eval_heatmap.html, plotly_ablation_ci.html,
      plotly_score_dist.html, plotly_cross_heatmap.html, và plotly_dashboard.html (gộp).
Mở bằng trình duyệt; nhúng vào web/báo cáo đều được.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve

import common as C

OUT = C.OUT / "html"
OUT.mkdir(parents=True, exist_ok=True)
TPL = "plotly_white"
COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]


def _save(fig, name):
    p = OUT / name
    fig.write_html(str(p), include_plotlyjs=True, full_html=True)  # nhúng plotly.js → mở offline
    print(f"[saved] {p.relative_to(C.REPO)}")


def _models_ds():
    models = C.list_models()
    datasets = sorted({d for m in models for d in C.datasets_of(m)})
    return models, datasets


# ── 1. ROC tương tác (overlay model, vạch eKYC FPR≤5%) ──
def fig_roc(dataset=None):
    models, datasets = _models_ds()
    ds = dataset or datasets[0]
    fig = go.Figure()
    for i, m in enumerate(models):
        if not (C.VIZ / m / f"scores_{ds}.npz").exists():
            continue
        prob, label, _ = C.load_scores(m, ds)
        fpr, tpr, thr = roc_curve(label, prob, pos_label=1)
        auc = C.compute_auc(label, prob)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{C.model_label(m)} (AUC={auc:.3f})",
            line=dict(color=COLORS[i % len(COLORS)], width=2.5),
            customdata=np.clip(thr, 0, 1),
            hovertemplate="FPR=%{x:.3f}<br>TPR=%{y:.3f}<br>τ=%{customdata:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="gray"), showlegend=False))
    fig.add_vline(x=0.05, line=dict(color="red", dash="dash"),
                  annotation_text="eKYC FPR≤5% (TT17)", annotation_position="top")
    fig.update_layout(template=TPL, title=f"ROC — {ds}", xaxis_title="FPR (real→fake)",
                      yaxis_title="TPR (bắt fake)", width=720, height=560,
                      legend=dict(x=0.4, y=0.08))
    return fig


# ── 2. Heatmap metric (model × metric) ──
def fig_eval_heatmap():
    models, _ = _models_ds()
    metrics = ["frame_auc", "video_auc", "ap", "tpr@fpr=5%", "tpr@fpr=1%", "eer"]
    z, text = [], []
    for m in models:
        r = list(C.load_results(m).values())[0] if C.load_results(m) else {}
        row = [r.get(k, np.nan) for k in metrics]
        z.append(row); text.append([f"{v:.3f}" if v == v else "—" for v in row])
    fig = go.Figure(go.Heatmap(
        z=z, x=metrics, y=[C.model_label(m) for m in models], text=text, texttemplate="%{text}",
        colorscale="Viridis", hovertemplate="%{y}<br>%{x}=%{z:.4f}<extra></extra>"))
    fig.update_layout(template=TPL, title="Ma trận metric (model × metric)", width=760, height=380)
    return fig


# ── 3. Ablation AUC ± bootstrap CI ──
def fig_ablation(dataset=None):
    models, datasets = _models_ds()
    ds = dataset or datasets[0]
    names, aucs, los, his = [], [], [], []
    for m in models:
        if not (C.VIZ / m / f"scores_{ds}.npz").exists():
            continue
        prob, label, _ = C.load_scores(m, ds)
        a, lo, hi = C.bootstrap_auc_ci(label, prob, 2000)
        names.append(C.model_label(m)); aucs.append(a); los.append(a - lo); his.append(hi - a)
    fig = go.Figure(go.Bar(
        x=names, y=aucs, marker_color=COLORS[:len(names)],
        error_y=dict(type="data", symmetric=False, array=his, arrayminus=los),
        text=[f"{a:.3f}" for a in aucs], textposition="outside",
        hovertemplate="%{x}<br>AUC=%{y:.4f}<extra></extra>"))
    fig.update_layout(template=TPL, title=f"AUC ± KTC95 bootstrap — {ds}",
                      yaxis=dict(range=[0.5, 1.0], title="frame-AUC"), width=620, height=480)
    return fig


# ── 4. Phân bố score real vs fake (overlay) ──
def fig_score_dist(model=None, dataset=None):
    models, datasets = _models_ds()
    m = model or (models[1] if len(models) > 1 else models[0])
    ds = dataset or datasets[0]
    prob, label, _ = C.load_scores(m, ds)
    thr, _, _ = C.threshold_at_fpr(label, prob, C.TT17_FPR)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=prob[label == 0], name="REAL", marker_color="#059669", opacity=0.6, nbinsx=50, histnorm="probability density"))
    fig.add_trace(go.Histogram(x=prob[label == 1], name="FAKE", marker_color="#dc2626", opacity=0.6, nbinsx=50, histnorm="probability density"))
    fig.add_vline(x=thr, line=dict(color="black", dash="dash"), annotation_text=f"τ@FPR≤5%={thr:.3f}")
    fig.update_layout(template=TPL, barmode="overlay", title=f"Phân bố score — {C.model_label(m)} ({ds})",
                      xaxis_title="prob(fake)", yaxis_title="mật độ", width=720, height=460)
    return fig


# ── 5. Heatmap cross-dataset (model × test-set) ──
def fig_cross_heatmap():
    models, datasets = _models_ds()
    z, text = [], []
    for m in models:
        r = C.load_results(m)
        row = [r.get(d, {}).get("frame_auc", np.nan) for d in datasets]
        z.append(row); text.append([f"{v:.3f}" if v == v else "—" for v in row])
    fig = go.Figure(go.Heatmap(z=z, x=datasets, y=[C.model_label(m) for m in models],
                               text=text, texttemplate="%{text}", colorscale="RdYlGn", zmin=0.5, zmax=1.0,
                               hovertemplate="%{y}<br>%{x} AUC=%{z:.4f}<extra></extra>"))
    fig.update_layout(template=TPL, title="Cross-dataset frame-AUC (model × test-set)", width=620, height=380)
    return fig


def fig_dashboard():
    """Gộp ROC + ablation + score-dist + heatmap vào 1 trang (layout sạch: legend trên, colorbar riêng góc)."""
    models, datasets = _models_ds()
    ds = datasets[0]
    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.12,
        subplot_titles=(f"ROC — {ds}", "AUC ± KTC95 bootstrap",
                        "Phân bố score real-vs-fake", "Metric (model × metric)"),
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "heatmap"}]])
    # (1,1) ROC — chỉ ROC hiện ở legend (ngang, trên cùng)
    for i, m in enumerate(models):
        if not (C.VIZ / m / f"scores_{ds}.npz").exists():
            continue
        prob, label, _ = C.load_scores(m, ds)
        fpr, tpr, _ = roc_curve(label, prob, pos_label=1)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{C.model_label(m)} (AUC={C.compute_auc(label,prob):.3f})",
                                 legendgroup=m, line=dict(color=COLORS[i % len(COLORS)], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="lightgray"),
                             showlegend=False), row=1, col=1)
    fig.add_vline(x=0.05, line=dict(color="red", dash="dash"), row=1, col=1)
    # (1,2) ablation bar — không legend, tên model trên trục x
    for t in fig_ablation(ds).data:
        t.showlegend = False; fig.add_trace(t, row=1, col=2)
    # (2,1) score dist — không legend (màu tự rõ: real lục / fake đỏ)
    for t in fig_score_dist(dataset=ds).data:
        t.showlegend = False; fig.add_trace(t, row=2, col=1)
    # (2,2) heatmap — colorbar gọn ở góc dưới-phải
    for t in fig_eval_heatmap().data:
        t.update(colorbar=dict(len=0.40, y=0.18, x=1.0, thickness=12, title="value"))
        fig.add_trace(t, row=2, col=2)
    fig.update_layout(
        template=TPL, title=dict(text="<b>DeepGuard — Bảng đồ thị tương tác (SFDCT)</b>", x=0.5),
        width=1180, height=900, barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(t=110, r=90))
    fig.update_xaxes(title_text="FPR (real→fake)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="TPR (bắt fake)", range=[0, 1.02], row=1, col=1)
    fig.update_yaxes(title_text="AUC", range=[0.5, 0.85], row=1, col=2)
    fig.update_xaxes(title_text="prob(fake)", row=2, col=1)
    fig.update_yaxes(title_text="mật độ", row=2, col=1)
    return fig


def main():
    _save(fig_roc(), "plotly_roc.html")
    _save(fig_eval_heatmap(), "plotly_eval_heatmap.html")
    _save(fig_ablation(), "plotly_ablation_ci.html")
    _save(fig_score_dist(), "plotly_score_dist.html")
    _save(fig_cross_heatmap(), "plotly_cross_heatmap.html")
    _save(fig_dashboard(), "plotly_dashboard.html")
    print("[plotly] Xong. Mở report_prepare/outputs/html/plotly_dashboard.html bằng trình duyệt.")


if __name__ == "__main__":
    main()
