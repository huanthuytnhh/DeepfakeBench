#!/usr/bin/env python3
"""MT-7 bản PLOTLY tương tác (đẹp, hover ra số) — bằng chứng tần số real-vs-fake.
Tái dùng hàm từ mt07_freq_panel / mt07_per_manip. Nguồn: datasets/ (ảnh face-crop thật).

Sinh: outputs/html/plotly_mt07_panel.html, plotly_mt07_radial.html,
      plotly_mt07_permanip_radial.html, plotly_mt07_permanip_grid.html
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import common as C
from mt07_freq_panel import (load256, gray, dct2_logmag, radial_profile, mouth,
                             auto_pair, mean_spectrum, _sample_images, DS)
from mt07_per_manip import REAL_GLOB, MANIPS, FFPP

OUT = C.OUT / "html"
OUT.mkdir(parents=True, exist_ok=True)
TPL = "plotly_white"


def _save(fig, name):
    p = OUT / name
    fig.write_html(str(p), include_plotlyjs=True, full_html=True)
    print(f"[saved] {p.relative_to(C.REPO)}")


def _img_trace(img):
    return go.Image(z=(img * 255).astype(np.uint8))


def _spec_trace(spec, zmin, zmax, showscale=False):
    # flipud để DC ([0,0]) nằm góc TRÊN-trái (plotly heatmap vẽ bottom-up)
    return go.Heatmap(z=np.flipud(spec), colorscale="Jet", zmin=zmin, zmax=zmax,
                      showscale=showscale, hovertemplate="u=%{x}<br>v=%{y}<br>log|DCT|=%{z:.2f}<extra></extra>")


def panel(real_img, fake_img, src):
    blocks = [("(a)", "(b)", real_img, fake_img), ("(c)", "(d)", mouth(real_img), mouth(fake_img))]
    fig = make_subplots(rows=4, cols=2, vertical_spacing=0.03, horizontal_spacing=0.04,
                        column_titles=["<b>Real</b>", "<b>Fake</b>"],
                        row_heights=[1, 1, 1, 1])
    for bi, (la, lb, ri, fi) in enumerate(blocks):
        r_face, r_spec = bi * 2 + 1, bi * 2 + 2
        fig.add_trace(_img_trace(ri), row=r_face, col=1)
        fig.add_trace(_img_trace(fi), row=r_face, col=2)
        sr, sf = dct2_logmag(gray(ri)), dct2_logmag(gray(fi))
        both = np.concatenate([sr.ravel(), sf.ravel()])
        zmin, zmax = np.percentile(both, 2), np.percentile(both, 82)
        fig.add_trace(_spec_trace(sr, zmin, zmax), row=r_spec, col=1)
        fig.add_trace(_spec_trace(sf, zmin, zmax), row=r_spec, col=2)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_layout(template=TPL, width=560, height=1040,
                      title=dict(text=f"<b>MT-7 · Real vs Fake + log|2D-DCT|</b><br><sup>{src}</sup>", x=0.5))
    return fig


def radial(n=150):
    sr, nr = mean_spectrum(_sample_images(
        [str(DS / "FaceForensics++/original_sequences/youtube/c23/frames/*/*.png"),
         str(DS / "Celeb-DF-v2/Celeb-real/frames/*/*.png")], n))
    sf, nf = mean_spectrum(_sample_images(
        [str(DS / "FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/*/*.png"),
         str(DS / "Celeb-DF-v2/Celeb-synthesis/frames/*/*.png")], n))
    pr, pf = radial_profile(sr), radial_profile(sf)
    x = np.linspace(0, 1, len(pr))
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Năng lượng theo dải tần (log-y)", "Chênh lệch FAKE − REAL"))
    fig.add_trace(go.Scatter(x=x, y=pr, name=f"REAL (n={nr})", line=dict(color="#059669", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=pf, name=f"FAKE (n={nf})", line=dict(color="#dc2626", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=pf - pr, name="FAKE−REAL", line=dict(color="#7c3aed", width=2.5), showlegend=False), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="black", width=0.8), row=1, col=2)
    for c in (1, 2):
        fig.add_vrect(x0=0.33, x1=0.8, fillcolor="orange", opacity=0.12, line_width=0, row=1, col=c)
    fig.update_yaxes(type="log", title_text="mean log|DCT|", row=1, col=1)
    fig.update_xaxes(title_text="tần số chuẩn hoá (0=DC→1)", row=1, col=1)
    fig.update_xaxes(title_text="tần số chuẩn hoá", row=1, col=2)
    fig.update_layout(template=TPL, width=1080, height=460,
                      title=dict(text="<b>MT-7 · Năng lượng DCT theo dải tần — fake mất tần cao</b>", x=0.5),
                      legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"))
    return fig


def permanip(n=80):
    sr, nr = mean_spectrum(_sample_images([REAL_GLOB], n))
    pr = radial_profile(sr); x = np.linspace(0, 1, len(pr))
    fig = go.Figure()
    for m in MANIPS:
        sm, nm = mean_spectrum(_sample_images([str(FFPP / f"manipulated_sequences/{m}/c23/frames/*/*.png")], n))
        if nm:
            fig.add_trace(go.Scatter(x=x, y=radial_profile(sm) - pr, name=m, line=dict(width=2)))
    fig.add_hline(y=0, line=dict(color="black", width=0.8))
    fig.add_vrect(x0=0.33, x1=0.8, fillcolor="orange", opacity=0.12, line_width=0,
                  annotation_text="dải mid/high")
    fig.update_layout(template=TPL, width=900, height=520,
                      title=dict(text=f"<b>MT-7 · Vân tay tần số theo manipulation (FF++ c23, n={n})</b>", x=0.5),
                      xaxis_title="tần số chuẩn hoá (0=DC→1)", yaxis_title="FAKE − REAL (mean log|DCT|)")
    return fig


def main():
    rp, fp, src = auto_pair()
    if rp and fp:
        _save(panel(load256(rp), load256(fp), src), "plotly_mt07_panel.html")
    _save(radial(), "plotly_mt07_radial.html")
    _save(permanip(), "plotly_mt07_permanip_radial.html")
    print("[plotly-mt07] xong → report_prepare/outputs/html/")


if __name__ == "__main__":
    main()
