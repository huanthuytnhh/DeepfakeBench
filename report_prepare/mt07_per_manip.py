#!/usr/bin/env python3
"""MT-7d · Dấu vết TẦN SỐ theo từng loại manipulation (FF++ c23).
(← per-manipulation: mỗi kiểu deepfake để lại "vân tay" tần số khác nhau → hỗ trợ block-DCT)

real = youtube; so với Deepfakes / Face2Face / FaceSwap / NeuralTextures / FaceShifter.
Hình: (trái) radial fake−real mỗi manipulation (vân tay tần số); (phải) grid phổ DCT trung bình.

CHẠY: python3 report_prepare/mt07_per_manip.py [--n 80]
Output: outputs/mt07_per_manip_radial.png, outputs/mt07_per_manip_grid.png
"""
import argparse
import glob
from pathlib import Path

import numpy as np

import common as C
from mt07_freq_panel import load256, gray, dct2_logmag, radial_profile, _sample_images, mean_spectrum

FFPP = C.REPO / "datasets/FaceForensics++"
REAL_GLOB = str(FFPP / "original_sequences/youtube/c23/frames/*/*.png")
MANIPS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="số ảnh mẫu mỗi lớp")
    a = ap.parse_args()
    plt = C.setup_mpl()

    sr, nr = mean_spectrum(_sample_images([REAL_GLOB], a.n))
    if nr == 0:
        print("[MT-7d] không có ảnh real youtube."); return
    pr = radial_profile(sr)
    x = np.linspace(0, 1, len(pr))

    specs = {"REAL (youtube)": sr}
    radials = {}
    for m in MANIPS:
        g = str(FFPP / f"manipulated_sequences/{m}/c23/frames/*/*.png")
        sm, nm = mean_spectrum(_sample_images([g], a.n))
        if nm == 0:
            continue
        specs[m] = sm
        radials[m] = radial_profile(sm) - pr  # fake − real

    # (1) radial fake−real mỗi manipulation
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for m, d in radials.items():
        ax.plot(x, d, lw=2, label=m)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(0.33, 0.8, color="orange", alpha=0.12, label="dải mid/high (DCT branch)")
    ax.set_xlabel("tần số chuẩn hoá (0=DC → 1=cao)"); ax.set_ylabel("FAKE − REAL  (mean log|DCT|)")
    ax.set_title(f"MT-7 · Vân tay tần số theo manipulation (FF++ c23, n={a.n}/lớp)")
    ax.legend(fontsize=9)
    C.save_fig(fig, "mt07_per_manip_radial.png")

    # (2) grid phổ DCT trung bình
    keys = list(specs.keys())
    vmin = float(np.percentile(np.concatenate([s.ravel() for s in specs.values()]), 2))
    vmax = float(np.percentile(np.concatenate([s.ravel() for s in specs.values()]), 82))
    fig, ax = plt.subplots(1, len(keys), figsize=(3 * len(keys), 3.4))
    for a_, k in zip(ax, keys):
        a_.imshow(specs[k], cmap="jet", vmin=vmin, vmax=vmax)
        a_.set_title(k, fontsize=10); a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle("MT-7 · Phổ DCT trung bình: real vs từng manipulation", y=1.04)
    C.save_fig(fig, "mt07_per_manip_grid.png")
    print(f"[MT-7d] xong: {len(radials)} manipulation, n={a.n}/lớp (real n={nr}).")


if __name__ == "__main__":
    main()
