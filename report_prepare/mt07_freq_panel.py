#!/usr/bin/env python3
"""MT-7c · Panel "Real vs Fake + phổ tần số" (đúng style paper F3Net/SFDCT).
(← bằng chứng trực quan cho hypothesis block-DCT: artifact fake rò vào dải tần mid/high)

Layout giống hình mẫu:
  hàng 1: mặt Real | mặt Fake
  hàng 2: log|2D-DCT| của Real | của Fake   (colormap rainbow; góc trên-trái = DC/low-freq)
  + khối thứ 2 (c)(d) cho vùng MIỆNG (mouth crop) — nơi artifact reenactment lộ rõ.

Tự chọn CẶP cùng nhận dạng từ FaceForensics++ (fake=Deepfakes/<tgt>_<src>, real=youtube/<tgt>),
fallback sang Celeb-DF (real/synthesis bất kỳ). Có thể chỉ định tay --real --fake.

CHẠY: python3 report_prepare/mt07_freq_panel.py            # tự chọn cặp
      python3 report_prepare/mt07_freq_panel.py --cmap turbo --real <png> --fake <png>
Output: outputs/mt07_freq_panel.png (+ mt07_freq_diff.png: phổ Fake−Real)
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct

import common as C

DS = C.REPO / "datasets"


def dct2_logmag(gray):
    """log(1+|2D-DCT|) chuẩn ortho. gray: HxW float[0,1]."""
    D = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
    return np.log1p(np.abs(D))


def load256(path):
    return np.asarray(Image.open(path).convert("RGB").resize((256, 256)), np.float32) / 255.0


def gray(img):
    return img @ np.array([0.299, 0.587, 0.114], np.float32)


def mouth(img):
    """Cắt vùng miệng (nửa dưới, giữa) rồi resize 256 — nơi reenactment để lộ artifact."""
    h, w = img.shape[:2]
    crop = img[int(h * 0.58):int(h * 0.95), int(w * 0.22):int(w * 0.78)]
    return np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((256, 256)), np.float32) / 255.0


def pick_frame(folder):
    fs = sorted(glob.glob(str(folder / "*.png")))
    return fs[len(fs) // 2] if fs else None  # frame giữa cho ổn định


def auto_pair():
    """Ưu tiên cặp FF++ cùng nhận dạng; fallback Celeb-DF."""
    # FF++: fake = Deepfakes/c23/frames/<tgt>_<src>, real = youtube/c23/frames/<tgt>
    fake_dirs = sorted(glob.glob(str(DS / "FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/*")))
    for fd in fake_dirs:
        tgt = Path(fd).name.split("_")[0]
        rd = DS / f"FaceForensics++/original_sequences/youtube/c23/frames/{tgt}"
        if rd.exists():
            rf, ff = pick_frame(rd), pick_frame(Path(fd))
            if rf and ff:
                return rf, ff, f"FF++ Deepfakes (id {tgt})"
    # Celeb-DF fallback (không khớp nhận dạng tuyệt đối)
    rd = sorted(glob.glob(str(DS / "Celeb-DF-v2/Celeb-real/frames/*")))
    fd = sorted(glob.glob(str(DS / "Celeb-DF-v2/Celeb-synthesis/frames/*")))
    if rd and fd:
        return pick_frame(Path(rd[0])), pick_frame(Path(fd[0])), "Celeb-DF-v2"
    return None, None, None


def _vrange(sr, sf):
    """Stretch theo percentile (chung cho real+fake) để hiện rainbow speckle thay vì bị DC nén về xanh."""
    both = np.concatenate([sr.ravel(), sf.ravel()])
    # stretch để bulk (high-freq) rơi vào giữa colormap → thân vàng-lục + góc DC ấm, giống style paper
    return float(np.percentile(both, 2)), float(np.percentile(both, 82))


def spectrum_panel(plt, real_img, fake_img, cmap, out_name, title):
    blocks = [("(a)", "(b)", real_img, fake_img, "full face"),
              ("(c)", "(d)", mouth(real_img), mouth(fake_img), "mouth")]
    fig, ax = plt.subplots(4, 2, figsize=(6, 11))
    ax[0, 0].set_title("Real", fontsize=14, fontweight="bold")
    ax[0, 1].set_title("Fake", fontsize=14, fontweight="bold")
    for bi, (la, lb, ri, fi, _) in enumerate(blocks):
        r0 = bi * 2
        ax[r0, 0].imshow(ri); ax[r0, 1].imshow(fi)
        sr, sf = dct2_logmag(gray(ri)), dct2_logmag(gray(fi))
        vmin, vmax = _vrange(sr, sf)
        ax[r0 + 1, 0].imshow(sr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[r0 + 1, 1].imshow(sf, cmap=cmap, vmin=vmin, vmax=vmax)
        ax[r0 + 1, 0].set_xlabel(la, fontsize=12); ax[r0 + 1, 1].set_xlabel(lb, fontsize=12)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, y=0.995, fontsize=12)
    C.save_fig(fig, out_name)


def _sample_images(globs, n):
    files = []
    for g in globs:
        files += glob.glob(g)
    files = sorted(files)[:: max(1, len(files) // max(n, 1))][:n]
    return files


def mean_spectrum(files):
    acc = None
    cnt = 0
    for f in files:
        try:
            acc = (dct2_logmag(gray(load256(f))) if acc is None
                   else acc + dct2_logmag(gray(load256(f))))
            cnt += 1
        except Exception:
            pass
    return acc / max(cnt, 1), cnt


def radial_profile(spec):
    """Năng lượng trung bình theo bán kính tần số (DC ở góc 0,0) → 1D: low→high freq."""
    h, w = spec.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(xx ** 2 + yy ** 2).astype(int)
    return np.array([spec[r == i].mean() for i in range(r.max() + 1)])


def aggregate_diff(plt, cmap, out_name, radial_name, n=150):
    """Phổ DCT TRUNG BÌNH real vs fake (trên ~n ảnh mỗi lớp) → artifact hệ thống ở dải mid/high."""
    real_g = [str(DS / "FaceForensics++/original_sequences/youtube/c23/frames/*/*.png"),
              str(DS / "Celeb-DF-v2/Celeb-real/frames/*/*.png")]
    fake_g = [str(DS / "FaceForensics++/manipulated_sequences/Deepfakes/c23/frames/*/*.png"),
              str(DS / "Celeb-DF-v2/Celeb-synthesis/frames/*/*.png")]
    sr, nr = mean_spectrum(_sample_images(real_g, n))
    sf, nf = mean_spectrum(_sample_images(fake_g, n))
    if nr == 0 or nf == 0:
        print("[MT-7c] không đủ ảnh để tính phổ trung bình."); return
    # (1) 3-panel phổ trung bình + diff
    d = sf - sr
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    vmin, vmax = _vrange(sr, sf)
    ax[0].imshow(sr, cmap=cmap, vmin=vmin, vmax=vmax); ax[0].set_title(f"mean log|DCT| REAL (n={nr})")
    ax[1].imshow(sf, cmap=cmap, vmin=vmin, vmax=vmax); ax[1].set_title(f"mean log|DCT| FAKE (n={nf})")
    vlim = float(np.percentile(np.abs(d), 99)) or 1   # bỏ outlier để lộ pattern hệ thống
    im = ax[2].imshow(d, cmap="coolwarm", vmin=-vlim, vmax=vlim)
    ax[2].set_title("FAKE − REAL (đỏ = fake nhiều hơn)")
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle("MT-7 · Phổ DCT trung bình: fake rò năng lượng vào dải tần — bằng chứng cho block-DCT", y=1.02)
    C.save_fig(fig, out_name)
    # (2) đường năng lượng theo dải tần — 2 panel: (trái) log-y real vs fake, (phải) fake−real
    pr, pf = radial_profile(sr), radial_profile(sf)
    x = np.linspace(0, 1, len(pr))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    ax[0].plot(x, pr, label=f"REAL (n={nr})", color="tab:green", lw=2)
    ax[0].plot(x, pf, label=f"FAKE (n={nf})", color="tab:red", lw=2)
    ax[0].set_yscale("log"); ax[0].axvspan(0.33, 0.8, color="orange", alpha=0.12, label="dải mid/high")
    ax[0].set_xlabel("tần số chuẩn hoá (0=DC → 1=cao)"); ax[0].set_ylabel("mean log|DCT| (log-y)")
    ax[0].set_title("Năng lượng theo dải tần"); ax[0].legend()
    diff = pf - pr
    ax[1].plot(x, diff, color="tab:purple", lw=2)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].axvspan(0.33, 0.8, color="orange", alpha=0.12, label="dải mid/high (DCT branch)")
    ax[1].set_xlabel("tần số chuẩn hoá"); ax[1].set_ylabel("FAKE − REAL")
    ax[1].set_title("Chênh lệch năng lượng (fake − real)"); ax[1].legend()
    fig.suptitle("MT-7 · Năng lượng DCT theo dải tần — bằng chứng định lượng (c23: artifact yếu)", y=1.02)
    C.save_fig(fig, radial_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real"); ap.add_argument("--fake")
    ap.add_argument("--cmap", default="jet", help="jet / turbo / nipy_spectral / gist_rainbow")
    a = ap.parse_args()
    plt = C.setup_mpl()
    if a.real and a.fake:
        rp, fp, src = a.real, a.fake, "tự chỉ định"
    else:
        rp, fp, src = auto_pair()
    if not rp or not fp:
        print("[MT-7c] Không tìm thấy cặp ảnh real/fake trong datasets/."); return
    print(f"[MT-7c] cặp: {src}\n  real={rp}\n  fake={fp}")
    real_img, fake_img = load256(rp), load256(fp)
    spectrum_panel(plt, real_img, fake_img, a.cmap, "mt07_freq_panel.png",
                   f"MT-7 · Real vs Fake + log|2D-DCT|  ({src})")
    aggregate_diff(plt, a.cmap, "mt07_freq_diff.png", "mt07_freq_radial.png", n=150)


if __name__ == "__main__":
    main()
