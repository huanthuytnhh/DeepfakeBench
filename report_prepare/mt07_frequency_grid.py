#!/usr/bin/env python3
"""MT-7 · Trực quan hóa BẰNG CHỨNG TẦN SỐ (chứng minh hypothesis block-DCT).
(← novelty #8 CQT của Trí → block-DCT mid/high-band cho deepfake)

Ghép các hình đã sinh sẵn (frequency.png = mean log|DCT| real vs fake; gradcam.png) của mọi
model thành 1 grid để đặt cạnh nhau trong báo cáo — chứng minh "artifact fake leak vào dải
DCT mid/high" + Grad-CAM nhìn đúng vùng.

Trạng thái: CHẠY ĐƯỢC NGAY (ghép PNG sẵn có trong viz_out/).
Phần DCT-band radial profile tươi (đọc ảnh thật) → cần thư mục ảnh real/fake: xem mt07 --imgdir (TODO).
Output: outputs/mt07_frequency_grid.png, outputs/mt07_gradcam_grid.png
"""
import argparse
from pathlib import Path

import common as C


def _grid(kind: str, out_name: str, title: str):
    plt = C.setup_mpl()
    import matplotlib.image as mpimg
    models = [m for m in C.list_models() if (C.VIZ / m / f"{kind}.png").exists()]
    if not models:
        print(f"[MT-7] Không thấy {kind}.png trong viz_out/."); return
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        ax.imshow(mpimg.imread(C.VIZ / m / f"{kind}.png"))
        ax.set_title(C.model_label(m), fontsize=11)
        ax.axis("off")
    fig.suptitle(title, y=1.02, fontsize=13)
    C.save_fig(fig, out_name)


def dct_band_profile(imgdir: str):
    """TODO (chạy mai): đọc ảnh real/fake từ imgdir → radial profile năng lượng theo dải DCT.
    Cần cấu trúc imgdir/real/*.png, imgdir/fake/*.png. Tái dùng sfdct_core.dct_matrix."""
    import numpy as np
    from PIL import Image
    plt = C.setup_mpl()
    import sys
    sys.path.insert(0, str(C.REPO / "training"))
    try:
        from sfdct_core import dct_matrix  # type: ignore
        M = dct_matrix(8)
    except Exception as e:
        print(f"[MT-7] không import được sfdct_core.dct_matrix ({e}); bỏ qua radial profile."); return

    def avg_dct_energy(folder):
        files = sorted(Path(folder).glob("*"))[:200]
        acc = None
        for f in files:
            try:
                g = np.asarray(Image.open(f).convert("L").resize((256, 256)), np.float32)
            except Exception:
                continue
            blocks = g.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
            coef = np.abs(np.einsum("ij,bjk,lk->bil", M, blocks, M)).mean(0)
            acc = coef if acc is None else acc + coef
        return None if acc is None else np.log1p(acc / max(len(files), 1))

    er = avg_dct_energy(Path(imgdir) / "real"); ef = avg_dct_energy(Path(imgdir) / "fake")
    if er is None or ef is None:
        print("[MT-7] thiếu ảnh real/fake trong imgdir."); return
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].imshow(er, cmap="viridis"); ax[0].set_title("REAL · mean log|DCT| 8×8")
    ax[1].imshow(ef, cmap="viridis"); ax[1].set_title("FAKE · mean log|DCT| 8×8")
    ax[2].imshow(ef - er, cmap="coolwarm"); ax[2].set_title("FAKE − REAL (mid/high band leak)")
    for a in ax: a.set_xlabel("freq u"); a.set_ylabel("freq v")
    fig.suptitle("MT-7 · Bằng chứng artifact ở dải DCT mid/high", y=1.03)
    C.save_fig(fig, "mt07_dct_band_diff.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgdir", default=None, help="thư mục ảnh real/ & fake/ để vẽ DCT-band tươi (tùy chọn)")
    a = ap.parse_args()
    _grid("frequency", "mt07_frequency_grid.png", "MT-7 · Phổ DCT real-vs-fake (mỗi model)")
    _grid("gradcam", "mt07_gradcam_grid.png", "MT-7 · Grad-CAM (mỗi model)")
    if a.imgdir:
        dct_band_profile(a.imgdir)
    else:
        print("[MT-7] (tùy chọn) thêm --imgdir <dir có real/ & fake/> để vẽ DCT-band diff tươi.")


if __name__ == "__main__":
    main()
