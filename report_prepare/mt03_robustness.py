#!/usr/bin/env python3
"""MT-3 · Robustness curve (nén JPEG / độ phân giải / nhiễu).
(← novelty #3 semi-strict tolerance của Trí → "tolerance theo điều kiện ảnh" cho deepfake)

CODE-FIRST — cần MODEL + ẢNH TEST (GPU). Hàm nhiễu đã sẵn; phần chạy eval gọi lại
training/eval_and_viz.py cho mỗi mức severity rồi gom AUC.

Chạy mai:
  python report_prepare/mt03_robustness.py \
      --detector_path training/config/detector/efficientnetb4_sfdct.yaml \
      --weights_path  <log_dir>/.../ckpt_best.pth \
      --test_dataset  Celeb-DF-v2

Output (khi chạy): outputs/mt03_robustness.{csv,md}, outputs/mt03_robustness.png
"""
import argparse
import io

import numpy as np

import common as C

# ── Hàm nhiễu (sẵn dùng — áp lên ảnh RGB uint8 HxWx3) ──
def jpeg_compress(img, quality):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def downscale(img, scale):
    from PIL import Image
    h, w = img.shape[:2]
    im = Image.fromarray(img).resize((max(1, int(w * scale)), max(1, int(h * scale))))
    return np.array(im.resize((w, h)).convert("RGB"))


def gaussian_noise(img, sigma):
    n = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


SEVERITIES = {
    "jpeg": [(q, lambda im, q=q: jpeg_compress(im, q)) for q in (100, 90, 70, 50, 30)],
    "downscale": [(s, lambda im, s=s: downscale(im, s)) for s in (1.0, 0.75, 0.5, 0.35, 0.25)],
    "noise": [(s, lambda im, s=s: gaussian_noise(im, s)) for s in (0, 5, 10, 20, 35)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_path")
    ap.add_argument("--weights_path")
    ap.add_argument("--test_dataset", nargs="+", default=["Celeb-DF-v2"])
    a = ap.parse_args()
    if not (a.detector_path and a.weights_path):
        print(__doc__)
        print("\n[MT-3] CODE-FIRST: cần --detector_path + --weights_path (GPU). Hàm nhiễu đã sẵn ở trên.")
        print("       Gợi ý: thêm 1 transform 'perturb' vào DataPreprocessor của eval_and_viz, lặp SEVERITIES,")
        print("       gom AUC mỗi mức rồi vẽ AUC-vs-severity cho jpeg/downscale/noise.")
        return
    # TODO (mai hoàn thiện): với mỗi (kind, level, fn): patch loader.dataset.transform để áp fn,
    # gọi lại pipeline infer của eval_and_viz, lấy AUC, append. Sau đó vẽ 3 đường.
    print("[MT-3] Khung chạy đã sẵn — nối hàm nhiễu vào loader của eval_and_viz để hoàn thiện.")


if __name__ == "__main__":
    main()
