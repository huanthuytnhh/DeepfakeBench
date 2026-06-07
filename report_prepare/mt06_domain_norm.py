#!/usr/bin/env python3
"""MT-6 · Chuẩn hóa MIỀN trước inference (domain normalization).
(← novelty #6 two-pass key-normalize của Trí: đoán "miền" → chuẩn hóa → predict → chuyển ngược)

Ý tưởng áp cho deepfake/eKYC: giảm phương sai đầu vào (mức nén JPEG, độ phân giải, ánh sáng,
white-balance) về MỘT MIỀN CHUẨN trước khi đưa qua detector → cải thiện cross-dataset.
Two-pass: pass-1 ước lượng điều kiện ảnh → chọn τ/normalize phù hợp → pass-2 detect.

CODE-FIRST — cần retrain/fine-tune để đo tác dụng. Đây là khung transform + wrapper.
Chạy mai: train lại với DomainNormalize bật, rồi so cross-dataset có/không (dùng mt04).
"""
import numpy as np


class DomainNormalize:
    """Transform chuẩn hóa miền ảnh khuôn mặt đã crop (áp TRƯỚC normalize mean/std của model).
    Đặt vào pipeline tiền xử lý của detector (cả train lẫn serve để nhất quán)."""

    def __init__(self, target_jpeg=90, target_size=256, gray_world=True, clahe=False):
        self.target_jpeg = target_jpeg
        self.target_size = target_size
        self.gray_world = gray_world
        self.clahe = clahe

    def __call__(self, img):  # img: HxWx3 uint8 RGB (mặt đã MTCNN-crop)
        import cv2
        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        if self.gray_world:  # cân bằng trắng gray-world → khử lệch màu giữa camera/dataset
            m = img.reshape(-1, 3).mean(0) + 1e-6
            img = np.clip(img.astype(np.float32) * (m.mean() / m), 0, 255).astype(np.uint8)
        if self.clahe:       # cân bằng sáng cục bộ → khử lệch phơi sáng
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        # tái nén về JPEG chuẩn → đồng nhất artifact nén giữa các nguồn
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.target_jpeg])
        if ok:
            img = cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)
        return img


def two_pass_infer(model, face_img, estimate_condition, transform):
    """Khung two-pass: pass-1 ước lượng điều kiện → chuẩn hóa → pass-2 detect.
    estimate_condition(face_img) -> dict (vd {'low_res':True}); ở đây dùng để chọn nhánh normalize."""
    cond = estimate_condition(face_img)         # pass-1: cheap probe
    norm = transform(face_img)                  # chuẩn hóa miền theo điều kiện
    prob = model(norm)                          # pass-2: detect trên miền chuẩn
    return prob, cond


if __name__ == "__main__":
    print(__doc__)
    print("[MT-6] CODE-FIRST: gắn DomainNormalize vào DataPreprocessor (train+serve), retrain, "
          "rồi so cross-dataset có/không normalize bằng mt04_cross_dataset.py.")
