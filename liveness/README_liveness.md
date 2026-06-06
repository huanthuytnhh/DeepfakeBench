# Liveness / Face Anti-Spoofing — module ĐỘC LẬP (B4 baseline)

Tách hoàn toàn khỏi code deepfake để **tránh xung đột file**: mọi thứ nằm trong `liveness/`, đuôi `_liveness`,
model B4 dựng bằng **torchvision** (không import `training/detectors/...`).

## Thiết kế (vì sao không xung đột)
| | Deepfake | Liveness (module này) |
|---|---|---|
| Model | `training/detectors/efficientnetb4*` | `liveness/model_liveness.py` (torchvision B4) |
| Data | `training/dataset/...` | `liveness/dataset_liveness.py` (LCC-FASD) |
| Metric | `training/metrics/...` (AUC) | `liveness/metrics_liveness.py` (APCER/BPCER/ACER/EER) |
| Train | `training/train.py` | `liveness/train_liveness.py` |
→ **Không file nào dùng chung.** Sửa bên này không ảnh hưởng bên kia.

## Chạy
```bash
cd DeepfakeBench
# Kaggle creds 1 lần: ~/.kaggle/kaggle.json (https://www.kaggle.com/settings)

SMOKE=1 ./liveness/run_liveness.sh        # smoke TRƯỚC (1 epoch, 200 ảnh/split) — rule
BATCH=16 ./liveness/run_liveness.sh       # full trên 4GB (AMP)
```
Kết quả → `liveness/out/b4/metrics_liveness.json` (APCER/BPCER/ACER/EER/AUC).

## Files
| File | Vai trò |
|---|---|
| `model_liveness.py` | B4 baseline (torchvision, head 2 lớp live/spoof) |
| `dataset_liveness.py` | LCC-FASD loader (RGB, resize, ImageNet norm) |
| `metrics_liveness.py` | APCER/BPCER/ACER/EER/AUC, ngưỡng @dev-EER |
| `download_liveness.py` | tải LCC-FASD (kagglehub) |
| `train_liveness.py` | train (AMP + early-stop) + eval |
| `run_liveness.sh` | 1 lệnh: tải → train B4 → metrics |

## Kỳ vọng (LCC-FASD, CNN nhẹ — từ literature verify)
AUC ~0.88–0.92, ACER ~16–21%. Cao hơn nhiều (AUC ~0.99) thường = **leak** identity/video → kiểm split.

## Mở rộng sau (nếu muốn ablation block-DCT cho liveness)
Thêm `build_b4dct_liveness()` vào `model_liveness.py` (nhánh DCT tự xây, vẫn KHÔNG import code deepfake)
→ so B4 vs B4+DCT trên liveness, chứng minh tần số giúp cả 2 bài toán.
