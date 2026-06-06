# Liveness / Face Anti-Spoofing — module ĐỘC LẬP (B4 baseline ≈ B4 detector deepfake)

Tách hoàn toàn khỏi code deepfake (**tránh xung đột file**): mọi thứ trong `liveness/`, đuôi `_liveness`.
B4 được làm **TƯƠNG ĐƯƠNG NHẤT** với detector B4 của DeepfakeBench để so sánh công bằng.

## Tương đương ở đâu (đã verify: state_dict 706/706 keys khớp)
| | Deepfake B4 detector | Liveness B4 (module này) |
|---|---|---|
| Backbone | `efficientnet_pytorch` EfficientNet-B4 | **cùng** (qua `efficientnet_pytorch`) |
| Pretrained | `efficientnet-b4-6ed6700e.pth` | **cùng file** (fallback tải lukemelas y hệt) |
| Stem / head | `_conv_stem=Conv2d(3,48,3,s2)`, `Linear(1792,2)` | **cùng** |
| Chuẩn hoá | 0.5/0.5 → [-1,1] | **cùng** (`NORM_*`) |
| Độ phân giải | 256, resize INTER_CUBIC | **cùng** |
| Augmentation | flip/rotate/blur/bright-FancyPCA-HSV/JPEG | **mirror cùng tham số** |
| Optimizer | Adam lr 2e-4, wd 5e-4 | **cùng** |
| Batch | 32 | **32** |
→ Khác biệt duy nhất: **dữ liệu** (live/spoof thay vì real/fake) — đúng mục đích.
→ Self-contained: chỉ phụ thuộc pip `efficientnet_pytorch`, **KHÔNG import `training/detectors`**.

## Chạy trên vast 5090 (sau khi fix1/fix2 xong)
```bash
cd /workspace/DeepfakeBench
git fetch origin feat/fas-liveness && git checkout feat/fas-liveness
# Kaggle creds 1 lần trên box: ~/.kaggle/kaggle.json (https://www.kaggle.com/settings)
export HF_TOKEN=hf_...

SMOKE=1 ./liveness/run_liveness.sh        # smoke TRƯỚC (rule)
BATCH=32 ./liveness/run_liveness.sh       # full -> push model+metrics lên HF runs/liveness-<ts>/b4/
```
Kéo model về local:
```bash
huggingface-cli download huanthuytnhh/deepfake --include 'runs/liveness-<ts>/b4/*' --local-dir ./liveness_pulled
```

## Files (đuôi _liveness)
| File | Vai trò |
|---|---|
| `model_liveness.py` | B4 ≈ detector deepfake (efficientnet_pytorch, 0.5-norm, 256) |
| `dataset_liveness.py` | LCC-FASD loader (norm/res/aug khớp deepfake) |
| `metrics_liveness.py` | APCER/BPCER/ACER/EER/AUC, ngưỡng @dev-EER |
| `train_liveness.py` | train (AMP + early-stop, Adam 2e-4, batch 32) + eval |
| `download_liveness.py` · `run_liveness.sh` | tải LCC-FASD · 1 lệnh train + push HF |

## Kỳ vọng (LCC-FASD, từ literature verify)
AUC ~0.88–0.92, ACER ~16–21%. AUC ~0.99 thường = **leak** identity/video → kiểm split.

## Mở rộng: ablation block-DCT cho liveness
Thêm `build_b4dct_liveness()` (nhánh DCT tự xây, vẫn standalone) → so B4 vs B4+DCT trên liveness,
chứng minh tần số giúp CẢ deepfake và liveness.
