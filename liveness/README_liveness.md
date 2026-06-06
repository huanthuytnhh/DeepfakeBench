# Liveness / Face Anti-Spoofing — module ĐỘC LẬP (B4 vs B4+block-DCT)

Tách hoàn toàn khỏi code deepfake (**tránh xung đột**): mọi thứ trong `liveness/`, đuôi `_liveness`.
Mỗi model làm **TƯƠNG ĐƯƠNG NHẤT** với detector deepfake tương ứng → ablation công bằng "block-DCT giúp CẢ 2".

## Tương đương (đã verify máy móc)
| Model liveness | ≈ detector deepfake | state_dict keys |
|---|---|---|
| `b4` (`model_liveness.py`) | EfficientNetB4 | **706/706 khớp** ✅ |
| `b4dct` (`model_dct_liveness.py`) | naive `efficientnetb4_sfdct` | **723/723 khớp** ✅ |

Cùng `efficientnet_pytorch` + cùng pretrained + **norm 0.5/0.5 · res 256 · INTER_CUBIC** + cùng aug +
Adam 2e-4 wd 5e-4 + **batch 32** + block-DCT (`dct_core_liveness.py` = bản copy byte-identical của
`sfdct_core.py`) + gated fusion zero-init + gate warm-up (gate_lr_mult 3.0). Chỉ phụ thuộc pip
`efficientnet_pytorch`, **KHÔNG import `training/detectors`**.

## Chạy trên vast 5090 (sau khi fix1/fix2 xong)
```bash
cd /workspace/DeepfakeBench
git fetch origin feat/fas-liveness && git checkout feat/fas-liveness   # ⚠️ chỉ khi fix1/fix2 đã xong
export HF_TOKEN=hf_...                                  # để kéo data từ HF + push model

SMOKE=1 ./liveness/run_liveness.sh        # smoke TRƯỚC (rule)
BATCH=32 ./liveness/run_liveness.sh       # ablation B4 vs B4+DCT -> compare -> push HF runs/liveness-<ts>/
```
- Data **tự kéo từ HF** `huanthuytnhh/deepfake-data/lcc-fasd.zip` (không cần Kaggle creds trên box).
- Kéo model về local: `huggingface-cli download huanthuytnhh/deepfake --include 'runs/liveness-<ts>/*' --local-dir ./liveness_pulled`

## Files (đuôi _liveness)
| File | Vai trò |
|---|---|
| `model_liveness.py` | B4 baseline (≈ B4 detector) |
| `model_dct_liveness.py` | B4 + block-DCT (≈ naive SFDCT detector) |
| `dct_core_liveness.py` | copy byte-identical của `sfdct_core.py` (ContentDCT + GatedCrossAttnFusion) |
| `dataset_liveness.py` | LCC-FASD loader (norm/res/aug khớp deepfake) |
| `metrics_liveness.py` | APCER/BPCER/ACER/EER/AUC, ngưỡng @dev-EER |
| `train_liveness.py` | train `--model b4|b4dct` (AMP + early-stop + gate warm-up) |
| `download_liveness.py` | lấy LCC-FASD (HF zip → fallback Kaggle) |
| `compare_liveness.py` · `run_liveness.sh` | bảng B4 vs B4+DCT · 1 lệnh ablation + push HF |

## LCC-FASD (18,827 ảnh, split sẵn — ⚠️ lệch spoff>>real)
training 1223/7076 · development 405/2543 · evaluation 314/7266 (live/spoof). AUC chịu được lệch; đừng nhìn accuracy.

## Kỳ vọng & cảnh báo
AUC ~0.88–0.92, ACER ~16–21% (CNN nhẹ, từ literature). AUC ~0.99 thường = **leak** → kiểm split.
Block-DCT **có thể** giúp liveness (replay/moiré hợp DCT) hoặc không (print giấy ít) — thí nghiệm trung thực.
