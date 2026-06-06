# SFDCT — Hướng dẫn chạy (dễ nhất, ai cũng chạy được)

**SFDCT** = Hybrid Spatial–Frequency Learning với **block-wise DCT** để phát hiện **deepfake** cho **eKYC**.
Train trên **FaceForensics++ c23**, đánh giá **cross-dataset trên Celeb-DF-v2**, so với baseline EfficientNet-B4.

> Branch: **`feat/sfdct-dct-fomixup`** · Kết quả (model + figures) tự push lên **Hugging Face `huanthuytnhh/deepfake`**.

---

## 0. Cần gì trước khi bắt đầu
| Thứ | Yêu cầu |
|---|---|
| **GPU** | **≥ 24GB VRAM** (RTX 3090 / 4090 / 5090). 16GB chỉ chạy được batch 16 (xem Troubleshooting). |
| **Python** | **3.10 – 3.12** (3.13+ sẽ lỗi build — đổi box). |
| **Box** | Verified host, ≥16 vCPU, ≥300GB disk, có internet. |
| **HF token** | Tạo ở https://huggingface.co/settings/tokens (role **Write**) — để push model/figures. |

---

## 1. Chạy train — copy từng dòng (vast.ai / bất kỳ box Linux)

```bash
# (0) KIỂM TRA PYTHON TRƯỚC — phải ≤ 3.12, nếu 3.13/3.14 thì đổi box
python --version

# (1) Lấy code
git clone https://github.com/huanthuytnhh/DeepfakeBench.git
cd DeepfakeBench
git checkout feat/sfdct-dct-fomixup

# (2) Token HF (để push model + figures)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

# (3) Setup deps (pinned, tự fail-fast nếu thiếu/lỗi thư viện)
./start.sh setup

# (4) Tải + giải nén dataset (FF++ c23 + Celeb-DF-v2, ~33GB, tự bung)
./start.sh data

# (5) Verify + smoke (smoke = 1 epoch + test push HF — CỬA CHỐT)
./start.sh verify
./start.sh smoke

# (6) Train Row1 + Row2 (~6h trên 24GB) — chạy nền, sống sót rớt SSH
nohup ./start.sh ablation > pipeline.log 2>&1 &
echo "PID $!"
```

**Xong!** Để nó chạy ~6h. Sống sót rớt SSH (nohup). Sáng dậy có kết quả trên HF.

---

## 2. Theo dõi tiến độ
```bash
# tiến độ epoch/iter của Row1 (live, Ctrl-C để thoát xem — job vẫn chạy)
tail -f logs/ablation_cdfv2/row1_sfdct_s1024.train.log

# GPU đang chạy?  (cần thấy python + VRAM dùng + util >0%)
nvidia-smi

# kết quả CDFv2 cuối cùng (khi xong)
cat logs/ablation_cdfv2/cdfv2_summary_*.txt
```

---

## 3. Kết quả ở đâu
Tự push lên **`https://huggingface.co/huanthuytnhh/deepfake/tree/main/runs/improved-<timestamp>/`**:
```
runs/improved-<ts>/
├── row1_sfdct_s1024/        # Row1 = block-DCT thiết kế tay (0 tham số thêm)
│   ├── ckpt/                #   ckpt_best.pth + metric.pickle
│   ├── viz/                 #   10 figures: roc, pr, radar, ap_bar, heatmap, tsne,
│   │                        #   frequency, gradcam, gate, training_curve
│   ├── config.yaml          #   kiến trúc (để inference)
│   └── *.log
├── row2_sfdct_v2_s1024/     # Row2 = học band (FcaNet) + single-center loss
└── cdfv2_summary.txt        # bảng AUC Celeb-DF-v2 Row1/Row2 vs B4 (0.7487)
```

---

## 4. Inference — dùng model đã train (ra: số + REAL/FAKE + ảnh Grad-CAM)
```bash
# tải model + config từ HF
huggingface-cli download huanthuytnhh/deepfake runs/improved-<ts>/row2_sfdct_v2_s1024/ --local-dir ./m

# chạy trên 1 ảnh (hoặc 1 folder)
python tools/infer.py \
  --detector_path m/config.yaml \
  --weights m/ckpt/test/Celeb-DF-v2/ckpt_best.pth \
  --input khuon_mat.jpg --gradcam --out_dir ket_qua
# -> khuon_mat.jpg   fake_prob=0.87   FAKE   gradcam=ket_qua/khuon_mat_gradcam.png
```
> Ảnh đầu vào nên là **khuôn mặt đã crop** (model train trên face crop). Ngưỡng mặc định 0.5 (`--thr` để đổi; eKYC nên calibrate FPR≤5%).

---

## 5. Các lệnh `./start.sh`
| Lệnh | Việc |
|---|---|
| `setup` | cài deps (pinned) + weight B4 + dataset JSON + patch config + verify |
| `data` | tải + giải nén FF++ & Celeb-DF-v2 (tự bung, bỏ qua nếu đã có) |
| `verify` | kiểm 1 frame trên đĩa + 2 detector đã đăng ký |
| `smoke` | smoke 1 epoch end-to-end + test push HF (bắt lỗi trước khi train thật) |
| `ablation` | train **Row1 + Row2** (B4 reuse) → figures → push HF. `RUN_B4=1` để train cả B4. |
| `viz` | tạo bộ figures từ checkpoint mới nhất |
| `model` | upload `.pth` + figures lên HF |
| `all` | làm hết trong tmux/nohup (setup→data→verify→smoke→ablation) |

---

## 6. Method — các "đòn" cải tiến (bật/tắt trong `training/config/detector/efficientnetb4_sfdct.yaml`)
| Knob | Đòn | Adapt từ |
|---|---|---|
| `dct_drop_low_bands: k` | bỏ DC + k-1 band thấp (chống content-leakage) | — |
| `dct_use_sign: true` | thêm DCT coeff sign (phase-analog) | SPSL |
| `dct_srm_residual: true` | block-DCT trên SRM high-pass residual | SRM |
| `use_dct_fomixup: true` | trộn band DCT khi train + consistency loss | FreqDebias |
| `dct_fca_attention: true` | học band (FcaNet multi-spectral attention) | FcaNet |
| `use_single_center_loss: true` | loss metric tách real/fake | FDFL |

- **Row1** = `use_dct_fomixup + dct_use_sign + dct_srm_residual + drop_low_bands` (0 tham số thêm)
- **Row2** = `dct_fca_attention + use_single_center_loss + use_dct_fomixup` (học band)

---

## 7. Troubleshooting (lỗi hay gặp + cách sửa)
| Lỗi | Sửa |
|---|---|
| **Setup build fail / `_PyLong_AsByteArray`** | Box Python ≥3.13. → Thuê box **Python 3.10-3.12**. |
| **`CUDA out of memory`** (GPU 16GB) | Giảm batch: `sed -i 's/^train_batchSize:.*/train_batchSize: 16/' training/config/detector/efficientnetb4_sfdct.yaml` rồi chạy lại. |
| **Data chỉ có `.zip`, chưa bung** | Chạy lại `./start.sh data` (tự bung), hoặc bung tay: `cd datasets && unzip -qn Celeb-DF-v2.zip && unzip -qn FaceForensics++.zip && cd ..` |
| **`HF_TOKEN not set`** | `export HF_TOKEN=hf_...` (role Write) TRƯỚC khi chạy. Thiếu → kết quả zip về `/workspace`. |
| **tmux không có** | `./start.sh all` tự fallback nohup; hoặc dùng `nohup ./start.sh ablation > pipeline.log 2>&1 &`. |
| **Rớt SSH** | Không sao — `nohup`/`tmux` giữ job chạy. Nối lại: `tail -f pipeline.log`. |

---

## 8. Kết quả tham chiếu (đã chạy, batch 32, FF++ c23 → Celeb-DF-v2)
| Model | Celeb-DF-v2 frame-AUC |
|---|---|
| EfficientNet-B4 (baseline, bar) | **0.7497** (≈ leaderboard 0.7487) |
| SFDCT naive (block-DCT + gated fusion) | **0.7572** (+0.75) |
| Row1 / Row2 (improved) | *(đang train — xem `cdfv2_summary.txt`)* |

> ⚠️ Đây là kết quả **1 seed**. Để claim chắc chắn cần ≥5 seed (mean±std). Đòn bẩy lớn nhất cho AUC cao là **SBI self-blended training** (ngoài phạm vi block-DCT thuần).
