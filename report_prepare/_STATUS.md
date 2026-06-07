# _STATUS.md — Trạng thái 10 mục tiêu + việc cần làm tiếp (mai)

Cập nhật: sau lần dựng report_prepare/ đầu tiên. ✅ = xong & đã chạy ra output. 🟠 = code sẵn, cần chạy thêm.

## ✅ ĐÃ HOÀN THÀNH (chạy ngay, không cần GPU) — output trong `outputs/`
- **MT-9** training curve THẬT (4 run, loss+AUC) — `outputs/training_curves/*.png`
- **MT-9b** CSV raw audit (3 model × 16420 dòng) — `outputs/raw_scores/*.csv`
- **MT-1** ma trận metric + heatmap — `outputs/mt01_*`
- **MT-2** bảng τ eKYC (FPR≤5%/1%) + score-dist — `outputs/mt02_threshold_table.*`, `mt02_score_dist_*.png`
- **MT-2b** confusion matrix THẬT @τ + per-class — `outputs/mt02_confusion_*.png`, `mt02_per_class.*`
- **MT-5** AUC ± bootstrap CI + ΔAUC (unpaired) — `outputs/mt05_auc_ci.*`, `mt05_paired_diff.*`, `mt05_ablation_bar.png`
  - **Kết quả chốt:** naive−B4 = **+0.0075, CI95 [−0.004, +0.019], p=0.184 → TRONG NHIỄU** (trung thực); hf row1 kém hơn có ý nghĩa (−0.0165, p=0.007).
- **MT-5b** `experiments/results.csv` đã điền (21 run). mean±std: efficientnetb4_dct 0.793±0.033 (cao nhất, n=3).
- **MT-4b** bảng + heatmap cross-dataset (hiện 1 test-set: Celeb-DF-v2) — `outputs/mt04_cross_*`
- **MT-7** grid phổ DCT + Grad-CAM — `outputs/mt07_frequency_grid.png`, `mt07_gradcam_grid.png`

## ⚠️ CẦN VERIFY
- **MT-4a** phân bố dataset: counts đang là **ước lượng theo cấu trúc JSON** (45/45, 60×5 — quá tròn).
  → Mai: đối chiếu số frame thật (đếm theo `frames`/`video_path` trong json) trước khi đưa vào báo cáo.

## 🟠 CODE-FIRST — hoàn thiện mai (cần GPU / train / service)

### MT-8 · Liveness (1-2h local, $0) — VẼ ĐÃ SẴN
```bash
cd /home/huanthuytnhh/Desktop/thanhln/datn/DeepfakeBench
export HF_TOKEN=...
AUG=fas BATCH=32 ./liveness/run_liveness.sh        # sinh liveness/out/metrics_liveness.json + scores
python3 report_prepare/mt08_liveness_report.py     # → ROC+DET+ACER/APCER/BPCER+hist (neo TT17 APCER≤5%)
```

### MT-3 · Robustness curve (cần weights + ảnh test, GPU)
```bash
python3 report_prepare/mt03_robustness.py \
    --detector_path training/config/detector/efficientnetb4_sfdct.yaml \
    --weights_path  <log_dir>/.../ckpt_best.pth --test_dataset Celeb-DF-v2
# hàm nhiễu jpeg/downscale/noise đã sẵn; nối vào loader của eval_and_viz để hoàn thiện.
```

### MT-4 mở rộng · eval thêm bộ cross-dataset (DFDC/DeeperForensics, GPU)
```bash
python3 training/eval_and_viz.py --detector_path <cfg> --weights_path <ckpt> \
    --test_dataset DFDC --out viz_out/<model>
python3 report_prepare/mt04_cross_dataset.py    # heatmap tự thêm cột
```

### MT-6 · Domain-normalize (cần retrain để đo tác dụng)
- Gắn `DomainNormalize` (trong `mt06_domain_norm.py`) vào DataPreprocessor (train+serve) → retrain → so cross-dataset có/không bằng MT-4b.

### MT-10 · Test API thật + CI (cần backend :8000 + SFDCT :8501)
```bash
uvicorn app.main:app --port 8000 &
python3 DeepfakeBench/serving/infer_server.py &
DG_SAMPLE_REAL=... DG_SAMPLE_FAKE=... pytest report_prepare/mt10_api_test_template.py -v
# gắn middleware/health theo mt10_observability.md; bật GitHub Actions KHÔNG '|| true'
```

## 🆕 NOVELTY THÊM
- **Bộ deepfake người Việt (30 video)** — giao thức đầy đủ trong `VIETNAM_DEEPFAKE_DATASET.md`.
  Khi có → eval bằng `eval_and_viz.py --test_dataset VN-Deepfake` rồi chạy lại MT-1/2/4 (tự thêm cột).

## Ưu tiên đề xuất
Tuần 1 (xong phần ✅) → **mai**: MT-8 (liveness, ăn điểm TT17) → verify MT-4a → MT-4 mở rộng (DFDC) → MT-3.
Tuần sau: MT-6 (retrain), MT-10 (ops/CI), bộ VN-Deepfake.
