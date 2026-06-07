# report_prepare/ — Chuẩn bị số liệu & biểu đồ cho báo cáo DATN

Thư mục này gom **10 cải tiến (MT-1…MT-10)** rút ra từ việc học đồ án thủ khoa Trần Đức Trí
(HarmonySeeker), adapt cho đồ án **deepfake/SFDCT (eKYC)**. Cấu trúc mô phỏng `SongChordRecognizer_Report/`
của Trí: mỗi script sinh một loại hình/bảng, đọc **số liệu THẬT** từ `viz_out/`, `logs/`, `experiments/`.

> Nguyên tắc xương sống (rút từ bẫy của Trí): **mọi số sinh từ log/.npz THẬT — không hard-code, không `np.random`, không test giả.**

## Chạy nhanh (nhóm không cần GPU)
```bash
cd DeepfakeBench/report_prepare
bash run_quickwins.sh         # MT-1,2,4(bảng),5,7(grid),9 — đọc viz_out/ + logs/
# kết quả → report_prepare/outputs/  (mỗi bảng có .csv + .md để dán thẳng báo cáo)
```

## 10 mục tiêu ↔ novelty của Trí ↔ output

| MT | Nội dung | ← Novelty | Script | Trạng thái |
|----|----------|-----------|--------|------------|
| **1** | Ma trận đánh giá đa-cấu hình + heatmap | #2 ma trận 16-config | `mt01_eval_matrix.py` | ✅ chạy được |
| **2** | Ngưỡng τ eKYC (FPR≤5%) + score-dist | #1 mode-aware→threshold | `mt02_threshold_ekyc.py` | ✅ |
| **2b** | Confusion matrix THẬT + per-class | #1 (sửa "Hình 3.9") | `mt02_confusion_matrix.py` | ✅ |
| **3** | Robustness curve (nén/res/nhiễu) | #3 semi-strict tolerance | `mt03_robustness.py` | 🟠 code-first (cần GPU) |
| **4a** | Phân bố dataset real/fake/manip | #4 cross-dataset | `mt04_dataset_distribution.py` | ✅ (counts ước lượng — verify) |
| **4b** | Bảng + heatmap cross-dataset | #4 | `mt04_cross_dataset.py` | ✅ (mở rộng khi eval thêm bộ) |
| **5** | Ablation B4 vs B4+DCT + bootstrap CI | #5 negative result + #7 | `mt05_ablation_ci.py` | ✅ |
| **5b** | Điền `experiments/results.csv` từ log | #5 multi-seed | `mt05_fill_results_csv.py` | ✅ |
| **6** | Domain-normalize trước inference | #6 two-pass key-normalize | `mt06_domain_norm.py` | 🟠 code-first (cần retrain) |
| **7** | Grid phổ DCT + Grad-CAM | #8 CQT→block-DCT | `mt07_frequency_grid.py` | ✅ (DCT-band tươi cần --imgdir) |
| **8** | Liveness ISO 30107-3 + DET + BPCER@APCER | #9 MIR metrics | `mt08_liveness_report.py` | 🟠 vẽ sẵn, cần chạy train (1-2h) |
| **9** | Training curve THẬT (loss+AUC) | #5 (tránh np.random) | `mt09_training_curves.py` | ✅ |
| **9b** | Export CSV raw audit | #5 | `mt09_export_raw_csv.py` | ✅ |
| **10** | Test API thật + observability + CI | #10 AWS ops | `mt10_api_test_template.py`, `mt10_observability.md` | 🟠 code-first (cần service chạy) |

> 🟠 = đã code sẵn, cần GPU/train/service → **hoàn thiện sau** (chi tiết lệnh trong `_STATUS.md`).

## Dữ liệu nguồn (đã verify schema)
- `viz_out/<model>/results.json` → `{dataset:{frame_auc,ap,eer,tpr@fpr=5%,tpr@fpr=1%,video_auc,n}}`
- `viz_out/<model>/scores_<dataset>.npz` → `prob(N,)`, `label(N,)` (1=fake), `feat(N,1792)`
- `logs/training/<run>/training.log` → Iter loss/AUC + per-epoch test AUC (FF++/Celeb-DF-v2)
- 3 model hiện có: `b4_local` (B4 baseline), `naive_local` (B4+block-DCT naive SFDCT), `row1_local` (B4+block-DCT hf)

## Liên quan
- `VIETNAM_DEEPFAKE_DATASET.md` — giao thức bộ deepfake người Việt tự thu (novelty thêm).
- `common.py` — thư viện chung (metrics, bootstrap CI, DET, ECE, style, IO).
- Báo cáo đích: `../../report/BAO_CAO_DATN_SFDCT.md` (các ô `[[FILL]]` Ch.3).
