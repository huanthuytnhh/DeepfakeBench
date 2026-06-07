# TUTOR CHECKLIST — Đồ án Deepfake/SFDCT (học từ thủ khoa Trần Đức Trí)

> Nguồn gộp từ toàn bộ phân tích: 10 method/novelty của Trí · 4 "vỏ bọc" làm nên thủ khoa · 10 mục tiêu MT · lộ trình P0/P1/P2.
> Quy ước: [x] xong · [ ] chưa · 🔴 quyết định · 🟡 nên · 🟢 đánh bóng.

---

## 0. Định vị (nhớ trước khi làm gì)
Trí được điểm cao vì **vỏ bọc** (đánh giá music-aware + dataset VN + sản phẩm), KHÔNG vì model mới — nửa lõi là **fork Lanž không credit**. Bạn có lợi thế: **novelty method gốc (block-DCT)** + **app multi-tenant** vượt Trí. Hai việc biến đồ án thành "vượt thủ khoa": **chứng minh block-DCT thật** + **dataset người Việt**. Và phải **credit DeepfakeBench/SFDCT** (tránh lỗi liêm chính của Trí).

---

## 1. ĐÃ XONG (vỏ #3 — biểu đồ & đánh giá, từ data thật, không GPU)
- [x] Hạ tầng `report_prepare/` (`common.py` + scripts + `run_quickwins.sh`)
- [x] MT-9 training curve THẬT (loss+AUC, 4 run) · MT-9b CSV raw audit
- [x] MT-1 ma trận metric + heatmap
- [x] MT-2 bảng τ eKYC (FPR≤5%/1%) + histogram score real-vs-fake
- [x] MT-2b confusion matrix THẬT @τ + per-class (thay "Hình 3.9" lệch)
- [x] MT-5 ablation AUC ± bootstrap CI (**naive−B4=+0.0075, CI[-0.004,+0.019], trong nhiễu**) · MT-5b điền `experiments/results.csv` (21 run)
- [x] MT-4b cross-dataset table/heatmap (hiện 1 bộ: Celeb-DF-v2) · MT-4a phân bố dataset (⚠️ counts ước lượng — verify)
- [x] MT-7 tần số: panel real/fake+DCT, radial, diff, per-manipulation (matplotlib)
- [x] MT-7 + dashboard bản **Plotly tương tác** (`mt_plotly_report.py`, `mt07_plotly.py`) + render PNG qua chrome-headless

---

## 2. 🔴 P0 — HAI VIỆC QUYẾT ĐỊNH (làm trước hết)

### ① Chứng minh block-DCT đóng góp THẬT (novelty method)
- [ ] **Thử rẻ trước:** `mt_dct_corrects_spatial.py` — đo block-DCT sửa được bao nhiêu lỗi của B4 spatial-only (từ `scores`, KHÔNG cần train)
- [ ] Nếu chưa thắng → **FSBI/SBI** (block-DCT là chính, SBI chỉ là trick train) — train lại
- [ ] **Multi-seed ≥3** + bootstrap CI → chứng minh Δ có ý nghĩa (CI không chứa 0)
- [ ] Per-manipulation: chứng minh DCT giúp ở loại nào (NeuralTextures/F2F…)

### ② Bộ deepfake người Việt cho eKYC (novelty dataset — chắc nhất)
- [ ] Thu 30 video real (đa dạng, có consent, ghi metadata)
- [ ] MTCNN-crop → `dataset_vn/real/`
- [ ] Sinh fake ≥2 method (SimSwap + SadTalker/LivePortrait) → `dataset_vn/fake/<method>/`
- [ ] Đóng gói `VN-Deepfake.json` (chuẩn DeepfakeBench) → giữ **test-only**
- [ ] Eval → chạy lại MT-1/2/4 (tự thêm cột) ; báo **bootstrap CI** vì n nhỏ
- Chi tiết: `VIETNAM_DEEPFAKE_DATASET.md`

---

## 3. 🟡 P1 — Chạy phần cần GPU + hoàn thiện báo cáo
- [ ] **MT-8 liveness** ISO/IEC 30107-3 + DET + BPCER@APCER≤5% (1-2h local, $0): `AUG=fas BATCH=32 ./liveness/run_liveness.sh` → `python3 mt08_liveness_report.py`
- [ ] **MT-4 mở rộng cross-dataset** DFDC/DeeperForensics: `eval_and_viz.py --test_dataset DFDC` → `mt04_cross_dataset.py`
- [ ] **MT-3 robustness** (nén JPEG/resolution/nhiễu): `mt03_robustness.py` (cần weights)
- [ ] **Verify MT-4a** counts (đếm frame thật, không dùng ước lượng)
- [ ] **Điền `[[FILL]]`** trong `report/BAO_CAO_DATN_SFDCT.md` bằng hình/bảng đã tạo
- [ ] Viết **"Khảo sát liên quan & Điểm khác biệt"** (dùng phân tích Trí) + **CREDIT DeepfakeBench/SFDCT/nguồn** rõ ràng

---

## 4. 🟢 P2 — Đánh bóng
- [ ] **MT-10** test API thật (`mt10_api_test_template.py`) + observability (`mt10_observability.md`) + CI chạy thật (KHÔNG `|| true`)
- [ ] **MT-6** domain-normalize 2-pass (cần retrain để đo)
- [ ] Xuất **PNG nét cao từ plotly** cho bản in (chrome-headless `--screenshot`)
- [ ] (tùy) nhúng plotly vào frontend DeepGuard (trang Analytics)

---

## 5. Checklist "đồ án defensible" (tránh lỗi của Trí)
- [ ] Credit MỌI nguồn (code/model/data) — Trí mất điểm liêm chính ở đây
- [ ] Có ≥1 đóng góp gốc (block-DCT) + chứng minh thống kê (CI)
- [ ] Cross-dataset, không test cùng phân phối
- [ ] Báo cả số xấu + hạn chế (trung thực, như Trí báo "tách giọng không giúp")
- [ ] Metric đặt tên đúng chuẩn (đừng như WCSR của Trí)
- [ ] Sản phẩm chạy thật + test thật

---

## 6. Thứ tự đề xuất
Tuần 1: ① thử rẻ (DCT-corrects-spatial) → quyết định train FSBI hay không · MT-8 liveness · điền báo cáo phần đã có.
Tuần 2: ② thu dataset VN · MT-4 DFDC · (nếu cần) train FSBI multi-seed.
Tuần 3: MT-3 robustness · MT-10 app · viết related-work + credit · đánh bóng.

> **Việc quan trọng nhất ngay bây giờ:** ① — phép thử `mt_dct_corrects_spatial.py` để biết block-DCT có cửa thành novelty không, TRƯỚC khi tốn GPU train FSBI.
