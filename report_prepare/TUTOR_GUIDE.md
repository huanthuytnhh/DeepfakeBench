# TUTOR GUIDE (chi tiết) — Đồ án Deepfake/SFDCT, học từ thủ khoa Trần Đức Trí

> Bản chi tiết của `TUTOR_CHECKLIST.md`. Mỗi mục: **Mục tiêu · Học từ Trí · Vì sao · Các bước · Output → báo cáo · Effort · Tránh bẫy**.
> Quy ước: 🔴 quyết định novelty · 🟡 nên có · 🟢 đánh bóng.

---

## 0. Định vị chiến lược (đọc trước)
- Trí điểm cao nhờ **vỏ bọc** (đánh giá music-aware + dataset nhạc Việt + sản phẩm full-stack), **KHÔNG vì model mới**; nửa lõi nhận dạng là **fork repo Lanž, không credit** (cả code lẫn luận văn).
- Bạn có 2 thứ Trí không có: **novelty method gốc (block-DCT spatial-frequency)** + **app multi-tenant**. Nhưng block-DCT **chưa chứng minh được** (naive Δ trong nhiễu) → phải xử lý.
- Mục tiêu: biến đồ án từ "train lại đồ người khác + vẽ đẹp" → "có novelty thật + minh bạch nguồn". 2 trụ: **① block-DCT thắng thật** + **② dataset người Việt**.

---

# 🔴 P0 — HAI VIỆC QUYẾT ĐỊNH

## ① Chứng minh block-DCT đóng góp THẬT (novelty method gốc)

**Mục tiêu:** chứng minh nhánh frequency (block-DCT) bổ trợ spatial (B4) một cách **có ý nghĩa thống kê**, không phải may rủi.

**Học từ Trí:** mode-aware của Trí dùng **tín hiệu thứ 2 trực giao** (Essentia mode) sửa lỗi có cấu trúc mà model chính không tự giải, và **chứng minh nó nâng metric 58→79**. Block-DCT của bạn = "tín hiệu thứ 2" (frequency) sửa cái spatial mơ hồ → thiết kế thí nghiệm **giống hệt** khung đó.

**Vì sao quan trọng nhất:** đây là novelty method — thứ Trí không có. Nếu không thắng được, novelty method sụp → phải reframe sang dataset+đánh giá. Nên **kiểm tra sớm, rẻ trước, đắt sau**.

**Các bước:**
1. **Thử rẻ (không train):** viết `mt_dct_corrects_spatial.py` đọc `viz_out/b4_local` và `viz_out/naive_local` `scores`:
   - Lấy tập mẫu **B4 spatial đoán SAI** (sai phía so với nhãn tại τ).
   - Đếm trong đó **block-DCT (naive) đoán ĐÚNG bao nhiêu %** (recovery rate) và ngược lại (DCT làm hỏng bao nhiêu mẫu B4 đúng → regression rate).
   - Net lift = recovery − regression. In bảng + (tùy) Venn lỗi.
   - ⚠️ Lưu ý: 2 `.npz` **không cùng thứ tự mẫu** (đã phát hiện ở MT-5) → phải so trên cùng tập ảnh (re-eval cả 2 model trên cùng loader, hoặc match theo image-path). Nếu không match được → kết luận chỉ ở mức phân phối, không per-sample.
2. **Đọc kết quả:**
   - Net lift dương rõ → block-DCT có cửa → đáng đầu tư train FSBI.
   - Net lift ~0 / âm → naive DCT chưa giúp → **bắt buộc** đổi cách (FSBI/SBI) hoặc reframe.
3. **Nếu cần nâng (train):** **FSBI route** — block-DCT branch + **SBI (Self-Blended Images) làm augmentation lúc train** (SBI chỉ là trick, **block-DCT vẫn là novelty chính**). Tập trung **mid/high band** (đúng giả thuyết), fusion gated cross-attention chuẩn.
4. **Chứng minh thống kê:** train **≥3 seed**, điền `experiments/results.csv` (`mt05_fill_results_csv.py`), bootstrap CI (`mt05_ablation_ci.py`) → Δ có ý nghĩa khi **CI không chứa 0**.
5. **Per-manipulation:** chứng minh DCT giúp ở loại nào (NeuralTextures/F2F thường lộ tần số khác FaceSwap).

**Output → báo cáo:** bảng "DCT sửa lỗi spatial" + bảng ablation đa-seed (mean±std, CI) + bar `mt05_ablation_bar.png` + per-manipulation. → phần "Đóng góp phương pháp" + "Kết quả".

**Effort:** thử rẻ 0.5 ngày (data sẵn); FSBI multi-seed = vài lần train GPU.

**Tránh bẫy:** đừng để **SBI lấn át** (SBI là của paper khác — block-DCT mới là của bạn); đừng báo Δ dương nhỏ là "thắng" khi CI chứa 0 (đó là nhiễu — như naive hiện tại).

---

## ② Bộ deepfake người Việt cho eKYC (novelty dataset — chắc chắn nhất)

**Mục tiêu:** tạo tập **test cross-domain** mặt người Việt + kịch bản eKYC — không benchmark nào có.

**Học từ Trí:** Trí tự crawl **125 bài nhạc Việt** làm test cross-cultural (train Tây → test Việt). Bạn làm phiên bản deepfake, **mạnh hơn**: frame-level (không phải song-level) + eKYC-relevant.

**Vì sao:** dataset là **đóng góp công bố được**, độc lập với việc block-DCT có thắng hay không → novelty "bảo hiểm".

**Các bước:**
1. **Thu 30 video real:** đa dạng giới/tuổi/sáng/thiết bị; **có consent**; ghi metadata (id, nguồn, fps, res, điều kiện). Không PII ngoài mục đích nghiên cứu.
2. **Crop mặt:** dùng **MTCNN có sẵn** (`backend` `_crop_face` hoặc `preprocessing/` DeepfakeBench) để khớp pipeline. Sample 1 frame/0.5–1s → `dataset_vn/real/<id>/*.png` (≥256px).
3. **Sinh fake ≥2 method** (để có per-manipulation): face-swap (SimSwap/inswapper) + reenactment (SadTalker/LivePortrait — sát tấn công eKYC) → `dataset_vn/fake/<method>/<id>/...`.
4. **Đóng gói JSON** giống `dataset_json_medium/Celeb-DF-v2.json` (label 1=fake/0=real) → `dataset_json_*/VN-Deepfake.json`. **Giữ test-only** (không train → tránh leak).
5. **Eval + cắm report_prepare:**
   ```
   python3 training/eval_and_viz.py --detector_path <cfg> --weights_path <ckpt> --test_dataset VN-Deepfake --out viz_out/<model>
   python3 report_prepare/mt04_cross_dataset.py    # tự thêm cột VN
   python3 report_prepare/mt02_threshold_ekyc.py   # τ@FPR5% mặt Việt
   python3 report_prepare/mt07_freq_panel.py --real <vn_real.png> --fake <vn_fake.png>
   ```

**Output → báo cáo:** mô tả dataset (số video/frame/method, consent) + bảng AUC trên VN + hình tần số mặt Việt + **bootstrap CI** (n nhỏ).

**Effort:** thu+crop 1-2 ngày; sinh fake tùy công cụ; eval+vẽ 0.5 ngày.

**Tránh bẫy:** n nhỏ → **luôn báo CI**, đừng tuyên bố tuyệt đối; ghi rõ method fake + giới hạn (fake tự sinh có thể "dễ" hơn in-the-wild). Chi tiết đầy đủ: `VIETNAM_DEEPFAKE_DATASET.md`.

---

# 🟡 P1 — Cần GPU + hoàn thiện báo cáo

## MT-8 · Module liveness chuẩn ISO/IEC 30107-3 + DET
**Mục tiêu:** hoàn tất hệ phụ FAS với metric chuẩn (APCER/BPCER/ACER/EER/AUC) + DET + **BPCER@APCER≤5%** (neo TT17).
**Học từ Trí:** hệ phụ (tách giọng) vẫn được đánh giá đầy đủ — hệ phụ ≠ bỏ trống.
**Các bước:** `cd DeepfakeBench; export HF_TOKEN=...; AUG=fas BATCH=32 ./liveness/run_liveness.sh` (1-2h local, $0) → `python3 report_prepare/mt08_liveness_report.py` (vẽ ROC+DET+bar ACER/APCER/BPCER+hist — đã code sẵn).
**Output:** hình + bảng liveness trong chương kết quả; neo TT17 APCER≤5%.
**Tránh bẫy:** split theo subject/video (không theo frame → leak); báo cả ACER lẫn BPCER@APCER.

## MT-4 · Cross-dataset mở rộng (DFDC/DeeperForensics)
**Mục tiêu:** chứng minh generalization ngoài Celeb-DF.
**Các bước:** `eval_and_viz.py --test_dataset DFDC --out viz_out/<model>` → `mt04_cross_dataset.py` (heatmap tự thêm cột) + `mt04_dataset_distribution.py` (phân bố — lập luận distribution-shift như Trí).
**Output:** bảng + heatmap AUC cross-dataset; bar phân bố real/fake.

## MT-3 · Robustness curve (nén/resolution/nhiễu)
**Mục tiêu:** "tolerance" kiểu Trí (semi-strict) cho ảnh eKYC (camera điện thoại nén mạnh).
**Các bước:** `mt03_robustness.py --detector_path <cfg> --weights_path <ckpt>` (hàm nhiễu jpeg/downscale/noise đã sẵn; nối vào loader của eval_and_viz).
**Output:** đường AUC theo mức nén/resolution.

## Verify MT-4a (phân bố dataset)
**Mục tiêu:** counts hiện là **ước lượng cấu trúc JSON** (45/45, 60×5 — quá tròn). Đếm **frame thật** trước khi đưa vào báo cáo.

## Điền báo cáo + Khảo sát liên quan
- [ ] Điền mọi `[[FILL]]` trong `report/BAO_CAO_DATN_SFDCT.md` bằng hình/bảng đã tạo (matplotlib cho bản in).
- [ ] Viết **"Khảo sát liên quan & Điểm khác biệt"** dùng phân tích Trí (mode-aware/dataset/honest eval) → định vị novelty của bạn.
- [ ] **CREDIT** DeepfakeBench + SFDCT + mọi nguồn (code/model/data) — **điểm bạn auto vượt Trí về liêm chính**.

---

# 🟢 P2 — Đánh bóng

## MT-10 · Production-readiness app eKYC
- **Test API thật:** `mt10_api_test_template.py` — pytest GỌI THẬT endpoint detect + SFDCT :8501 (không assert dict cứng như Trí).
- **Observability:** middleware request_id+latency, `/health` psutil, log rotate (snippet `mt10_observability.md`).
- **CI chạy thật:** GitHub Actions pytest **KHÔNG** `pip install || true`.
**Tránh bẫy:** Trí có `tests/test_api.py` GIẢ + CI stub → dễ bị phản biện. Bạn làm thật = vượt.

## MT-6 · Domain-normalize 2-pass (cần retrain)
**Học từ Trí:** two-pass key-normalize → chuẩn hoá miền trước suy luận. Bạn: MTCNN-align + chuẩn hoá DCT/illumination về miền chuẩn → đo cross-dataset có/không (DomainNormalize trong `mt06_domain_norm.py`).

## Trình bày
- Xuất **PNG nét cao từ plotly** cho bản in: `google-chrome --headless --screenshot` (đã chạy được).
- (tùy) nhúng plotly vào **frontend DeepGuard** (trang Analytics) → dashboard thật trong app, vượt Trí.

---

# ✔️ Checklist "đồ án defensible" (rút từ lỗi + điểm mạnh của Trí)
- [ ] Credit MỌI nguồn (code/model/data) — Trí mất điểm liêm chính
- [ ] ≥1 đóng góp gốc (block-DCT) + chứng minh thống kê (CI không chứa 0)
- [ ] Cross-dataset / cross-domain, không test cùng phân phối
- [ ] Báo cả số xấu + hạn chế (trung thực, như Trí báo "tách giọng không giúp")
- [ ] Metric đặt tên đúng chuẩn (đừng như WCSR của Trí — đặt tên trùng chuẩn nhưng định nghĩa khác)
- [ ] Sản phẩm chạy thật + test thật + observability

---

# 📅 Thứ tự đề xuất
- **Tuần 1:** P0-① thử rẻ (DCT-corrects-spatial) → quyết train FSBI hay không · MT-8 liveness · điền báo cáo phần đã có.
- **Tuần 2:** P0-② thu dataset VN · MT-4 DFDC · (nếu cần) train FSBI multi-seed.
- **Tuần 3:** MT-3 robustness · MT-10 app · related-work + credit · đánh bóng plotly/PNG.

> **Bắt đầu ngay:** P0-① `mt_dct_corrects_spatial.py` — phép thử rẻ nhất để biết block-DCT có cửa thành novelty không, TRƯỚC khi tốn GPU train FSBI.
