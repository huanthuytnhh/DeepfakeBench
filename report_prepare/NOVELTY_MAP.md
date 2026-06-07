# Căn chỉnh SFDCT theo "công thức thắng" của Trí — novelty & trình tự báo cáo

> Review + tối ưu để (a) novelty SFDCT ánh xạ 1:1 novelty Trí, (b) trình tự báo cáo bám sát Trí.
> Kết luận review: **báo cáo bạn ĐÃ rất sát Trí về trình tự** (Mở đầu → Ch1 lý thuyết → Ch2 phân tích/method → Ch3 thực nghiệm/đánh giá → Kết luận); chỉ cần nâng vài điểm.

## 1. Ánh xạ NOVELTY (Trí ↔ SFDCT của bạn)

| Trí (novelty thật) | Tầng | → SFDCT (ĐGx) | Nằm ở đâu trong báo cáo bạn |
|---|---|---|---|
| Mode-aware relative-key (tách trục lỗi) | Đánh giá | **ĐG3** đánh giá điểm vận hành eKYC + decompose lỗi (confusion FP/FN, APCER/BPCER) | Mục 3.4/3.6, Bảng 3.4 |
| Ma trận 16-config + heatmap | Phương pháp luận | **ĐG2** khung đa cấu hình + heatmap (B4→B4-DCT→Row1→Row2) | Mục 3.4 (cần **nâng heatmap thành hình trung tâm**) |
| Semi-strict tolerance | Thiết kế metric | (ĐG3 mở rộng) robustness nén/resolution | Mục 3.x robustness (MT-3, cần thêm) |
| Cross-cultural VN test | Dữ liệu | **ĐG5** bộ deepfake người Việt | Hướng phát triển / Mục 3.2 (cần thêm khi có data) |
| Dual-eval + negative result | Trung thực KH | **ĐG4** bootstrap CI + báo Δ-trong-nhiễu | Mục 3.4 (đã có MT-5) |
| (Trí KHÔNG có method gốc) | — | **ĐG1** nhánh block-DCT spatial-freq + zero-init **(bạn VƯỢT Trí)** | Chương 2 |
| Full-stack + ops | Kỹ nghệ | DeepGuard multi-tenant **(vượt Trí)** | Hệ thống (ngoài báo cáo lõi) |

→ Bạn **ngang Trí** ở ĐG2–ĐG5, **vượt** ở ĐG1 (method gốc) + app. Đã thêm mục **"Đóng góp chính (novelty)"** vào Mở đầu để phát biểu rõ 5 trục này (giống cách thesis mạnh liệt kê đóng góp).

## 2. Ánh xạ TRÌNH TỰ BÁO CÁO (Trí ↔ bạn)

| Trí | SFDCT của bạn | Trạng thái |
|---|---|---|
| INTRODUCTION (Problem/Purpose/Objectives/7-step process/Structure) | MỞ ĐẦU (Đặt vấn đề/Mục tiêu+CH/Đóng góp/Phạm vi/Phương pháp/Ý nghĩa/Cấu trúc) | ✅ sát (đã thêm "Đóng góp") |
| Ch1 THEORIES & TECHNOLOGIES | Ch1 CƠ SỞ LÝ THUYẾT (deepfake, B4, DCT, attention, S1–S5, eKYC) | ✅ sát (sâu hơn Trí) |
| Ch2 ANALYSIS & DESIGN (use case → kiến trúc → 2.3 Method: Data/Eval+Loss/Model arch) | Ch2 PHÂN TÍCH & THIẾT KẾ (2.1 yêu cầu → 2.2 use case/kiến trúc → 2.3 dữ liệu → 2.4 SFDCT → 2.5 ablation → 2.6 loss/train → 2.7 metric/XAI) | ✅ sát; thứ tự Data→Model→Eval/Loss giống Trí |
| Ch3 IMPLEMENTATION & EVAL (data → train từng model → **16-config comparative + heatmap** → conclusion) | Ch3 (3.1 setup → 3.2 data → 3.3 training → **3.4 ablation** → 3.5 định tính → 3.6 demo eKYC → 3.7 thảo luận) | ✅ sát; **cần NÂNG 3.4 = "đánh giá so sánh đa cấu hình + heatmap" làm hình trung tâm** (giống 16-config Trí) |
| CONCLUSION (achievements/limitations/future) | Kết luận + Hướng phát triển | ✅ sát |
| REFERENCES | Tài liệu tham khảo | ✅ (cần credit DeepfakeBench + verify `[[KIỂM TRA]]`) |

## 3. Việc tối ưu còn lại để "sát Trí" hơn nữa
- [ ] **Nâng Mục 3.4**: đặt **heatmap AUC (model × cấu hình/dataset)** làm hình trung tâm của chương kết quả (hệt 16-config heatmap của Trí). Dùng `mt01_eval_matrix.py` / `mt04_cross_dataset.py`.
- [ ] **Thêm Mục robustness** (nén/resolution) = bản deepfake của "semi-strict tolerance" (MT-3, cần train/infer).
- [ ] **Thêm tiểu mục dữ liệu VN** (ĐG5) trong 3.2 hoặc Hướng phát triển khi thu được — bản deepfake của "cross-cultural VN test".
- [ ] **Credit DeepfakeBench + các paper S1–S5** đầy đủ ở Tài liệu tham khảo (tránh lỗi liêm chính của Trí).
- [ ] Sau khi train Row2/FSBI: điền các ô `[[FILL]]` còn lại (xem `report/_HOAN_THIEN_STATUS.md`).

## 4. Khác biệt định vị (1 câu)
> SFDCT bám đúng "công thức thắng" của Trí (đánh giá có chiều sâu + dữ liệu bản địa + sản phẩm), **cộng thêm một method gốc (block-DCT spatial-frequency)** mà Trí không có, và **ghi nguồn minh bạch** — hai điểm đưa đồ án vượt mức tham khảo.
