# Bộ dữ liệu deepfake người Việt tự thu thập (NOVELTY thêm)

> Bạn dự định: **thu 30 video người Việt thật → crop face → tạo deepfake → thành bộ test.**
> Đây chính là phiên bản deepfake của novelty #4 (Trí tự crawl 125 bài nhạc Việt làm test cross-cultural).
> Với eKYC Việt Nam, đây là **đóng góp mạnh nhất bạn có thể tạo** — không bộ benchmark nào có mặt người Việt + kịch bản eKYC.

## Vì sao đây là novelty thật (đáng nhấn khi bảo vệ)
- **Cross-domain thật**: train FF++ (mặt Tây) → test mặt Việt = đo generalization đúng bài toán eKYC VN.
- **In-the-wild + eKYC-relevant**: khác dataset academic (quay studio), video người Việt sát kịch bản định danh ngân hàng (Thông tư 17).
- **Kiểm chứng giả thuyết block-DCT** trên phân phối mới (artifact GAN trên mặt Việt).

## Giao thức đề xuất (làm chuẩn để bảo vệ không bị bắt bẻ)

### 1. Thu thập real (30 video)
- Nguồn: tự quay / video công khai có phép. Đa dạng: giới tính, tuổi, ánh sáng, thiết bị (điện thoại/webcam), độ phân giải.
- Ghi metadata mỗi video: id, nguồn, độ dài, fps, resolution, điều kiện sáng. **KHÔNG PII** ngoài mục đích nghiên cứu; có **consent** nếu là người thật → ghi rõ trong báo cáo (đạo đức dữ liệu).

### 2. Crop face (tái dùng pipeline có sẵn)
- Dùng **MTCNN** đã có trong repo (backend `_crop_face`, hoặc `preprocessing/` của DeepfakeBench) để đồng nhất với pipeline train/serve.
- Sample frame: 1 frame / 0.5–1s; lưu `dataset_vn/real/<video_id>/<frame>.png` (đã crop, vuông, ≥256px).

### 3. Sinh deepfake (tạo nhánh fake)
- Công cụ gợi ý (chọn ≥2 loại để đa dạng manipulation, giống FF++ có 4 họ):
  - **Face swap**: SimSwap / InsightFace inswapper / Roop-unleashed.
  - **Face reenactment**: SadTalker / LivePortrait (động khẩu hình — sát tấn công eKYC "video giả").
  - **(tùy)** Diffusion face-edit để có artifact khác hệ GAN.
- Mỗi video real → ≥1 video/ảnh fake; lưu `dataset_vn/fake/<method>/<video_id>/...`.
- Ghi rõ **method** mỗi mẫu → cho phép báo cáo **per-manipulation** (MT-1/MT-2b).

### 4. Đóng gói JSON đúng định dạng DeepfakeBench
- Tạo `dataset_json_*/VN-Deepfake.json` theo schema giống `Celeb-DF-v2.json` (label 1=fake, 0=real; đường dẫn frames).
- Tham khảo `preprocessing/` của DeepfakeBench để sinh json chuẩn (rerun script preprocessing trỏ vào `dataset_vn/`).

### 5. Đánh giá → cắm thẳng vào report_prepare
```
python training/eval_and_viz.py --detector_path <cfg> --weights_path <ckpt> \
    --test_dataset VN-Deepfake --out viz_out/<model>
# rồi:
python report_prepare/mt04_cross_dataset.py        # thêm cột VN-Deepfake vào heatmap
python report_prepare/mt02_threshold_ekyc.py       # τ@FPR5% trên mặt Việt (eKYC thật)
python report_prepare/mt04_dataset_distribution.py # phân bố real/fake bộ VN
```

## Quy mô & lưu ý
- 30 video real là **đủ cho một TEST set** (Trí dùng 125 bài, nhưng song-level; bạn dùng frame-level nên 30 video × nhiều frame là ổn để báo AUC + CI bootstrap). **KHÔNG train trên bộ này** (giữ làm hold-out cross-domain) để tránh leak.
- Báo cáo phải nêu: số video, số frame, #method fake, consent/đạo đức, và rằng đây là **test-only**.
- Khi có bộ này → cập nhật `_STATUS.md` MT-4 và thêm vào bảng cross-dataset như một cột "VN-Deepfake (in-house, eKYC)".

## Hạn chế cần ghi trung thực (học từ tinh thần #5 của Trí)
- 30 video là nhỏ → báo **bootstrap CI** (mt05) cho AUC trên bộ này, đừng tuyên bố tuyệt đối.
- Deepfake tự sinh có thể "dễ" hơn in-the-wild → nêu rõ method + giới hạn.
