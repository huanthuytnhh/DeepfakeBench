#!/usr/bin/env bash
# Chạy mọi MT "quick-win" (data đã sẵn trong viz_out/ + logs/ — KHÔNG cần GPU).
# Các MT cần train/GPU (3,6) hoặc cần chạy trước (8 liveness, 10 API) xem _STATUS.md.
set -e
cd "$(dirname "$0")"
export MPLBACKEND=Agg

echo "==== MT-9  training curves (log thật) ===="; python3 mt09_training_curves.py
echo "==== MT-9b raw CSV audit ===="            ; python3 mt09_export_raw_csv.py
echo "==== MT-1  eval matrix + heatmap ===="     ; python3 mt01_eval_matrix.py
echo "==== MT-2  threshold eKYC + score dist ====" ; python3 mt02_threshold_ekyc.py
echo "==== MT-2b confusion matrix thật ===="     ; python3 mt02_confusion_matrix.py
echo "==== MT-5  ablation + bootstrap CI ===="    ; python3 mt05_ablation_ci.py
echo "==== MT-5b fill results.csv từ log ===="    ; python3 mt05_fill_results_csv.py
echo "==== MT-7  frequency / gradcam grid ===="   ; python3 mt07_frequency_grid.py
echo "==== MT-4a dataset distribution ===="       ; python3 mt04_dataset_distribution.py || true
echo "==== MT-4b cross-dataset table ===="        ; python3 mt04_cross_dataset.py
echo ""
echo "Xong. Kết quả trong report_prepare/outputs/. MT cần GPU/chạy-trước: xem _STATUS.md"
