#!/usr/bin/env python3
"""G0.3 · Hiệu chỉnh ngưỡng τ cho SERVE + đo gap train-serve do lệch crop.
Train/eval dùng dlib-align; serve (backend) dùng MTCNN bbox → cùng model, 2 crop khác nhau ⇒ lệch.
Script chạy CÙNG model trên CÙNG ảnh với 2 crop → AUC(dlib) vs AUC(MTCNN) + **τ@FPR≤5% trên crop-MTCNN**
để nạp vào backend (đúng điểm vận hành lúc serve).

CẦN GPU + (dlib + shape_predictor_81_face_landmarks.dat) + (mtcnn) + thư mục ẢNH FULL-FRAME có nhãn.
CHẠY (trên 3060/vast):
  python3 report_prepare/mt_recalibrate_serve.py \
    --detector_path training/config/detector/efficientnetb4_sfdct.yaml \
    --weights_path ./hf_runs/runs/naive-20260605-233339/sfdct_naive/ckpt/ckpt_best.pth \
    --images_real <dir_real_fullframe> --images_fake <dir_fake_fullframe>
Output: report_prepare/outputs/serve_calibration.json {auc_dlib, auc_mtcnn, gap, tau_mtcnn_fpr5}.
"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

import common as C
REPO = C.REPO


def dlib_crop(bgr, det, pred, dst_tmpl):
    import cv2
    from skimage import transform as trans
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); faces = det(rgb, 1)
    if not faces: return None
    f = max(faces, key=lambda r: r.width() * r.height()); sh = pred(rgb, f)
    pts = np.array([[sh.part(i).x, sh.part(i).y] for i in range(81)], np.float32)
    five = np.array([pts[36:42].mean(0), pts[42:48].mean(0), pts[30], pts[48], pts[54]], np.float32)
    t = trans.SimilarityTransform(); t.estimate(five, dst_tmpl)
    return cv2.resize(cv2.warpAffine(rgb, t.params[:2], (256, 256)), (256, 256))


def mtcnn_crop(bgr, det):
    import cv2
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB); r = det.detect_faces(rgb)
    if not r: return None
    x, y, w, h = max(r, key=lambda b: b['box'][2] * b['box'][3])['box']
    px, py = int(w * 0.25), int(h * 0.25)
    crop = rgb[max(0, y - py):y + h + py, max(0, x - px):x + w + px]
    return cv2.resize(crop, (256, 256)) if crop.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_path", required=True); ap.add_argument("--weights_path", required=True)
    ap.add_argument("--images_real", required=True); ap.add_argument("--images_fake", required=True)
    ap.add_argument("--limit", type=int, default=400)
    a = ap.parse_args()

    # 1) detectors
    import cv2, dlib
    dat = str(REPO / "preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat")
    det_d = dlib.get_frontal_face_detector(); pred = dlib.shape_predictor(dat)
    dst = np.array([[30.2946,51.6963],[65.5318,51.5014],[48.0252,71.7366],[33.5493,92.3655],[62.7299,92.2041]],np.float32); dst[:,0]+=8.0; dst*=256/112
    from mtcnn import MTCNN; det_m = MTCNN()

    # 2) model (theo đúng pattern eval_and_viz.main)
    sys.path.insert(0, str(REPO / "training"))
    import torch, yaml
    from detectors import DETECTOR  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(open(a.detector_path))
    model = DETECTOR[cfg["model_name"]](cfg).to(device)
    ck = torch.load(a.weights_path, map_location=device)
    model.load_state_dict(ck.get("state_dict", ck), strict=True); model.eval()

    def predict(crop_rgb):
        import torch
        x = torch.from_numpy(((crop_rgb/255.-0.5)/0.5).transpose(2,0,1)[None]).float().to(device)
        with torch.no_grad():
            out = model({"image": x}, inference=True)
        return float(out["prob"].cpu().numpy().ravel()[0])

    # 3) chạy 2 crop trên cùng ảnh
    files = ([(f,0) for f in sorted(glob.glob(a.images_real+"/**/*.*", recursive=True))[:a.limit]] +
             [(f,1) for f in sorted(glob.glob(a.images_fake+"/**/*.*", recursive=True))[:a.limit]])
    pd_, pm, lab = [], [], []
    for f, y in files:
        bgr = cv2.imread(f)
        if bgr is None: continue
        cd = dlib_crop(bgr, det_d, pred, dst); cm = mtcnn_crop(bgr, det_m)
        if cd is None or cm is None: continue
        pd_.append(predict(cd)); pm.append(predict(cm)); lab.append(y)
    pd_, pm, lab = np.array(pd_), np.array(pm), np.array(lab)

    auc_d = C.compute_auc(lab, pd_); auc_m = C.compute_auc(lab, pm)
    tau_m, fpr_m, tpr_m = C.threshold_at_fpr(lab, pm, C.TT17_FPR)
    out = {"n": int(len(lab)), "auc_dlib": round(auc_d,4), "auc_mtcnn": round(auc_m,4),
           "gap_dlib_minus_mtcnn": round(auc_d-auc_m,4),
           "tau_mtcnn_fpr5": round(tau_m,4), "fpr_at_tau": round(fpr_m,4), "tpr_at_tau": round(tpr_m,4)}
    (C.OUT/"serve_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n[G0.3] Nạp tau_mtcnn_fpr5={out['tau_mtcnn_fpr5']} vào backend MODEL_THRESHOLD để serve đúng điểm vận hành.")


if __name__ == "__main__":
    main()
