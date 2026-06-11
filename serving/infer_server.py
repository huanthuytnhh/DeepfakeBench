"""infer_server.py — microservice phục vụ model deepfake SFDCT (naive, AUC 0.7572).

Cô lập torch/DeepfakeBench khỏi backend chính. Backend (FastAPI :8000) gọi POST /predict qua HTTP.
Tái dùng tools/infer.py (load_model + gradcam). Trả: prob_fake + verdict + Grad-CAM (base64 data URL).

Chạy (từ DeepfakeBench root):
  uvicorn serving.infer_server:app --host 0.0.0.0 --port 8501
Env tuỳ chọn: SFDCT_CONFIG, SFDCT_CKPT, SFDCT_THRESHOLD (mặc định trỏ serving/naive_sfdct/).
"""
import os
import sys
import time
import base64
import numpy as np
import torch
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))        # tools/infer.py
sys.path.insert(0, os.path.join(REPO, "training"))     # detectors/...
import infer as infer_mod                              # tools/infer.py

CFG = os.environ.get("SFDCT_CONFIG", os.path.join(REPO, "serving/naive_sfdct/config.yaml"))
CKPT = os.environ.get("SFDCT_CKPT", os.path.join(REPO, "serving/naive_sfdct/ckpt_best.pth"))
THR = float(os.environ.get("SFDCT_THRESHOLD", "0.5"))
MODEL_VERSION = os.environ.get("SFDCT_VERSION", "naive-sfdct-cdfv2-0.7572")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="SFDCT Inference Service", version="1.0")
_model = None
_cfg = None


def _get_model():
    global _model, _cfg
    if _model is None:
        _model, _cfg = infer_mod.load_model(CFG, CKPT, DEVICE)
    return _model, _cfg


@app.get("/health")
def health():
    return {"ok": True, "device": DEVICE, "ckpt": os.path.basename(CKPT), "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.perf_counter()
    raw = await file.read()
    bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return JSONResponse({"error": "cannot decode image"}, status_code=400)
    model, cfg = _get_model()
    mean = cfg.get("mean", [0.5, 0.5, 0.5]); std = cfg.get("std", [0.5, 0.5, 0.5])
    res = int(cfg.get("resolution", 256))
    rgb = cv2.cvtColor(cv2.resize(bgr, (res, res), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - np.array(mean)) / np.array(std)).transpose(2, 0, 1)
    x = torch.from_numpy(x).float().unsqueeze(0)
    prob, cam = infer_mod.gradcam(model, x, DEVICE)             # prob in [0,1], cam[h,w] in [0,1]
    verdict = "FAKE" if prob >= THR else "REAL"
    # Grad-CAM overlay -> base64 data URL
    cam = cv2.resize(cam, (res, res), interpolation=cv2.INTER_LINEAR)
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = (0.55 * rgb + 0.45 * heat).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cam_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
    return {
        "prob_fake": round(float(prob), 4),
        "verdict": verdict,
        "threshold": THR,
        "gradcam": cam_b64,
        "model_version": MODEL_VERSION,
        "device": DEVICE,
        "processing_time_ms": int((time.perf_counter() - start) * 1000),
    }
