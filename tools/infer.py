#!/usr/bin/env python3
"""infer.py — run a trained SFDCT/B4 checkpoint on real images. The model->serving bridge.

Outputs per image: (1) fake probability in [0,1], (2) REAL/FAKE verdict, (3) a Grad-CAM overlay PNG
(where the model looked) when --gradcam is set.

Usage:
  python tools/infer.py --detector_path <cfg.yaml> --weights <ckpt_best.pth> --input face.jpg --gradcam
  python tools/infer.py --detector_path <cfg.yaml> --weights <ckpt_best.pth> --input folder/ --gradcam --csv out.csv

--detector_path MUST be the config the ckpt was TRAINED with (architecture must match the state_dict;
each run on HF stores its own config.yaml). Input should be an aligned FACE CROP (training data is face
crops); whole frames work but are weaker.
"""
import os, sys, glob, argparse, csv
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_model(cfg_path, weights, device):
    import cv2  # noqa: ensure opencv imports before the heavy detector load
    from detectors import DETECTOR
    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("backbone_config", {"num_classes": 2, "inc": 3, "dropout": False, "mode": "Original"})
    cfg.setdefault("pretrained", "training/pretrained/efficientnet-b4-6ed6700e.pth")
    model = DETECTOR[cfg["model_name"]](cfg)
    sd = torch.load(weights, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)   # tolerate aux (scl center) / buffers
    if missing:    print(f"[warn] {len(missing)} missing keys (e.g. {missing[:3]})", file=sys.stderr)
    if unexpected: print(f"[warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})", file=sys.stderr)
    model.eval().to(device)
    return model, cfg


def preprocess(path, mean, std, res=256):
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        return None, None
    rgb = cv2.cvtColor(cv2.resize(bgr, (res, res), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = (rgb.astype(np.float32) / 255.0 - np.array(mean)) / np.array(std)
    return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0), rgb


def gradcam(model, x, device):
    """(prob, cam[H/8,W/8]) — Grad-CAM of the fake-class score on the fused features."""
    x = x.to(device)
    model.zero_grad(set_to_none=True)
    feat = model.features({"image": x})            # fused spatial features [1,C,h,w]
    feat.retain_grad()
    prob = torch.softmax(model.classifier(feat), dim=1)[0, 1]
    prob.backward()
    w = feat.grad.mean(dim=(2, 3), keepdim=True)   # channel weights
    cam = torch.relu((w * feat).sum(dim=1, keepdim=True))[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return float(prob), cam.detach().cpu().numpy()


def save_cam(rgb, cam, out_path):
    import cv2
    res = rgb.shape[0]
    cam = cv2.resize(cam, (res, res), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * rgb + 0.45 * heat).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_path", required=True, help="the config the ckpt was TRAINED with")
    ap.add_argument("--weights", required=True, help="ckpt_best.pth")
    ap.add_argument("--input", required=True, help="image file or folder of face crops")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thr", type=float, default=0.5, help="fake if prob >= thr (eKYC: calibrate on val)")
    ap.add_argument("--gradcam", action="store_true", help="also save a Grad-CAM overlay per image")
    ap.add_argument("--out_dir", default="infer_out", help="where Grad-CAM overlays are written")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    model, cfg = load_model(a.detector_path, a.weights, a.device)
    mean = cfg.get("mean", [0.5, 0.5, 0.5]); std = cfg.get("std", [0.5, 0.5, 0.5])
    paths = ([a.input] if os.path.isfile(a.input)
             else sorted(p for p in glob.glob(os.path.join(a.input, "**", "*"), recursive=True)
                         if p.lower().endswith(IMG_EXT)))
    if not paths:
        print("no images found at", a.input); return
    if a.gradcam:
        os.makedirs(a.out_dir, exist_ok=True)

    rows = []
    for p in paths:
        x, rgb = preprocess(p, mean, std)
        if x is None:
            print(f"{p}\tSKIP (unreadable)"); continue
        cam_path = ""
        if a.gradcam:
            prob, cam = gradcam(model, x, a.device)                       # forward WITH grad
            cam_path = os.path.join(a.out_dir, os.path.splitext(os.path.basename(p))[0] + "_gradcam.png")
            save_cam(rgb, cam, cam_path)
        else:
            with torch.no_grad():
                prob = float(model({"image": x.to(a.device)}, inference=True)["prob"][0])
        verdict = "FAKE" if prob >= a.thr else "REAL"
        print(f"{p}\tfake_prob={prob:.4f}\t{verdict}" + (f"\tgradcam={cam_path}" if cam_path else ""))
        rows.append((p, f"{prob:.4f}", verdict, cam_path))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            csv.writer(f).writerows([("path", "fake_prob", "verdict", "gradcam"), *rows])
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
