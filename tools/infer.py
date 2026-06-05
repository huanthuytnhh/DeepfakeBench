#!/usr/bin/env python3
"""infer.py — run a trained SFDCT/B4 checkpoint on real images (the model->serving bridge).

Usage:
  python tools/infer.py --detector_path training/config/detector/efficientnetb4_sfdct.yaml \
      --weights runs/.../ckpt_best.pth --input path/to/face.jpg
  python tools/infer.py --detector_path <cfg.yaml> --weights <ckpt.pth> --input path/to/folder --csv out.csv

The detector_path MUST be the config the checkpoint was TRAINED with (architecture must match), e.g. the
config.yaml saved next to each run on HF (Row1 has sign+SRM, Row2 has FcaNet+SCL -> different state_dicts).
Input image should already be a FACE CROP (DeepfakeBench trains on aligned face crops); whole frames work
but are weaker. Output: fake probability in [0,1] (>0.5 => predicted FAKE).
"""
import os, sys, glob, argparse, csv
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_model(cfg_path, weights, device):
    import cv2  # noqa (ensure opencv import works before heavy load)
    from detectors import DETECTOR
    cfg = yaml.safe_load(open(cfg_path))
    cfg.setdefault("backbone_config", {"num_classes": 2, "inc": 3, "dropout": False, "mode": "Original"})
    cfg.setdefault("pretrained", "training/pretrained/efficientnet-b4-6ed6700e.pth")
    model = DETECTOR[cfg["model_name"]](cfg)
    sd = torch.load(weights, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)   # tolerate aux (scl center) / buffer keys
    if missing:    print(f"[warn] {len(missing)} missing keys (e.g. {missing[:3]})", file=sys.stderr)
    if unexpected: print(f"[warn] {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})", file=sys.stderr)
    model.eval().to(device)
    return model, cfg


def preprocess(path, mean, std, res=256):
    import cv2
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (res, res), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    img = (img - np.array(mean)) / np.array(std)
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_path", required=True, help="the config the ckpt was TRAINED with")
    ap.add_argument("--weights", required=True, help="ckpt_best.pth")
    ap.add_argument("--input", required=True, help="image file or folder of face crops")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--thr", type=float, default=0.5, help="fake if prob >= thr (eKYC: calibrate on val)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    model, cfg = load_model(a.detector_path, a.weights, a.device)
    mean = cfg.get("mean", [0.5, 0.5, 0.5]); std = cfg.get("std", [0.5, 0.5, 0.5])
    paths = ([a.input] if os.path.isfile(a.input)
             else sorted(p for p in glob.glob(os.path.join(a.input, "**", "*"), recursive=True)
                         if p.lower().endswith(IMG_EXT)))
    if not paths:
        print("no images found at", a.input); return

    rows = []
    for p in paths:
        x = preprocess(p, mean, std)
        if x is None:
            print(f"{p}\tSKIP (unreadable)"); continue
        prob = float(model({"image": x.to(a.device)}, inference=True)["prob"][0])  # P(fake)
        verdict = "FAKE" if prob >= a.thr else "REAL"
        print(f"{p}\tfake_prob={prob:.4f}\t{verdict}")
        rows.append((p, prob, verdict))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            csv.writer(f).writerows([("path", "fake_prob", "verdict"), *rows])
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
