#!/usr/bin/env python3
"""train_fas.py — fine-tune the EXISTING deepfake detectors (B4 baseline / SFDCT = B4+block-DCT)
for liveness / face anti-spoofing on LCC-FASD.

Reuses the detector classes UNCHANGED (they are already binary; live=0/spoof=1 maps onto real=0/fake=1).
Adds AMP + early stopping so it fits a 4GB GPU. Reports APCER/BPCER/ACER/EER/AUC with the decision
threshold fixed at EER on the dev split (no test-set peeking).

Run from the DeepfakeBench repo root:
  python tools/train_fas.py --config training/config/detector/efficientnetb4.yaml \
      --data_root <LCC_FASD dir> --out logs/fas/b4 [--init_ckpt <deepfake_ckpt.pth>]
"""
import os
import sys
import json
import time
import glob
import random
import argparse
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # .../DeepfakeBench
sys.path.insert(0, os.path.join(REPO, "training"))                     # detectors/networks/loss/...
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))         # tools/  -> import fas.*

import cv2  # noqa: F401  (import cv2 before the heavy detector load, like tools/infer.py)
import yaml
from fas.dataset import list_split, LCCFASD
from fas import metrics as fas_metrics


def build_detector(cfg, init_ckpt, device):
    """Instantiate B4/SFDCT from a detector config dict and (optionally) warm-start from a ckpt."""
    from detectors import DETECTOR
    cfg.setdefault("backbone_name", "efficientnetb4")
    cfg.setdefault("backbone_config", {"num_classes": 2, "inc": 3, "dropout": False, "mode": "Original"})
    cfg["backbone_config"]["num_classes"] = 2
    pt = cfg.get("pretrained", "None")
    if isinstance(pt, str) and pt.startswith("./"):                    # resolve relative to repo root
        cfg["pretrained"] = os.path.join(REPO, pt[2:])
    cfg.setdefault("loss_func", "cross_entropy")
    cfg.setdefault("mean", [0.5, 0.5, 0.5])
    cfg.setdefault("std", [0.5, 0.5, 0.5])
    cfg.setdefault("resolution", 256)
    model = DETECTOR[cfg["model_name"]](cfg)
    if init_ckpt and os.path.isfile(init_ckpt):
        sd = torch.load(init_ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        miss, unexp = model.load_state_dict(sd, strict=False)
        print(f"[init] warm-start from {init_ckpt}  (missing={len(miss)}, unexpected={len(unexp)})")
    return model.to(device)


@torch.no_grad()
def infer_probs(model, loader, device, use_amp):
    model.eval()
    P, L = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=use_amp):
            out = model({"image": x})
        P.append(out["prob"].float().cpu().numpy())
        L.append(y.numpy())
    return np.concatenate(L), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="detector yaml (efficientnetb4.yaml | efficientnetb4_sfdct.yaml)")
    ap.add_argument("--data_root", required=True, help="LCC_FASD root (contains *_training/_development/_evaluation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init_ckpt", default=None, help="optional warm-start from an existing deepfake ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=3, help="early-stop after N epochs w/o dev AUC gain")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision (use if NaN/instability)")
    ap.add_argument("--max_per_split", type=int, default=0, help="cap images/split for a smoke run (0=all)")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    use_amp = (not a.no_amp) and a.device.startswith("cuda")

    cfg = yaml.safe_load(open(a.config))
    res = cfg.get("resolution", 256)
    mean, std = cfg.get("mean", [0.5] * 3), cfg.get("std", [0.5] * 3)
    aug_cfg = cfg.get("data_aug")

    # ---- locate the official LCC-FASD splits ----
    def find_split(*keys):
        for d in sorted(glob.glob(os.path.join(a.data_root, "**"), recursive=True)):
            if os.path.isdir(d) and any(k in os.path.basename(d).lower() for k in keys) and list_split(d):
                return d
        return None
    train_dir, dev_dir, test_dir = find_split("train"), find_split("develop", "dev", "val"), find_split("eval", "test")
    assert train_dir and test_dir, f"could not find train/eval splits under {a.data_root}"

    def cap(items):
        if a.max_per_split and len(items) > a.max_per_split:
            random.Random(a.seed).shuffle(items)
            items = items[:a.max_per_split]
        return items
    tr_items, te_items = cap(list_split(train_dir)), cap(list_split(test_dir))
    if dev_dir:
        dv_items = cap(list_split(dev_dir))
    else:                                          # no dev split -> carve 10% off train (avoid test leak)
        random.Random(a.seed).shuffle(tr_items)
        k = max(1, len(tr_items) // 10)
        dv_items, tr_items = tr_items[:k], tr_items[k:]
        print("[dev] no development split found -> using 10% of train for threshold/early-stop")

    cnt = lambda it: dict(collections.Counter(l for _, l in it))
    print(f"splits: train={train_dir}\n        dev={dev_dir}\n        test={test_dir}")
    print(f"counts: train={len(tr_items)} {cnt(tr_items)} | dev={len(dv_items)} {cnt(dv_items)} | test={len(te_items)} {cnt(te_items)}")

    train_ds = LCCFASD(tr_items, res, mean, std, augment=True, aug_cfg=aug_cfg)
    dev_ds = LCCFASD(dv_items, res, mean, std, augment=False)
    test_ds = LCCFASD(te_items, res, mean, std, augment=False)
    tl = DataLoader(train_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True)
    dl = DataLoader(dev_ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)
    el = DataLoader(test_ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, pin_memory=True)

    model = build_detector(cfg, a.init_ckpt, a.device)
    params = model.get_optim_groups(a.lr) if hasattr(model, "get_optim_groups") else model.parameters()
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=5e-4, betas=(0.9, 0.999), eps=1e-8)
    scaler = GradScaler("cuda", enabled=use_amp)

    best_auc, best_path, bad = -1.0, os.path.join(a.out, "ckpt_best.pth"), 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run, n = 0.0, 0
        for x, y in tl:
            data = {"image": x.to(a.device, non_blocking=True), "label": y.to(a.device, non_blocking=True)}
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                pred = model(data)
                loss = model.get_losses(data, pred)["overall"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += float(loss) * len(y); n += len(y)
        dvL, dvP = infer_probs(model, dl, a.device, use_amp)
        dev_auc = fas_metrics.auc_score(dvL, dvP)
        dev_eer, _ = fas_metrics.eer_threshold(dvL, dvP)
        print(f"epoch {ep:2d}/{a.epochs}  loss={run/max(n,1):.4f}  dev_auc={dev_auc:.4f}  dev_eer={dev_eer:.4f}  ({time.time()-t0:.0f}s)")
        if dev_auc > best_auc:
            best_auc, bad = dev_auc, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg, "dev_auc": dev_auc}, best_path)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[early-stop] no dev_auc gain for {a.patience} epochs")
                break

    # ---- final test with the best ckpt, threshold fixed at dev EER ----
    model.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
    model.to(a.device)
    dvL, dvP = infer_probs(model, dl, a.device, use_amp)
    teL, teP = infer_probs(model, el, a.device, use_amp)
    m = fas_metrics.evaluate(dvL, dvP, teL, teP)
    m.update({"model": cfg["model_name"], "config": os.path.basename(a.config),
              "dev_best_auc": best_auc, "n_test": int(len(teL)),
              "init_ckpt": a.init_ckpt or "imagenet"})
    json.dump(m, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print("== TEST ==\n" + json.dumps(m, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
