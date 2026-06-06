#!/usr/bin/env python3
"""train_liveness.py — train B4 baseline cho LIVENESS / Face Anti-Spoofing trên LCC-FASD.

SELF-CONTAINED: chỉ dùng các file *_liveness.py trong thư mục này + torchvision. KHÔNG đụng code
deepfake (training/detectors/...). AMP + early-stop để vừa GPU 4GB. Báo cáo APCER/BPCER/ACER/EER/AUC
với ngưỡng cố định @EER trên dev (không nhìn test).

Chạy từ thư mục liveness/ hoặc repo root:
  python liveness/train_liveness.py --data_root <LCC_FASD dir> --out liveness/out/b4
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
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # cho import *_liveness
from model_liveness import build_b4_liveness
from dataset_liveness import list_split_liveness, LCCFASDLiveness
import metrics_liveness as M


@torch.no_grad()
def infer_probs_liveness(model, loader, device, use_amp):
    model.eval()
    P, L = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logit = model(x)
        prob = torch.softmax(logit.float(), dim=1)[:, 1]
        P.append(prob.cpu().numpy()); L.append(y.numpy())
    return np.concatenate(L), np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="LCC_FASD (chứa *_training/_development/_evaluation)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--max_per_split", type=int, default=0, help="cap ảnh/split cho smoke (0=all)")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    use_amp = (not a.no_amp) and a.device.startswith("cuda")

    def find_split(*keys):
        for d in sorted(glob.glob(os.path.join(a.data_root, "**"), recursive=True)):
            if os.path.isdir(d) and any(k in os.path.basename(d).lower() for k in keys) and list_split_liveness(d):
                return d
        return None
    train_dir, dev_dir, test_dir = find_split("train"), find_split("develop", "dev", "val"), find_split("eval", "test")
    assert train_dir and test_dir, f"không tìm thấy train/eval split trong {a.data_root}"

    def cap(items):
        if a.max_per_split and len(items) > a.max_per_split:
            random.Random(a.seed).shuffle(items); items = items[:a.max_per_split]
        return items
    tr, te = cap(list_split_liveness(train_dir)), cap(list_split_liveness(test_dir))
    if dev_dir:
        dv = cap(list_split_liveness(dev_dir))
    else:                                            # không có dev -> cắt 10% train (tránh leak test)
        random.Random(a.seed).shuffle(tr); k = max(1, len(tr) // 10); dv, tr = tr[:k], tr[k:]
        print("[dev] không có development split -> dùng 10% train")

    cnt = lambda it: dict(collections.Counter(l for _, l in it))
    print(f"counts: train={len(tr)} {cnt(tr)} | dev={len(dv)} {cnt(dv)} | test={len(te)} {cnt(te)}")

    mk = lambda items, aug: DataLoader(LCCFASDLiveness(items, a.res, augment=aug), batch_size=a.batch,
                                       shuffle=aug, num_workers=a.workers, pin_memory=True, drop_last=aug)
    tl, dl, el = mk(tr, True), mk(dv, False), mk(te, False)

    model = build_b4_liveness(num_classes=2, pretrained=True).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=5e-4)
    scaler = GradScaler("cuda", enabled=use_amp)
    ce = nn.CrossEntropyLoss()

    best_auc, best_path, bad = -1.0, os.path.join(a.out, "ckpt_best_liveness.pth"), 0
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); run, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(a.device, non_blocking=True), y.to(a.device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=use_amp):
                loss = ce(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += float(loss) * len(y); n += len(y)
        dvL, dvP = infer_probs_liveness(model, dl, a.device, use_amp)
        dev_auc = M.auc_liveness(dvL, dvP); dev_eer, _ = M.eer_threshold_liveness(dvL, dvP)
        print(f"epoch {ep:2d}/{a.epochs}  loss={run/max(n,1):.4f}  dev_auc={dev_auc:.4f}  dev_eer={dev_eer:.4f}  ({time.time()-t0:.0f}s)")
        if dev_auc > best_auc:
            best_auc, bad = dev_auc, 0
            torch.save({"state_dict": model.state_dict(), "dev_auc": dev_auc, "res": a.res}, best_path)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[early-stop] dev_auc không cải thiện {a.patience} epoch"); break

    model.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"]); model.to(a.device)
    dvL, dvP = infer_probs_liveness(model, dl, a.device, use_amp)
    teL, teP = infer_probs_liveness(model, el, a.device, use_amp)
    res = M.evaluate_liveness(dvL, dvP, teL, teP)
    res.update({"model": "b4_liveness_baseline", "dev_best_auc": best_auc, "n_test": int(len(teL))})
    json.dump(res, open(os.path.join(a.out, "metrics_liveness.json"), "w"), indent=2)
    print("== TEST (liveness) ==\n" + json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
