#!/usr/bin/env python3
"""MT-9 · Training/validation curves từ LOG THẬT (loss + AUC).
(← novelty #5 + TRÁNH bẫy của Trí: training_progress_cnn.py mô phỏng 55 epoch bằng np.random)

Tự parse logs/training/*/training.log (self-contained, regex trên text) → mỗi run 1 PNG:
(1) train loss vs iter, (2) train AUC vs iter, (3) test AUC (FF++/Celeb-DF-v2) theo epoch.
TUYỆT ĐỐI không mô phỏng số.

Trạng thái: CHẠY ĐƯỢC NGAY (log sẵn).
Output: outputs/training_curves/<tag>.png  (mặc định chỉ render run mới nhất mỗi prefix; --all để render hết)
"""
import re
import argparse
from pathlib import Path

import common as C

RE_LOSS = re.compile(r"Iter:\s*(\d+)\s+training-loss,\s*overall:\s*([0-9.]+)")
RE_TRAUC = re.compile(r"Iter:\s*(\d+)\s+training-metric.*?auc:\s*([0-9.]+)")
RE_EPOCH = re.compile(r"Epoch\[(\d+)\]\s*end with testing auc")
RE_DSAUC = re.compile(r"\|\s*([\w\-+]+):\s*auc=([0-9.]+)\s*\|")
RE_AVG = re.compile(r"\|\s*avg auc:\s*([0-9.]+)\s*\|")


def parse(log_path: Path):
    il, ia = [], []
    epochs = []  # list of (epoch, {dataset:auc}, avg)
    cur_ep, cur = None, {}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = RE_LOSS.search(line)
        if m: il.append((int(m.group(1)), float(m.group(2))))
        m = RE_TRAUC.search(line)
        if m: ia.append((int(m.group(1)), float(m.group(2))))
        m = RE_EPOCH.search(line)
        if m:
            cur_ep, cur = int(m.group(1)), {}
            continue
        if cur_ep is not None:
            for ds, a in RE_DSAUC.findall(line):
                if ds.lower() != "avg":
                    cur[ds] = float(a)
            ma = RE_AVG.search(line)
            if ma:
                epochs.append((cur_ep, cur, float(ma.group(1))))
                cur_ep = None
    return il, ia, epochs


def plot_run(tag: str, log_path: Path):
    plt = C.setup_mpl()
    il, ia, epochs = parse(log_path)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if il:
        ax[0].plot(*zip(*il)); ax[0].set_title("train loss"); ax[0].set_xlabel("iter")
    if ia:
        ax[1].plot(*zip(*ia), color="tab:green"); ax[1].set_title("train AUC")
        ax[1].set_xlabel("iter"); ax[1].set_ylim(0.4, 1.0)
    if epochs:
        eps = [e for e, _, _ in epochs]
        all_ds = sorted({d for _, dd, _ in epochs for d in dd})
        for ds in all_ds:
            ax[2].plot(eps, [dd.get(ds, float("nan")) for _, dd, _ in epochs], marker="o", label=ds)
        ax[2].plot(eps, [a for _, _, a in epochs], marker="s", color="black", ls="--", label="avg")
        be = max(epochs, key=lambda t: t[2])
        ax[2].axvline(be[0], color="red", ls=":", lw=1, label=f"best avg @ep{be[0]} ({be[2]:.3f})")
        ax[2].set_title("test AUC / epoch"); ax[2].set_xlabel("epoch"); ax[2].set_ylim(0.4, 1.0); ax[2].legend(fontsize=8)
    fig.suptitle(f"MT-9 · {tag}", y=1.02)
    C.save_fig(fig, f"training_curves/{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="render mọi run (mặc định chỉ run mới nhất mỗi prefix)")
    a = ap.parse_args()
    if not C.LOGS.exists():
        print("[MT-9] Không thấy logs/training/."); return
    runs = sorted([p for p in C.LOGS.iterdir() if (p / "training.log").exists()])
    if not a.all:
        latest = {}
        for p in runs:
            prefix = re.sub(r"_20[0-9\-]+$", "", p.name)
            latest[prefix] = p  # sorted → cái cuối là mới nhất
        runs = list(latest.values())
    for p in runs:
        try:
            plot_run(p.name, p / "training.log")
        except Exception as e:
            print(f"[MT-9] lỗi {p.name}: {e}")
    print(f"[MT-9] Render {len(runs)} training curve (loss+AUC từ log THẬT).")


if __name__ == "__main__":
    main()
