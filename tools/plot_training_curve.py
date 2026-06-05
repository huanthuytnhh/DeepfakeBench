#!/usr/bin/env python3
"""Parse a DeepfakeBench training log -> training_curve.png:
   (1) train loss vs iter, (2) train AUC vs iter, (3) test Celeb-DF-v2 AUC per eval.
Standalone (regex on the log text) so it works without the model/data. Output dropped into the run's
viz folder so it is pushed to HF alongside the eval figures."""
import re, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(log_path):
    it_loss, it_auc, test_auc = [], [], []
    with open(log_path, errors="ignore") as fh:
        for line in fh:
            m = re.search(r"Iter:\s*(\d+)\s+training-loss,\s*overall:\s*([0-9.]+)", line)
            if m:
                it_loss.append((int(m.group(1)), float(m.group(2))))
            m = re.search(r"Iter:\s*(\d+)\s+training-metric.*?auc:\s*([0-9.]+)", line)
            if m:
                it_auc.append((int(m.group(1)), float(m.group(2))))
            m = re.search(r"Celeb-DF-v2.*?testing-metric.*?auc:\s*([0-9.]+)", line)
            if m:
                test_auc.append((len(test_auc) + 1, float(m.group(1))))
    return it_loss, it_auc, test_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="training")
    a = ap.parse_args()
    il, ia, ta = parse(a.log)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    if il:
        ax[0].plot(*zip(*il)); ax[0].set_title("train loss"); ax[0].set_xlabel("iter"); ax[0].grid(alpha=.3)
    if ia:
        ax[1].plot(*zip(*ia), color="tab:green"); ax[1].set_title("train AUC"); ax[1].set_xlabel("iter")
        ax[1].set_ylim(0.4, 1.0); ax[1].grid(alpha=.3)
    if ta:
        xs, ys = zip(*ta)
        ax[2].plot(xs, ys, "o-", color="tab:red")
        ax[2].set_title(f"test Celeb-DF-v2 AUC (best={max(ys):.4f})"); ax[2].set_xlabel("eval #")
        ax[2].axhline(0.7487, ls="--", c="gray", lw=1, label="B4 bar 0.7487"); ax[2].legend(fontsize=8)
        ax[2].set_ylim(0.5, 0.9); ax[2].grid(alpha=.3)
    fig.suptitle(f"Training curves — {a.title}"); fig.tight_layout()
    fig.savefig(a.out, dpi=130); plt.close(fig)
    print(f"saved {a.out}  (loss pts={len(il)}, train-auc pts={len(ia)}, test-auc pts={len(ta)})")


if __name__ == "__main__":
    main()
