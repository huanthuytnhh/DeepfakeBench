#!/usr/bin/env python3
"""MT-5b · Điền experiments/results.csv (đang RỖNG) từ log training THẬT.
(← novelty #5: multi-seed/đối chứng — đây là bước gom mọi run để thống kê)

Parse mọi logs/training/*/training.log → 1 dòng/run với best/last AUC (FF++/Celeb/avg).
Khi bạn chạy thêm seed, chỉ cần chạy lại script này.

Trạng thái: CHẠY ĐƯỢC NGAY.
Output: GHI ĐÈ experiments/results.csv (+ bản sao outputs/mt05_runs_summary.{csv,md})
"""
import re
import pandas as pd

import common as C

COLS = ["tag", "seed", "model", "n_epochs_logged", "best_avg_epoch", "best_avg_ff",
        "best_avg_cdf", "best_avg", "best_ff_value", "best_ff_cdf", "last_ff", "last_cdf", "last_avg", "log"]

RE_EPOCH = re.compile(r"Epoch\[(\d+)\]\s*end with testing auc")
RE_DSAUC = re.compile(r"\|\s*([\w\-+]+):\s*auc=([0-9.]+)\s*\|")
RE_AVG = re.compile(r"\|\s*avg auc:\s*([0-9.]+)\s*\|")
RE_SEED = re.compile(r"(?:manual_?seed|seed)['\":\s]+(\d+)", re.I)


def parse_run(log_path):
    epochs = []
    cur_ep, cur = None, {}
    seed = ""
    text = log_path.read_text(errors="ignore")
    for line in text.splitlines():
        ms = RE_SEED.search(line)
        if ms and not seed:
            seed = ms.group(1)
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
    return epochs, seed


def ff_cdf(d):
    ff = next((v for k, v in d.items() if k.lower().startswith("face")), float("nan"))
    cdf = next((v for k, v in d.items() if "celeb" in k.lower()), float("nan"))
    return ff, cdf


def main():
    rows = []
    runs = sorted([p for p in C.LOGS.iterdir() if (p / "training.log").exists()]) if C.LOGS.exists() else []
    for p in runs:
        epochs, seed = parse_run(p / "training.log")
        if not epochs:
            continue
        model = re.sub(r"_20[0-9\-]+$", "", p.name)
        best_avg = max(epochs, key=lambda t: t[2])
        best_ff_e = max(epochs, key=lambda t: ff_cdf(t[1])[0] if ff_cdf(t[1])[0] == ff_cdf(t[1])[0] else -1)
        last = epochs[-1]
        ba_ff, ba_cdf = ff_cdf(best_avg[1])
        bf_ff, bf_cdf = ff_cdf(best_ff_e[1])
        l_ff, l_cdf = ff_cdf(last[1])
        rows.append({
            "tag": p.name, "seed": seed, "model": model, "n_epochs_logged": len(epochs),
            "best_avg_epoch": best_avg[0], "best_avg_ff": round(ba_ff, 4), "best_avg_cdf": round(ba_cdf, 4),
            "best_avg": round(best_avg[2], 4), "best_ff_value": round(bf_ff, 4), "best_ff_cdf": round(bf_cdf, 4),
            "last_ff": round(l_ff, 4), "last_cdf": round(l_cdf, 4), "last_avg": round(last[2], 4),
            "log": str(p / "training.log"),
        })
    df = pd.DataFrame(rows, columns=COLS).sort_values(["model", "tag"])
    out_csv = C.EXPERIMENTS / "results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv.relative_to(C.REPO)}  ({len(df)} run)")
    C.save_table(df, "mt05_runs_summary.csv")
    # tóm tắt mean±std theo model
    if not df.empty:
        agg = df.groupby("model")["best_avg"].agg(["count", "mean", "std"]).round(4).reset_index()
        C.save_table(agg, "mt05_model_meanstd.csv")
        print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
