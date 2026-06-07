#!/usr/bin/env python3
"""MT-4a · Phân bố lớp / dataset (real vs fake, per-manipulation).
(← novelty #4: Trí verify train/test cùng phân bố ~70/30 để lập luận no-distribution-shift)

Đọc dataset_json_*/<dataset>.json (cấu trúc DeepfakeBench) → đếm real/fake + theo loại
manipulation (DF/F2F/FS/NT...) → bar/pie + bảng. Dùng để lập luận distribution-shift khi
so within-dataset (FF++) vs cross-dataset (Celeb-DF-v2/DFDC).

Trạng thái: CHẠY ĐƯỢC NGAY nếu dataset_json tồn tại.
Output: outputs/mt04_distribution.{csv,md}, outputs/mt04_distribution.png
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

import common as C

# thứ tự ưu tiên thư mục json
JSON_DIRS = [C.REPO / "dataset_json_medium", C.REPO / "dataset_json_small",
             Path("/home/huanthuytnhh/Desktop/thanhln/datn/dataset_json")]


def find_jsons():
    for d in JSON_DIRS:
        if d.exists():
            js = sorted(d.glob("*.json"))
            if js:
                return d, js
    return None, []


def walk_labels(obj, real=0, fake=0, manip=None):
    """Duyệt đệ quy dict json DeepfakeBench, đếm theo key 'label' và nhánh manipulation."""
    manip = manip if manip is not None else Counter()
    if isinstance(obj, dict):
        if "label" in obj and isinstance(obj["label"], (int, float)):
            pass
        for k, v in obj.items():
            real, fake = walk_labels(v, real, fake, manip)[:2] if False else (real, fake)
    return real, fake, manip


def summarize(path: Path):
    """Đếm số clip real/fake và theo manipulation. Cấu trúc DFBench: {dataset:{split:{label_folder:{...}}}}.
    Robust: dò các key chứa 'real'/'fake'/'youtube'/manipulation."""
    data = json.loads(path.read_text())
    counts = Counter()
    REAL_KEYS = ("real", "youtube", "actors", "live", "original")
    MANIP = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter",
             "DFD", "FF-DF", "FF-F2F", "FF-FS", "FF-NT", "fake", "spoof", "synthesis")

    def rec(o, tag=None):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = str(k).lower()
                nt = tag
                if any(r in lk for r in REAL_KEYS):
                    nt = "REAL"
                elif any(mp.lower() in lk for mp in MANIP):
                    nt = k  # giữ tên manipulation
                if isinstance(v, dict) and ("frames" in v or "label" in v or "video_path" in str(v)[:200]):
                    counts[nt or "UNK"] += 1
                rec(v, nt)
    rec(data)
    return counts


def main():
    plt = C.setup_mpl()
    d, jsons = find_jsons()
    if not jsons:
        print("[MT-4a] Không thấy dataset_json_*; bỏ qua (chạy sau khi có json)."); return
    print(f"[MT-4a] dùng {d}")
    rows = []
    fig, axes = plt.subplots(1, len(jsons), figsize=(6 * len(jsons), 4.5))
    if len(jsons) == 1:
        axes = [axes]
    for ax, jp in zip(axes, jsons):
        c = summarize(jp)
        total = sum(c.values()) or 1
        real = sum(v for k, v in c.items() if k == "REAL")
        for k, v in sorted(c.items(), key=lambda x: -x[1]):
            rows.append({"dataset": jp.stem, "class": k, "count": v, "pct": round(100 * v / total, 1)})
        labels = list(c.keys()); vals = list(c.values())
        ax.bar(range(len(labels)), vals, color=["tab:green" if l == "REAL" else "tab:red" for l in labels])
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"{jp.stem}\nreal={real} / total={total} ({100*real/total:.0f}% real)")
        ax.set_ylabel("#clip")
    fig.suptitle("MT-4 · Phân bố real/fake & manipulation theo dataset", y=1.03)
    C.save_fig(fig, "mt04_distribution.png")
    C.save_table(pd.DataFrame(rows), "mt04_distribution.csv")
    print("[MT-4a] Phân bố dataset xong. Dùng để lập luận distribution-shift (như Trí).")


if __name__ == "__main__":
    main()
