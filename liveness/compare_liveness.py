#!/usr/bin/env python3
"""compare_liveness.py — bảng so B4 vs B4+DCT cho liveness từ 2 metrics_liveness.json.
  python liveness/compare_liveness.py liveness/out/b4/metrics_liveness.json liveness/out/b4dct/metrics_liveness.json
"""
import sys
import json


def main():
    rows = []
    for p in sys.argv[1:]:
        try:
            rows.append(json.load(open(p)))
        except Exception as e:
            print(f"!! {p}: {e}")
    if not rows:
        print("no metrics"); return
    hdr = f"{'model':26s} {'AUC↑':>8s} {'ACER↓':>8s} {'APCER↓':>8s} {'BPCER↓':>8s} {'EER↓':>8s}"
    print(hdr); print("-" * len(hdr))
    for m in rows:
        print(f"{m.get('model','?'):26s} {m.get('auc',float('nan')):8.4f} {m.get('acer',float('nan')):8.4f} "
              f"{m.get('apcer',float('nan')):8.4f} {m.get('bpcer',float('nan')):8.4f} {m.get('eer',float('nan')):8.4f}")
    if len(rows) == 2:
        d_auc = rows[1].get("auc", 0) - rows[0].get("auc", 0)
        d_acer = rows[1].get("acer", 0) - rows[0].get("acer", 0)
        print("-" * len(hdr))
        print(f"{'Δ (B4+DCT - B4)':26s} {d_auc:+8.4f} {d_acer:+8.4f}")
        helps = d_auc > 0 and d_acer < 0
        print(f"\n=> block-DCT {'GIÚP' if helps else 'CHƯA rõ giúp'} liveness (AUC {d_auc:+.4f}, ACER {d_acer:+.4f}).")


if __name__ == "__main__":
    main()
