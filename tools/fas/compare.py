#!/usr/bin/env python3
"""Print a side-by-side FAS comparison table from two metrics.json files (B4 vs B4+DCT).

  python tools/fas/compare.py logs/fas/b4/metrics.json logs/fas/sfdct/metrics.json
"""
import sys
import json


def main():
    paths = sys.argv[1:]
    rows = []
    for p in paths:
        try:
            m = json.load(open(p))
        except Exception as e:
            print(f"!! cannot read {p}: {e}")
            continue
        rows.append(m)
    if not rows:
        print("no metrics to compare")
        return
    hdr = f"{'model':28s} {'AUC↑':>8s} {'ACER↓':>8s} {'APCER↓':>8s} {'BPCER↓':>8s} {'EER↓':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for m in rows:
        print(f"{m.get('model','?'):28s} "
              f"{m.get('auc',float('nan')):8.4f} {m.get('acer',float('nan')):8.4f} "
              f"{m.get('apcer',float('nan')):8.4f} {m.get('bpcer',float('nan')):8.4f} "
              f"{m.get('eer',float('nan')):8.4f}")
    if len(rows) == 2:
        d_auc = rows[1].get("auc", 0) - rows[0].get("auc", 0)
        d_acer = rows[1].get("acer", 0) - rows[0].get("acer", 0)
        print("-" * len(hdr))
        print(f"{'Δ (B4+DCT - B4)':28s} {d_auc:+8.4f} {d_acer:+8.4f}")
        print(f"\n=> block-DCT {'GIÚP' if d_auc > 0 and d_acer < 0 else 'CHƯA rõ giúp'} "
              f"trên liveness (AUC {d_auc:+.4f}, ACER {d_acer:+.4f}).")


if __name__ == "__main__":
    main()
