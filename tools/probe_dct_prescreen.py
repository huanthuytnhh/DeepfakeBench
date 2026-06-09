#!/usr/bin/env python3
"""
probe_dct_prescreen.py — $0 LOCAL pre-screen before renting a GPU.

Question it answers (NOT "will AUC go up end-to-end" — that needs a real train):
    "Does a richer block-DCT REPRESENTATION carry more CROSS-DATASET signal than the
     naive global-mean repr the current SFDCT uses?"

Method: freeze the 0-param ContentDCT front-end, compute features for a sample of
FF++ (train probe) and Celeb-DF-v2 (cross test) face crops, fit a linear logistic
regression on FF++, and report BOTH in-domain (FF++ held-out) and cross (CDFv2) AUC
for 3 representation variants. Runs on CPU in minutes; no checkpoint, no GPU rental.

V1 global-mean          : per-image mean of the band map        -> == naive repr (control)
V2 mean+std             : + per-image STD across 8x8 blocks      -> tests bottleneck (2) localization
V3 mean+std (suppressed): V2 on drop_low_bands + SRM residual + DCT-sign  -> tests (1)(4)(5)

DECISION RULE (printed at the end):
  GO (weak)  : best cross AUC (V2/V3) >= V1 cross + GO_MARGIN  AND its in->cross drop
               is not worse than V1's  -> the repr has extra transferable signal; renting
               a 5090 to test the architectural fix end-to-end is justified (NOT a guarantee).
  NO-GO      : otherwise -> the representation change does not add transferable signal;
               do NOT rent — pivot to the honest-negative narrative.

Run from the DeepfakeBench repo root:
    python tools/probe_dct_prescreen.py --per_class 2000
    python tools/probe_dct_prescreen.py --per_class 3000 --workers 8   # more samples / faster
"""
import os, sys, json, time, argparse, importlib.util
import numpy as np

# --- import ContentDCT directly from its file (it has NO heavy DeepfakeBench deps) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_CORE = os.path.join(_HERE, "..", "training", "detectors", "sfdct_core.py")
_spec = importlib.util.spec_from_file_location("sfdct_core", _CORE)
sfdct_core = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(sfdct_core)
ContentDCT = sfdct_core.ContentDCT

import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

torch.set_grad_enabled(False)

REAL_MARKERS = ("real", "original", "youtube", "actor")   # everything else = fake


def collect_frames(json_path, rgb_dir, per_class, seed):
    """Walk a DeepfakeBench dataset json, return balanced [(abs_path, label0/1), ...].
    label 0 = real, 1 = fake. Robust to nested {subset:{split:{vid:{label,frames}}}}."""
    with open(json_path) as f:
        data = json.load(f)
    real, fake = [], []

    def is_real(label_str):
        s = str(label_str).lower()
        return any(m in s for m in REAL_MARKERS)

    def walk(node):
        if isinstance(node, dict):
            if "frames" in node and isinstance(node["frames"], list):
                lbl = node.get("label", "")
                bucket = real if is_real(lbl) else fake
                for fr in node["frames"]:
                    bucket.append(fr)
            else:
                for v in node.values():
                    walk(v)
    walk(data)

    rng = np.random.default_rng(seed)
    def pick(lst):
        lst = list(dict.fromkeys(lst))            # dedupe, keep order
        idx = rng.permutation(len(lst))[:per_class]
        return [lst[i] for i in idx]
    real_s, fake_s = pick(real), pick(fake)
    items = [(os.path.join(rgb_dir, p), 0) for p in real_s] + \
            [(os.path.join(rgb_dir, p), 1) for p in fake_s]
    rng.shuffle(items)
    return items, len(real), len(fake)


def load_batch(paths, size=256):
    """PNG paths -> normalized model-input tensor [B,3,size,size] (mean=std=0.5)."""
    out = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            out.append((t - 0.5) / 0.5)                       # -> [-1,1], ContentDCT denorms back
        except Exception:
            out.append(None)
    return out


def perblk_map(dct, x, use_sign):
    """Return per-block band map [B, C, Nblocks, K] using ContentDCT internals."""
    logmag, sign, nh, nw = dct._block_dct_logmag(x)
    perblk = dct._bands(logmag)
    if use_sign:
        perblk = torch.cat([perblk, dct._bands(sign)], dim=3)
    return perblk                                              # [B,C,N,K]


def feats_for_items(items, dct_plain, dct_supp, batch, size, workers):
    """Compute V1/V2/V3 feature matrices + labels for all items (skipping unreadable)."""
    V1, V2, V3, Y = [], [], [], []
    torch.set_num_threads(max(1, workers))
    n = len(items)
    for i in range(0, n, batch):
        chunk = items[i:i + batch]
        tens = load_batch([p for p, _ in chunk], size)
        keep = [(t, lab) for (t, (_, lab)) in zip(tens, chunk) if t is not None]
        if not keep:
            continue
        x = torch.stack([t for t, _ in keep])                 # [b,3,H,W]
        labs = [lab for _, lab in keep]
        pm = perblk_map(dct_plain, x, use_sign=False)         # [b,C,N,K]
        ps = perblk_map(dct_supp, x, use_sign=True)
        mean_p = pm.mean(2).reshape(pm.shape[0], -1)
        std_p = pm.std(2).reshape(pm.shape[0], -1)
        mean_s = ps.mean(2).reshape(ps.shape[0], -1)
        std_s = ps.std(2).reshape(ps.shape[0], -1)
        V1.append(mean_p.numpy())
        V2.append(torch.cat([mean_p, std_p], 1).numpy())
        V3.append(torch.cat([mean_s, std_s], 1).numpy())
        Y.extend(labs)
        if (i // batch) % 10 == 0:
            print(f"   ...{min(i + batch, n)}/{n}", end="\r", flush=True)
    print()
    return (np.concatenate(V1), np.concatenate(V2), np.concatenate(V3), np.asarray(Y))


def probe(Xtr, ytr, Xin, yin, Xcr, ycr, C):
    """Standardize on train, fit L2 logistic regression, return (in-domain AUC, cross AUC)."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    auc_in = roc_auc_score(yin, clf.decision_function(sc.transform(Xin)))
    auc_cr = roc_auc_score(ycr, clf.decision_function(sc.transform(Xcr)))
    return auc_in, auc_cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffpp_json", default="dataset_json_medium/FaceForensics++.json")
    ap.add_argument("--cdf_json", default="dataset_json_medium/Celeb-DF-v2.json")
    ap.add_argument("--rgb_dir", default="./datasets")
    ap.add_argument("--per_class", type=int, default=2000, help="frames per class per dataset")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--C", type=float, default=0.05, help="L2 strength (small=strong reg, anti-overfit)")
    ap.add_argument("--go_margin", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--out", default="./viz_out/probe_prescreen.json")
    args = ap.parse_args()
    t0 = time.time()

    print(f"[1/4] sampling {args.per_class}/class  FF++={args.ffpp_json}  CDFv2={args.cdf_json}")
    ff_items, ff_nr, ff_nf = collect_frames(args.ffpp_json, args.rgb_dir, args.per_class, args.seed)
    cdf_items, cdf_nr, cdf_nf = collect_frames(args.cdf_json, args.rgb_dir, args.per_class, args.seed + 1)
    print(f"      FF++ pool real/fake={ff_nr}/{ff_nf}  sampled={len(ff_items)} | "
          f"CDFv2 pool real/fake={cdf_nr}/{cdf_nf} sampled={len(cdf_items)}")

    dct_plain = ContentDCT(block=8, nbands=16, to_ycbcr=True, drop_dc=True,
                           freq_repr="global48", grid=8, channels=3,
                           drop_low_bands=0, use_sign=False, srm_residual=False)
    dct_supp = ContentDCT(block=8, nbands=16, to_ycbcr=True, drop_dc=True,
                          freq_repr="global48", grid=8, channels=3,
                          drop_low_bands=3, use_sign=True, srm_residual=True)

    print("[2/4] extracting frozen DCT features (FF++)")
    f1, f2, f3, yff = feats_for_items(ff_items, dct_plain, dct_supp, args.batch, args.size, args.workers)
    print("[3/4] extracting frozen DCT features (CDFv2)")
    c1, c2, c3, ycdf = feats_for_items(cdf_items, dct_plain, dct_supp, args.batch, args.size, args.workers)

    # FF++ -> 80% train probe / 20% in-domain held-out
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(yff)); cut = int(0.8 * len(perm))
    tr, ind = perm[:cut], perm[cut:]

    print("[4/4] probing 3 variants (linear, L2)\n")
    rows = []
    for name, Xff, Xcd in [("V1 global-mean (naive)", f1, c1),
                           ("V2 mean+std (localization)", f2, c2),
                           ("V3 mean+std suppressed", f3, c3)]:
        auc_in, auc_cr = probe(Xff[tr], yff[tr], Xff[ind], yff[ind], Xcd, ycdf, args.C)
        rows.append({"variant": name, "dim": int(Xff.shape[1]),
                     "indomain_auc": round(float(auc_in), 4), "cross_auc": round(float(auc_cr), 4),
                     "transfer_gap": round(float(auc_in - auc_cr), 4)})

    v1 = rows[0]["cross_auc"]
    best = max(rows[1:], key=lambda r: r["cross_auc"])
    v1_gap = rows[0]["transfer_gap"]
    go = (best["cross_auc"] >= v1 + args.go_margin) and (best["transfer_gap"] <= v1_gap + 0.03)

    print("=" * 74)
    print(f"{'variant':<30}{'dim':>6}{'FF++(in)':>11}{'CDFv2(cross)':>14}{'gap':>8}")
    print("-" * 74)
    for r in rows:
        print(f"{r['variant']:<30}{r['dim']:>6}{r['indomain_auc']:>11}{r['cross_auc']:>14}{r['transfer_gap']:>8}")
    print("=" * 74)
    print(f"\nBest non-control cross AUC: {best['variant']} = {best['cross_auc']}  "
          f"(V1 naive = {v1}, margin needed = +{args.go_margin})")
    verdict = "GO (weak signal)" if go else "NO-GO"
    print(f"\n  >>> DECISION: {verdict} <<<")
    if go:
        print("  The richer repr carries EXTRA cross-dataset signal beyond naive global-mean.")
        print("  Renting a 5090 to test this architectural fix END-TO-END is JUSTIFIED.")
        print("  NOT a guarantee end-to-end AUC rises — it removes the 'flying blind' risk.")
    else:
        print("  No extra transferable signal in the representation (cross AUC did not beat")
        print("  naive by the margin, or only in-domain improved = pure overfit).")
        print("  Recommendation: do NOT rent — pivot to the honest-negative narrative.")
    print("\n  CAVEAT: frozen+linear probe with no B4 fusion. Strong at KILLING bad ideas;")
    print("  a positive is necessary-but-not-sufficient. The real arbiter is a val-selected")
    print("  end-to-end CDFv2 run, reported honestly whichever way it goes.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "decision": verdict, "go_margin": args.go_margin,
                   "per_class": args.per_class, "seconds": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nSaved {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
