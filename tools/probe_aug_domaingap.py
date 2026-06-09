#!/usr/bin/env python3
"""
probe_aug_domaingap.py — $0 LOCAL pre-screen for the SPATIAL "domain-shift aug" lever (B1).

Question (NOT "will end-to-end AUC rise" — that needs a GPU train):
    "Does augmenting FF++ to look like CDFv2 (heavier JPEG + downscale) (1) make the FF++->CDFv2
     real/fake boundary TRANSFER better, and (2) SHRINK the FF++<->CDFv2 domain gap?"

Method: freeze a TRAINED EfficientNet-B4 trunk, mean-pool its [B,1792,8,8] map to a 1792-vec, then
run two linear probes on FF++ (clean) vs FF++ (augmented), evaluated against CDFv2:
   M1 real/fake CROSS : train logreg on FF++ real/fake feats -> test CDFv2.  aug>clean => aug helps transfer.
   M2 domain GAP      : train logreg FF++(0) vs CDFv2(1).  aug pushes AUC toward 0.5 => gap closed.

HONEST: weaker than the DCT probe (aug's real benefit is end-to-end invariance learning; a frozen-feature
probe sees only a shadow of it). Strong at KILLING the idea; a positive is weak-confirm. EMA is NOT
pre-screenable and is excluded here (it's ~free to just turn on in the real run).

Run from the DeepfakeBench repo root:
    python tools/probe_aug_domaingap.py --per_class 1500
"""
import os, io, json, time, random, argparse, importlib.util
import numpy as np
from PIL import Image
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

torch.set_grad_enabled(False)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_MARKERS = ("real", "original", "youtube", "actor")

# reuse the json walker from the DCT probe (same dir)
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("dctprobe", os.path.join(_HERE, "probe_dct_prescreen.py"))
dctprobe = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(dctprobe)
collect_frames = dctprobe.collect_frames


def load_b4(ckpt):
    from efficientnet_pytorch import EfficientNet
    net = EfficientNet.from_name("efficientnet-b4")
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    bk = {k.replace("backbone.efficientnet.", ""): v for k, v in sd.items()
          if k.startswith("backbone.efficientnet.")}
    miss, unexp = net.load_state_dict(bk, strict=False)
    loaded = len(bk)
    print(f"   loaded {loaded} trunk tensors from {ckpt}  (missing={len(miss)} unexpected={len(unexp)})")
    return net.to(DEV).eval()


def augment(im, p_down=0.7):
    """Domain-shift aug on a 256 crop: low-quality JPEG + random downscale-upscale."""
    q = random.randint(15, 35)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=q); im = Image.open(buf).convert("RGB")
    if random.random() < p_down:
        f = random.uniform(0.4, 0.8); w, h = im.size
        sm = im.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR)
        im = sm.resize((w, h), Image.BILINEAR)
    return im


def feats(net, items, size, batch, aug=False, seed=0):
    """Mean-pooled B4 trunk features [N,1792] + labels (skip unreadable). aug=True applies domain-shift aug."""
    random.seed(seed)
    X, Y = [], []
    buf, labs = [], []

    def flush():
        if not buf:
            return
        x = torch.stack(buf).to(DEV)
        fmap = net.extract_features(x)            # [b,1792,8,8]
        X.append(fmap.mean((2, 3)).cpu().numpy())
        Y.extend(labs)
        buf.clear(); labs.clear()

    for k, (path, lab) in enumerate(items):
        try:
            im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
            if aug:
                im = augment(im)
            t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
            buf.append((t - 0.5) / 0.5); labs.append(lab)
        except Exception:
            pass
        if len(buf) >= batch:
            flush()
        if k % 512 == 0:
            print(f"      ...{k}/{len(items)}", end="\r", flush=True)
    flush(); print()
    return np.concatenate(X), np.asarray(Y)


def cross_probe(Xtr, ytr, Xte, yte, C=0.05):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.decision_function(sc.transform(Xte)))


def domain_auc(Xff, Xcdf, C=0.05, seed=0):
    """Separability of FF++(0) vs CDFv2(1): ~1.0 = big domain gap, ~0.5 = no gap."""
    X = np.concatenate([Xff, Xcdf]); d = np.r_[np.zeros(len(Xff)), np.ones(len(Xcdf))]
    rng = np.random.default_rng(seed); perm = rng.permutation(len(d)); cut = int(0.7 * len(d))
    tr, te = perm[:cut], perm[cut:]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X[tr]), d[tr])
    return roc_auc_score(d[te], clf.decision_function(sc.transform(X[te])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="serving/naive_sfdct/ckpt_best.pth", help="any deepfake-trained B4 trunk")
    ap.add_argument("--ffpp_json", default="dataset_json_medium/FaceForensics++.json")
    ap.add_argument("--cdf_json", default="dataset_json_medium/Celeb-DF-v2.json")
    ap.add_argument("--rgb_dir", default="./datasets")
    ap.add_argument("--per_class", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--out", default="./viz_out/probe_aug_domaingap.json")
    args = ap.parse_args()
    t0 = time.time()
    print(f"[1/4] load trunk ({DEV})")
    net = load_b4(args.ckpt)
    print(f"[2/4] sampling {args.per_class}/class")
    ff_items, _, _ = collect_frames(args.ffpp_json, args.rgb_dir, args.per_class, args.seed)
    cdf_items, _, _ = collect_frames(args.cdf_json, args.rgb_dir, args.per_class, args.seed + 1)

    print("[3/4] extracting B4 features: FF++ clean / FF++ aug / CDFv2")
    Xff_c, yff = feats(net, ff_items, args.size, args.batch, aug=False, seed=1)
    Xff_a, yff_a = feats(net, ff_items, args.size, args.batch, aug=True, seed=1)
    Xcdf, ycdf = feats(net, cdf_items, args.size, args.batch, aug=False, seed=2)

    print("[4/4] probing\n")
    m1_clean = cross_probe(Xff_c, yff, Xcdf, ycdf)
    m1_aug = cross_probe(Xff_a, yff_a, Xcdf, ycdf)
    d_clean = domain_auc(Xff_c, Xcdf, seed=args.seed)
    d_aug = domain_auc(Xff_a, Xcdf, seed=args.seed)

    go = (m1_aug >= m1_clean + 0.01) or (d_aug <= d_clean - 0.03)
    print("=" * 64)
    print(f"  M1 real/fake CROSS (FF++ -> CDFv2)   clean = {m1_clean:.4f}   aug = {m1_aug:.4f}   Δ = {m1_aug-m1_clean:+.4f}")
    print(f"  M2 domain GAP (FF++ vs CDFv2 sep.)   clean = {d_clean:.4f}   aug = {d_aug:.4f}   Δ = {d_aug-d_clean:+.4f}")
    print("=" * 64)
    verdict = "GO (weak)" if go else "NO-GO"
    print(f"\n  >>> DECISION: {verdict} <<<")
    if go:
        print("  Domain-shift aug improves FF++->CDFv2 transfer and/or shrinks the domain gap on frozen")
        print("  B4 features. Renting a 5090 to train B4+aug+EMA and SFDCT+aug+EMA is justified.")
    else:
        print("  Aug neither improves cross-transfer (>= +0.01) nor closes the domain gap (<= -0.03) on")
        print("  frozen features. WEAK evidence vs renting — end-to-end may still differ, but no $0 signal.")
    print("\n  CAVEAT: frozen-feature shadow of an end-to-end effect; EMA not testable here. A positive is")
    print("  weak-confirm; a negative is a real 'do not expect much'. Real arbiter = the val-selected run.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"m1_clean": round(float(m1_clean), 4), "m1_aug": round(float(m1_aug), 4),
               "domain_clean": round(float(d_clean), 4), "domain_aug": round(float(d_aug), 4),
               "decision": verdict, "per_class": args.per_class, "seconds": round(time.time() - t0, 1)},
              open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
