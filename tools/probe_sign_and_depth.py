#!/usr/bin/env python3
"""
probe_sign_and_depth.py — $0 LOCAL pre-screen for two SPSL-derived ideas, keeping block-DCT native.

(a) DCT-SIGN modality (the only DCT-native "phase analog"): 8x8 block-DCT -> keep SIGN (unit magnitude)
    -> iDCT -> image. Direct DCT counterpart of the FFT-phase-only probe (which scored 0.574 cross).
    Two variants: full-band sign, and high-pass sign (zero DC+low bands then sign).
    Question: does DCT-sign carry cross-dataset signal comparable to FFT-phase (0.574) / blockDCT-resid (0.598)?

(c) SHALLOW / reduced-receptive-field (SPSL idea #2): probe the frozen B4 trunk's RGB features at MULTIPLE
    depths (EfficientNet reduction endpoints). Question: do SHALLOWER features generalize BETTER cross-dataset
    (SPSL's claim that high-level semantics hurt transfer) -> informs where to fuse / a shallow exit.

All through the same frozen deepfake-trained B4 trunk, FF++ (train probe) -> CDFv2 (cross). sklearn-free.

Run from the DeepfakeBench repo root:
    python tools/probe_sign_and_depth.py --per_class 1500
"""
import os, time, json, argparse, importlib.util
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, p):
    s = importlib.util.spec_from_file_location(m, p); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
P = _load("prm", os.path.join(_HERE, "probe_residual_modality.py"))   # sklearn-free infra
core = _load("sfdct_core", os.path.join(_HERE, "..", "training", "detectors", "sfdct_core.py"))
collect_frames, load_b4, feats, probe, _auc = P.collect_frames, P.load_b4, P.feats, P.probe, P._auc
dct_matrix, zigzag_band_of = core.dct_matrix, core.zigzag_band_of

torch.set_grad_enabled(False)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCTSign:
    """DCT-native phase analog: 8x8 block-DCT -> SIGN (unit magnitude) -> iDCT -> image.
    high_pass=True zeroes DC+low bands BEFORE taking sign (mid/high sign only)."""
    def __init__(self, drop_k=3, high_pass=False, nbands=16, block=8, mean=0.5, std=0.5):
        self.b, self.mean, self.std, self.hp = block, mean, std, high_pass
        self.M = dct_matrix(block).to(DEV)
        self.keep = (zigzag_band_of(block, nbands) >= drop_k).float().to(DEV) if high_pass else None

    def __call__(self, x):
        b = self.b
        x = (x * self.std + self.mean).clamp(0.0, 1.0)
        B, C, H, W = x.shape
        ph, pw = (b - H % b) % b, (b - W % b) % b
        if ph or pw: x = F.pad(x, (0, pw, 0, ph))
        Bn, Cn, Hp, Wp = x.shape
        blk = x.unfold(2, b, b).unfold(3, b, b)
        coef = torch.einsum("pa,ncijab,qb->ncijpq", self.M, blk, self.M)
        if self.hp: coef = coef * self.keep
        s = torch.sign(coef)                                                # 1-bit "phase analog"
        rec = torch.einsum("pa,ncijpq,qb->ncijab", self.M, s, self.M)
        rec = rec.permute(0, 1, 2, 4, 3, 5).reshape(Bn, Cn, Hp, Wp)
        return rec[:, :, :H, :W]


def feats_depth(net, items, size, batch):
    """Mean-pooled RGB features at each EfficientNet reduction endpoint -> dict{level: [N,Ck]}, labels."""
    from PIL import Image
    acc, Y, buf, labs = {}, [], [], []
    def flush():
        if not buf: return
        x = torch.stack(buf).to(DEV)
        eps = net.extract_endpoints(x)                                      # reduction_1..reduction_5
        for k, v in eps.items():
            acc.setdefault(k, []).append(v.mean((2, 3)).cpu().numpy())
        Y.extend(labs); buf.clear(); labs.clear()
    for i, (p, lab) in enumerate(items):
        try:
            im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
            buf.append((t - 0.5) / 0.5); labs.append(lab)
        except Exception: pass
        if len(buf) >= batch: flush()
        if i % 512 == 0: print(f"      [depth] ...{i}/{len(items)}", end="\r", flush=True)
    flush(); print()
    return {k: np.concatenate(v) for k, v in acc.items()}, np.asarray(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="serving/naive_sfdct/ckpt_best.pth")
    ap.add_argument("--ffpp_json", default="dataset_json_medium/FaceForensics++.json")
    ap.add_argument("--cdf_json", default="dataset_json_medium/Celeb-DF-v2.json")
    ap.add_argument("--rgb_dir", default="./datasets")
    ap.add_argument("--per_class", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--out", default="./viz_out/probe_sign_depth.json")
    args = ap.parse_args()
    t0 = time.time()
    print(f"[1/4] load trunk ({DEV})"); net = load_b4(args.ckpt)
    print(f"[2/4] sample {args.per_class}/class")
    ff, _, _ = collect_frames(args.ffpp_json, args.rgb_dir, args.per_class, args.seed)
    cdf, _, _ = collect_frames(args.cdf_json, args.rgb_dir, args.per_class, args.seed + 1)
    rng = np.random.default_rng(args.seed)

    # ---- (a) DCT-sign modalities ----
    print("[3/4] (a) DCT-sign modalities")
    sign_mods = [("DCT-sign (all)", DCTSign(high_pass=False)), ("DCT-sign (high-pass)", DCTSign(high_pass=True))]
    rows_a = {}
    for name, fn in sign_mods:
        Xff, yff = feats(net, ff, name, fn, args.size, args.batch)
        Xcd, ycd = feats(net, cdf, name, fn, args.size, args.batch)
        n = len(yff); perm = rng.permutation(n); cut = int(0.8 * n)
        ai, ac = probe(Xff, yff, Xcd, ycd, perm[:cut], perm[cut:])
        rows_a[name] = {"in": round(ai, 4), "cross": round(ac, 4)}

    # ---- (c) depth / shallow probe (RGB at each reduction) ----
    print("[4/4] (c) depth probe (RGB at each B4 reduction)")
    Dff, yff = feats_depth(net, ff, args.size, args.batch)
    Dcd, ycd = feats_depth(net, cdf, args.size, args.batch)
    n = len(yff); perm = rng.permutation(n); cut = int(0.8 * n); tr, ind = perm[:cut], perm[cut:]
    rows_c = {}
    for lvl in sorted(Dff):
        ai, ac = probe(Dff[lvl], yff, Dcd[lvl], ycd, tr, ind)
        rows_c[lvl] = {"dim": int(Dff[lvl].shape[1]), "in": round(ai, 4), "cross": round(ac, 4)}

    print("\n" + "=" * 70)
    print("(a) DCT-SIGN (phase analog)   ref: FFT-phase=0.574  blockDCT-resid=0.598")
    print(f"{'modality':<24}{'FF++(in)':>11}{'CDFv2(cross)':>14}")
    for k, v in rows_a.items(): print(f"{k:<24}{v['in']:>11}{v['cross']:>14}")
    print("-" * 70)
    print("(c) DEPTH (RGB) — shallow vs deep cross-dataset")
    print(f"{'reduction':<24}{'dim':>6}{'FF++(in)':>11}{'CDFv2(cross)':>14}")
    for k, v in rows_c.items(): print(f"{k:<24}{v['dim']:>6}{v['in']:>11}{v['cross']:>14}")
    print("=" * 70)

    best_sign = max(rows_a.values(), key=lambda r: r["cross"])["cross"]
    deep = rows_c[sorted(rows_c)[-1]]["cross"]
    best_lvl = max(rows_c, key=lambda k: rows_c[k]["cross"]); best_depth = rows_c[best_lvl]["cross"]
    print(f"\n(a) best DCT-sign cross = {best_sign}  (vs FFT-phase 0.574, blockDCT-resid 0.598)")
    print(f"    -> {'GO: DCT-sign is a viable native phase-analog' if best_sign >= 0.55 else 'WEAK: DCT-sign < phase/residual (1-bit too coarse) — keep using residual'}")
    print(f"(c) deepest level cross = {deep:.4f}; BEST level = {best_lvl} @ {best_depth:.4f}")
    print(f"    -> {'GO: a SHALLOWER exit generalizes better (SPSL idea holds on B4)' if best_depth > deep + 0.01 else 'NO: deepest is best — shallow exit does not help on B4'}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"sign": rows_a, "depth": rows_c, "best_sign": best_sign, "deep": deep,
               "best_level": best_lvl, "best_depth": best_depth, "seconds": round(time.time() - t0, 1)},
              open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
