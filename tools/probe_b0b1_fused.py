#!/usr/bin/env python3
"""
probe_b0b1_fused.py — put B0 (B4 baseline) and B1 (naive SFDCT) on the SAME $0 frozen-probe scale
as the modality screen, so all rows live in one table.

  B0 (B4 baseline)   = trunk RGB features, mean-pooled, linear-probed   (== the RGB anchor)
  B1 (naive SFDCT)   = the FUSED feature (trunk + ContentDCT + GatedCrossAttnFusion, loaded from the
                       naive checkpoint's fusion.* weights), mean-pooled, linear-probed

Same naive trunk, same FF++(train)->CDFv2(cross) linear probe as probe_residual_modality.py. sklearn-free.
NOTE: these frozen-probe numbers are a RELATIVE signal screen on a medium subset; they are NOT comparable
to the end-to-end CDFv2 numbers (B4 0.7497, naive 0.7572 on the full 16,420 test).

Run from the DeepfakeBench repo root:
    python tools/probe_b0b1_fused.py --per_class 1500
"""
import os, time, json, argparse, importlib.util
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, p):
    s = importlib.util.spec_from_file_location(m, p); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
P = _load("prm", os.path.join(_HERE, "probe_residual_modality.py"))
core = _load("sfdct_core", os.path.join(_HERE, "..", "training", "detectors", "sfdct_core.py"))
collect_frames, load_b4, probe = P.collect_frames, P.load_b4, P.probe
ContentDCT, GatedCrossAttnFusion = core.ContentDCT, core.GatedCrossAttnFusion

torch.set_grad_enabled(False)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_naive_head(ckpt):
    """ContentDCT (buffers only) + GatedCrossAttnFusion with naive-checkpoint fusion weights."""
    sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("state_dict", sd)
    dct = ContentDCT(block=8, nbands=16, to_ycbcr=True, drop_dc=True, freq_repr="global48",
                     grid=8, channels=3, input_mean=0.5, input_std=0.5).to(DEV).eval()
    fusion = GatedCrossAttnFusion(spatial_ch=1792, token_in=dct.token_in, n_tokens=dct.n_tokens,
                                  d_model=128, heads=4, mode="crossattn", n_query=None, gate_mode="zero").to(DEV).eval()
    fw = {k[len("fusion."):]: v for k, v in sd.items() if k.startswith("fusion.")}
    miss, unexp = fusion.load_state_dict(fw, strict=False)
    print(f"   fusion: loaded {len(fw)} tensors (missing={len(miss)} unexpected={len(unexp)})")
    return dct, fusion


def feats_b0b1(net, dct, fusion, items, size, batch):
    """Return (B0 trunk-pooled [N,1792], B1 fused-pooled [N,1792], labels)."""
    from PIL import Image
    B0, B1, Y, buf, labs = [], [], [], [], []
    def flush():
        if not buf: return
        x = torch.stack(buf).to(DEV)
        fmap = net.extract_features(x)                       # [b,1792,8,8]  (B0 trunk)
        _, tok = dct(x)                                      # band tokens
        fused = fusion(fmap, tok)                            # [b,1792,8,8]  (B1 naive fused)
        B0.append(fmap.mean((2, 3)).cpu().numpy()); B1.append(fused.mean((2, 3)).cpu().numpy()); Y.extend(labs)
        buf.clear(); labs.clear()
    for i, (p, lab) in enumerate(items):
        try:
            im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
            buf.append((t - 0.5) / 0.5); labs.append(lab)
        except Exception: pass
        if len(buf) >= batch: flush()
        if i % 512 == 0: print(f"      ...{i}/{len(items)}", end="\r", flush=True)
    flush(); print()
    return np.concatenate(B0), np.concatenate(B1), np.asarray(Y)


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
    ap.add_argument("--out", default="./viz_out/probe_b0b1_fused.json")
    args = ap.parse_args()
    t0 = time.time()
    print(f"[1/3] load naive trunk + fusion head ({DEV})")
    net = load_b4(args.ckpt); dct, fusion = build_naive_head(args.ckpt)
    print(f"[2/3] sample {args.per_class}/class")
    ff, _, _ = collect_frames(args.ffpp_json, args.rgb_dir, args.per_class, args.seed)
    cdf, _, _ = collect_frames(args.cdf_json, args.rgb_dir, args.per_class, args.seed + 1)
    print("[3/3] extract B0 trunk + B1 fused features")
    b0ff, b1ff, yff = feats_b0b1(net, dct, fusion, ff, args.size, args.batch)
    b0cd, b1cd, ycd = feats_b0b1(net, dct, fusion, cdf, args.size, args.batch)
    rng = np.random.default_rng(args.seed); n = len(yff); perm = rng.permutation(n); cut = int(0.8 * n)
    tr, ind = perm[:cut], perm[cut:]
    b0i, b0c = probe(b0ff, yff, b0cd, ycd, tr, ind)
    b1i, b1c = probe(b1ff, yff, b1cd, ycd, tr, ind)
    print("\n" + "=" * 60)
    print(f"{'row':<28}{'FF++(in)':>11}{'CDFv2(cross)':>14}")
    print("-" * 60)
    print(f"{'B0  B4 baseline (RGB trunk)':<28}{b0i:>11.4f}{b0c:>14.4f}")
    print(f"{'B1  naive SFDCT (fused)':<28}{b1i:>11.4f}{b1c:>14.4f}")
    print("=" * 60)
    print(f"\nfused - trunk (frozen-probe Δ) = {b1c - b0c:+.4f}  "
          f"(end-to-end Δ was +0.0075: B4 0.7497 -> naive 0.7572)")
    print("CAVEAT: frozen-probe on medium subset; NOT comparable to end-to-end full-test numbers.")
    json.dump({"B0_trunk": {"in": round(b0i, 4), "cross": round(b0c, 4)},
               "B1_fused": {"in": round(b1i, 4), "cross": round(b1c, 4)},
               "fused_minus_trunk": round(b1c - b0c, 4), "per_class": args.per_class,
               "seconds": round(time.time() - t0, 1)}, open(args.out, "w"), indent=2)
    print(f"Saved {args.out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
