#!/usr/bin/env python3
"""
probe_residual_modality.py — $0 LOCAL pre-screen for the block-DCT-HFF bet.

THE BET (keeping block-DCT central): replace HFF's SRM-residual with a BLOCK-DCT
high-pass RESIDUAL IMAGE (8x8 DCT -> zero DC+low bands -> iDCT) as the texture-suppressed
modality feeding a learnable multi-scale + residual-guided-attention stream on B4.

THE DECISIVE QUESTION this probe answers ($0, no training):
    "Is a block-DCT high-pass RESIDUAL IMAGE as informative a modality as SRM residual
     for CROSS-DATASET (FF++ -> CDFv2), when seen by the B4 trunk?"
    (HFF proved SRM-residual works: CelebDF 0.794. If block-DCT-residual carries comparable
     cross-dataset signal, the block-DCT-HFF re-architecture is worth GPU; if it is far below
     SRM, the bet is risky.)

Method: freeze a deepfake-trained B4 trunk, feed THREE input modalities, mean-pool the
[B,1792,8,8] map, linear-probe on FF++ (train) -> test CDFv2 (cross) and FF++ held-out (in):
    RGB                     : normalized image            (upper anchor; the trunk's native input)
    SRM-residual            : SRMHighPass(image)          (HFF's modality; reference that WORKS)
    blockDCT-residual       : iDCT(mid/high block-DCT)    (OUR block-DCT-preserving modality)

HONEST CAVEAT: a frozen RGB-trained trunk sees residuals as OOD, so absolute numbers are
depressed for BOTH residual modalities equally — what matters is blockDCT-residual VS SRM-residual
(same handicap). A positive (blockDCT ~ SRM) is a green light; blockDCT << SRM is a real warning.
The true arbiter remains an end-to-end trained block-DCT-HFF run.

Run from the DeepfakeBench repo root:
    python tools/probe_residual_modality.py --per_class 1500
"""
import os, time, json, argparse, importlib.util
import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))

# reuse ContentDCT building blocks (dct matrix, zigzag bands, SRM) + the aug-probe infra
def _load(mod, path):
    s = importlib.util.spec_from_file_location(mod, path); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
core = _load("sfdct_core", os.path.join(_HERE, "..", "training", "detectors", "sfdct_core.py"))  # sklearn-free
dct_matrix, zigzag_band_of, SRMHighPass = core.dct_matrix, core.zigzag_band_of, core.SRMHighPass

torch.set_grad_enabled(False)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REAL_MARKERS = ("real", "original", "youtube", "actor")


def collect_frames(json_path, rgb_dir, per_class, seed):
    """Walk a DeepfakeBench dataset json -> balanced [(abs_path, 0/1)] (0=real,1=fake). sklearn-free."""
    import json
    data = json.load(open(json_path)); real, fake = [], []
    def is_real(s): return any(m in str(s).lower() for m in REAL_MARKERS)
    def walk(node):
        if isinstance(node, dict):
            if "frames" in node and isinstance(node["frames"], list):
                (real if is_real(node.get("label", "")) else fake).extend(node["frames"])
            else:
                for v in node.values(): walk(v)
    walk(data)
    rng = np.random.default_rng(seed)
    def pick(lst):
        lst = list(dict.fromkeys(lst)); idx = rng.permutation(len(lst))[:per_class]; return [lst[i] for i in idx]
    items = [(os.path.join(rgb_dir, p), 0) for p in pick(real)] + [(os.path.join(rgb_dir, p), 1) for p in pick(fake)]
    rng.shuffle(items); return items, len(real), len(fake)


def load_b4(ckpt):
    """Load a deepfake-trained EfficientNet-B4 trunk from a DeepfakeBench checkpoint."""
    from efficientnet_pytorch import EfficientNet
    net = EfficientNet.from_name("efficientnet-b4")
    sd = torch.load(ckpt, map_location="cpu"); sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    bk = {k.replace("backbone.efficientnet.", ""): v for k, v in sd.items() if k.startswith("backbone.efficientnet.")}
    miss, unexp = net.load_state_dict(bk, strict=False)
    print(f"   loaded {len(bk)} trunk tensors from {ckpt}  (missing={len(miss)} unexpected={len(unexp)})")
    return net.to(DEV).eval()


class BlockDCTHighPass:
    """8x8 block-DCT high-pass RESIDUAL IMAGE: DCT -> zero DC + (drop_k-1) lowest zigzag bands -> iDCT.
    Keeps mid/high block-DCT energy as a content-suppressed residual (block-DCT analog of SRM)."""
    def __init__(self, drop_k=3, nbands=16, block=8, mean=0.5, std=0.5):
        self.b, self.mean, self.std = block, mean, std
        self.M = dct_matrix(block).to(DEV)                                  # [8,8]
        band2d = zigzag_band_of(block, nbands)                              # [8,8] band idx 0..15
        self.keep = (band2d >= drop_k).float().to(DEV)                      # zero positions in bands < drop_k

    def __call__(self, x):
        b = self.b
        x = (x * self.std + self.mean).clamp(0.0, 1.0)                       # denorm to [0,1]
        B, C, H, W = x.shape
        ph, pw = (b - H % b) % b, (b - W % b) % b
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        Bn, Cn, Hp, Wp = x.shape
        blk = x.unfold(2, b, b).unfold(3, b, b)                              # [B,C,nh,nw,8,8]
        coef = torch.einsum("pa,ncijab,qb->ncijpq", self.M, blk, self.M)     # block DCT
        coef = coef * self.keep                                             # high-pass: kill DC+low bands
        rec = torch.einsum("pa,ncijpq,qb->ncijab", self.M, coef, self.M)     # inverse DCT
        rec = rec.permute(0, 1, 2, 4, 3, 5).reshape(Bn, Cn, Hp, Wp)
        return rec[:, :, :H, :W]                                            # [B,3,H,W] residual image


class PhaseImage:
    """SPSL phase modality: gray -> FFT -> keep PHASE (unit amplitude) -> iFFT -> real image (3ch).
    Tests whether FFT-phase carries cross-dataset signal on B4 (SPSL: phase is up-sampling sensitive)."""
    def __init__(self, mean=0.5, std=0.5):
        self.mean, self.std = mean, std
        self.luma = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(DEV)

    def __call__(self, x):
        x = (x * self.std + self.mean).clamp(0.0, 1.0)
        gray = (x * self.luma).sum(1, keepdim=True)                          # [B,1,H,W]
        ph = torch.angle(torch.fft.fft2(gray))                              # phase spectrum
        rec = torch.fft.ifft2(torch.exp(1j * ph)).real                      # iDFT of unit-amplitude phase
        return rec.repeat(1, 3, 1, 1)                                       # 3ch for the trunk


def feats(net, items, modality, fn, size, batch):
    """Mean-pooled B4 trunk features [N,1792] for a given input modality. fn maps image->fed tensor."""
    X, Y, buf, labs = [], [], [], []
    def flush():
        if not buf: return
        x = torch.stack(buf).to(DEV)
        fed = fn(x) if fn is not None else x
        X.append(net.extract_features(fed).mean((2, 3)).cpu().numpy()); Y.extend(labs)
        buf.clear(); labs.clear()
    from PIL import Image
    for k, (p, lab) in enumerate(items):
        try:
            im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
            buf.append((t - 0.5) / 0.5); labs.append(lab)
        except Exception:
            pass
        if len(buf) >= batch: flush()
        if k % 512 == 0: print(f"      [{modality}] ...{k}/{len(items)}", end="\r", flush=True)
    flush(); print()
    return np.concatenate(X), np.asarray(Y)


def _auc(scores, labels):
    """ROC-AUC via Mann-Whitney U (numpy-only, tie-corrected)."""
    s = np.asarray(scores, np.float64); y = np.asarray(labels)
    order = s.argsort(); ranks = np.empty(len(s), np.float64); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    npos = int((y == 1).sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def probe(Xff, yff, Xcdf, ycdf, tr, ind, C=0.05, iters=400):
    """Standardize on train, L2 logistic regression (torch, balanced), return (in-domain AUC, cross AUC).
    sklearn-free to avoid the local pandas/numpy ABI clash."""
    mu = Xff[tr].mean(0); sd = Xff[tr].std(0) + 1e-6
    Z = lambda M: torch.tensor((M - mu) / sd, dtype=torch.float32, device=DEV)
    Xt, yt = Z(Xff[tr]), torch.tensor(yff[tr], dtype=torch.float32, device=DEV)
    w = torch.zeros(Xt.shape[1], device=DEV, requires_grad=True); b = torch.zeros(1, device=DEV, requires_grad=True)
    pos_w = (yt == 0).sum() / (yt == 1).sum().clamp_min(1)            # class-balance positive weight
    opt = torch.optim.Adam([w, b], lr=0.05)
    with torch.enable_grad():                                        # module sets grad off globally
        for _ in range(iters):
            opt.zero_grad()
            logit = Xt @ w + b
            loss = F.binary_cross_entropy_with_logits(logit, yt, pos_weight=pos_w) + (1.0 / C) * 0.5 * (w * w).mean()
            loss.backward(); opt.step()
    with torch.no_grad():
        sc_in = (Z(Xff[ind]) @ w + b).cpu().numpy()
        sc_cr = (Z(Xcdf) @ w + b).cpu().numpy()
    return _auc(sc_in, yff[ind]), _auc(sc_cr, ycdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="serving/naive_sfdct/ckpt_best.pth")
    ap.add_argument("--ffpp_json", default="dataset_json_medium/FaceForensics++.json")
    ap.add_argument("--cdf_json", default="dataset_json_medium/Celeb-DF-v2.json")
    ap.add_argument("--rgb_dir", default="./datasets")
    ap.add_argument("--per_class", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--drop_k", type=int, default=3, help="zero DC + (drop_k-1) lowest zigzag bands")
    ap.add_argument("--seed", type=int, default=1024)
    ap.add_argument("--out", default="./viz_out/probe_residual_modality.json")
    args = ap.parse_args()
    t0 = time.time()

    print(f"[1/4] load B4 trunk ({DEV})")
    net = load_b4(args.ckpt)
    srm = SRMHighPass().to(DEV)
    bdct = BlockDCTHighPass(drop_k=args.drop_k)
    mods = [("RGB", None), ("SRM-residual", srm), ("blockDCT-residual", bdct), ("FFT-phase", PhaseImage())]

    print(f"[2/4] sampling {args.per_class}/class")
    ff, _, _ = collect_frames(args.ffpp_json, args.rgb_dir, args.per_class, args.seed)
    cdf, _, _ = collect_frames(args.cdf_json, args.rgb_dir, args.per_class, args.seed + 1)

    print("[3/4] extract features per modality")
    cache = {}
    for name, fn in mods:
        Xff, yff = feats(net, ff, name, fn, args.size, args.batch)
        Xcd, ycd = feats(net, cdf, name, fn, args.size, args.batch)
        cache[name] = (Xff, yff, Xcd, ycd)

    rng = np.random.default_rng(args.seed)
    n = len(cache["RGB"][1]); perm = rng.permutation(n); cut = int(0.8 * n)
    tr, ind = perm[:cut], perm[cut:]

    print("[4/4] probe\n" + "=" * 66)
    print(f"{'modality':<22}{'FF++(in)':>11}{'CDFv2(cross)':>14}")
    print("-" * 66)
    rows = {}
    for name, _ in mods:
        Xff, yff, Xcd, ycd = cache[name]
        ai, ac = probe(Xff, yff, Xcd, ycd, tr, ind)
        rows[name] = {"in": round(ai, 4), "cross": round(ac, 4)}
        print(f"{name:<22}{ai:>11.4f}{ac:>14.4f}")
    print("=" * 66)

    srm_c = rows["SRM-residual"]["cross"]; bd_c = rows["blockDCT-residual"]["cross"]; rgb_c = rows["RGB"]["cross"]
    gap = bd_c - srm_c
    go = bd_c >= srm_c - 0.02 and bd_c > 0.55
    print(f"\nblockDCT-residual cross = {bd_c}  vs  SRM-residual = {srm_c}  (Δ={gap:+.4f})  | RGB anchor = {rgb_c}")
    verdict = "GO (block-DCT viable as residual modality)" if go else "RISKY (block-DCT << SRM)"
    print(f"\n  >>> {verdict} <<<")
    if go:
        print("  Block-DCT high-pass residual carries cross-dataset signal comparable to SRM (the modality")
        print("  HFF rode to 0.794). Building block-DCT-HFF on B4 (multi-scale + residual-guided attn) is justified.")
    else:
        print("  Block-DCT residual carries materially LESS cross-dataset signal than SRM. Re-architecting")
        print("  block-DCT-HFF risks landing below the SRM version; consider SRM-primary or a hybrid.")
    print("\n  CAVEAT: frozen RGB-trained trunk handicaps BOTH residuals equally; the blockDCT-vs-SRM")
    print("  comparison is the fair signal. End-to-end trained block-DCT-HFF is the real arbiter.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"rows": rows, "delta_blockdct_minus_srm": round(gap, 4), "drop_k": args.drop_k,
               "verdict": verdict, "per_class": args.per_class, "seconds": round(time.time() - t0, 1)},
              open(args.out, "w"), indent=2)
    print(f"\nSaved {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
