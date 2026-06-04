"""
viz_dct.py — visualization suite for the Hybrid Spatial-Frequency (block-DCT) thesis.
Reuses the SAME dct_matrix / zigzag bands as the real detector (sfdct_core), so the
figures explain exactly what ContentDCT computes.

Generates (no trained model / dataset needed for the 3 motivation figures):
  1. dct_basis_grid.png      — the 8x8 2D-DCT basis (what each band "looks for")
  2. real_vs_fake_spectrum.png — avg log|DCT| real vs fake + difference (the core hypothesis)
  3. band_energy_profile.png  — mean |DCT| per zigzag band, real vs each manipulation

If --data DIR is given with subfolders real/ and fake_*/ of face crops, figs 2-3 use REAL
data; otherwise a physically-motivated synthetic demo (high-band checkerboard artifact in
fakes) stands in so the pipeline is verifiable today and swaps to real frames later.

Run:  python3 tools/viz_dct.py                 # synthetic demo
      python3 tools/viz_dct.py --data ./crops  # real face crops
"""
import os, sys, glob, argparse, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "detectors"))
from sfdct_core import dct_matrix, zigzag_band_of, ContentDCT

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viz_out")
os.makedirs(OUT, exist_ok=True)
N = 8

# ---------- fig 1: DCT basis grid ----------
def fig_basis():
    M = dct_matrix(N).numpy()                       # [8,8] 1D-DCT
    band = zigzag_band_of(N, 16).numpy()
    fig, ax = plt.subplots(N, N, figsize=(8.2, 8.6))
    fig.suptitle("2D-DCT basis 8x8  (u: freq down, v: freq right) — (0,0)=DC≈mean, bottom-right=high freq",
                 fontsize=11)
    for u in range(N):
        for v in range(N):
            basis = np.outer(M[u], M[v])            # 2D basis = outer product of 1D rows
            a = ax[u, v]; a.imshow(basis, cmap="gray"); a.set_xticks([]); a.set_yticks([])
            if u == 0 and v == 0:
                a.set_title("DC", color="green", fontsize=8)
            if u >= 4 and v >= 4 and band[u, v] >= 13:   # mark a high-band cell (GAN artifact home)
                for s in a.spines.values(): s.set_color("red"); s.set_linewidth(2)
    ax[5, 6].set_title("high band\n(artifact)", color="red", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(OUT, "dct_basis_grid.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ---------- data loaders ----------
def _load_imgs(folder, n=64, res=128):
    from PIL import Image
    fs = sorted(glob.glob(os.path.join(folder, "*")))[:n]
    out = []
    for f in fs:
        try:
            im = Image.open(f).convert("L").resize((res, res))
            out.append(np.asarray(im, np.float32) / 255.0)
        except Exception:
            pass
    return np.stack(out) if out else None

def _synth(kind, n=64, res=128, seed=0):
    """real = smooth natural-ish texture; fake = real + faint HIGH-freq checkerboard (GAN artifact)."""
    rng = np.random.RandomState(seed)
    yy, xx = np.meshgrid(np.arange(res), np.arange(res), indexing="ij")
    batch = []
    for i in range(n):
        base = 0.5 + 0.25 * np.sin(2*math.pi*(rng.uniform(1,3)*xx + rng.uniform(1,3)*yy)/res)
        base += 0.05 * rng.randn(res, res)
        if kind == "DF":   art = 0.06*(((yy+xx) % 2)*2-1)                       # high-band checker
        elif kind == "F2F":art = 0.05*np.sin(2*math.pi*22*xx/res)               # mid-high vertical
        elif kind == "NT": art = 0.04*(((yy//2+xx//2) % 2)*2-1)                 # block-grid
        else:              art = 0.0                                            # real
        batch.append(np.clip(base + art, 0, 1).astype(np.float32))
    return np.stack(batch)

def _avg_spectrum(imgs):
    """avg log|DCT| over all 8x8 blocks of all images -> [8,8]."""
    M = dct_matrix(N)
    x = torch.from_numpy(imgs)[:, None]                          # [B,1,H,W]
    B, _, H, W = x.shape
    blk = x.unfold(2, N, N).unfold(3, N, N)                      # [B,1,nh,nw,8,8]
    coef = torch.einsum("pa,ncijab,qb->ncijpq", M, blk, M)
    logmag = torch.log1p(coef.abs()).mean((0,1,2,3))            # [8,8]
    return logmag.numpy()

def _band_profile(imgs, nb=16):
    dct = ContentDCT(freq_repr="global48", channels=1)
    x = torch.from_numpy(imgs)[:, None].repeat(1,3,1,1)         # fake 3ch (Y dominates)
    flat, _ = dct(x)
    return flat.reshape(flat.shape[0], 3, -1).mean((0,1)).numpy()   # [nb] mean over imgs & channels

# ---------- fig 2: real vs fake spectrum ----------
def fig_spectrum(data):
    real = _load_imgs(os.path.join(data,"real")) if data else None
    fake = _load_imgs(os.path.join(data,"fake")) if data else None
    if real is None: real = _synth("real", seed=1)
    if fake is None: fake = _synth("DF", seed=2)
    sr, sf = _avg_spectrum(real), _avg_spectrum(fake)
    diff = np.abs(sf - sr)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    fig.suptitle("Spatially near-identical, but the fake's energy leaks into the HIGH DCT band", fontsize=12)
    for a, (s, t) in zip(ax, [(sr,"REAL  log|DCT|"), (sf,"FAKE  log|DCT|"), (diff,"|DCT_fake| - |DCT_real|")]):
        im = a.imshow(s, cmap=("magma" if "diff" in t.lower() or "-" in t else "viridis"))
        a.set_title(t); a.set_xlabel("v (freq →)"); a.set_ylabel("u (freq ↓)")
        plt.colorbar(im, ax=a, fraction=0.046)
    ax[2].annotate("artifact\n(high band)", xy=(6,6), xytext=(1.2,2.2), color="white",
                   fontweight="bold", arrowprops=dict(arrowstyle="->", color="white"))
    fig.tight_layout(rect=[0,0,1,0.93])
    p = os.path.join(OUT, "real_vs_fake_spectrum.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ---------- fig 3: band-energy profile ----------
def fig_bandprofile(data):
    series = {}
    for name, kind in [("real","real"),("DF","DF"),("F2F","F2F"),("NT","NT")]:
        folder = os.path.join(data, name) if data else None
        imgs = _load_imgs(folder) if (folder and os.path.isdir(folder)) else _synth(kind, seed=hash(name)%100)
        series[name] = _band_profile(imgs)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    colors = {"real":"0.4","DF":"tab:red","F2F":"tab:blue","NT":"tab:green"}
    for k,v in series.items():
        ax.plot(range(len(v)), v, label=k, color=colors[k], lw=2 if k!="real" else 3)
    ax.set_yscale("log"); ax.axvline(len(series['real'])//2, ls="--", c="k", lw=1)
    ax.set_xlabel("zigzag band index (0=DC … 15=highest)"); ax.set_ylabel("mean log|DCT|")
    ax.set_title("Per-band energy: manipulations diverge from real in MID/HIGH bands → the 16 K/V tokens")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(OUT, "band_energy_profile.png"); fig.savefig(p, dpi=130); plt.close(fig)
    return p

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=None)
    a = ap.parse_args()
    mode = "REAL data: "+a.data if a.data else "SYNTHETIC demo (physically-motivated artifact)"
    print(f"viz_dct — {mode}\noutput dir: {os.path.abspath(OUT)}")
    for fn, args in [(fig_basis, ()), (fig_spectrum, (a.data,)), (fig_bandprofile, (a.data,))]:
        p = fn(*args); print("  wrote", os.path.relpath(p, os.getcwd()) if os.path.isabs(p) else p)
    print("done.")
