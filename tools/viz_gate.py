"""
viz_gate.py — STANDALONE: visualize the learned zero-init gate alpha from a trained checkpoint.
Needs ONLY the .pth + torch + matplotlib (no dataset, no DeepfakeBench import) — so it always works,
even after the vast instance is gone, from the pulled checkpoint alone.

The fusion gate `alpha` starts at 0 (== EfficientNet-B4 baseline, no regression possible). After training,
how far it opened = how much the frequency branch is actually used. This is the signature figure (B1).

Run:  python3 tools/viz_gate.py --ckpt path/to/ckpt_best.pth [--out viz_out]
"""
import os, sys, argparse
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "viz_out"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

sd = torch.load(a.ckpt, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd:        # handle {'state_dict':...} / {'R':..,'state_dict':..}
    sd = sd["state_dict"]
alphas = {k: v for k, v in sd.items() if k.endswith("alpha") or k.endswith("alpha.weight") or ".alpha" in k}
if not alphas:
    print("No gate 'alpha' parameter found in checkpoint. Keys sampled:")
    for k in list(sd.keys())[:20]:
        print("  ", k)
    sys.exit("=> Is this an SFDCT checkpoint? (gate lives at fusion.alpha)")

for k, v in alphas.items():
    v = v.detach().float().flatten()
    n = v.numel()
    absmean = v.abs().mean().item()
    frac_open = (v.abs() > 1e-3).float().mean().item()
    print(f"[{k}] channels={n}  mean|α|={absmean:.4f}  max|α|={v.abs().max():.4f}  "
          f"frac|α|>1e-3={frac_open*100:.1f}%  (init was 0 → all of this is LEARNED)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.suptitle(f"Zero-init fusion gate after training — '{k}'  "
                 f"(init=0 ⇒ ==B4 baseline; opened ⇒ freq branch is used)", fontsize=11)
    ax[0].hist(v.numpy(), bins=60, color="tab:blue", alpha=0.8)
    ax[0].axvline(0, color="k", ls="--", lw=1, label="init = 0")
    ax[0].set_title(f"α distribution (mean|α|={absmean:.3f})")
    ax[0].set_xlabel("gate value α per channel"); ax[0].set_ylabel("#channels"); ax[0].legend()
    sv, _ = v.abs().sort(descending=True)
    ax[1].plot(sv.numpy(), color="tab:red")
    ax[1].set_title(f"|α| sorted — {frac_open*100:.0f}% of {n} channels engaged")
    ax[1].set_xlabel("channel rank"); ax[1].set_ylabel("|α|")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    p = os.path.join(a.out, "gate_alpha.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p)
