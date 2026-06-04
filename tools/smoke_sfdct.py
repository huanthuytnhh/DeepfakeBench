"""
smoke_sfdct.py — CPU smoke test for the Hybrid Spatial–Frequency (block-DCT) modules.
Tests ONLY sfdct_core (no DeepfakeBench backbone / registry needed), for BOTH freq reprs.

  per variant: shapes -> gate-0 identity -> gate engages ; then overfit-1-batch (freq separates real/fake + grads flow)

Run:  CUDA_VISIBLE_DEVICES="" python3 tools/smoke_sfdct.py
"""
import os, sys, math
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training", "detectors"))
from sfdct_core import ContentDCT, GatedCrossAttnFusion

torch.manual_seed(0)
dev = "cpu"; B, Csp, Hf, H = 2, 1792, 7, 256
ok = True
def check(name, cond):
    global ok; ok = ok and bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

img = torch.randn(B, 3, H, H, device=dev)
x = torch.randn(B, Csp, Hf, Hf, device=dev)

for repr in ["global48", "blockgrid"]:
    print(f"\n== variant: freq_repr = {repr} ==")
    dct = ContentDCT(freq_repr=repr, grid=Hf).to(dev)
    flat, tokens = dct(img)
    exp_T = 16 if repr == "global48" else Hf * Hf      # 16 vs 49
    exp_in = 3 if repr == "global48" else 48           # 3 vs C*nbands=48
    check("dct48 flat shape == (B,48)", tuple(flat.shape) == (B, 48))
    check(f"kv tokens shape == (B,{exp_T},{exp_in})", tuple(tokens.shape) == (B, exp_T, exp_in))
    check("dct token_in/n_tokens match", dct.token_in == exp_in and dct.n_tokens == exp_T)
    check("dct has 0 learnable params", sum(p.numel() for p in dct.parameters()) == 0)

    fuse = GatedCrossAttnFusion(spatial_ch=Csp, token_in=dct.token_in, n_tokens=dct.n_tokens).to(dev)
    out = fuse(x, tokens)
    check("fusion out shape == x shape", out.shape == x.shape)
    check("gate-0 => fusion(x,freq) == x EXACTLY", torch.allclose(out, x, atol=1e-6))
    with torch.no_grad(): fuse.alpha.fill_(0.5)
    check("alpha=0.5 => output changes", not torch.allclose(fuse(x, tokens), x, atol=1e-4))

print("\n== overfit-1-batch: DCT-48 features separate real vs fake + grads flow (repr-independent) ==")
def make_batch(n, dev):
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    low = 0.5 * torch.cos(math.pi * xx / H * 2).float()[None, None].to(dev)
    checker = (((yy + xx) % 2) * 2 - 1).float()[None, None].to(dev)
    y = torch.arange(n) % 2
    base = low.repeat(n, 3, 1, 1).clone()
    for i in range(n):
        if y[i] == 1: base[i] += 0.2 * checker[0]
    return base.to(dev), y.to(dev)

dct = ContentDCT().to(dev)
clf = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 2)).to(dev)
opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
xb, yb = make_batch(16, dev)
feat = dct(xb)[0].detach(); feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-6)
loss0 = None
for _ in range(300):
    loss = nn.functional.cross_entropy(clf(feat), yb)
    if loss0 is None: loss0 = float(loss)
    opt.zero_grad(); loss.backward(); opt.step()
acc = (clf(feat).argmax(1) == yb).float().mean().item()
check(f"loss decreased ({loss0:.3f} -> {float(loss):.3f})", float(loss) < loss0 * 0.5)
check(f"overfit-1-batch acc == 1.0 (got {acc:.2f})", acc > 0.99)
proj = nn.Linear(48, 48).to(dev); clf2 = nn.Linear(48, 2).to(dev)
opt2 = torch.optim.Adam(list(proj.parameters()) + list(clf2.parameters()), lr=1e-2)
for _ in range(50):
    l = nn.functional.cross_entropy(clf2(proj(dct(xb)[0])), yb); opt2.zero_grad(); l.backward(); opt2.step()
check("gradients flow through the DCT front-end", proj.weight.grad is not None and float(proj.weight.grad.abs().sum()) > 0)

print("\n" + ("ALL SMOKE CHECKS PASSED ✓" if ok else "SOME CHECKS FAILED ✗"))
sys.exit(0 if ok else 1)
