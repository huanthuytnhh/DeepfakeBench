#!/usr/bin/env python3
"""
smoke_hff.py — $0 LOCAL smoke for block-DCT-HFF modules (sfdct_hff_core). NO DeepfakeBench framework.
Three checks per the project rule (shape -> dry-run -> overfit-1-batch) + the FLOOR>=B4 guarantee:

  1. SHAPE   : every module forwards a [4,3,256,256] tensor to the expected shape, no crash.
  2. FLOOR   : at init (HFFGate alpha=0), fused output == B4 final map exactly (floor >= B4 at init),
               for BOTH R1 (no RSA, single-scale) and R3 (RSA + multi-scale).
  3. OVERFIT : on 8 real + 8 fake frames, a small head on the fused feature drives loss -> ~0
               (proves the R1 and R3 models CAN learn). Runs on the local GPU/CPU.

Run from the DeepfakeBench repo root:
    python tools/smoke_hff.py
"""
import os, sys, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
def _load(m, p):
    s = importlib.util.spec_from_file_location(m, p); x = importlib.util.module_from_spec(s); s.loader.exec_module(x); return x
hff = _load("sfdct_hff_core", os.path.join(_HERE, "..", "training", "detectors", "sfdct_hff_core.py"))
prm = _load("prm", os.path.join(_HERE, "probe_residual_modality.py"))           # sklearn-free infra

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OK = "\033[92mPASS\033[0m"; BAD = "\033[91mFAIL\033[0m"
fails = []


class HFFHead(nn.Module):
    """Minimal block-DCT-HFF feature head over a frozen-or-live B4 trunk (mirrors detector.features())."""
    def __init__(self, trunk, out_ch=1792, multi_scale=False, use_rsa=False, freeze_trunk=False):
        super().__init__()
        self.trunk = trunk; self.freeze_trunk = freeze_trunk
        self.dct = hff.BlockDCTHighPass(drop_k=3)
        self.stream = hff.HFStream(out_ch=out_ch, grid=8, multi_scale=multi_scale)
        self.rsa = hff.RSAttention() if use_rsa else None
        self.gate = hff.HFFGate(out_ch)

    def features(self, image):
        if self.freeze_trunk:
            with torch.no_grad():
                x = self.trunk.extract_features(image)    # frozen for low-VRAM smoke (no trunk activations)
        else:
            x = self.trunk.extract_features(image)        # [B,1792,8,8]
        res = self.dct(image)
        h = self.stream(res)
        if self.rsa is not None:
            h = self.rsa(h)
        return self.gate(x, h)                            # alpha=0 => == x at init


def build(trunk, **kw):
    return HFFHead(trunk, **kw).to(DEV).eval()


def get_trunk():
    """B4 trunk for the smoke — prefer the local naive ckpt, else ImageNet pretrained (./start.sh setup),
    else random init. Module smoke (shape/floor/overfit) does NOT need trained weights."""
    from efficientnet_pytorch import EfficientNet
    p_naive = "serving/naive_sfdct/ckpt_best.pth"
    p_imnet = "training/pretrained/efficientnet-b4-6ed6700e.pth"
    if os.path.exists(p_naive):
        return prm.load_b4(p_naive)
    net = EfficientNet.from_name("efficientnet-b4")
    if os.path.exists(p_imnet):
        sd = torch.load(p_imnet, map_location="cpu"); sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        miss, _ = net.load_state_dict(sd, strict=False)
        print(f"   trunk: ImageNet B4 from {p_imnet} (missing={len(miss)})")
    else:
        print("   trunk: random init (no ckpt — fine for module smoke)")
    return net.to(DEV).eval()


def main():
    print(f"device = {DEV}")
    trunk = get_trunk()
    x = torch.randn(4, 3, 256, 256, device=DEV)

    # ---- 1. SHAPE ----
    print("\n[1] SHAPE")
    dct = hff.BlockDCTHighPass(drop_k=3).to(DEV)
    res = dct(x);                                  print(f"  BlockDCTHighPass {tuple(res.shape)}  {OK if res.shape==x.shape else BAD}")
    for ms in (False, True):
        st = hff.HFStream(1792, 8, ms).to(DEV); h = st(res)
        ok = tuple(h.shape) == (4, 1792, 8, 8); fails.append(not ok)
        print(f"  HFStream(multi_scale={ms}) {tuple(h.shape)}  {OK if ok else BAD}")
    rsa = hff.RSAttention().to(DEV); hr = rsa(torch.randn(4, 1792, 8, 8, device=DEV))
    ok = tuple(hr.shape) == (4, 1792, 8, 8); fails.append(not ok); print(f"  RSAttention {tuple(hr.shape)}  {OK if ok else BAD}")

    # ---- 2. FLOOR (alpha=0 => fused == trunk) ----
    print("\n[2] FLOOR >= B4 at init (gate alpha=0)")
    with torch.no_grad():
        xb = trunk.extract_features(x)
        for tag, kw in [("R1", dict(multi_scale=False, use_rsa=False)), ("R3", dict(multi_scale=True, use_rsa=True))]:
            m = build(trunk, **kw)
            out = m.features(x)
            same = torch.allclose(out, xb, atol=1e-5)
            amax = m.gate.alpha.abs().max().item()
            fails.append(not same)
            print(f"  {tag}: fused==B4 {same} (max|alpha|={amax:.3g})  {OK if same else BAD}")

    # ---- 3. OVERFIT-1-BATCH (8 real + 8 fake) ----
    print("\n[3] OVERFIT-1-BATCH (loss should drop to ~0)")
    from PIL import Image
    ff, _, _ = prm.collect_frames("dataset_json_medium/FaceForensics++.json", "./datasets", 6, 1024)
    imgs, labs = [], []
    for p, lab in ff[:12]:
        try:
            im = Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)
            t = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
            imgs.append((t - 0.5) / 0.5); labs.append(lab)
        except Exception:
            pass
    X = torch.stack(imgs).to(DEV); Y = torch.tensor(labs, device=DEV)
    torch.set_grad_enabled(True)                                            # prm import turned it off globally
    for tag, kw in [("R1", dict(multi_scale=False, use_rsa=False)), ("R3", dict(multi_scale=True, use_rsa=True))]:
        m = HFFHead(trunk, multi_scale=kw["multi_scale"], use_rsa=kw["use_rsa"], freeze_trunk=True).to(DEV).train()
        head = nn.Linear(1792, 2).to(DEV)
        opt = torch.optim.Adam(list(m.stream.parameters()) + ([] if m.rsa is None else list(m.rsa.parameters()))
                               + list(m.gate.parameters()) + list(head.parameters()), lr=1e-3)
        l0 = lN = None
        for it in range(80):
            opt.zero_grad()
            feat = m.features(X).mean((2, 3))
            loss = F.cross_entropy(head(feat), Y)
            loss.backward(); opt.step()
            if it == 0: l0 = loss.item()
            lN = loss.item()
        ok = lN < 0.2 and lN < l0; fails.append(not ok)
        print(f"  {tag}: loss {l0:.3f} -> {lN:.3f} (trunk frozen for 4GB smoke)  {OK if ok else BAD}")
        del m, head, opt; torch.cuda.empty_cache() if DEV.type == "cuda" else None

    print("\n" + ("=" * 40))
    if any(fails):
        print(f"{BAD}: {sum(fails)} check(s) failed"); sys.exit(1)
    print(f"{OK}: all smoke checks passed — modules correct, floor>=B4 at init, both R1/R3 can learn.")


if __name__ == "__main__":
    main()
