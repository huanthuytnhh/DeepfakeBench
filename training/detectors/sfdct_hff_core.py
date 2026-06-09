"""
sfdct_hff_core.py — pure-torch modules for the block-DCT-HFF detector (thesis SFDCT-HFF line).
NO DeepfakeBench imports -> unit/smoke-testable standalone (see tools/smoke_hff.py).

Design (committee plan 2026-06-09, docs/plans/2026-06-09-blockdct-hff.md):
- Keep BLOCK-DCT as the ONLY frequency operator. Adopt HFF's winning mechanism
  (learnable + multi-scale + residual-guided attention) but on a block-DCT high-pass RESIDUAL IMAGE
  instead of SRM, and fuse into B4 via a ZERO-INIT gate so the model == B4 at init (floor >= B4).

Modules:
- BlockDCTHighPass : 8x8 block-DCT -> zero DC+low bands -> iDCT -> residual image (0 learnable params).
                     (verified in tools/probe_residual_modality.py: cross-dataset signal > chance.)
- HFStream         : conv encoder on the residual image -> [B, out_ch, 8, 8]; multi_scale optionally
                     adds two mid-scale projections (the HFF multi-scale idea, self-contained).
- RSAttention      : residual-guided spatial attention (HFF RSA) applied to the HF feature.
- HFFGate          : per-channel zero-init gate  out = x + alpha * hf   (alpha=0 => out == x at init).

Floor guarantee: alpha is zero-initialised on the HF add, so at init out == B4 final map regardless of
HFStream/RSA — a no-regression-AT-INIT property (the trained model can still differ).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_matrix(n: int = 8) -> torch.Tensor:
    k = torch.arange(n).float(); i = k.view(-1, 1); j = k.view(1, -1)
    M = torch.cos(math.pi * (2 * j + 1) * i / (2 * n))
    M[0, :] *= math.sqrt(1.0 / n); M[1:, :] *= math.sqrt(2.0 / n)
    return M


def zigzag_band_of(n: int = 8, nbands: int = 16) -> torch.Tensor:
    order = []
    for s in range(2 * n - 1):
        ks = range(s + 1) if s % 2 else range(s, -1, -1)
        for k in ks:
            r, c = (k, s - k) if s % 2 else (s - k, k)
            if r < n and c < n:
                order.append((r, c))
    band_of = torch.zeros(n, n, dtype=torch.long)
    for rank, (r, c) in enumerate(order):
        band_of[r, c] = min(rank * nbands // (n * n), nbands - 1)
    return band_of


class BlockDCTHighPass(nn.Module):
    """8x8 block-DCT high-pass RESIDUAL IMAGE: DCT -> zero DC + (drop_k-1) lowest zigzag bands -> iDCT.
    0 learnable params. Returns [B,3,H,W] residual (mid/high block-DCT energy; content suppressed)."""
    def __init__(self, drop_k: int = 3, nbands: int = 16, block: int = 8,
                 input_mean: float = 0.5, input_std: float = 0.5):
        super().__init__()
        self.b = block
        self.register_buffer("M", dct_matrix(block))
        keep = (zigzag_band_of(block, nbands) >= drop_k).float()             # zero positions in bands < drop_k
        self.register_buffer("keep", keep)
        self.register_buffer("mean", torch.tensor(float(input_mean)))
        self.register_buffer("std", torch.tensor(float(input_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.b
        x = (x * self.std + self.mean).clamp(0.0, 1.0)                        # denorm to pixel [0,1]
        B, C, H, W = x.shape
        ph, pw = (b - H % b) % b, (b - W % b) % b
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        Bn, Cn, Hp, Wp = x.shape
        blk = x.unfold(2, b, b).unfold(3, b, b)                               # [B,C,nh,nw,8,8]
        coef = torch.einsum("pa,ncijab,qb->ncijpq", self.M, blk, self.M)
        coef = coef * self.keep                                              # high-pass
        rec = torch.einsum("pa,ncijpq,qb->ncijab", self.M, coef, self.M)      # inverse DCT
        rec = rec.permute(0, 1, 2, 4, 3, 5).reshape(Bn, Cn, Hp, Wp)
        return rec[:, :, :H, :W]


def _cbr(cin, cout, stride=2):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class HFStream(nn.Module):
    """Conv encoder over the residual image -> [B, out_ch, grid, grid]. 256 -> 8 via 5 stride-2 blocks.
    multi_scale=True also projects two mid-scale features to out_ch and adds them (HFF multi-scale)."""
    def __init__(self, out_ch: int = 1792, grid: int = 8, multi_scale: bool = False):
        super().__init__()
        chs = [3, 32, 64, 128, 256, out_ch]                                  # 5 stride-2 => /32 (256->8)
        self.blocks = nn.ModuleList([_cbr(chs[i], chs[i + 1]) for i in range(5)])
        self.grid, self.multi_scale = grid, multi_scale
        if multi_scale:
            self.proj = nn.ModuleList([nn.Conv2d(chs[3], out_ch, 1), nn.Conv2d(chs[4], out_ch, 1)])  # 128,256 -> out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for blk in self.blocks:
            x = blk(x); feats.append(x)                                      # feats: [128,64,32,16,8] spatial
        out = feats[-1]                                                      # [B,out_ch,8,8]
        if self.multi_scale:
            for proj, fi in zip(self.proj, [feats[2], feats[3]]):            # 32x32(128ch), 16x16(256ch)
                out = out + F.adaptive_avg_pool2d(proj(fi), (self.grid, self.grid))
        return out


class RSAttention(nn.Module):
    """Residual-guided spatial attention (HFF RSA): feature -> [max,avg] over channels -> 7x7 conv -> sigmoid
    -> spatial map M in [0,1]; returns f * M. Applied to the HF stream (gated overall by HFFGate alpha)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=True)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        mx = f.max(1, keepdim=True)[0]; av = f.mean(1, keepdim=True)
        m = torch.sigmoid(self.conv(torch.cat([mx, av], 1)))                 # [B,1,H,W]
        return f * m


class HFFGate(nn.Module):
    """Per-channel ZERO-INIT gate: out = x + alpha * hf. alpha=0 at init => out == x (floor >= B4)."""
    def __init__(self, ch: int = 1792):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(ch, 1, 1))

    def forward(self, x: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * hf
