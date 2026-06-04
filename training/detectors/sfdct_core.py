"""
sfdct_core.py — pure-torch building blocks for the Hybrid Spatial–Frequency (block-DCT)
detector. NO DeepfakeBench imports here, so this file is unit/smoke-testable standalone
(see tools/smoke_sfdct.py). The DeepfakeBench drop-in detector
(efficientnetb4_sfdct_detector.py) imports ContentDCT + GatedCrossAttnFusion from here.

Both frequency representations use the SAME block-wise 8x8 DCT (the locked title);
they differ only in how the per-block coefficients are aggregated into K/V tokens:

  freq_repr = "global48"  : block-wise 8x8 DCT -> log-mag band-stats, GLOBALLY pooled over all
                            blocks -> 16 band tokens of dim 3 (3 ch).  [48-d global spectral
                            signature; 0 params; robust/light; no block localization]
  freq_repr = "blockgrid" : block-wise 8x8 DCT -> per-block log-mag band map [C*nbands, H/8, W/8],
                            adaptive-pooled to a (grid x grid) lattice -> grid*grid tokens of dim
                            C*nbands. [preserves block spatial layout; true spatial<->spatial-freq
                            cross-attention with the B4 7x7 grid; richer, slightly heavier]

GatedCrossAttnFusion: spatial 7x7 grid (Q) cross-attends to the freq tokens (K/V); a per-channel
gate `alpha` init 0 => IDENTITY at init (out == spatial) => can never regress the backbone.
"""
import math
import torch
import torch.nn as nn

_RGB2YCBCR = torch.tensor([[0.299, 0.587, 0.114],
                           [-0.168736, -0.331264, 0.5],
                           [0.5, -0.418688, -0.081312]])


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


class ContentDCT(nn.Module):
    """Block-wise 8x8 DCT front-end (0 learnable params). Returns (flat48, kv_tokens).

    kv_tokens shape depends on freq_repr:
      global48  -> [B, nbands, C]            (n_tokens=nbands, token_in=C)
      blockgrid -> [B, grid*grid, C*nbands]  (n_tokens=grid*grid, token_in=C*nbands)
    `self.token_in` and `self.n_tokens` expose the dims for the fusion module.
    """
    def __init__(self, block: int = 8, nbands: int = 16, to_ycbcr: bool = True,
                 drop_dc: bool = True, freq_repr: str = "global48", grid: int = 7, channels: int = 3):
        super().__init__()
        assert freq_repr in ("global48", "blockgrid")
        self.block, self.nbands, self.to_ycbcr, self.drop_dc = block, nbands, to_ycbcr, drop_dc
        self.freq_repr, self.grid = freq_repr, grid
        self.register_buffer("M", dct_matrix(block))
        self.register_buffer("band_of", zigzag_band_of(block, nbands).reshape(-1))   # [64]
        self.register_buffer("rgb2ycbcr", _RGB2YCBCR.clone())
        # token dims exposed for the fusion module
        self.token_in = channels if freq_repr == "global48" else channels * nbands
        self.n_tokens = nbands if freq_repr == "global48" else grid * grid

    def _block_dct_logmag(self, x):
        """x[B,3,H,W] -> logmag[B,C,nblocks,64], plus (nh,nw)."""
        b = self.block
        if self.to_ycbcr:
            x = torch.einsum("ij,njhw->nihw", self.rgb2ycbcr.to(x.dtype), x)
        B, C, H, W = x.shape
        ph, pw = (b - H % b) % b, (b - W % b) % b
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph)); B, C, H, W = x.shape
        nh, nw = H // b, W // b
        blk = x.unfold(2, b, b).unfold(3, b, b)                            # [B,C,nh,nw,8,8]
        coef = torch.einsum("pa,ncijab,qb->ncijpq", self.M, blk, self.M)   # [B,C,nh,nw,8,8]
        logmag = torch.log1p(coef.abs()).reshape(B, C, nh * nw, b * b)     # [B,C,nblocks,64]
        return logmag, nh, nw

    def _bands(self, logmag):
        """[B,C,nblocks,64] -> [B,C,nblocks,nbands] (mean of log-mag within each zigzag band)."""
        B, C, N, _ = logmag.shape
        out = logmag.new_zeros(B, C, N, self.nbands)
        cnt = logmag.new_zeros(self.nbands)
        out.index_add_(3, self.band_of, logmag)
        cnt.index_add_(0, self.band_of, torch.ones_like(self.band_of, dtype=logmag.dtype))
        out = out / cnt.clamp_min(1.0)
        if self.drop_dc:
            out[..., 0] = 0.0
        return out                                                          # [B,C,nblocks,nbands]

    def forward(self, x: torch.Tensor):
        logmag, nh, nw = self._block_dct_logmag(x)
        perblk = self._bands(logmag)                                        # [B,C,nblocks,nbands]
        B, C, N, K = perblk.shape
        flat = perblk.mean(2).reshape(B, C * K)                             # [B, 48] (global, for aux)
        if self.freq_repr == "global48":
            tokens = perblk.mean(2).transpose(1, 2).contiguous()           # [B, nbands, C]
        else:  # blockgrid: keep block layout, pool to (grid x grid)
            grid_map = perblk.reshape(B, C, nh, nw, K).permute(0, 1, 4, 2, 3).reshape(B, C * K, nh, nw)
            grid_map = nn.functional.adaptive_avg_pool2d(grid_map, (self.grid, self.grid))  # [B,C*K,g,g]
            tokens = grid_map.flatten(2).transpose(1, 2).contiguous()      # [B, g*g, C*K]
        return flat, tokens


class GatedCrossAttnFusion(nn.Module):
    """Spatial grid (Q) cross-attends to freq tokens (K/V); zero-init gate => identity at init."""
    def __init__(self, spatial_ch: int = 1792, token_in: int = 3, n_tokens: int = 16,
                 d_model: int = 128, heads: int = 4):
        super().__init__()
        self.q = nn.Linear(spatial_ch, d_model)
        self.kv = nn.Linear(token_in, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))        # token positional embed
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.out = nn.Linear(d_model, spatial_ch)
        self.alpha = nn.Parameter(torch.zeros(spatial_ch, 1, 1))          # per-channel gate, init 0

    def forward(self, x: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        B, Csp, H, W = x.shape
        q = self.q(x.flatten(2).transpose(1, 2))                          # [B, H*W, d]
        kv = self.kv(freq_tokens) + self.pos                              # [B, n_tokens, d]
        ctx, _ = self.attn(q, kv, kv)                                     # [B, H*W, d]
        ctx = self.out(ctx).transpose(1, 2).reshape(B, Csp, H, W)
        return x + self.alpha * ctx                                       # gate-0 => identity
