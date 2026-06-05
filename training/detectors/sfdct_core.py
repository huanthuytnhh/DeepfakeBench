"""
sfdct_core.py — pure-torch building blocks for the Hybrid Spatial–Frequency (block-DCT)
detector. NO DeepfakeBench imports here, so this file is unit/smoke-testable standalone
(see tools/smoke_sfdct.py). The drop-in detector imports ContentDCT + GatedCrossAttnFusion.

HONEST DESCRIPTION (committee-reviewed 2026-06-05):
- This computes block-wise 8x8 DCT log-magnitude band statistics of the INPUT face crop. On the FF++/
  Celeb-DF benchmark the crop is H.264-sourced and resized to 256, so the 8x8 grid is NOT aligned to any
  source JPEG grid — describe it as a CONTENT-SPECTRUM band-statistics branch, not a JPEG/quantization signal.
  (JPEG-grid alignment is only meaningful on the later eval-only VN-eKYC JPEG captures.)
- The backbone spatial grid is 8x8 (=64 cells) at 256px input (stride 32), NOT 7x7. `grid` defaults to 8 so
  blockgrid produces 64 freq tokens matching the 64 query cells; an optional query positional embedding makes
  the cross-attention spatially grounded.
- YCbCr is computed on PIXEL [0,1] values: ContentDCT denormalises the model input (mean/std) back to [0,1]
  BEFORE the RGB->YCbCr transform (otherwise it runs on [-1,1] and is not YCbCr).

freq_repr:
  "global48"  : per-block band stats GLOBALLY pooled -> nbands tokens of dim C (global spectral signature).
  "blockgrid" : per-block band map adaptive-pooled to grid x grid -> grid*grid tokens of dim C*nbands
                (spatially-resolved; pair with query pos-embed for grounded cross-attention).

GatedCrossAttnFusion: per-channel gate `alpha` init 0 => IDENTITY at init (out == spatial). This is a
NO-REGRESSION-AT-INIT property only: the model equals plain B4 at initialisation and cannot regress the
FROZEN backbone; the TRAINED hybrid can still underperform the trained baseline. Two fusion modes
(crossattn | concat) let the C-vs-D ablation vary one thing at a time. shuffle_bands is a negative control.
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
    """Block-wise 8x8 DCT content-spectrum front-end (0 learnable params). Returns (flat, kv_tokens).

    kv_tokens: global48 -> [B, nbands, C]; blockgrid -> [B, grid*grid, C*nbands].
    `self.token_in`, `self.n_tokens` expose dims for the fusion module.
    input_mean/input_std: denormalise the model input back to [0,1] before YCbCr (set to the data mean/std).
    shuffle_bands: NEGATIVE CONTROL — permute the band assignment (same params, destroyed low->high semantics).
    """
    def __init__(self, block: int = 8, nbands: int = 16, to_ycbcr: bool = True,
                 drop_dc: bool = True, freq_repr: str = "global48", grid: int = 8, channels: int = 3,
                 input_mean: float = 0.5, input_std: float = 0.5, shuffle_bands: bool = False, seed: int = 0):
        super().__init__()
        assert freq_repr in ("global48", "blockgrid")
        self.block, self.nbands, self.to_ycbcr, self.drop_dc = block, nbands, to_ycbcr, drop_dc
        self.freq_repr, self.grid = freq_repr, grid
        self.register_buffer("M", dct_matrix(block))
        band = zigzag_band_of(block, nbands).reshape(-1)                  # [64]
        if shuffle_bands:
            g = torch.Generator().manual_seed(seed)
            band = band[torch.randperm(band.numel(), generator=g)]       # destroys freq semantics, same #params
        self.register_buffer("band_of", band)
        self.register_buffer("rgb2ycbcr", _RGB2YCBCR.clone())
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))
        self.token_in = channels if freq_repr == "global48" else channels * nbands
        self.n_tokens = nbands if freq_repr == "global48" else grid * grid

    def _block_dct_logmag(self, x):
        """x[B,3,H,W] (model input) -> logmag[B,C,nblocks,64], plus (nh,nw)."""
        b = self.block
        x = (x * self.input_std + self.input_mean).clamp(0.0, 1.0)        # denorm to pixel [0,1] before YCbCr
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
        flat = perblk.mean(2).reshape(B, C * K)                             # [B, C*nbands] global (aux)
        if self.freq_repr == "global48":
            tokens = perblk.mean(2).transpose(1, 2).contiguous()           # [B, nbands, C]
        else:  # blockgrid: keep block layout, pool to (grid x grid)
            grid_map = perblk.reshape(B, C, nh, nw, K).permute(0, 1, 4, 2, 3).reshape(B, C * K, nh, nw)
            grid_map = nn.functional.adaptive_avg_pool2d(grid_map, (self.grid, self.grid))  # [B,C*K,g,g]
            tokens = grid_map.flatten(2).transpose(1, 2).contiguous()      # [B, g*g, C*K]
        return flat, tokens


class GatedCrossAttnFusion(nn.Module):
    """Inject the frequency branch into the spatial features via a ZERO-INIT gate (identity at init).

    mode='crossattn': spatial grid (Q, +optional pos) cross-attends to freq tokens (K/V).
    mode='concat'   : freq tokens -> MLP -> a global per-channel vector, broadcast over the grid.
    Both end with `x + alpha * ctx`, alpha init 0 => out == x at init (no-regression AT INIT only).
    n_query: if set (e.g. 64 for an 8x8 blockgrid), adds a learned query positional embedding so the
             cross-attention is spatially grounded to the backbone grid.
    """
    def __init__(self, spatial_ch: int = 1792, token_in: int = 3, n_tokens: int = 16,
                 d_model: int = 128, heads: int = 4, mode: str = "crossattn", n_query: int = None):
        super().__init__()
        assert mode in ("crossattn", "concat")
        self.mode = mode
        self.alpha = nn.Parameter(torch.zeros(spatial_ch, 1, 1))          # per-channel gate, init 0 (all modes)
        if mode == "crossattn":
            self.q = nn.Linear(spatial_ch, d_model)
            self.kv = nn.Linear(token_in, d_model)
            self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))    # K/V positional embed
            self.qpos = nn.Parameter(torch.zeros(1, n_query, d_model)) if n_query else None
            self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.out = nn.Linear(d_model, spatial_ch)
        else:  # concat
            self.mlp = nn.Sequential(nn.Linear(token_in * n_tokens, d_model), nn.ReLU(),
                                     nn.Linear(d_model, spatial_ch))

    def forward(self, x: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        B, Csp, H, W = x.shape
        if self.mode == "crossattn":
            q = self.q(x.flatten(2).transpose(1, 2))                      # [B, H*W, d]
            if self.qpos is not None and self.qpos.shape[1] == q.shape[1]:
                q = q + self.qpos                                         # spatial grounding when grids match
            kv = self.kv(freq_tokens) + self.pos                          # [B, n_tokens, d]
            ctx, _ = self.attn(q, kv, kv)                                 # [B, H*W, d]
            ctx = self.out(ctx).transpose(1, 2).reshape(B, Csp, H, W)
        else:  # concat: global freq vector broadcast over the grid
            v = self.mlp(freq_tokens.flatten(1))                          # [B, Csp]
            ctx = v[:, :, None, None].expand(B, Csp, H, W)
        return x + self.alpha * ctx                                       # gate-0 => identity at init
