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


# 3 fixed SRM high-pass steganalysis filters (Fridrich & Kodovsky / RGB-N) — 0 learnable params.
_SRM_K = torch.tensor([
    [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
    [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
], dtype=torch.float32)
_SRM_NORM = torch.tensor([4.0, 12.0, 2.0]).view(3, 1, 1)


class SRMHighPass(nn.Module):
    """S2 (adapt SRM, arXiv:2103.12376, +0.65 CDFv2 in DeepfakeBench): 3 fixed high-pass steganalysis
    filters on luma -> 3-channel NOISE residual (0 learnable params). Feed as the DCT-branch input so the
    block-DCT describes the spectrum of the forgery noise residual (generalisable) instead of raw content."""
    def __init__(self, input_mean: float = 0.5, input_std: float = 0.5):
        super().__init__()
        self.register_buffer("k", (_SRM_K / _SRM_NORM).unsqueeze(1))           # [3,1,5,5]
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.register_buffer("input_mean", torch.tensor(float(input_mean)))
        self.register_buffer("input_std", torch.tensor(float(input_std)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x * self.input_std + self.input_mean).clamp(0.0, 1.0)             # denorm to [0,1]
        y = (x * self.luma.to(x.dtype)).sum(1, keepdim=True)                   # luma [B,1,H,W]
        return torch.nn.functional.conv2d(y, self.k.to(x.dtype), padding=2)    # [B,3,H,W] residuals


class ContentDCT(nn.Module):
    """Block-wise 8x8 DCT content-spectrum front-end (0 learnable params). Returns (flat, kv_tokens).

    kv_tokens: global48 -> [B, nbands, C]; blockgrid -> [B, grid*grid, C*nbands].
    `self.token_in`, `self.n_tokens` expose dims for the fusion module.
    input_mean/input_std: denormalise the model input back to [0,1] before YCbCr (set to the data mean/std).
    shuffle_bands: NEGATIVE CONTROL — permute the band assignment (same params, destroyed low->high semantics).
    """
    def __init__(self, block: int = 8, nbands: int = 16, to_ycbcr: bool = True,
                 drop_dc: bool = True, freq_repr: str = "global48", grid: int = 8, channels: int = 3,
                 input_mean: float = 0.5, input_std: float = 0.5, shuffle_bands: bool = False, seed: int = 0,
                 drop_low_bands: int = 0, use_sign: bool = False, srm_residual: bool = False):
        super().__init__()
        assert freq_repr in ("global48", "blockgrid")
        self.block, self.nbands, self.to_ycbcr, self.drop_dc = block, nbands, to_ycbcr, drop_dc
        # drop_low_bands>0: zero the DC + (k-1) lowest zigzag AC bands -> suppress content/identity leakage
        # (the #1 reason naive DCT-band decomposition fails cross-dataset; emphasises mid/high forgery cues).
        # Falls back to drop_dc (k=1 effective) when drop_low_bands==0 for backward compatibility.
        self.drop_low_bands = int(drop_low_bands)
        # S1 (adapt SPSL): append per-band mean SIGN of DCT coeffs (phase-analog) to the magnitude stats.
        # SPSL wins cross-dataset by KEEPING phase; magnitude-only ContentDCT discards exactly that cue.
        self.use_sign = bool(use_sign)
        feat_bands = nbands * (2 if self.use_sign else 1)
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
        # S2 (adapt SRM): when on, run block-DCT on the SRM noise residual instead of the YCbCr image.
        self.srm = SRMHighPass(input_mean, input_std) if srm_residual else None
        self.feat_bands = feat_bands
        self.token_in = channels if freq_repr == "global48" else channels * feat_bands
        self.n_tokens = feat_bands if freq_repr == "global48" else grid * grid

    def _block_dct_logmag(self, x):
        """x[B,3,H,W] (model input) -> logmag[B,C,nblocks,64], sign[B,C,nblocks,64], (nh,nw)."""
        b = self.block
        if self.srm is not None:
            x = self.srm(x)                                              # S2: block-DCT on SRM noise residual (no YCbCr)
        else:
            x = (x * self.input_std + self.input_mean).clamp(0.0, 1.0)    # denorm to pixel [0,1] before YCbCr
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
        sign = coef.sign().reshape(B, C, nh * nw, b * b)                   # S1: phase-analog (DCT coeff sign)
        return logmag, sign, nh, nw

    def _bands(self, logmag):
        B, C, N, _ = logmag.shape
        out = logmag.new_zeros(B, C, N, self.nbands)
        cnt = logmag.new_zeros(self.nbands)
        out.index_add_(3, self.band_of, logmag)
        cnt.index_add_(0, self.band_of, torch.ones_like(self.band_of, dtype=logmag.dtype))
        out = out / cnt.clamp_min(1.0)
        k = self.drop_low_bands if self.drop_low_bands > 0 else (1 if self.drop_dc else 0)
        if k > 0:
            out[..., :min(k, self.nbands - 1)] = 0.0    # keep >=1 band; suppress DC + lowest AC (content)
        return out                                                          # [B,C,nblocks,nbands]

    def forward(self, x: torch.Tensor):
        logmag, sign, nh, nw = self._block_dct_logmag(x)
        perblk = self._bands(logmag)                                        # [B,C,nblocks,nbands]
        if self.use_sign:
            perblk = torch.cat([perblk, self._bands(sign)], dim=3)          # S1: + per-band mean sign -> [..,2*nbands]
        B, C, N, K = perblk.shape                                           # K == self.feat_bands
        flat = perblk.mean(2).reshape(B, C * K)                             # [B, C*feat_bands] global (aux)
        if self.freq_repr == "global48":
            tokens = perblk.mean(2).transpose(1, 2).contiguous()           # [B, feat_bands, C]
        else:  # blockgrid: keep block layout, pool to (grid x grid)
            grid_map = perblk.reshape(B, C, nh, nw, K).permute(0, 1, 4, 2, 3).reshape(B, C * K, nh, nw)
            grid_map = nn.functional.adaptive_avg_pool2d(grid_map, (self.grid, self.grid))  # [B,C*K,g,g]
            tokens = grid_map.flatten(2).transpose(1, 2).contiguous()      # [B, g*g, C*feat_bands]
        return flat, tokens


class DCTFoMixup(nn.Module):
    """Frequency-domain Forgery-Mixup on BLOCK-wise 8x8 DCT — the cross-dataset LEARNING lever
    (adapts FreqDebias's FFT Fo-Mixup, arXiv:2509.22412, to block-DCT; 0 learnable params).

    Mechanism: per training step, take the signed 8x8 block-DCT of the input, pick a random subset of
    zigzag bands, MIX those coefficients with a shuffled partner sample (ratio xi), inverse-DCT back to an
    augmented IMAGE. This diversifies the band content the model sees -> breaks the "spectral bias" where a
    detector over-relies on FF++-specific bands and fails to generalise to Celeb-DF. Pair with a consistency
    loss between model(x) and model(x_aug). DC band (0) is excluded from mixing by default (it carries
    content/identity, not forgery cues). Identity when p_band==0 / mix_ratio==0 / batch<2.

    train-time only (wrap call in `if self.training`). Operates in the model-input space directly (per input
    channel), so the augmented image is a drop-in replacement fed to BOTH the spatial backbone and ContentDCT.
    """
    def __init__(self, block: int = 8, nbands: int = 16, p_band: float = 0.3, mix_ratio: float = 0.5):
        super().__init__()
        self.block, self.nbands = block, nbands
        self.p_band, self.mix_ratio = float(p_band), float(mix_ratio)
        self.register_buffer("M", dct_matrix(block))                       # orthonormal DCT-II: M M^T = I
        self.register_buffer("band_of", zigzag_band_of(block, nbands).reshape(-1))  # [64]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p_band <= 0 or self.mix_ratio <= 0 or x.shape[0] < 2:
            return x
        b = self.block; B, C, H, W = x.shape
        ph, pw = (b - H % b) % b, (b - W % b) % b
        xp = nn.functional.pad(x, (0, pw, 0, ph)) if (ph or pw) else x
        Bn, Cn, Hp, Wp = xp.shape; nh, nw = Hp // b, Wp // b
        blk = xp.unfold(2, b, b).unfold(3, b, b)                            # [B,C,nh,nw,8,8]
        coef = torch.einsum("pa,ncijab,qb->ncijpq", self.M, blk, self.M)    # signed block-DCT = M blk M^T
        perm = torch.randperm(B, device=x.device)
        partner = coef[perm]
        nsel = max(1, int(round(self.p_band * self.nbands)))
        cand = torch.arange(1, self.nbands, device=x.device)               # exclude DC band 0 (content)
        sel = cand[torch.randperm(cand.numel(), device=x.device)[:nsel]]
        band_mask = torch.zeros(self.nbands, dtype=torch.bool, device=x.device); band_mask[sel] = True
        pos = band_mask[self.band_of].view(1, 1, 1, 1, b, b).to(coef.dtype) * self.mix_ratio
        mixed = coef * (1.0 - pos) + partner * pos
        rec = torch.einsum("pa,ncijpq,qb->ncijab", self.M, mixed, self.M)   # inverse DCT = M^T coef M
        rec = rec.permute(0, 1, 2, 4, 3, 5).reshape(Bn, Cn, Hp, Wp)
        return rec[:, :, :H, :W]


class GatedCrossAttnFusion(nn.Module):
    """Inject the frequency branch into the spatial features via a ZERO-INIT gate (identity at init).

    mode='crossattn': spatial grid (Q, +optional pos) cross-attends to freq tokens (K/V).
    mode='concat'   : freq tokens -> MLP -> a global per-channel vector, broadcast over the grid.
    Both end with `x + alpha * ctx`, alpha init 0 => out == x at init (no-regression AT INIT only).
    n_query: if set (e.g. 64 for an 8x8 blockgrid), adds a learned query positional embedding so the
             cross-attention is spatially grounded to the backbone grid.
    """
    def __init__(self, spatial_ch: int = 1792, token_in: int = 3, n_tokens: int = 16,
                 d_model: int = 128, heads: int = 4, mode: str = "crossattn", n_query: int = None,
                 gate_mode: str = "zero"):
        super().__init__()
        assert mode in ("crossattn", "concat")
        assert gate_mode in ("zero", "sigmoid", "const")    # LOAD-BEARING ABLATION (vs SFCL sigmoid / FGINet 0.01)
        self.mode, self.gate_mode = mode, gate_mode
        if gate_mode == "zero":                                          # OURS: exact identity at init
            self.alpha = nn.Parameter(torch.zeros(spatial_ch, 1, 1))
        elif gate_mode == "sigmoid":                                     # SFCL-style: sigmoid gate -> 0.5 at init (NOT identity)
            self.gate_lin = nn.Parameter(torch.zeros(spatial_ch, 1, 1))
        else:                                                            # FGINet-style: fixed alpha=0.01 (approximate)
            self.register_buffer("alpha_const", torch.tensor(0.01))
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
        if self.gate_mode == "zero":      g = self.alpha                  # 0 at init => identity (OURS)
        elif self.gate_mode == "sigmoid": g = torch.sigmoid(self.gate_lin)  # 0.5 at init (SFCL-style)
        else:                             g = self.alpha_const            # 0.01 fixed (FGINet-style)
        return x + g * ctx                                                # gate_mode='zero' => identity at init


class SingleCenterLoss(nn.Module):
    """S5 (adapt FDFL, arXiv:2103.09096): pull the REAL class to a learnable center C and push FAKE samples
    at least a margin farther -> a tighter, more transferable real/fake boundary cross-dataset.
    label convention: 0=real, 1=fake. feat: [B, D] pooled embedding. Adds 1 learnable vector (D params)."""
    def __init__(self, feat_dim: int, margin: float = 0.3):
        super().__init__()
        self.C = nn.Parameter(torch.randn(feat_dim) * 0.01)
        self.margin = float(margin)

    def forward(self, feat: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        d = torch.norm(feat - self.C, dim=1)                              # [B] L2 distance to center
        real, fake = (label == 0), (label == 1)
        m_nat = d[real].mean() if real.any() else feat.new_tensor(0.0)    # mean dist of real to C
        m_man = d[fake].mean() if fake.any() else feat.new_tensor(0.0)    # mean dist of fake to C
        margin = self.margin * (feat.shape[1] ** 0.5)                     # scale margin by sqrt(D) (FDFL)
        return m_nat + torch.relu(m_nat - m_man + margin)                 # compact real + separate fake by margin
