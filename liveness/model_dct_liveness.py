"""model_dct_liveness.py — B4 + block-DCT (naive SFDCT) cho LIVENESS, TƯƠNG ĐƯƠNG với
naive SFDCT detector của DeepfakeBench (efficientnetb4_sfdct, mọi lever S1-S5 OFF).

Nhân bản đúng cách ráp của EfficientSFDCTDetector ở cấu hình naive:
  backbone (B4Liveness) -> features [B,1792,8,8]
  + ContentDCT(global48, nbands=16, levers OFF) -> band tokens [B,16,3]
  + GatedCrossAttnFusion(zero-init gate) -> fused [B,1792,8,8]  (gate=0 => == B4 tại init)
  classifier = backbone.classifier (GAP -> Linear(1792,2)).
Dùng dct_core_liveness.py (bản COPY của sfdct_core.py) -> KHÔNG import training/detectors. Standalone.
"""
import torch.nn as nn

from model_liveness import B4Liveness
from dct_core_liveness import ContentDCT, GatedCrossAttnFusion


class B4DCTLiveness(nn.Module):
    """B4 + block-DCT (naive). Mirror byte-for-byte EfficientSFDCTDetector ở cấu hình naive."""

    def __init__(self, num_classes=2, nbands=16, freq_repr="global48", grid=8,
                 fusion_dim=128, fusion_heads=4, gate_lr_mult=3.0,
                 use_pretrained=True, pretrained_path=None):
        super().__init__()
        # self.backbone giống detector (EfficientDetector) -> keys 'backbone.efficientnet.*'/'backbone.last_layer.*'
        self.backbone = B4Liveness(num_classes=num_classes, inc=3, dropout=False,
                                   use_pretrained=use_pretrained, pretrained_path=pretrained_path)
        self.dct = ContentDCT(block=8, nbands=nbands, to_ycbcr=True, drop_dc=True,
                              freq_repr=freq_repr, grid=grid, channels=3,
                              input_mean=0.5, input_std=0.5, shuffle_bands=False, seed=0,
                              drop_low_bands=0, use_sign=False, srm_residual=False)   # NAIVE: levers OFF
        n_query = grid * grid if freq_repr == "blockgrid" else None
        self.fusion = GatedCrossAttnFusion(
            spatial_ch=1792, token_in=self.dct.token_in, n_tokens=self.dct.n_tokens,
            d_model=fusion_dim, heads=fusion_heads, mode="crossattn",
            n_query=n_query, gate_mode="zero")
        self.gate_lr_mult = float(gate_lr_mult)

    def features(self, x):
        sp = self.backbone.features(x)                  # [B,1792,8,8]
        _, band_tokens = self.dct(x)                    # global48: [B,16,3]
        return self.fusion(sp, band_tokens)             # gate=0 => == sp tại init

    def classifier(self, feat):
        return self.backbone.classifier(feat)           # GAP -> Linear(1792,2)

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_optim_groups(self, base_lr):
        """Gate warm-up giống detector: fusion/gate được lr cao hơn (gate_lr_mult) để cổng zero-init engage."""
        fp = list(self.fusion.parameters()); fids = {id(p) for p in fp}
        bp = [p for p in self.parameters() if id(p) not in fids and p.requires_grad]
        return [
            {"params": bp, "lr": base_lr},
            {"params": [p for p in fp if p.requires_grad], "lr": base_lr * self.gate_lr_mult},
        ]


def build_b4dct_liveness(num_classes=2, pretrained=True, pretrained_path=None):
    """API: B4 + block-DCT (naive), tương đương naive SFDCT detector deepfake. forward(x)->logits[B,2]."""
    return B4DCTLiveness(num_classes=num_classes, use_pretrained=pretrained, pretrained_path=pretrained_path)
