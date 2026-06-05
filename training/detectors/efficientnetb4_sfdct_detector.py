'''
efficientnetb4_sfdct_detector.py
--------------------------------
Hybrid Spatial-Frequency (block-wise DCT) deepfake detector — the thesis method.
Built ON TOP of DeepfakeBench's EfficientDetector (EfficientNet-B4) by inheritance;
only features() (and get_optim_groups) is overridden, so loss / metrics / classifier / forward are reused.

Design (committee-reviewed 2026-06-05; see refine-logs/IMPROVEMENT_PLAN.md):
- Spatial stream  : EfficientNet-B4 (reused, pretrained) -> [B,1792,8,8] at 256px (stride 32 => 8x8, NOT 7x7).
- Frequency stream: ContentDCT on the INPUT image — denormalise to [0,1], RGB->YCbCr, 8x8 block DCT ->
  log-magnitude band statistics (content-spectrum band stats; 0 learnable params). On FF++/Celeb-DF this is
  a CONTENT-SPECTRUM signal, NOT a JPEG-grid-aligned one (the crop is H.264-sourced + resized).
- Fusion          : GatedCrossAttnFusion — per-channel gate `alpha` init 0 => identity at init => the model
  EQUALS plain B4 at initialisation (no-regression AT INIT only; the trained hybrid CAN still underperform
  the trained baseline). freq_repr global48|blockgrid, fusion_type crossattn|concat (for one-variable ablation).
- Gate warm-up    : get_optim_groups gives the fusion/gate a higher LR (gate_lr_mult) so the zero-init gate
  engages instead of staying inert (the diagnosed cause of the prior content-DCT TIE).
'''
import logging
import torch
import torch.nn as nn

from detectors import DETECTOR
from .efficientnetb4_detector import EfficientDetector
from .sfdct_core import ContentDCT, GatedCrossAttnFusion

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='efficientnetb4_sfdct')
class EfficientSFDCTDetector(EfficientDetector):
    def __init__(self, config):
        super().__init__(config)                                  # B4 backbone + loss (reused)
        c = config.get('dct_channels', 1792)                      # B4 final feature channels
        nbands = config.get('dct_nbands', 16)
        freq_repr = config.get('freq_repr', 'global48')           # 'global48' | 'blockgrid'
        grid = config.get('dct_grid', 8)                          # block-grid -> (grid,grid); 8 == real B4 grid @256
        fusion_type = config.get('fusion_type', 'crossattn')      # 'crossattn' | 'concat'  (ablation A3)
        shuffle_bands = config.get('shuffle_bands', False)        # negative control
        mean = config.get('mean', [0.5, 0.5, 0.5]); std = config.get('std', [0.5, 0.5, 0.5])
        self.gate_lr_mult = float(config.get('gate_lr_mult', 3.0))
        self.dct = ContentDCT(block=8, nbands=nbands, to_ycbcr=True, drop_dc=True,
                              freq_repr=freq_repr, grid=grid, channels=3,
                              input_mean=float(mean[0]), input_std=float(std[0]),
                              shuffle_bands=shuffle_bands, seed=int(config.get('manualSeed', 0)))
        n_query = grid * grid if freq_repr == 'blockgrid' else None   # spatial grounding when grids match
        self.fusion = GatedCrossAttnFusion(
            spatial_ch=c, token_in=self.dct.token_in, n_tokens=self.dct.n_tokens,
            d_model=config.get('fusion_dim', 128), heads=config.get('fusion_heads', 4),
            mode=fusion_type, n_query=n_query)
        logger.info(f'[SFDCT] ContentDCT(freq_repr={freq_repr}, nbands={nbands}, shuffle_bands={shuffle_bands}) + '
                    f'GatedCrossAttnFusion(mode={fusion_type}, token_in={self.dct.token_in}, '
                    f'n_tokens={self.dct.n_tokens}, n_query={n_query}); alpha init 0 => == B4 at init '
                    f'(no-regression AT INIT only). gate_lr_mult={self.gate_lr_mult}.')

    def features(self, data_dict: dict) -> torch.tensor:
        x = self.backbone.features(data_dict['image'])            # [B,1792,8,8] @256px
        _, band_tokens = self.dct(data_dict['image'])            # global48: [B,16,3]; blockgrid: [B,grid^2,C*nb]
        return self.fusion(x, band_tokens)                        # gate-0 => == x at init

    def get_optim_groups(self, base_lr):
        """Gate warm-up: the (zero-init) fusion + gate get a higher LR so they engage; backbone at base_lr.
        train.py uses this when present (else falls back to model.parameters())."""
        fusion_params = list(self.fusion.parameters())
        fusion_ids = {id(p) for p in fusion_params}
        backbone_params = [p for p in self.parameters() if id(p) not in fusion_ids and p.requires_grad]
        return [
            {'params': backbone_params, 'lr': base_lr},
            {'params': [p for p in fusion_params if p.requires_grad], 'lr': base_lr * self.gate_lr_mult},
        ]
