'''
efficientnetb4_sfdct_detector.py
--------------------------------
Hybrid Spatial–Frequency (block-wise DCT) deepfake detector — the thesis method.
Built ON TOP of DeepfakeBench's EfficientDetector (EfficientNet-B4) by inheritance;
only features() is overridden, so loss / metrics / classifier / forward are reused.

Design (see refine-logs/EXPERIMENT_PLAN.md § FINAL STRATEGY):
- Spatial stream  : EfficientNet-B4 (reused, pretrained) -> [B,1792,7,7].
- Frequency stream: ContentDCT on the INPUT image — YCbCr 8x8 block DCT -> log-magnitude
  band statistics (48-d = 3 channels x 16 zigzag bands), exposed as [B,16,3] band tokens.
  (PRIMARY, easy, 0 learnable params. QA-DCT quantization-forensic features = later stretch.)
- Fusion          : GatedCrossAttnFusion — spatial 7x7 grid (Q) cross-attends to the 16 freq
  band tokens (K/V); per-channel gate `alpha` init 0 => identity at start => can NEVER regress
  the pretrained B4 baseline (the no-regression guarantee; verified in tools/smoke_sfdct.py).

SETUP:
  1. Place sfdct_core.py next to this file (already in training/detectors/).
  2. Register: in training/detectors/__init__.py add
       from .efficientnetb4_sfdct_detector import EfficientSFDCTDetector
  3. Train with training/config/detector/efficientnetb4_sfdct.yaml
NOTE: ContentDCT runs on data_dict['image'] (the model input). drop_dc=True removes the
DC/mean term so input normalisation does not dominate the band statistics.
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
        grid = config.get('dct_grid', 7)                          # block-grid pooled to (grid,grid) == B4 7x7
        self.dct = ContentDCT(block=8, nbands=nbands, to_ycbcr=True, drop_dc=True,
                              freq_repr=freq_repr, grid=grid, channels=3)
        self.fusion = GatedCrossAttnFusion(
            spatial_ch=c, token_in=self.dct.token_in, n_tokens=self.dct.n_tokens,
            d_model=config.get('fusion_dim', 128), heads=config.get('fusion_heads', 4))
        logger.info(f'[SFDCT] ContentDCT(freq_repr={freq_repr}, nbands={nbands}) + '
                    f'GatedCrossAttnFusion(token_in={self.dct.token_in}, n_tokens={self.dct.n_tokens}, '
                    f'channels={c}); alpha init 0 => starts == EfficientNet-B4 baseline.')

    def features(self, data_dict: dict) -> torch.tensor:
        x = self.backbone.features(data_dict['image'])            # [B,1792,7,7]
        _, band_tokens = self.dct(data_dict['image'])             # [B,16,3]
        return self.fusion(x, band_tokens)                        # gate-0 => == x at init
