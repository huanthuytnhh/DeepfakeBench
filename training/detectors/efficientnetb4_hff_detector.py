'''
efficientnetb4_hff_detector.py
------------------------------
Block-DCT-HFF detector (thesis SFDCT-HFF line). Built ON TOP of EfficientDetector (EfficientNet-B4) by
inheritance; only features() and get_optim_groups are overridden, so loss/metrics/classifier/forward are reused.

Design (docs/plans/2026-06-09-blockdct-hff.md; modules in sfdct_hff_core.py):
- Spatial : EfficientNet-B4 -> [B,1792,8,8] @256.
- Frequency RESIDUAL: BlockDCTHighPass on the INPUT (8x8 DCT -> zero DC+low bands -> iDCT) = high-pass
  residual image (0 params). residual_source='srm' switches to the SRM residual (control only).
- HF stream : learnable conv encoder (HFStream), optional multi_scale (HFF multi-scale).
- RSA       : residual-guided spatial attention on the HF feature (HFF RSA), optional.
- Fusion    : HFFGate per-channel zero-init  out = x + alpha*hf  => == B4 at init (floor >= B4 AT INIT only;
  the trained model can still differ). gate/stream/rsa get a warm-up LR via get_optim_groups.

Ablation (2 runs):  R1 = multi_scale:false, use_rsa:false ;  R3 (full) = multi_scale:true, use_rsa:true.
'''
import logging
import torch

from detectors import DETECTOR
from .efficientnetb4_detector import EfficientDetector
from .sfdct_hff_core import BlockDCTHighPass, HFStream, RSAttention, HFFGate
from .sfdct_core import SRMHighPass

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='efficientnetb4_hff')
class EfficientHFFDetector(EfficientDetector):
    def __init__(self, config):
        super().__init__(config)                                  # B4 backbone + loss (reused)
        c = int(config.get('dct_channels', 1792))
        mean = config.get('mean', [0.5, 0.5, 0.5]); std = config.get('std', [0.5, 0.5, 0.5])
        self.residual_source = config.get('residual_source', 'blockdct')   # 'blockdct' (ours) | 'srm' (control)
        self.multi_scale = bool(config.get('hff_multi_scale', True))
        self.use_rsa = bool(config.get('hff_use_rsa', True))
        self.lr_mult = float(config.get('hff_lr_mult', 3.0))
        drop_k = int(config.get('dct_drop_low_bands', 3))
        if self.residual_source == 'srm':
            self.residual = SRMHighPass(input_mean=float(mean[0]), input_std=float(std[0]))
        else:
            self.residual = BlockDCTHighPass(drop_k=drop_k, input_mean=float(mean[0]), input_std=float(std[0]))
        self.stream = HFStream(out_ch=c, grid=int(config.get('dct_grid', 8)), multi_scale=self.multi_scale)
        self.rsa = RSAttention() if self.use_rsa else None
        self.gate = HFFGate(c)
        logger.info(f'[HFF] residual_source={self.residual_source}, drop_low_bands={drop_k}, '
                    f'multi_scale={self.multi_scale}, use_rsa={self.use_rsa}, lr_mult={self.lr_mult}; '
                    f'HFFGate alpha init 0 => == B4 at init (no-regression AT INIT only).')

    def features(self, data_dict: dict) -> torch.tensor:
        img = data_dict['image']
        x = self.backbone.features(img)                           # [B,1792,8,8]
        res = self.residual(img)                                  # high-pass residual image
        h = self.stream(res)                                      # [B,1792,8,8]
        if self.rsa is not None:
            h = self.rsa(h)
        return self.gate(x, h)                                    # alpha=0 => == x at init

    def get_optim_groups(self, base_lr):
        """New modules (residual stream + rsa + gate) get a warm-up LR; backbone at base_lr."""
        new = list(self.stream.parameters()) + list(self.gate.parameters())
        if self.rsa is not None:
            new += list(self.rsa.parameters())
        new_ids = {id(p) for p in new}
        backbone = [p for p in self.parameters() if id(p) not in new_ids and p.requires_grad]
        return [
            {'params': backbone, 'lr': base_lr},
            {'params': [p for p in new if p.requires_grad], 'lr': base_lr * self.lr_mult},
        ]
