import os
import sys
import importlib
import warnings
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR
try:
    from .utils import slowfast            # only video detectors (i3d/ftcn/slowfast) need it; pulls simplejson/fvcore/...
except Exception as _e:
    warnings.warn(f"[detectors] slowfast utils not loaded ({type(_e).__name__}); video detectors off — fine for B4/SFDCT")

# Core detectors the thesis needs — import HARD (fail loudly if these break).
from .efficientnetb4_detector import EfficientDetector             # baseline B4 (efficientnetb4 / efficientnetb4_repro)
from .efficientnetb4_sfdct_detector import EfficientSFDCTDetector  # the method (efficientnetb4_sfdct)
from .efficientnetb4_hff_detector import EfficientHFFDetector      # block-DCT-HFF (efficientnetb4_hff): R1/R3
from .xception_detector import XceptionDetector                    # common anchor

# Every other DeepfakeBench detector registers via import side-effect. TOLERATE missing deps
# (clip / transformers / timm / dlib / ...): a detector that fails to import is just not registered and
# does NOT break training of efficientnetb4 / efficientnetb4_sfdct. Install its dep (see start.sh) to enable.
_OPTIONAL_DETECTORS = [
    ".facexray_detector", ".resnet34_detector", ".f3net_detector", ".meso4_detector",
    ".meso4Inception_detector", ".spsl_detector", ".core_detector", ".capsule_net_detector",
    ".srm_detector", ".ucf_detector", ".recce_detector", ".fwa_detector", ".ffd_detector",
    ".videomae_detector", ".clip_detector", ".timesformer_detector", ".xclip_detector",
    ".sbi_detector", ".ftcn_detector", ".i3d_detector", ".altfreezing_detector", ".stil_detector",
    ".lsda_detector", ".sladd_detector", ".pcl_xception_detector", ".iid_detector",
    ".lrl_detector", ".rfm_detector", ".uia_vit_detector", ".multi_attention_detector",
    ".sia_detector", ".tall_detector", ".effort_detector",
    # Experimental B4 variants kept LOCAL (uncomment when their files/deps are verified present):
    # ".efficientnetb4_dct_detector", ".efficientnetb4_hfdct_detector",
    # ".efficientnetb4_hfinput_detector", ".ecsf_detector",
]
_failed = []
for _m in _OPTIONAL_DETECTORS:
    try:
        importlib.import_module(_m, __name__)
    except Exception as _e:
        _failed.append(_m.lstrip("."))
if _failed:
    warnings.warn("[detectors] skipped (missing deps; OK unless you train them): " + ", ".join(_failed))
