import os
import sys
import importlib
import warnings
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

# Core dataset — always needed, no heavy deps (used by efficientnetb4 / efficientnetb4_sfdct / most detectors).
from .abstract_dataset import DeepfakeAbstractBaseDataset

# Optional datasets. ff_blend / fwa_blend call dlib.shape_predictor() AT IMPORT TIME and need
# preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat (gitignored, ~19MB). On a fresh clone that
# file/dlib may be absent — import lazily and SKIP (with a warning) instead of crashing the whole package
# (which would break train.py for EVERY detector). They only back the blend detectors (FWA/FaceXray/SBI/LSDA/
# SLADD), which the block-DCT thesis runs do not use. To enable later: download the .dat + `pip install dlib`.
_OPTIONAL = {
    "I2GDataset": ".I2G_dataset",
    "IIDDataset": ".iid_dataset",
    "FFBlendDataset": ".ff_blend",
    "FWABlendDataset": ".fwa_blend",
    "LRLDataset": ".lrl_dataset",
    "pairDataset": ".pair_dataset",
    "SBIDataset": ".sbi_dataset",
    "LSDADataset": ".lsda_dataset",
    "TALLDataset": ".tall_dataset",
}
for _sym, _modname in _OPTIONAL.items():
    try:
        _m = importlib.import_module(_modname, __name__)
        globals()[_sym] = getattr(_m, _sym)
    except Exception as _e:
        warnings.warn(f"[dataset] optional '{_sym}' ({_modname}) not loaded "
                      f"({type(_e).__name__}). OK unless you train a detector that needs it.")
