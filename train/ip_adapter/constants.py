"""
Shared constants for SigLIP SO400M preprocessing.

Centralised here to prevent silent drift between the precompute path
(precompute_all.py) and the live-encode path (train_ip_adapter.py).
"""

import numpy as np

SIGLIP_IMAGE_SIZE: int = 384

# SigLIP SO400M normalisation: (pixel/255 - MEAN) / STD == pixel/127.5 - 1.0
SIGLIP_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
