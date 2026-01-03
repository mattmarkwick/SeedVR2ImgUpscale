"""
Model Registry for SeedVR2
Central registry for model class definitions
"""

# Model class imports using relative imports
from ..models.dit_3b.nadit import NaDiT as NaDiT3B
from ..models.dit_7b.nadit import NaDiT as NaDiT7B
from ..models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper

# Model classes - simple registry with clear keys
MODEL_CLASSES = {
    "dit_3b.nadit": NaDiT3B,
    "dit_7b.nadit": NaDiT7B,
    "video_vae_v3.modules.attn_video_vae": VideoAutoencoderKLWrapper,
}
