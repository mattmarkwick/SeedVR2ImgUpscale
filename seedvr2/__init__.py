"""
SeedVR2 - Diffusion-based video/image upscaler

Standalone package implementing ByteDance's SeedVR2 diffusion-based upscaling.

Example usage:
    from seedvr2 import load_dit, load_vae, SeedVR2Upscaler

    # Load models
    dit = load_dit("models/seedvr2_3b_fp8.safetensors", device="cuda")
    vae = load_vae("models/vae_fp16.safetensors", device="cuda")

    # Create upscaler
    upscaler = SeedVR2Upscaler(dit, vae, device="cuda")

    # Upscale image
    result = upscaler.upscale(my_image, scale=2.0)
"""

from .api import (
    load_dit,
    load_vae,
    get_model_info,
    SeedVR2Upscaler,
    disable_blockswap,
    is_blockswap_enabled,
)

__version__ = "1.0.0"

__all__ = [
    "load_dit",
    "load_vae",
    "get_model_info",
    "SeedVR2Upscaler",
    "disable_blockswap",
    "is_blockswap_enabled",
]
