"""
High-level API for SeedVR2 image upscaling.

This module provides a clean interface for external applications to use SeedVR2
without dealing with internal implementation details like config files or
the VideoDiffusionInfer class directly.
"""

import os
import logging
from typing import Optional, Literal, Callable, Dict, Any

import torch
from PIL import Image

# Type alias for progress callback
ProgressCallback = Callable[[int, int, str], None]  # (step, total_steps, message)

logger = logging.getLogger(__name__)


# =============================================================================
# Model Loading Functions
# =============================================================================

def disable_blockswap(dit: torch.nn.Module) -> None:
    """
    Fully disable and clean up BlockSwap on a model, allowing safe device movement.

    This removes BlockSwap's forward hooks and device references, making the model
    safe to move between devices with .to(). After calling this, the model behaves
    like it was loaded without BlockSwap.

    WARNING: BlockSwap wraps each block's forward() method with hooks that reference
    hardcoded device names. Simply calling .to() without cleanup would cause the
    hooks to malfunction. This function removes those hooks entirely.

    Args:
        dit: DiT model that was loaded with blocks_to_swap > 0

    Example:
        dit = load_dit("model.safetensors", blocks_to_swap=20, offload_device="cpu")

        # Before cleanup, .to() is blocked and hooks would cause issues
        disable_blockswap(dit)

        # Now safe to move
        dit.to("cpu")   # Works correctly
        dit.to("cuda")  # Works correctly

    Notes:
        - Safe to call on models without BlockSwap (no-op)
        - After disabling, model must be reloaded to re-enable BlockSwap
        - All blocks are moved to their current majority device before cleanup
    """
    # Get the actual model (handle CompatibleDiT wrapper)
    model = dit
    if hasattr(model, 'dit_model'):
        model = model.dit_model

    # Check if BlockSwap was configured
    if not hasattr(model, '_block_swap_config') and not hasattr(model, '_original_to'):
        return  # No BlockSwap, nothing to do

    # First, disable the .to() protection
    model._blockswap_bypass_protection = True

    # Restore original .to() method if it was wrapped
    if hasattr(model, '_original_to'):
        model.to = model._original_to
        delattr(model, '_original_to')

    # Restore original forward methods on blocks
    if hasattr(model, 'blocks'):
        for block in model.blocks:
            if hasattr(block, '_original_forward'):
                block.forward = block._original_forward
                delattr(block, '_original_forward')
            # Clean up block index marker
            if hasattr(block, '_block_idx'):
                delattr(block, '_block_idx')

    # Clean up I/O component hooks
    for attr in ['vid_in', 'vid_out', 'txt_in', 'time_embed', 'final_norm']:
        if hasattr(model, attr):
            module = getattr(model, attr)
            if module is not None and hasattr(module, '_original_forward'):
                module.forward = module._original_forward
                delattr(module, '_original_forward')

    # Remove BlockSwap state from model
    for attr in ['main_device', 'offload_device', 'blocks_to_swap',
                 '_block_swap_config', '_blockswap_bypass_protection']:
        if hasattr(model, attr):
            delattr(model, attr)


def is_blockswap_enabled(dit: torch.nn.Module) -> bool:
    """
    Check if a DiT model has BlockSwap enabled.

    Args:
        dit: DiT model to check

    Returns:
        True if BlockSwap is active on this model
    """
    model = dit
    if hasattr(model, 'dit_model'):
        model = model.dit_model

    return hasattr(model, '_block_swap_config') and model._block_swap_config is not None


def load_dit(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    attention_mode: Literal["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"] = "sdpa",
    blocks_to_swap: int = 0,
    offload_device: Optional[str] = None,
) -> torch.nn.Module:
    """
    Load a SeedVR2 DiT (Diffusion Transformer) model.

    Args:
        checkpoint_path: Path to model file (.safetensors, .gguf, or .pth)
        device: Target device for inference ("cuda", "cuda:0", "cpu", etc.)
        dtype: Model precision (torch.float16, torch.bfloat16, torch.float32)
        attention_mode: Attention implementation to use
        blocks_to_swap: Number of transformer blocks to swap to offload_device during inference
                        for memory optimization. 0 = disabled. Max ~32 for 3B, ~36 for 7B.
        offload_device: Device for block swapping (required if blocks_to_swap > 0).
                        Typically "cpu" or a secondary GPU like "cuda:1".

    Returns:
        Loaded DiT model ready for inference. The model is moved to `device`.

    Raises:
        FileNotFoundError: If checkpoint_path doesn't exist
        ValueError: If blocks_to_swap > 0 but offload_device is None

    Notes:
        - Model variant (3B vs 7B) is auto-detected from checkpoint
        - FP8 weights are automatically handled (converted at runtime)
        - GGUF quantized models are supported
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if blocks_to_swap > 0 and offload_device is None:
        raise ValueError("offload_device is required when blocks_to_swap > 0")

    # Auto-detect model variant from checkpoint
    model_info = get_model_info(checkpoint_path)
    model_variant = model_info.get("type", "dit_3b")

    # Import internal modules
    from src.utils.debug import Debug
    from src.optimization.compatibility import CompatibleDiT, validate_attention_mode, COMPUTE_DTYPE
    from src.core.model_loader import load_quantized_state_dict
    from src.utils.model_registry import MODEL_CLASSES

    # Create silent debug instance
    debug = Debug(enabled=False)

    # Determine compute dtype
    compute_dtype = dtype if dtype in (torch.float16, torch.bfloat16) else COMPUTE_DTYPE

    # Validate attention mode
    validated_attention_mode = validate_attention_mode(attention_mode, debug)

    # Get model class based on variant
    if "7b" in model_variant.lower():
        model_class = MODEL_CLASSES["dit_7b.nadit"]
        model_config = _get_dit_7b_config(validated_attention_mode)
    else:
        model_class = MODEL_CLASSES["dit_3b.nadit"]
        model_config = _get_dit_3b_config(validated_attention_mode)

    # Create model on meta device first for memory efficiency
    target_device = torch.device(device)

    with torch.device("meta"):
        model = model_class(**model_config)

    # Load weights
    state_dict = load_quantized_state_dict(checkpoint_path, target_device, debug)

    # Materialize model with weights
    model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    # Initialize any meta buffers
    _initialize_meta_buffers(model, target_device)

    # Wrap with CompatibleDiT for FP8/precision handling
    model = CompatibleDiT(model, debug, compute_dtype=compute_dtype)

    # Propagate attention_mode and compute_dtype to all FlashAttentionVarlen modules
    # This is critical for proper dtype handling during attention computation
    # (mirrors what apply_model_specific_config does in the production pipeline)
    actual_model = model.dit_model if hasattr(model, 'dit_model') else model
    for module in actual_model.modules():
        if type(module).__name__ == 'FlashAttentionVarlen':
            module.attention_mode = validated_attention_mode
            module.compute_dtype = compute_dtype

    # Apply BlockSwap if requested
    if blocks_to_swap > 0:
        from src.optimization.blockswap import apply_block_swap_to_dit

        # Create a minimal runner-like object for BlockSwap
        class _MinimalRunner:
            def __init__(self, dit_model, device, offload_dev, dbg):
                self.dit = dit_model
                self._dit_device = device
                self.debug = dbg
                self._blockswap_active = False

        runner = _MinimalRunner(model, target_device, torch.device(offload_device), debug)

        block_swap_config = {
            "blocks_to_swap": blocks_to_swap,
            "offload_device": torch.device(offload_device),
            "swap_io_components": False,
        }

        apply_block_swap_to_dit(runner, block_swap_config, debug)
        model = runner.dit

    # Set model to eval mode
    model.eval()

    # Store metadata on model for later access
    model._seedvr2_variant = "7b" if "7b" in model_variant.lower() else "3b"
    model._seedvr2_precision = model_info.get("precision", "unknown")
    model._seedvr2_attention_mode = validated_attention_mode

    return model


def load_vae(
    checkpoint_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    enable_tiling: bool = True,
    tile_sample_min_size: int = 256,
) -> torch.nn.Module:
    """
    Load the SeedVR2 VAE (Variational Autoencoder) model.

    Args:
        checkpoint_path: Path to VAE checkpoint (.safetensors or .pth)
        device: Target device ("cuda", "cuda:0", "cpu", etc.)
        dtype: Model precision
        enable_tiling: Enable tiled encoding/decoding for large images (recommended)
        tile_sample_min_size: Minimum tile size when tiling is enabled

    Returns:
        Loaded VAE model ready for encoding/decoding.

    Notes:
        - The VAE uses a scaling factor of 0.9152 (NOT 0.18215 like Stable Diffusion)
        - 16 latent channels
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")

    # Import internal modules
    from src.utils.debug import Debug
    from src.core.model_loader import load_quantized_state_dict
    from src.utils.model_registry import MODEL_CLASSES

    # Create silent debug instance
    debug = Debug(enabled=False)

    target_device = torch.device(device)

    # Get VAE config
    vae_config = _get_vae_config()

    # Get VAE class
    vae_class = MODEL_CLASSES["video_vae_v3.modules.attn_video_vae"]

    # Create model on meta device
    with torch.device("meta"):
        model = vae_class(**vae_config)

    # Load weights
    state_dict = load_quantized_state_dict(checkpoint_path, target_device, debug)
    model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict

    # Initialize meta buffers
    _initialize_meta_buffers(model, target_device)

    # Convert to target dtype
    model = model.to(dtype)

    # Configure tiling if enabled
    if enable_tiling:
        model.set_causal_slicing(
            split_size=tile_sample_min_size,
            memory_device="same"
        )

    # Set model to eval mode
    model.eval()

    # Set debug attribute (required by tiled_encode/tiled_decode methods)
    # Use None to disable debug logging
    model.debug = None

    # Store metadata
    model._seedvr2_scaling_factor = 0.9152
    model._seedvr2_latent_channels = 16
    model._seedvr2_tiling_enabled = enable_tiling

    return model


def get_model_info(checkpoint_path: str) -> dict:
    """
    Get information about a model checkpoint without loading it.

    Args:
        checkpoint_path: Path to model file

    Returns:
        Dictionary with:
        - "type": "dit_3b" | "dit_7b" | "vae"
        - "precision": "fp16" | "fp8" | "gguf_q4" | "gguf_q8" | etc.
        - "file_size_gb": float
        - "estimated_vram_gb": float (approximate VRAM needed for inference)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    filename = os.path.basename(checkpoint_path).lower()
    file_size_bytes = os.path.getsize(checkpoint_path)
    file_size_gb = file_size_bytes / (1024 ** 3)

    # Detect model type and precision from filename
    model_type = "dit_3b"  # default
    precision = "fp16"  # default
    estimated_vram_gb = 6.0  # default for 3B FP16

    # Detect VAE
    if "vae" in filename:
        model_type = "vae"
        precision = "fp16"
        estimated_vram_gb = 1.5
    # Detect 7B models
    elif "7b" in filename:
        model_type = "dit_7b"
        estimated_vram_gb = 14.0  # FP16 7B
    # Detect 3B models (default)
    else:
        model_type = "dit_3b"
        estimated_vram_gb = 6.0  # FP16 3B

    # Detect precision from filename or extension
    if filename.endswith(".gguf"):
        if "q4" in filename:
            precision = "gguf_q4"
            estimated_vram_gb *= 0.3  # Q4 is ~30% of FP16
        elif "q8" in filename:
            precision = "gguf_q8"
            estimated_vram_gb *= 0.55  # Q8 is ~55% of FP16
        else:
            precision = "gguf"
            estimated_vram_gb *= 0.4
    elif "fp8" in filename:
        precision = "fp8"
        estimated_vram_gb *= 0.55  # FP8 is ~55% of FP16
    elif "bf16" in filename or "bfloat16" in filename:
        precision = "bf16"
        # Same as FP16
    else:
        precision = "fp16"

    return {
        "type": model_type,
        "precision": precision,
        "file_size_gb": round(file_size_gb, 2),
        "estimated_vram_gb": round(estimated_vram_gb, 2),
    }


# =============================================================================
# High-Level Upscaler Class
# =============================================================================

class SeedVR2Upscaler:
    """
    High-level upscaler that orchestrates DiT and VAE for image upscaling.

    The caller is responsible for loading models via load_dit() and load_vae()
    and managing their lifecycle (loading, unloading, device placement).
    This class just performs the upscaling operation using provided models.

    Example:
        dit = load_dit("models/seedvr2_3b_fp8.safetensors", device="cuda")
        vae = load_vae("models/vae_fp16.safetensors", device="cuda")

        upscaler = SeedVR2Upscaler(dit, vae, device="cuda")
        result = upscaler.upscale(my_image, scale=2.0)

        # Later, caller can move models to CPU or delete them
        dit.to("cpu")  # Managed by caller
    """

    # VAE scaling factor (different from SD's 0.18215)
    VAE_SCALING_FACTOR = 0.9152

    def __init__(
        self,
        dit: torch.nn.Module,
        vae: torch.nn.Module,
        device: str = "cuda",
    ):
        """
        Initialize upscaler with pre-loaded models.

        Args:
            dit: Pre-loaded DiT model from load_dit()
            vae: Pre-loaded VAE model from load_vae()
            device: Device for inference operations
        """
        self.dit = dit
        self.vae = vae
        self.device = torch.device(device)

        # Import internal modules for inference
        from src.utils.debug import Debug
        from src.common.diffusion.schedules.lerp import LinearInterpolationSchedule
        from src.common.diffusion.samplers.euler import EulerSampler
        from src.common.diffusion.timesteps.sampling.trailing import UniformTrailingSamplingTimesteps

        # Create silent debug instance
        self._debug = Debug(enabled=False)

        # Set up diffusion components (matching config from configs_3b/main.yaml)
        # Schedule: lerp with T=1000
        self._schedule = LinearInterpolationSchedule(T=1000.0)

        # Timesteps: uniform_trailing with 1 step (distilled one-step model)
        # SeedVR2 is a distilled model that produces results in a single step
        self._timesteps = UniformTrailingSamplingTimesteps(
            T=self._schedule.T,
            steps=1,
            shift=1.0,
            device=self.device,
            dtype=torch.float32,
        )

        # Sampler: euler with v_lerp prediction type
        self._sampler = EulerSampler(
            schedule=self._schedule,
            timesteps=self._timesteps,
            prediction_type="v_lerp",
        )

    def upscale(
        self,
        image: Image.Image,
        scale: float = 2.0,
        color_correction: Literal["lab", "hsv", "wavelet", "none"] = "lab",
        seed: Optional[int] = None,
        progress_callback: Optional[ProgressCallback] = None,
        vae_tiling: bool = True,
        vae_tile_size: int = 512,
        vae_tile_overlap: int = 128,
        input_noise_scale: float = 0.0,
        latent_noise_scale: float = 0.0,
    ) -> Image.Image:
        """
        Upscale a single image.

        Args:
            image: Input PIL Image (RGB mode)
            scale: Upscale factor (e.g., 2.0 for 2x upscaling)
            color_correction: Method to preserve original colors
                - "lab": LAB color space transfer (recommended, best color fidelity)
                - "hsv": HSV color space transfer
                - "wavelet": Wavelet-based color transfer
                - "none": No color correction (may have color shift)
            seed: Random seed for reproducibility (None = random)
            progress_callback: Optional callback for progress updates
                Called as: callback(current_step, total_steps, message)
            vae_tiling: Enable tiled VAE encoding/decoding for large images (default True)
                Required for high resolution output to avoid OOM errors.
            vae_tile_size: Size of each VAE tile in pixels (default 512)
            vae_tile_overlap: Overlap between tiles in pixels (default 128)
                Increase if you see visible seams/grid artifacts in output.
            input_noise_scale: Noise injection before VAE encoding (0.0-1.0, default 0.0)
                Adds subtle noise to input image before encoding. Can help reduce
                artifacts at very high resolutions. Try 0.1-0.3 if you see artifacts.
            latent_noise_scale: Noise injection during diffusion conditioning (0.0-1.0, default 0.0)
                Controls how much the source latent is "degraded" before conditioning.
                Higher values = model works harder to reconstruct = more detail generation
                but potentially more hallucination. Also called "denoise_strength" in some guides.
                - 0.0: No noise (crisp, faithful to source)
                - 0.2-0.3: Light noise (subtle enhancement for clean sources)
                - 0.4: Standard (good balance for most content)
                - 0.5-0.6: Stronger (more detail generation, good for compressed sources)

        Returns:
            Upscaled PIL Image (RGB mode)

        Notes:
            - SeedVR2 uses single-step diffusion (very fast)
            - Input image dimensions should be divisible by 16 (will be padded if not)
            - Output dimensions = input dimensions * scale
            - VAE tiling is essential for outputs larger than ~1024px to avoid OOM
        """
        import torch.nn.functional as F
        from torchvision.transforms import ToTensor, ToPILImage

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to tensor: [1, 3, H, W]
        to_tensor = ToTensor()
        input_tensor = to_tensor(image).unsqueeze(0).to(self.device)

        # Store original size
        original_h, original_w = input_tensor.shape[2], input_tensor.shape[3]

        # Calculate target size
        target_h = int(original_h * scale)
        target_w = int(original_w * scale)

        # Pad to multiple of 16 if needed
        pad_h = (16 - (original_h % 16)) % 16
        pad_w = (16 - (original_w % 16)) % 16

        if pad_h > 0 or pad_w > 0:
            input_tensor = F.pad(input_tensor, (0, pad_w, 0, pad_h), mode='reflect')

        padded_h, padded_w = input_tensor.shape[2], input_tensor.shape[3]

        # Calculate padded target size
        padded_target_h = int(padded_h * scale)
        padded_target_w = int(padded_w * scale)

        if progress_callback:
            progress_callback(0, 4, "Encoding source image")

        # Normalize to [-1, 1] for VAE
        input_normalized = input_tensor * 2.0 - 1.0

        # Encode source image with VAE
        with torch.no_grad():
            # Get VAE dtype from model parameters
            vae_dtype = next(self.vae.parameters()).dtype

            # Upsample input to target resolution first
            upsampled_input = F.interpolate(
                input_normalized,
                size=(padded_target_h, padded_target_w),
                mode='bilinear',
                align_corners=False
            )

            # Convert to VAE dtype before encoding
            upsampled_input = upsampled_input.to(dtype=vae_dtype)

            # Apply input noise if requested (reduces artifacts at high resolutions)
            # Matching production pipeline: generation_phases.py lines 416-430
            if input_noise_scale > 0:
                # Generate noise matching the input shape
                input_noise = torch.randn_like(upsampled_input)
                # Subtle noise amplitude
                input_noise = input_noise * 0.05
                # Linear blend factor: 0 at scale=0, 0.5 at scale=1
                blend_factor = input_noise_scale * 0.5
                # Apply blend
                upsampled_input = upsampled_input * (1 - blend_factor) + (upsampled_input + input_noise) * blend_factor
                del input_noise

            # Encode upsampled image to get source latent
            # CausalEncoderOutput has .latent attribute (not .sample like diffusers)
            # Use tiled encoding for large images to avoid OOM
            source_latent = self.vae.encode(
                upsampled_input,
                tiled=vae_tiling,
                tile_size=(vae_tile_size, vae_tile_size),
                tile_overlap=(vae_tile_overlap, vae_tile_overlap),
            ).latent
            source_latent = source_latent * self.VAE_SCALING_FACTOR

        if progress_callback:
            progress_callback(1, 4, "Running diffusion")

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Import helpers for proper data format
        from src.optimization.performance import optimized_channels_to_last, optimized_channels_to_second
        from src.models.dit_3b import na
        from src.common.diffusion.samplers.base import SamplerModelArgs

        # Get compute dtype - use bfloat16 when supported for 7B model stability
        # The 7B model specifically requires bfloat16 for numerical stability in attention
        # FP8 and FP16 weights still compute in bfloat16 (autocast handles the conversion)
        from src.optimization.compatibility import COMPUTE_DTYPE
        compute_dtype = COMPUTE_DTYPE  # bfloat16 if supported, else float16

        # Convert source_latent to channels-last format [T, H, W, C] as expected by the model
        # source_latent is [1, 16, H, W] (B, C, H, W)
        # -> unsqueeze(2) -> [1, 16, 1, H, W] (B, C, T, H, W)
        # -> optimized_channels_to_last -> [1, 1, H, W, 16] (B, T, H, W, C)
        # -> squeeze(0) -> [1, H, W, 16] (T, H, W, C) where T=1 for single image
        source_latent_5d = source_latent.unsqueeze(2)  # [1, 16, 1, H, W]
        source_latent_cl = optimized_channels_to_last(source_latent_5d)  # [1, 1, H, W, 16]
        source_latent_cl = source_latent_cl.squeeze(0)  # [1, H, W, 16] - remove batch dim

        # Build condition: [T, H, W, C+1] where C=16 latent channels + 1 task indicator
        # For SR task, condition is: [source_latent(16ch), task_indicator(1ch all 1s)]
        t, h, w, c = source_latent_cl.shape

        # Create base noise for both initial noise and augmentation
        base_noise = torch.randn([t, h, w, c], device=self.device, dtype=compute_dtype)

        # Apply latent noise scale if requested (degrades source latent for conditioning)
        # Matching production pipeline: generation_phases.py lines 689-697
        # This controls how much "work" the model does to reconstruct the image
        conditioned_latent = source_latent_cl.to(compute_dtype)
        if latent_noise_scale > 0:
            # Create augmentation noise (slight variation from base noise)
            aug_noise = base_noise * 0.1 + torch.randn_like(base_noise) * 0.05

            # Calculate timestep for noise level (higher scale = more noise)
            t_noise = torch.tensor([1000.0], device=self.device, dtype=compute_dtype) * latent_noise_scale

            # Apply noise via diffusion schedule's forward process
            # This adds noise proportional to the timestep level
            conditioned_latent = self._schedule.forward(conditioned_latent, aug_noise, t_noise)
            del aug_noise

        condition = torch.zeros([t, h, w, c + 1], device=self.device, dtype=compute_dtype)
        condition[..., :-1] = conditioned_latent  # Source latent (possibly noised)
        condition[..., -1:] = 1.0  # Task indicator = 1 for SR

        # Use base noise as initial noise
        noise = base_noise

        # Flatten for batch processing (matching infer.py pattern)
        latents, latents_shapes = na.flatten([noise])
        latents_cond, _ = na.flatten([condition])

        # Load pre-trained text embeddings from package directory
        # These are required for proper model operation (even though SR doesn't use text prompts,
        # the model architecture expects specific text conditioning)
        batch_size = 1

        # Get package directory (where pos_emb.pt and neg_emb.pt are stored)
        package_dir = os.path.dirname(os.path.abspath(__file__))
        pos_emb_path = os.path.join(package_dir, 'pos_emb.pt')

        # Load and prepare text embeddings (matching production pipeline)
        text_pos_embed = torch.load(pos_emb_path, weights_only=True)
        text_pos_embed = text_pos_embed.to(device=self.device, dtype=compute_dtype)
        text_pos_embeds, text_pos_shapes = na.flatten([text_pos_embed])

        # CFG is disabled for distilled one-step models
        # SeedVR2 is a distilled model - CFG is incompatible with distillation
        # So we only need a single forward pass (no negative/unconditioned pass)

        # Always use autocast on CUDA for numerical stability
        # - FP8 models: autocast converts FP8 weights to compute_dtype during matmul
        # - FP16 models: autocast provides mixed precision which prevents overflow/underflow
        #   in attention computations (pure FP16 attention can produce NaN/Inf spots)
        device_type = self.device.type
        use_autocast = device_type == 'cuda'

        # Run diffusion sampling using the sampler
        def model_fn(args: SamplerModelArgs) -> torch.Tensor:
            """Callback for sampler - runs DiT model (single pass, no CFG for distilled model)."""
            x_t = args.x_t
            t = args.t

            # Concatenate latent with condition: [latent, condition] along channel dim
            vid_input = torch.cat([x_t, latents_cond], dim=-1)

            # Use autocast for numerical stability (required for both FP8 and FP16)
            # Mixed precision prevents overflow in attention softmax computations
            if use_autocast:
                with torch.autocast(device_type, dtype=compute_dtype):
                    output = self.dit(
                        vid=vid_input,
                        txt=text_pos_embeds,
                        vid_shape=latents_shapes,
                        txt_shape=text_pos_shapes,
                        timestep=t.repeat(batch_size),
                    ).vid_sample
            else:
                # Non-CUDA path (CPU, MPS) - no autocast
                output = self.dit(
                    vid=vid_input,
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=t.repeat(batch_size),
                ).vid_sample

            return output

        # Run the sampler (iterates through all timesteps)
        with torch.no_grad():
            denoised_latents = self._sampler.sample(x=latents, f=model_fn)

        # Unflatten back to list
        denoised_latents = na.unflatten(denoised_latents, latents_shapes)
        denoised_latent = denoised_latents[0]  # [T, H, W, C]

        # Convert back to channels-first format [B, C, H, W]
        denoised_latent = optimized_channels_to_second(denoised_latent.unsqueeze(0))  # [1, T, H, W, C] -> [1, C, T, H, W]
        denoised_latent = denoised_latent.squeeze(2)  # Remove T dimension -> [1, C, H, W]

        if progress_callback:
            progress_callback(2, 4, "Decoding output")

        # Decode with VAE
        # Use tiled decoding for large images to avoid OOM
        with torch.no_grad():
            output_latent = denoised_latent / self.VAE_SCALING_FACTOR
            # Convert to VAE dtype before decoding
            output_latent = output_latent.to(dtype=vae_dtype)
            output_tensor = self.vae.decode(
                output_latent,
                tiled=vae_tiling,
                tile_size=(vae_tile_size, vae_tile_size),
                tile_overlap=(vae_tile_overlap, vae_tile_overlap),
            ).sample

        # Convert from [-1, 1] to [0, 1]
        output_tensor = (output_tensor + 1.0) / 2.0
        output_tensor = output_tensor.clamp(0, 1)

        if progress_callback:
            progress_callback(3, 4, "Applying color correction")

        # Apply color correction
        if color_correction != "none":
            # Upsample original for color reference
            reference_tensor = F.interpolate(
                input_tensor,
                size=(padded_target_h, padded_target_w),
                mode='bilinear',
                align_corners=False
            )
            output_tensor = self._apply_color_correction(
                output_tensor,
                reference_tensor,
                color_correction
            )

        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            output_tensor = output_tensor[:, :, :target_h, :target_w]

        if progress_callback:
            progress_callback(4, 4, "Complete")

        # Convert back to PIL Image
        to_pil = ToPILImage()
        result = to_pil(output_tensor.squeeze(0).cpu())

        return result

    def _apply_color_correction(
        self,
        output: torch.Tensor,
        reference: torch.Tensor,
        method: str,
    ) -> torch.Tensor:
        """Apply color correction to match reference colors."""
        from src.utils.color_fix import (
            lab_color_transfer,
            wavelet_reconstruction,
            adain_color_fix,
        )
        from PIL import Image
        from torchvision.transforms import ToTensor, ToPILImage

        # Convert to [-1, 1] range for color correction functions
        output_norm = output * 2.0 - 1.0
        reference_norm = reference * 2.0 - 1.0

        if method == "lab":
            result = lab_color_transfer(output_norm, reference_norm, self._debug)
        elif method == "wavelet":
            result = wavelet_reconstruction(output_norm, reference_norm, self._debug)
        elif method == "hsv":
            # HSV uses PIL images
            to_pil = ToPILImage()
            to_tensor = ToTensor()

            output_pil = to_pil(output.squeeze(0).cpu())
            reference_pil = to_pil(reference.squeeze(0).cpu())

            # Use adain as fallback for HSV (similar effect)
            result_pil = adain_color_fix(output_pil, reference_pil)
            result = to_tensor(result_pil).unsqueeze(0).to(self.device)
            return result  # Already in [0, 1]
        else:
            return output

        # Convert back to [0, 1]
        result = (result + 1.0) / 2.0
        return result.clamp(0, 1)

    @property
    def supports_fp8(self) -> bool:
        """Whether the loaded DiT model uses FP8 precision."""
        precision = getattr(self.dit, '_seedvr2_precision', 'unknown')
        return 'fp8' in precision.lower()

    @property
    def model_variant(self) -> str:
        """Returns '3b' or '7b' based on loaded DiT model."""
        return getattr(self.dit, '_seedvr2_variant', '3b')


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _get_dit_3b_config(attention_mode: str) -> Dict[str, Any]:
    """Get configuration for 3B DiT model."""
    return {
        "vid_in_channels": 33,
        "vid_out_channels": 16,
        "vid_dim": 2560,
        "vid_out_norm": "fusedrms",
        "txt_in_dim": 5120,
        "txt_in_norm": "fusedln",
        "txt_dim": 2560,
        "emb_dim": 15360,  # 6 * vid_dim
        "heads": 20,
        "head_dim": 128,
        "expand_ratio": 4,
        "norm": "fusedrms",
        "norm_eps": 1e-5,
        "ada": "single",
        "qk_bias": False,
        "qk_norm": "fusedrms",
        "patch_size": [1, 2, 2],
        "num_layers": 32,
        "mm_layers": 10,
        "mlp_type": "swiglu",
        "msa_type": None,
        "block_type": ["mmdit_sr"] * 32,
        "window": [(4, 3, 3)] * 32,
        "window_method": ["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 16,
        "rope_type": "mmrope3d",
        "rope_dim": 128,
        "attention_mode": attention_mode,
    }


def _get_dit_7b_config(attention_mode: str) -> Dict[str, Any]:
    """Get configuration for 7B DiT model (matching configs_7b/main.yaml)."""
    return {
        "vid_in_channels": 33,
        "vid_out_channels": 16,
        "vid_dim": 3072,
        "txt_in_dim": 5120,
        "txt_dim": 3072,
        "emb_dim": 18432,  # 6 * vid_dim
        "heads": 24,
        "head_dim": 128,
        "expand_ratio": 4,
        "norm": "fusedrms",
        "norm_eps": 1e-5,
        "ada": "single",
        "qk_bias": False,
        "qk_norm": "fusedrms",
        "qk_rope": True,  # Required for 7B model
        "patch_size": [1, 2, 2],
        "num_layers": 36,
        "shared_mlp": False,  # Required for 7B
        "shared_qkv": False,  # Required for 7B
        "mlp_type": "normal",  # Required for 7B (3B uses "swiglu")
        "block_type": ["mmdit_sr"] * 36,
        "window": [(4, 3, 3)] * 36,
        "window_method": ["720pwin_by_size_bysize", "720pswin_by_size_bysize"] * 18,
        "attention_mode": attention_mode,
    }


def _get_vae_config() -> Dict[str, Any]:
    """Get configuration for VAE model."""
    return {
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": [
            "DownEncoderBlock3D",
            "DownEncoderBlock3D",
            "DownEncoderBlock3D",
            "DownEncoderBlock3D",
        ],
        "in_channels": 3,
        "latent_channels": 16,
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "out_channels": 3,
        "slicing_sample_min_size": 4,
        "temporal_scale_num": 2,
        "inflation_mode": "pad",
        "up_block_types": [
            "UpDecoderBlock3D",
            "UpDecoderBlock3D",
            "UpDecoderBlock3D",
            "UpDecoderBlock3D",
        ],
        "spatial_downsample_factor": 8,
        "temporal_downsample_factor": 4,
        "use_quant_conv": False,
        "use_post_quant_conv": False,
        "freeze_encoder": False,
    }


def _initialize_meta_buffers(model: torch.nn.Module, target_device: torch.device) -> None:
    """Initialize any buffers still on meta device after weight loading."""
    for name, buffer in model.named_buffers():
        if buffer is not None and buffer.device.type == 'meta':
            # Get the module that owns this buffer
            module_path = name.rsplit('.', 1)[0] if '.' in name else ''
            buffer_name = name.rsplit('.', 1)[1] if '.' in name else name

            # Get the actual module
            if module_path:
                module = model
                for part in module_path.split('.'):
                    module = getattr(module, part)
            else:
                module = model

            # Create a zero tensor of the same shape on target device
            initialized_buffer = torch.zeros_like(buffer, device=target_device)
            module.register_buffer(buffer_name, initialized_buffer, persistent=False)
