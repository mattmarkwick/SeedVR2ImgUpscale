# SeedVR2 Image Upscaler

A pip-installable Python package implementing ByteDance's SeedVR2 diffusion-based image upscaler.

## About This Fork

This package is a streamlined fork of the [ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) project by **[NumZ](https://github.com/numz)** and **[AInVFX](https://www.youtube.com/@AInVFX)** (Adrien Toupet).

### What Changed

The original project is a comprehensive ComfyUI extension supporting:
- Video and image upscaling with temporal consistency
- ComfyUI V3 nodes with four-node modular architecture
- Multi-GPU CLI with streaming mode for long videos
- Batch processing of video directories

**This fork strips all of that down to a focused, pip-installable package for image-only upscaling:**
- Removed all ComfyUI integration (`src/interfaces/`)
- Removed video processing, CLI, and multi-GPU support
- Removed batch/temporal processing pipeline (`generation_phases.py`, `infer.py`)
- Added clean Python API (`seedvr2/api.py`)
- Converted to standard pip package structure

**Why?** To provide a simple, importable library for applications that just need high-quality image upscaling without the ComfyUI/video overhead.

### Original Project

For the full-featured ComfyUI extension with video support, CLI, and multi-GPU processing, see the original repository:

**[numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)**

## Installation

```bash
pip install git+https://github.com/mattmarkwick/SeedVR2ImgUpscale.git
```

Or for local development:

```bash
git clone https://github.com/mattmarkwick/SeedVR2ImgUpscale.git
cd SeedVR2ImgUpscale
pip install -e .
```

### Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA support
- ~6GB VRAM for 3B model (FP16), ~14GB for 7B model
- Lower VRAM possible with FP8, GGUF quantization, or BlockSwap

## Quick Start

```python
from seedvr2 import load_dit, load_vae, SeedVR2Upscaler
from PIL import Image

# Load models (downloads automatically on first use)
dit = load_dit("path/to/seedvr2_3b_fp8.safetensors", device="cuda")
vae = load_vae("path/to/vae_fp16.safetensors", device="cuda")

# Create upscaler
upscaler = SeedVR2Upscaler(dit, vae, device="cuda")

# Upscale image
image = Image.open("input.png")
result = upscaler.upscale(image, scale=2.0)
result.save("output.png")
```

## API Reference

### Model Loading

#### `load_dit(checkpoint_path, device="cuda", dtype=torch.float16, attention_mode="sdpa", blocks_to_swap=0, offload_device=None)`

Load a SeedVR2 DiT (Diffusion Transformer) model.

**Parameters:**
- `checkpoint_path`: Path to model file (`.safetensors`, `.gguf`, or `.pth`)
- `device`: Target device (`"cuda"`, `"cuda:0"`, `"cpu"`, etc.)
- `dtype`: Model precision (`torch.float16`, `torch.bfloat16`, `torch.float32`)
- `attention_mode`: Attention backend (`"sdpa"`, `"flash_attn_2"`, `"flash_attn_3"`, `"sageattn_2"`, `"sageattn_3"`)
- `blocks_to_swap`: Number of transformer blocks to swap to CPU for memory optimization (0 = disabled)
- `offload_device`: Device for block swapping (required if `blocks_to_swap > 0`)

**Returns:** Loaded DiT model

#### `load_vae(checkpoint_path, device="cuda", dtype=torch.float16, enable_tiling=True, tile_sample_min_size=256)`

Load the SeedVR2 VAE model.

**Parameters:**
- `checkpoint_path`: Path to VAE checkpoint
- `device`: Target device
- `dtype`: Model precision
- `enable_tiling`: Enable tiled encoding/decoding for large images
- `tile_sample_min_size`: Minimum tile size when tiling is enabled

**Returns:** Loaded VAE model

#### `get_model_info(checkpoint_path)`

Get information about a model checkpoint without loading it.

**Returns:** Dictionary with `type`, `precision`, `file_size_gb`, `estimated_vram_gb`

### Upscaling

#### `SeedVR2Upscaler(dit, vae, device="cuda")`

High-level upscaler class.

**Methods:**

##### `upscale(image, scale=2.0, color_correction="lab", seed=None, progress_callback=None, vae_tiling=True, vae_tile_size=512, vae_tile_overlap=128, input_noise_scale=0.0, latent_noise_scale=0.0)`

Upscale a single image.

**Parameters:**
- `image`: Input PIL Image (RGB mode)
- `scale`: Upscale factor (e.g., `2.0` for 2x)
- `color_correction`: Method to preserve original colors
  - `"lab"`: LAB color space transfer (recommended)
  - `"hsv"`: HSV color space transfer
  - `"wavelet"`: Wavelet-based color transfer
  - `"none"`: No color correction
- `seed`: Random seed for reproducibility
- `progress_callback`: Optional callback `(step, total_steps, message)`
- `vae_tiling`: Enable tiled VAE for large images (default `True`)
- `vae_tile_size`: Size of each VAE tile in pixels
- `vae_tile_overlap`: Overlap between tiles (increase if you see seams)
- `input_noise_scale`: Noise before encoding (0.0-1.0, helps with artifacts)
- `latent_noise_scale`: Noise during diffusion (0.0-1.0, controls detail generation)

**Returns:** Upscaled PIL Image

### BlockSwap Utilities

#### `disable_blockswap(dit)`

Fully disable and clean up BlockSwap on a model, allowing safe device movement with `.to()`.

#### `is_blockswap_enabled(dit)`

Check if a DiT model has BlockSwap enabled.

## Memory Optimization

### Low VRAM (8GB)

```python
# Use GGUF quantized model with BlockSwap
dit = load_dit(
    "seedvr2_3b-Q8_0.gguf",
    device="cuda",
    blocks_to_swap=32,
    offload_device="cpu"
)

# Enable VAE tiling for large outputs
result = upscaler.upscale(
    image,
    scale=2.0,
    vae_tiling=True,
    vae_tile_size=512
)
```

### Moderate VRAM (12-16GB)

```python
# FP8 model, optional BlockSwap
dit = load_dit(
    "seedvr2_3b_fp8.safetensors",
    device="cuda",
    blocks_to_swap=16,  # Optional
    offload_device="cpu"
)
```

### High VRAM (24GB+)

```python
# Full FP16 precision, no memory optimization needed
dit = load_dit("seedvr2_7b_fp16.safetensors", device="cuda")
```

## Model Files

Models are not included in the package. Download from:
- [numz/SeedVR2_comfyUI](https://huggingface.co/numz/SeedVR2_comfyUI/tree/main)
- [AInVFX/SeedVR2_comfyUI](https://huggingface.co/AInVFX/SeedVR2_comfyUI/tree/main)
- [cmeka/SeedVR2-GGUF](https://huggingface.co/cmeka/SeedVR2-GGUF/tree/main) (quantized)

**Available Models:**

| Model | Size | VRAM (approx) | Quality |
|-------|------|---------------|---------|
| `seedvr2_ema_3b_fp16.safetensors` | 6GB | ~6GB | Best |
| `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | 3GB | ~3.5GB | Good |
| `seedvr2_ema_3b-Q8_0.gguf` | 3GB | ~3.5GB | Good |
| `seedvr2_ema_3b-Q4_K_M.gguf` | 1.5GB | ~2GB | Acceptable |
| `seedvr2_ema_7b_fp16.safetensors` | 14GB | ~14GB | Best |
| `seedvr2_ema_7b_fp8_e4m3fn_mixed.safetensors` | 7GB | ~8GB | Good |
| `ema_vae_fp16.safetensors` | 330MB | ~1.5GB | Required |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| CUDA (NVIDIA) | Full | All features supported |
| MPS (Apple Silicon) | Partial | Works, no BlockSwap (unified memory) |
| ROCm (AMD) | Partial | Basic support, some features limited |
| CPU | Yes | Very slow, not recommended |

## Credits

- **Original SeedVR2**: [ByteDance Seed Team](https://github.com/ByteDance-Seed/SeedVR)
- **ComfyUI Implementation**: [NumZ](https://github.com/numz) & [AInVFX](https://www.youtube.com/@AInVFX) (Adrien Toupet)
- **This Package**: Streamlined API fork for pip installation

## License

Apache 2.0 - See [LICENSE](LICENSE) file.
