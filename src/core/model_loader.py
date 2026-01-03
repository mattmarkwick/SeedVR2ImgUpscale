"""
Model Weight Loading for SeedVR2

This module handles weight loading operations for DiT and VAE models:
- Loading state dictionaries from multiple formats (SafeTensors, PyTorch, GGUF)
- GGUF quantized model support with automatic dequantization
- Meta buffer initialization for non-persistent buffers

Key Features:
- Multi-format support: .safetensors, .pth, .gguf files
- Native FP8 weight handling with optimal performance
- GGUF quantization support (Q4_K_M, Q8_0, etc.)

Main Functions:
- load_quantized_state_dict: Load state dict from checkpoint file
- GGUFTensor: Tensor wrapper for quantized weights with automatic dequantization
- initialize_meta_buffers: Initialize non-persistent buffers on target device
"""

import os
import torch
from typing import Dict, Optional

# Import SafeTensors with fallback
try:
    from safetensors.torch import load_file as load_safetensors_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from ..optimization.compatibility import (
    GGUF_AVAILABLE,
    GGMLQuantizationType,
    validate_gguf_availability
)

# GGUF-specific imports (only when available)
if GGUF_AVAILABLE:
    import gguf
    import traceback
    from ..optimization.gguf_dequant import dequantize_tensor

from ..utils.constants import suppress_tensor_warnings


def load_quantized_state_dict(checkpoint_path: str, device: torch.device = torch.device("cpu"),
                              debug: Optional['Debug'] = None) -> Dict[str, torch.Tensor]:
    """
    Load model state dict from checkpoint with support for multiple formats.
    
    Handles .safetensors, .gguf, and .pth files. GGUF models support quantization
    for memory-efficient loading. Validates required libraries are installed.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Target device for tensor placement (torch.device object, defaults to CPU)
        debug: Optional Debug instance for logging
        
    Returns:
        dict: State dictionary loaded with appropriate format handler
        
    Notes:
        - SafeTensors files use optimized loading with direct device placement
        - PyTorch files use memory-mapped loading to reduce RAM usage
    """
    device_str = str(device)
    
    if checkpoint_path.endswith('.safetensors'):
        if not SAFETENSORS_AVAILABLE:
            error_msg = (
                f"Cannot load {os.path.basename(checkpoint_path)}\n"
                f"SafeTensors library is required but not installed.\n"
                f"Please install it with: pip install safetensors"
            )
            if debug:
                debug.log(error_msg, level="ERROR", category="dit", force=True)
                debug.log("This is a one-time installation that will enable loading of .safetensors files", 
                         level="INFO", category="info", force=True)
            raise ImportError(error_msg)
        
        # Try direct device loading first (optimal path)
        try:
            state = load_safetensors_file(checkpoint_path, device=device_str)
        except RuntimeError as e:
            # MPS allocator fallback: some PyTorch/macOS versions have issues with
            # direct MPS loading (allocation failures, watermark errors, etc.)
            error_msg = str(e).lower()
            is_mps_alloc_error = device.type == "mps" and any(
                keyword in error_msg for keyword in ["watermark", "allocat", "memory"]
            )
            
            if is_mps_alloc_error:
                # Transparent fallback - only log if debug enabled
                if debug:
                    debug.log("Using CPU intermediate loading for MPS compatibility", 
                            category="info", indent_level=1)
                state = load_safetensors_file(checkpoint_path, device="cpu")
                # Tensors will be moved to MPS during model.load_state_dict()
            else:
                # Re-raise if it's a different error (file corruption, etc.)
                raise
    elif checkpoint_path.endswith('.gguf'):
        validate_gguf_availability(f"load {os.path.basename(checkpoint_path)}", debug)
        state = _load_gguf_state(
                    checkpoint_path=checkpoint_path, 
                    device=device, 
                    debug=debug, 
                    handle_prefix="model.diffusion_model."
                )
    elif checkpoint_path.endswith('.pth'):
        state = torch.load(checkpoint_path, map_location=device_str, mmap=True, weights_only=True)
    else:
        raise ValueError(f"Unsupported checkpoint format. Expected .safetensors or .pth, got: {checkpoint_path}")
    
    return state


def _load_gguf_state(checkpoint_path: str, device: torch.device, debug: Optional['Debug'] = None,
                    handle_prefix: str = "model.diffusion_model.") -> Dict[str, torch.Tensor]:
    """
    Load GGUF state dict
    
    Args:
        checkpoint_path: Path to GGUF file
        device: Target device (torch.device object)
        debug: Debug instance
        handle_prefix: Prefix to strip from tensor names
        
    Returns:
        State dictionary with loaded tensors
    """
    reader = gguf.GGUFReader(checkpoint_path)

    # Filter and strip prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)
        
    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    state_dict = {}
    total_tensors = len(reader.tensors)
    
    device_str = str(device)
    debug.log(f"Loading {total_tensors} tensors to {str(device_str)}...", category="dit")
    
    # Suppress expected warnings: GGUF tensors are read-only numpy arrays that trigger warnings when converted
    suppress_tensor_warnings()
    
    for i, (sd_key, tensor) in enumerate(tensors):
        tensor_name = tensor.name
        
        # Create tensor directly on target device to avoid CPU->GPU copy overhead
        # For meta-initialized models, this directly materializes to the target device
        torch_tensor = torch.from_numpy(tensor.data).to(device, non_blocking=False)
            
        # Get original shape from metadata or infer from tensor shape
        shape = _get_tensor_logical_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
            
        # Handle tensors based on quantization type
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # For unquantized tensors, just reshape
            torch_tensor = torch_tensor.view(*shape)
        else:
            # For quantized tensors, keep them quantized but track original shape
            torch_tensor = GGUFTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape, debug=debug)
            
        state_dict[sd_key] = torch_tensor
        
        # Progress reporting
        if (i + 1) % 100 == 0:
            debug.log(f"Loaded {i+1}/{total_tensors} tensors...", category="dit", indent_level=1)

    debug.log(f"Successfully loaded {len(state_dict)} tensors to {device_str}", category="success")

    return state_dict


def _get_tensor_logical_shape(reader: 'gguf.GGUFReader', tensor_name: str) -> Optional[torch.Size]:
    """
    Extract the logical (unquantized) shape from GGUF metadata
    """
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    # Has original shape metadata, so we try to decode it.
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


class GGUFTensor(torch.Tensor):
    """
    Tensor wrapper for GGUF quantized tensors that preserves quantization info
    """
    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape
        
    def __new__(cls, *args, tensor_type, tensor_shape, debug, **kwargs):
        # Create tensor with requires_grad=False to avoid gradient issues
        tensor = super().__new__(cls, *args, **kwargs)
        tensor.requires_grad_(False)
        tensor.tensor_type = tensor_type
        tensor.tensor_shape = tensor_shape
        tensor.debug = debug
        return tensor
    
    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", self.tensor_shape if hasattr(self, "tensor_shape") else new.shape)
        new.debug = getattr(self, "debug", None)
        new.requires_grad_(False)  # Ensure no gradients
        return new
    
    @property
    def shape(self):
        # Always return the logical tensor shape, not the quantized data shape
        if hasattr(self, "tensor_shape"):
            return self.tensor_shape
        else:
            # Fallback to actual data shape if tensor_shape is not available
            return self.size()
        
    def size(self, *args):
        # Override size() to also return logical shape
        if hasattr(self, "tensor_shape") and len(args) == 0:
            return self.tensor_shape
        elif hasattr(self, "tensor_shape") and len(args) == 1:
            return self.tensor_shape[args[0]]
        else:
            return super().size(*args)
        
    def dequantize(self, device=None, dtype=torch.float16, dequant_dtype=None):
        """Dequantize this tensor to its original shape"""
        if device is None:
            device = self.device
            
        # Suppress expected warning when converting from GGUFTensor subclass to regular tensor
        suppress_tensor_warnings()

        # Check if already unquantized
        if self.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            # Return regular tensor, not GGUFTensor
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                # Convert to regular tensor to avoid __torch_function__ calls
                result = torch.tensor(result, dtype=dtype, device=device, requires_grad=False)
            return result
        
        # Try fast dequantization with crash protection
        try:
            result = dequantize_tensor(self, dtype, dequant_dtype)
            final_result = result.to(device)
            
            # Ensure we return a regular tensor, not GGUFTensor
            if isinstance(final_result, GGUFTensor):
                final_result = torch.tensor(final_result.data, dtype=dtype, device=device, requires_grad=False)
                
            return final_result
        except Exception as e:
            self.debug.log(f"Fast dequantization failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Falling back to numpy dequantization", level="WARNING", category="dit", force=True)
            
        # Fallback to numpy (slower but reliable)
        try:
            numpy_data = self.cpu().numpy()
            dequantized = gguf.quants.dequantize(numpy_data, self.tensor_type)
            result = torch.from_numpy(dequantized).to(device, dtype)
            result.requires_grad_(False)
            final_result = result.reshape(self.tensor_shape)
            # from_numpy already returns a regular tensor, no conversion needed
            return final_result
        except Exception as e:
            self.debug.log(f"Numpy fallback also failed: {e}", level="WARNING", category="dit", force=True)
            self.debug.log(f"Tensor type: {self.tensor_type}", level="WARNING", category="dit", force=True, indent_level=1)
            self.debug.log(f"Shape: {self.shape}", level="WARNING", category="dit", force=True, indent_level=1)
            self.debug.log(f"Target shape: {self.tensor_shape}", level="WARNING", category="dit", force=True, indent_level=1)
            traceback.print_exc()
            
            # Return regular tensor as last resort
            result = self.to(device, dtype)
            if isinstance(result, GGUFTensor):
                result = torch.tensor(result.data, dtype=dtype, device=device, requires_grad=False)
            return result
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Override torch function calls to automatically dequantize"""
        if kwargs is None:
            kwargs = {}
        
        # Find the GGUFTensor instance(s) in args
        gguf_tensors = [arg for arg in args if isinstance(arg, cls)]
        if not gguf_tensors:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Use the first GGUFTensor instance for attribute access
        self = gguf_tensors[0]
        
        # Check if the tensor is fully constructed and still quantized
        tensor_type = getattr(self, 'tensor_type', None)
        if tensor_type is None:
            # Tensor is either being constructed or already dequantized
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if tensor is already unquantized (F32/F16)
        if tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            return super().__torch_function__(func, types, args, kwargs)
        
        # Check if debug exists before using it
        debug = getattr(self, 'debug', None)
        
        # Handle linear operations specially
        if func == torch.nn.functional.linear:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is the second argument
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in linear dequantization: {e}", level="WARNING", category="dit", force=True)
                        debug.log(f"Function: {func}", level="WARNING", category="dit", force=True, indent_level=1)
                        debug.log(f"Args: {[arg.shape if hasattr(arg, 'shape') else type(arg) for arg in args]}", level="WARNING", category="dit", force=True, indent_level=1)
                    raise
        
        # Handle matrix multiplication operations that need dequantization
        if func in {torch.matmul, torch.mm, torch.bmm, torch.addmm, torch.addmv,
                    torch.addr, torch.baddbmm, torch.chain_matmul}:
            try:
                new_args = []
                for arg in args:
                    if isinstance(arg, cls):
                        new_args.append(arg.dequantize())
                    else:
                        new_args.append(arg)
                return func(*tuple(new_args), **kwargs)
            except Exception as e:
                if debug:
                    debug.log(f"Error in {func.__name__} dequantization: {e}", level="WARNING", category="dit", force=True)
                raise

        # Handle conv2d/conv3d operations (critical for GGUF VAE models)
        # Conv3d layers (InflatedCausalConv3d) are not replaced by layer replacement
        if func in {torch.nn.functional.conv2d, torch.nn.functional.conv3d}:
            if len(args) >= 2 and isinstance(args[1], cls):  # weight is second arg
                try:
                    weight_tensor = args[1]
                    dequantized_weight = weight_tensor.dequantize(device=args[0].device, dtype=args[0].dtype)
                    new_args = (args[0], dequantized_weight) + args[2:]
                    return func(*new_args, **kwargs)
                except Exception as e:
                    if debug:
                        debug.log(f"Error in conv dequantization: {e}", level="WARNING", category="dit", force=True)
                    raise
        
        # For ALL other operations, delegate to parent WITHOUT dequantization
        # This includes .cpu(), .to(), .device, .dtype, .shape, etc.
        return super().__torch_function__(func, types, args, kwargs)