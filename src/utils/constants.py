"""
Shared constants and utilities for SeedVR2
"""

# Version information
__version__ = "2.5.24"

import warnings

# GGUF Quantization Constants
QK_K = 256
K_SCALE_SIZE = 12


def suppress_tensor_warnings() -> None:
    """
    Suppress common tensor conversion and numpy array warnings that are expected behavior
    when working with GGUF tensors and numpy arrays.
    """
    warnings.filterwarnings("ignore", message="To copy construct from a tensor", category=UserWarning)
    warnings.filterwarnings("ignore", message="The given NumPy array is not writable", category=UserWarning)