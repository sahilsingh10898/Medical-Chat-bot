"""
Quantization module for AWQ model quantization.
"""

from .awq_quantization import AWQQuantizationConfig, AWQQuantizerModule

__all__ = ['AWQQuantizationConfig', 'AWQQuantizerModule']
