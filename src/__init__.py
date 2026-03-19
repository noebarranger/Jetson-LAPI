"""
LAPI Pipeline - Jetson LAPI Module

This module provides the core functionality for the LAPI (Latency-Aware Parallel Inference)
pipeline on NVIDIA Jetson devices.

Author: 
Date: 
Version: 1.0.0
"""

__title__ = "jetson-lapi"
__version__ = "1.0.0"
__author__ = ""
__license__ = "MIT"

from . import (
    io
)

__all__ = [
    "io",
]