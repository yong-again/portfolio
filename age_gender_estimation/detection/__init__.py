"""
Detection package for human head detection using YOLO.

This package contains YOLO-based head detection training, validation, and prediction scripts.
"""

from .train_detector import train_head_detector
from .val_detector import validate_head_detector
from .predict_detector import predict_heads, HeadDetector

__all__ = [
    'train_head_detector',
    'validate_head_detector',
    'predict_heads',
    'HeadDetector'
]
