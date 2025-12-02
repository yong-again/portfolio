"""
Inference package for phone defect detection system.

This package contains inference pipelines for different sections.
"""

from .display import DisplayPipeline
from .side import SidePipeline

__all__ = ['DisplayPipeline', 'SidePipeline']

