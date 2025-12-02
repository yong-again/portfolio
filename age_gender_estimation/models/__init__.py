"""
Models package for age and gender estimation system.

This package contains the complete network architecture including
backbone, age head, gender head, and the combined network.
"""

from .backbone import FeatureExtractor, build_backbone, get_backbone_feature_dim
from .age_head import AgeHead, create_age_bins, age_to_bin, bin_to_age_range
from .gender_head import GenderHead, class_to_gender, gender_to_class
from .network import AgeGenderNetwork, build_network

__all__ = [
    # Backbone
    'FeatureExtractor',
    'build_backbone',
    'get_backbone_feature_dim',
    # Age Head
    'AgeHead',
    'create_age_bins',
    'age_to_bin',
    'bin_to_age_range',
    # Gender Head
    'GenderHead',
    'class_to_gender',
    'gender_to_class',
    # Network
    'AgeGenderNetwork',
    'build_network'
]

