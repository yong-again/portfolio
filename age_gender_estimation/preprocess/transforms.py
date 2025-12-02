"""
Preprocessing and Augmentation Transforms

이미지 전처리 및 Augmentation을 정의합니다.
학습 시에는 augmentation을 적용하고, 추론 시에는 정규화만 수행합니다.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any
import random


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    학습용 전처리 파이프라인을 생성합니다.
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        학습용 transform
    """
    preprocess_config = config.get('preprocessing', {})
    input_size = preprocess_config.get('input_size', {})
    width = input_size.get('width', 224)
    height = input_size.get('height', 224)
    
    # Normalization 설정
    norm_config = preprocess_config.get('normalization', {})
    norm_type = norm_config.get('type', 'imagenet')
    
    if norm_type == 'imagenet':
        mean = norm_config.get('mean', [0.485, 0.456, 0.406])
        std = norm_config.get('std', [0.229, 0.224, 0.225])
    elif norm_type == 'custom':
        mean = norm_config.get('mean', [0.5, 0.5, 0.5])
        std = norm_config.get('std', [0.5, 0.5, 0.5])
    else:
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    
    # Augmentation 설정
    aug_config = preprocess_config.get('augmentation', {})
    
    transform_list = []
    
    # Resize
    transform_list.append(transforms.Resize((height, width)))
    
    # Augmentation
    if aug_config.get('horizontal_flip', {}).get('enabled', False):
        prob = aug_config['horizontal_flip'].get('probability', 0.5)
        transform_list.append(transforms.RandomHorizontalFlip(p=prob))
    
    if aug_config.get('rotation', {}).get('enabled', False):
        max_angle = aug_config['rotation'].get('max_angle', 15)
        prob = aug_config['rotation'].get('probability', 0.5)
        if random.random() < prob:
            transform_list.append(transforms.RandomRotation(degrees=max_angle))
    
    if aug_config.get('color_jitter', {}).get('enabled', False):
        jitter_config = aug_config['color_jitter']
        prob = jitter_config.get('probability', 0.5)
        if random.random() < prob:
            transform_list.append(transforms.ColorJitter(
                brightness=jitter_config.get('brightness', 0.2),
                contrast=jitter_config.get('contrast', 0.2),
                saturation=jitter_config.get('saturation', 0.2),
                hue=jitter_config.get('hue', 0.1)
            ))
    
    if aug_config.get('random_crop', {}).get('enabled', False):
        scale = aug_config['random_crop'].get('scale', [0.8, 1.0])
        prob = aug_config['random_crop'].get('probability', 0.5)
        if random.random() < prob:
            transform_list.append(transforms.RandomResizedCrop(
                size=(height, width),
                scale=scale
            ))
    
    # ToTensor (PIL Image -> Tensor, [0, 255] -> [0, 1])
    transform_list.append(transforms.ToTensor())
    
    # Normalization
    transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    # Random Erasing (선택사항)
    if aug_config.get('random_erasing', {}).get('enabled', False):
        erase_config = aug_config['random_erasing']
        transform_list.append(transforms.RandomErasing(
            p=erase_config.get('probability', 0.3),
            scale=erase_config.get('scale', [0.02, 0.33]),
            ratio=erase_config.get('ratio', [0.3, 3.3])
        ))
    
    return transforms.Compose(transform_list)


def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    검증/추론용 전처리 파이프라인을 생성합니다.
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        검증용 transform (augmentation 없음)
    """
    preprocess_config = config.get('preprocessing', {})
    input_size = preprocess_config.get('input_size', {})
    width = input_size.get('width', 224)
    height = input_size.get('height', 224)
    
    # Normalization 설정
    norm_config = preprocess_config.get('normalization', {})
    norm_type = norm_config.get('type', 'imagenet')
    
    if norm_type == 'imagenet':
        mean = norm_config.get('mean', [0.485, 0.456, 0.406])
        std = norm_config.get('std', [0.229, 0.224, 0.225])
    elif norm_type == 'custom':
        mean = norm_config.get('mean', [0.5, 0.5, 0.5])
        std = norm_config.get('std', [0.5, 0.5, 0.5])
    else:
        mean = [0.0, 0.0, 0.0]
        std = [1.0, 1.0, 1.0]
    
    # 검증/추론 시에는 augmentation 없이 정규화만 수행
    transform_list = [
        transforms.Resize((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    return transforms.Compose(transform_list)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    정규화된 텐서를 원래 범위로 되돌립니다 (시각화용).
    
    Args:
        tensor: 정규화된 이미지 텐서 [C, H, W] 또는 [B, C, H, W]
        mean: 정규화에 사용된 평균
        std: 정규화에 사용된 표준편차
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    
    if tensor.dim() == 4:  # [B, C, H, W]
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Tensor를 PIL Image로 변환합니다.
    
    Args:
        tensor: 이미지 텐서 [C, H, W] (값 범위: [0, 1])
    
    Returns:
        PIL Image
    """
    # [C, H, W] -> [H, W, C]
    tensor = tensor.permute(1, 2, 0)
    
    # NumPy로 변환
    numpy_array = tensor.cpu().numpy()
    
    # [0, 1] -> [0, 255]
    numpy_array = (numpy_array * 255).astype(np.uint8)
    
    # PIL Image로 변환
    return Image.fromarray(numpy_array)


if __name__ == "__main__":
    # Transform 테스트
    sample_config = {
        'preprocessing': {
            'input_size': {'width': 224, 'height': 224},
            'normalization': {
                'type': 'imagenet'
            },
            'augmentation': {
                'horizontal_flip': {'enabled': True, 'probability': 0.5},
                'color_jitter': {'enabled': True, 'probability': 0.5}
            }
        }
    }
    
    # 테스트 이미지 생성
    test_image = Image.new('RGB', (256, 256), color='red')
    
    # 학습용 transform
    train_transform = get_train_transforms(sample_config)
    train_tensor = train_transform(test_image)
    print(f"Train transform output shape: {train_tensor.shape}")
    
    # 검증용 transform
    val_transform = get_val_transforms(sample_config)
    val_tensor = val_transform(test_image)
    print(f"Val transform output shape: {val_tensor.shape}")

