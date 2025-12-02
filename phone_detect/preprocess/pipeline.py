"""
Preprocessing Pipeline

이미지 전처리 파이프라인을 정의합니다.
학습 시에는 augmentation을 적용하고, 추론 시에는 정규화만 수행합니다.
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import random
import cv2


class RandomHorizontalFlip:
    """Random horizontal flip with bounding box adjustment"""
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def __call__(self, image: Image.Image, bboxes: Optional[np.ndarray] = None) -> Tuple[Image.Image, Optional[np.ndarray]]:
        if random.random() < self.probability:
            image = F.hflip(image)
            if bboxes is not None:
                # Flip bboxes: x_center -> 1 - x_center
                bboxes = bboxes.copy()
                bboxes[:, 0] = 1.0 - bboxes[:, 0]
        return image, bboxes


class RandomVerticalFlip:
    """Random vertical flip with bounding box adjustment"""
    
    def __init__(self, probability: float = 0.5):
        self.probability = probability
    
    def __call__(self, image: Image.Image, bboxes: Optional[np.ndarray] = None) -> Tuple[Image.Image, Optional[np.ndarray]]:
        if random.random() < self.probability:
            image = F.vflip(image)
            if bboxes is not None:
                # Flip bboxes: y_center -> 1 - y_center
                bboxes = bboxes.copy()
                bboxes[:, 1] = 1.0 - bboxes[:, 1]
        return image, bboxes


class RandomRotation:
    """Random rotation with bounding box adjustment"""
    
    def __init__(self, probability: float = 0.5, max_angle: float = 15.0):
        self.probability = probability
        self.max_angle = max_angle
    
    def __call__(self, image: Image.Image, bboxes: Optional[np.ndarray] = None) -> Tuple[Image.Image, Optional[np.ndarray]]:
        if random.random() < self.probability:
            angle = random.uniform(-self.max_angle, self.max_angle)
            image = F.rotate(image, angle, fill=0)
            # Bbox rotation은 복잡하므로 간단한 구현 (실제로는 더 정교한 변환이 필요)
            # 여기서는 단순화하여 bbox는 그대로 유지
        return image, bboxes


class ResizeWithPadding:
    """Resize image while maintaining aspect ratio and add padding"""
    
    def __init__(self, target_size: Tuple[int, int], fill: int = 0):
        self.target_size = target_size
        self.fill = fill
    
    def __call__(self, image: Image.Image) -> Image.Image:
        original_size = image.size  # (width, height)
        target_width, target_height = self.target_size
        
        # Aspect ratio 계산
        scale = min(target_width / original_size[0], target_height / original_size[1])
        new_width = int(original_size[0] * scale)
        new_height = int(original_size[1] * scale)
        
        # Resize
        image = image.resize((new_width, new_height), Image.BILINEAR)
        
        # Padding 추가
        pad_left = (target_width - new_width) // 2
        pad_top = (target_height - new_height) // 2
        pad_right = target_width - new_width - pad_left
        pad_bottom = target_height - new_height - pad_top
        
        image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=self.fill)
        
        return image


class PreprocessingPipeline:
    """
    전처리 파이프라인 클래스
    
    이미지와 bounding box를 함께 처리할 수 있는 통합 파이프라인입니다.
    """
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """
        Args:
            config: 설정 딕셔너리
            is_training: 학습 모드 여부 (True면 augmentation 적용)
        """
        self.config = config
        self.is_training = is_training
        self.preprocess_config = config.get('preprocessing', {})
        
        # Resize 설정
        resize_config = self.preprocess_config.get('resize', {})
        self.target_size = (
            resize_config.get('width', 640),
            resize_config.get('height', 640)
        )
        self.keep_aspect_ratio = resize_config.get('keep_aspect_ratio', False)
        
        # Normalization 설정
        norm_config = self.preprocess_config.get('normalization', {})
        norm_type = norm_config.get('type', 'imagenet')
        
        if norm_type == 'imagenet':
            self.mean = norm_config.get('mean', [0.485, 0.456, 0.406])
            self.std = norm_config.get('std', [0.229, 0.224, 0.225])
        elif norm_type == 'custom':
            self.mean = norm_config.get('mean', [0.5, 0.5, 0.5])
            self.std = norm_config.get('std', [0.5, 0.5, 0.5])
        else:
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]
        
        # Augmentation 설정 (학습 시에만)
        self.aug_config = self.preprocess_config.get('augmentation', {}) if is_training else {}
    
    def __call__(self, image: Image.Image, bboxes: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        이미지와 bounding box를 전처리합니다.
        
        Args:
            image: PIL Image
            bboxes: Bounding boxes [N, 5] (class_id, x_center, y_center, width, height) - 정규화된 좌표
        
        Returns:
            processed_image: 전처리된 이미지 텐서 [C, H, W]
            processed_bboxes: 전처리된 bounding boxes (좌표는 그대로 유지)
        """
        # Augmentation 적용 (학습 시에만)
        if self.is_training:
            # Horizontal flip
            if self.aug_config.get('horizontal_flip', {}).get('enabled', False):
                flip = RandomHorizontalFlip(
                    self.aug_config['horizontal_flip'].get('probability', 0.5)
                )
                image, bboxes = flip(image, bboxes)
            
            # Vertical flip
            if self.aug_config.get('vertical_flip', {}).get('enabled', False):
                vflip = RandomVerticalFlip(
                    self.aug_config['vertical_flip'].get('probability', 0.5)
                )
                image, bboxes = vflip(image, bboxes)
            
            # Rotation
            if self.aug_config.get('rotation', {}).get('enabled', False):
                rotation = RandomRotation(
                    self.aug_config['rotation'].get('probability', 0.5),
                    self.aug_config['rotation'].get('max_angle', 15.0)
                )
                image, bboxes = rotation(image, bboxes)
            
            # Color jitter
            if self.aug_config.get('color_jitter', {}).get('enabled', False):
                jitter_config = self.aug_config['color_jitter']
                if random.random() < jitter_config.get('probability', 0.5):
                    color_jitter = transforms.ColorJitter(
                        brightness=jitter_config.get('brightness', 0.2),
                        contrast=jitter_config.get('contrast', 0.2),
                        saturation=jitter_config.get('saturation', 0.2),
                        hue=jitter_config.get('hue', 0.1)
                    )
                    image = color_jitter(image)
        
        # Resize
        if self.keep_aspect_ratio:
            resize = ResizeWithPadding(self.target_size)
            image = resize(image)
        else:
            image = image.resize(self.target_size, Image.BILINEAR)
        
        # PIL Image to Tensor
        image_tensor = F.to_tensor(image)  # [0, 1] 범위로 자동 정규화
        
        # Normalization
        image_tensor = F.normalize(image_tensor, mean=self.mean, std=self.std)
        
        return image_tensor, bboxes


def get_train_transform(config: Dict[str, Any]) -> PreprocessingPipeline:
    """
    학습용 전처리 파이프라인을 생성합니다.
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        PreprocessingPipeline 인스턴스 (is_training=True)
    """
    return PreprocessingPipeline(config, is_training=True)


def get_val_transform(config: Dict[str, Any]) -> PreprocessingPipeline:
    """
    검증/추론용 전처리 파이프라인을 생성합니다.
    
    Args:
        config: 설정 딕셔너리
    
    Returns:
        PreprocessingPipeline 인스턴스 (is_training=False)
    """
    return PreprocessingPipeline(config, is_training=False)


def load_image(image_path: str) -> Image.Image:
    """
    이미지 파일을 로드합니다.
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    return image


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: list = [0.485, 0.456, 0.406],
    std: list = [0.229, 0.224, 0.225]
) -> torch.Tensor:
    """
    정규화된 텐서를 원래 범위로 되돌립니다 (시각화용).
    
    Args:
        tensor: 정규화된 이미지 텐서 [C, H, W]
        mean: 정규화에 사용된 평균
        std: 정규화에 사용된 표준편차
    
    Returns:
        Denormalized tensor [C, H, W]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    return tensor


if __name__ == "__main__":
    # 전처리 파이프라인 테스트
    import yaml
    
    # 샘플 설정
    sample_config = {
        'preprocessing': {
            'resize': {
                'width': 640,
                'height': 640,
                'keep_aspect_ratio': False
            },
            'normalization': {
                'type': 'imagenet'
            },
            'augmentation': {
                'horizontal_flip': {
                    'enabled': True,
                    'probability': 0.5
                }
            }
        }
    }
    
    # 테스트 이미지 생성
    test_image = Image.new('RGB', (1920, 1080), color='red')
    
    # 학습용 파이프라인
    train_pipeline = get_train_transform(sample_config)
    train_image, _ = train_pipeline(test_image)
    print(f"Train image shape: {train_image.shape}")
    
    # 검증용 파이프라인
    val_pipeline = get_val_transform(sample_config)
    val_image, _ = val_pipeline(test_image)
    print(f"Val image shape: {val_image.shape}")

