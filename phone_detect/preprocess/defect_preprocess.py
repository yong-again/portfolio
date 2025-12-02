"""
Defect Preprocessing Pipeline

결함 검출을 위한 전처리 파이프라인입니다.
CLAHE, histogram normalization 등을 포함합니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from PIL import Image


def get_image_scaler(scaler_str: str) -> Optional[Callable]:
    """
    Scaler 함수를 반환합니다.
    
    Args:
        scaler_str: Scaler 이름
    
    Returns:
        Scaler 함수 또는 None
    """
    if scaler_str == 'normalize':
        return normalize_image
    elif scaler_str == 'normalize_histogram':
        return normalize_histogram
    elif scaler_str == 'equalize_histogram':
        return equalize_histogram
    elif scaler_str == 'adaptive_equalize_histogram':
        return adaptive_equalize_histogram
    elif scaler_str == 'imagenet_standardize':
        return imagenet_standardize
    else:
        return None


def normalize_image(image: np.ndarray, max_pixel_value: int = 255) -> np.ndarray:
    """
    이미지를 [0, 1] 범위로 정규화합니다.
    
    Args:
        image: 입력 이미지 [H, W, 3]
        max_pixel_value: 최대 픽셀 값
    
    Returns:
        정규화된 이미지
    """
    return image.astype(np.float32) / max_pixel_value


def normalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Histogram normalization을 적용합니다.
    
    Args:
        image: 입력 이미지 [H, W, 3] (RGB)
    
    Returns:
        정규화된 이미지
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] = cv2.normalize(
        lab_image[:, :, 0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    histogram_normalized_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return histogram_normalized_image


def equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    Histogram equalization을 적용합니다.
    
    Args:
        image: 입력 이미지 [H, W, 3] (RGB)
    
    Returns:
        Equalized 이미지
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_image[:, :, 0] = cv2.equalizeHist(lab_image[:, :, 0].astype(np.uint8))
    equalized_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return equalized_image


def adaptive_equalize_histogram(image: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)를 적용합니다.
    
    Args:
        image: 입력 이미지 [H, W, 3] (RGB)
    
    Returns:
        CLAHE 적용된 이미지
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0].astype(np.uint8))
    histogram_equalized_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
    return histogram_equalized_image


def imagenet_standardize(image: np.ndarray) -> np.ndarray:
    """
    ImageNet 통계를 사용하여 표준화합니다.
    
    Args:
        image: 입력 이미지 [H, W, 3] (RGB, [0, 255])
    
    Returns:
        표준화된 이미지
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image.astype(np.float32) / 255.0
    standardized_image = (image - mean) / std
    
    return standardized_image


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    CLAHE를 적용합니다.
    
    Args:
        image: 입력 이미지 [H, W, 3] (RGB)
        clip_limit: CLAHE clip limit
        tile_grid_size: Tile grid size (width, height)
    
    Returns:
        CLAHE 적용된 이미지
    """
    # Grayscale로 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # CLAHE 적용
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_gray = clahe.apply(gray.astype(np.uint8))
    
    # RGB로 변환
    if len(image.shape) == 3:
        clahe_rgb = cv2.cvtColor(clahe_gray, cv2.COLOR_GRAY2RGB)
        return clahe_rgb
    else:
        return clahe_gray


class DefectPreprocessor:
    """
    결함 검출을 위한 전처리 클래스
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
    
    def preprocess(
        self,
        image: np.ndarray,
        section: str = "display",
        apply_clahe_flag: bool = False
    ) -> np.ndarray:
        """
        이미지를 전처리합니다.
        
        Args:
            image: 입력 이미지 [H, W, 3]
            section: 섹션 이름 (display, side, front, back)
            apply_clahe_flag: CLAHE 적용 여부
        
        Returns:
            전처리된 이미지
        """
        processed_image = image.copy()
        
        # CLAHE 적용 (전면 등 특정 섹션에서만)
        if apply_clahe_flag:
            clahe_config = self.config['defect_segmentation'][section]['preprocessing'].get('clahe', {})
            if clahe_config.get('enabled', False):
                processed_image = apply_clahe(
                    processed_image,
                    clip_limit=clahe_config.get('clip_limit', 2.0),
                    tile_grid_size=tuple(clahe_config.get('tile_grid_size', [8, 8]))
                )
        
        # Scaler 적용
        scaler_list = self.config['defect_segmentation'][section]['preprocessing'].get('scaler', [])
        
        for scaler_str in scaler_list:
            scaler_func = get_image_scaler(scaler_str)
            if scaler_func is not None:
                processed_image = scaler_func(processed_image)
        
        return processed_image
    
    def resize(
        self,
        image: np.ndarray,
        target_size: Dict[str, int],
        keep_aspect_ratio: bool = False
    ) -> np.ndarray:
        """
        이미지를 리사이즈합니다.
        
        Args:
            image: 입력 이미지 [H, W, 3]
            target_size: 목표 크기 {'width': int, 'height': int}
            keep_aspect_ratio: 종횡비 유지 여부
        
        Returns:
            리사이즈된 이미지
        """
        if keep_aspect_ratio:
            # Aspect ratio 유지하며 리사이즈 (padding 추가)
            h, w = image.shape[:2]
            target_w, target_h = target_size['width'], target_size['height']
            
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(image, (new_w, new_h))
            
            # Padding 추가
            pad_left = (target_w - new_w) // 2
            pad_top = (target_h - new_h) // 2
            pad_right = target_w - new_w - pad_left
            pad_bottom = target_h - new_h - pad_top
            
            padded = cv2.copyMakeBorder(
                resized, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )
            
            return padded
        else:
            return cv2.resize(image, (target_size['width'], target_size['height']))
    
    def to_tensor(self, image: np.ndarray) -> np.ndarray:
        """
        이미지를 텐서 형식으로 변환합니다.
        
        Args:
            image: 입력 이미지 [H, W, 3]
        
        Returns:
            텐서 형식 이미지 [1, 3, H, W]
        """
        # [H, W, C] -> [C, H, W]
        if len(image.shape) == 3:
            image = image.transpose(2, 0, 1)
        
        # [C, H, W] -> [1, C, H, W]
        image = np.expand_dims(image, axis=0)
        
        return image

