"""
Image Quality Filtering

이미지 품질을 평가하고 필터링하는 모듈입니다.
빛 번짐, 초점 불안정 등을 감지합니다.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
import math


class ImageQualityFilter:
    """
    이미지 품질 필터링 클래스
    
    이미지의 blur, brightness, contrast 등을 평가하여
    품질이 낮은 이미지를 필터링합니다.
    """
    
    def __init__(
        self,
        blur_threshold: float = 100.0,
        brightness_min: float = 0.1,
        brightness_max: float = 0.9,
        contrast_min: float = 0.3,
        variance_threshold: float = 10.0
    ):
        """
        Args:
            blur_threshold: Blur detection threshold (Laplacian variance)
                           낮을수록 더 blur함. 일반적으로 100 미만이면 blur로 판단
            brightness_min: 최소 밝기 (0.0 ~ 1.0)
            brightness_max: 최대 밝기 (0.0 ~ 1.0)
            contrast_min: 최소 대비 (0.0 ~ 1.0)
            variance_threshold: 이미지 분산 threshold (너무 어둡거나 밝으면 낮음)
        """
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max
        self.contrast_min = contrast_min
        self.variance_threshold = variance_threshold
    
    def calculate_blur_score(self, image: np.ndarray) -> float:
        """
        Laplacian variance를 이용한 blur detection
        
        Args:
            image: 입력 이미지 (grayscale)
        
        Returns:
            Blur score (높을수록 선명함)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        이미지 평균 밝기 계산
        
        Args:
            image: 입력 이미지
        
        Returns:
            평균 밝기 (0.0 ~ 1.0)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        brightness = np.mean(gray) / 255.0
        return brightness
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        이미지 대비 계산 (표준편차 기반)
        
        Args:
            image: 입력 이미지
        
        Returns:
            대비 점수 (0.0 ~ 1.0)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        contrast = np.std(gray) / 255.0
        return contrast
    
    def calculate_variance(self, image: np.ndarray) -> float:
        """
        이미지 분산 계산 (너무 어둡거나 밝으면 낮음)
        
        Args:
            image: 입력 이미지
        
        Returns:
            분산 값
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        variance = np.var(gray)
        return variance
    
    def is_blurry(self, image: np.ndarray) -> bool:
        """
        이미지가 blur한지 확인
        
        Args:
            image: 입력 이미지
        
        Returns:
            blur 여부 (True: blur, False: 선명)
        """
        blur_score = self.calculate_blur_score(image)
        return blur_score < self.blur_threshold
    
    def is_brightness_acceptable(self, image: np.ndarray) -> bool:
        """
        이미지 밝기가 적절한지 확인
        
        Args:
            image: 입력 이미지
        
        Returns:
            밝기가 적절한지 여부
        """
        brightness = self.calculate_brightness(image)
        return self.brightness_min <= brightness <= self.brightness_max
    
    def is_contrast_acceptable(self, image: np.ndarray) -> bool:
        """
        이미지 대비가 적절한지 확인
        
        Args:
            image: 입력 이미지
        
        Returns:
            대비가 적절한지 여부
        """
        contrast = self.calculate_contrast(image)
        return contrast >= self.contrast_min
    
    def evaluate_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지 품질을 종합적으로 평가
        
        Args:
            image: 입력 이미지 (numpy array)
        
        Returns:
            품질 평가 결과 딕셔너리:
                - 'blur_score': Blur score
                - 'brightness': 밝기 (0.0 ~ 1.0)
                - 'contrast': 대비 (0.0 ~ 1.0)
                - 'variance': 분산
                - 'is_blurry': Blur 여부
                - 'is_brightness_ok': 밝기 적절 여부
                - 'is_contrast_ok': 대비 적절 여부
                - 'is_acceptable': 전체적으로 수용 가능한지
        """
        blur_score = self.calculate_blur_score(image)
        brightness = self.calculate_brightness(image)
        contrast = self.calculate_contrast(image)
        variance = self.calculate_variance(image)
        
        is_blurry = blur_score < self.blur_threshold
        is_brightness_ok = self.brightness_min <= brightness <= self.brightness_max
        is_contrast_ok = contrast >= self.contrast_min
        is_variance_ok = variance >= self.variance_threshold
        
        is_acceptable = (
            not is_blurry and
            is_brightness_ok and
            is_contrast_ok and
            is_variance_ok
        )
        
        return {
            'blur_score': float(blur_score),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'variance': float(variance),
            'is_blurry': is_blurry,
            'is_brightness_ok': is_brightness_ok,
            'is_contrast_ok': is_contrast_ok,
            'is_variance_ok': is_variance_ok,
            'is_acceptable': is_acceptable
        }
    
    def filter_images(
        self,
        images: list,
        return_scores: bool = False
    ) -> Tuple[list, list]:
        """
        이미지 리스트를 필터링
        
        Args:
            images: PIL Image 또는 numpy array 리스트
            return_scores: 품질 점수도 반환할지 여부
        
        Returns:
            필터링된 이미지 리스트
            (optional) 품질 평가 결과 리스트
        """
        filtered_images = []
        quality_scores = []
        
        for img in images:
            # PIL Image를 numpy array로 변환
            if isinstance(img, Image.Image):
                img_array = np.array(img.convert('RGB'))
            elif isinstance(img, np.ndarray):
                img_array = img
            else:
                continue
            
            # 품질 평가
            quality = self.evaluate_quality(img_array)
            
            if quality['is_acceptable']:
                filtered_images.append(img)
                if return_scores:
                    quality_scores.append(quality)
        
        if return_scores:
            return filtered_images, quality_scores
        return filtered_images


def filter_image_list(
    images: list,
    blur_threshold: float = 100.0,
    brightness_min: float = 0.1,
    brightness_max: float = 0.9,
    contrast_min: float = 0.3,
    variance_threshold: float = 10.0,
    return_scores: bool = False
) -> Tuple[list, list]:
    """
    이미지 리스트를 필터링하는 편의 함수
    
    Args:
        images: PIL Image 또는 numpy array 리스트
        blur_threshold: Blur detection threshold
        brightness_min: 최소 밝기
        brightness_max: 최대 밝기
        contrast_min: 최소 대비
        variance_threshold: 분산 threshold
        return_scores: 품질 점수도 반환할지 여부
    
    Returns:
        필터링된 이미지 리스트
        (optional) 품질 평가 결과 리스트
    """
    filter_obj = ImageQualityFilter(
        blur_threshold=blur_threshold,
        brightness_min=brightness_min,
        brightness_max=brightness_max,
        contrast_min=contrast_min,
        variance_threshold=variance_threshold
    )
    
    return filter_obj.filter_images(images, return_scores=return_scores)


if __name__ == "__main__":
    # 테스트 코드
    filter_obj = ImageQualityFilter()
    
    # 더미 이미지 생성 (선명한 이미지)
    test_image_good = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    
    # Blur 이미지 생성
    test_image_blur = cv2.GaussianBlur(test_image_good, (15, 15), 0)
    
    # 평가
    quality_good = filter_obj.evaluate_quality(test_image_good)
    quality_blur = filter_obj.evaluate_quality(test_image_blur)
    
    print("Good image quality:")
    print(quality_good)
    
    print("\nBlur image quality:")
    print(quality_blur)

