"""
Phone Region Segmentation

Segmentation 모델을 사용하여 휴대폰 영역을 검출합니다 (측면, 디스플레이용).
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2


class PhoneSegmenter(nn.Module):
    """
    Phone Region Segmentation Model
    
    측면/디스플레이 이미지에서 휴대폰 영역을 segmentation으로 검출합니다.
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b3",
        encoder_weights: str = "imagenet",
        classes: int = 2,  # 배경 + 휴대폰 영역
        activation: Optional[str] = None
    ):
        """
        Args:
            encoder_name: Encoder 모델 이름
            encoder_weights: 사전 학습 가중치
            classes: Segmentation 클래스 수
            activation: Activation 함수 (None이면 logits 반환)
        """
        super(PhoneSegmenter, self).__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 입력 이미지 [B, 3, H, W]
        
        Returns:
            Segmentation logits [B, classes, H, W]
        """
        return self.model(x)
    
    def predict_mask(
        self,
        image: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        이미지에서 휴대폰 영역 mask를 예측합니다.
        
        Args:
            image: 입력 이미지 [H, W, 3]
            threshold: Binary threshold
        
        Returns:
            Binary mask [H, W] (0: 배경, 1: 휴대폰)
        """
        self.eval()
        
        # 전처리
        image_tensor = self._preprocess(image)
        
        with torch.no_grad():
            output = self.forward(image_tensor)
            probs = torch.softmax(output, dim=1)
            mask = (probs[0, 1] > threshold).cpu().numpy()  # 클래스 1 (휴대폰)
        
        return mask.astype(np.uint8)
    
    def get_bbox_from_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Mask에서 bounding box를 추출합니다.
        
        Args:
            mask: Binary mask [H, W]
        
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            h, w = mask.shape
            return np.array([0, 0, w, h])
        
        # 가장 큰 contour 선택
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        return np.array([x, y, x + w, y + h])
    
    def crop_phone_region(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        padding_ratio: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segmentation으로 검출된 휴대폰 영역을 crop합니다.
        
        Args:
            image: 입력 이미지 [H, W, 3]
            threshold: Segmentation threshold
            padding_ratio: Crop 시 추가할 padding 비율
        
        Returns:
            cropped_image: Crop된 이미지
            bbox: Bounding box [x1, y1, x2, y2]
        """
        # Mask 예측
        mask = self.predict_mask(image, threshold)
        
        # Bounding box 추출
        bbox = self.get_bbox_from_mask(mask)
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Padding 추가
        if padding_ratio > 0:
            width = x2 - x1
            height = y2 - y1
            pad_w = int(width * padding_ratio)
            pad_h = int(height * padding_ratio)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(image.shape[1], x2 + pad_w)
            y2 = min(image.shape[0], y2 + pad_h)
        
        # Crop
        cropped = image[y1:y2, x1:x2]
        
        return cropped, np.array([x1, y1, x2, y2])
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        이미지를 모델 입력 형식으로 전처리합니다.
        
        Args:
            image: 입력 이미지 [H, W, 3]
        
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Resize (필요시)
        # Normalize
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # [H, W, C] -> [C, H, W]
        image = image.transpose(2, 0, 1)
        
        # [C, H, W] -> [1, C, H, W]
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        
        return image_tensor

