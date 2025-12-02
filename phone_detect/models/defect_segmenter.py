"""
Defect Segmentation Model

결함 영역을 pixel-level로 segmentation하는 모델입니다.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Optional, Tuple
import numpy as np


class DefectSegmenter(nn.Module):
    """
    Defect Segmentation Model
    
    휴대폰 영역에서 결함을 pixel-level로 분류합니다.
    각 pixel은 A, B, C, D 등급 중 하나로 분류됩니다.
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        decoder_name: str = "unet",
        classes: int = 4,  # A, B, C, D
        activation: Optional[str] = None
    ):
        """
        Args:
            encoder_name: Encoder 모델 이름
            encoder_weights: 사전 학습 가중치
            decoder_name: Decoder 타입 (unet, fpn, deeplabv3+ 등)
            classes: Segmentation 클래스 수 (등급 개수)
            activation: Activation 함수 (None이면 logits 반환)
        """
        super(DefectSegmenter, self).__init__()
        
        if decoder_name.lower() == "unet":
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=classes,
                activation=activation
            )
        elif decoder_name.lower() == "fpn":
            self.model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=classes,
                activation=activation
            )
        elif decoder_name.lower() == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=classes,
                activation=activation
            )
        else:
            raise ValueError(f"Unknown decoder: {decoder_name}")
    
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
        x: torch.Tensor,
        return_probs: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Segmentation mask를 예측합니다.
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
            return_probs: 확률도 반환할지 여부
        
        Returns:
            mask: Predicted mask [B, H, W] (각 pixel의 등급)
            probs: (optional) Class probabilities [B, classes, H, W]
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            mask = torch.argmax(probs, dim=1).cpu().numpy()
        
        if return_probs:
            return mask, probs.cpu().numpy()
        return mask, None
    
    def predict_with_threshold(
        self,
        x: torch.Tensor,
        thresholds: list
    ) -> np.ndarray:
        """
        Threshold를 적용하여 mask를 예측합니다.
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
            thresholds: 각 클래스별 threshold [classes]
        
        Returns:
            mask: Threshold 적용된 mask [B, H, W]
        """
        self.eval()
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            # Threshold 적용
            mask = np.zeros((probs.shape[0], probs.shape[2], probs.shape[3]), dtype=np.uint8)
            
            for class_idx in range(probs.shape[1]):
                class_mask = probs[:, class_idx] > thresholds[class_idx]
                mask[class_mask] = class_idx
        
        return mask

