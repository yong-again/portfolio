"""
Gender Classification Head

Backbone에서 추출한 feature를 입력으로 받아 성별을 분류합니다.
"""

import torch
import torch.nn as nn
from typing import Optional


class GenderHead(nn.Module):
    """
    Gender Classification Head
    
    Binary classification으로 성별을 예측합니다 (Male / Female).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        """
        Args:
            input_dim: 입력 feature dimension (Backbone 출력 차원)
            num_classes: 클래스 수 (2: Male, Female)
            hidden_dim: Hidden layer dimension
            dropout: Dropout 확률
        """
        super(GenderHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: Backbone에서 추출한 feature [B, input_dim]
        
        Returns:
            Gender logits [B, num_classes]
        """
        logits = self.classifier(features)
        return logits
    
    def predict(self, features: torch.Tensor) -> dict:
        """
        예측 결과를 반환합니다.
        
        Args:
            features: Backbone feature [B, input_dim]
        
        Returns:
            Dictionary containing:
                - 'logits': Gender logits [B, num_classes]
                - 'probs': Gender probabilities [B, num_classes]
                - 'predicted_class': Predicted class index [B] (0: Male, 1: Female)
                - 'confidence': Confidence score [B]
        """
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        
        return {
            'logits': logits,
            'probs': probs,
            'predicted_class': predicted_class,
            'confidence': confidence
        }


def class_to_gender(class_idx: int) -> str:
    """
    클래스 인덱스를 성별 문자열로 변환합니다.
    
    Args:
        class_idx: 클래스 인덱스 (0: Male, 1: Female)
    
    Returns:
        성별 문자열 ("Male" 또는 "Female")
    """
    gender_map = {
        0: "Male",
        1: "Female"
    }
    return gender_map.get(class_idx, "Unknown")


def gender_to_class(gender: str) -> int:
    """
    성별 문자열을 클래스 인덱스로 변환합니다.
    
    Args:
        gender: 성별 문자열 ("Male", "Female", "M", "F" 등)
    
    Returns:
        클래스 인덱스 (0: Male, 1: Female)
    """
    gender_lower = gender.lower()
    
    if gender_lower in ['male', 'm', '0']:
        return 0
    elif gender_lower in ['female', 'f', '1']:
        return 1
    else:
        raise ValueError(f"Unknown gender: {gender}")


if __name__ == "__main__":
    # Gender Head 테스트
    input_dim = 1280
    num_classes = 2
    
    gender_head = GenderHead(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.5
    )
    
    dummy_features = torch.randn(2, input_dim)
    output = gender_head.predict(dummy_features)
    
    print(f"Gender Head")
    print(f"Input shape: {dummy_features.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Predicted classes: {output['predicted_class']}")
    print(f"Confidence: {output['confidence']}")
    
    # 클래스 변환 테스트
    print(f"\nClass 0 -> Gender: {class_to_gender(0)}")
    print(f"Class 1 -> Gender: {class_to_gender(1)}")
    print(f"Male -> Class: {gender_to_class('Male')}")
    print(f"Female -> Class: {gender_to_class('Female')}")

