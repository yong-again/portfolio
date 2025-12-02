"""
Age Classification Head

Backbone에서 추출한 feature를 입력으로 받아 나이를 분류합니다.
0~100세를 1세 단위로 분류합니다 (총 101 classes).
"""

import torch
import torch.nn as nn
from typing import Optional


class AgeHead(nn.Module):
    """
    Age Classification Head
    
    Multi-class classification으로 나이를 예측합니다 (0~100세, 1세 단위).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 101,  # 0~100세
        hidden_dim: int = 512,
        dropout: float = 0.5
    ):
        """
        Args:
            input_dim: 입력 feature dimension (Backbone 출력 차원)
            num_classes: 나이 클래스 개수 (0~100세 = 101)
            hidden_dim: Hidden layer dimension
            dropout: Dropout 확률
        """
        super(AgeHead, self).__init__()
        
        self.num_classes = num_classes
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: Backbone에서 추출한 feature [B, input_dim]
        
        Returns:
            Age logits [B, num_classes] (0~100세)
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
                - 'logits': Age logits [B, num_classes]
                - 'probs': Age probabilities [B, num_classes]
                - 'predicted_age': Predicted age (0~100) [B]
        """
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=1)
        predicted_age = torch.argmax(probs, dim=1)  # 0~100
        
        return {
            'logits': logits,
            'probs': probs,
            'predicted_age': predicted_age
        }


if __name__ == "__main__":
    # Age Head 테스트
    input_dim = 1280
    num_classes = 101  # 0~100세
    
    age_head = AgeHead(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=512,
        dropout=0.5
    )
    
    dummy_features = torch.randn(2, input_dim)
    output = age_head.predict(dummy_features)
    
    print(f"Age Head")
    print(f"Input shape: {dummy_features.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Predicted ages: {output['predicted_age']}")
