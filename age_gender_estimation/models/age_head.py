"""
Age Classification Head

Backbone에서 추출한 feature를 입력으로 받아 연령대를 분류합니다.
"""

import torch
import torch.nn as nn
from typing import Optional


class AgeHead(nn.Module):
    """
    Age Classification Head
    
    Multi-class classification으로 연령대를 예측합니다.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_bins: int = 10,
        hidden_dim: int = 512,
        dropout: float = 0.5
    ):
        """
        Args:
            input_dim: 입력 feature dimension (Backbone 출력 차원)
            num_bins: 연령대 구간 개수
            hidden_dim: Hidden layer dimension
            dropout: Dropout 확률
        """
        super(AgeHead, self).__init__()
        
        self.num_bins = num_bins
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_bins)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            features: Backbone에서 추출한 feature [B, input_dim]
        
        Returns:
            Age logits [B, num_bins]
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
                - 'logits': Age logits [B, num_bins]
                - 'probs': Age probabilities [B, num_bins]
                - 'predicted_bin': Predicted age bin index [B]
        """
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=1)
        predicted_bin = torch.argmax(probs, dim=1)
        
        return {
            'logits': logits,
            'probs': probs,
            'predicted_bin': predicted_bin
        }


def create_age_bins(
    num_bins: int = 10,
    min_age: int = 0,
    max_age: int = 100
) -> list:
    """
    연령대 구간(bins)을 생성합니다.
    
    Args:
        num_bins: 구간 개수
        min_age: 최소 나이
        max_age: 최대 나이
    
    Returns:
        Age bins 리스트 [(min, max), ...]
    """
    bin_size = (max_age - min_age) / num_bins
    bins = []
    
    for i in range(num_bins):
        bin_min = min_age + i * bin_size
        bin_max = min_age + (i + 1) * bin_size
        bins.append((int(bin_min), int(bin_max)))
    
    return bins


def age_to_bin(age: int, bins: list) -> int:
    """
    나이를 연령대 bin 인덱스로 변환합니다.
    
    Args:
        age: 나이 (정수)
        bins: Age bins 리스트
    
    Returns:
        Bin 인덱스
    """
    for idx, (bin_min, bin_max) in enumerate(bins):
        if bin_min <= age <= bin_max:
            return idx
    
    # 범위를 벗어난 경우 마지막 bin 반환
    return len(bins) - 1


def bin_to_age_range(bin_idx: int, bins: list) -> tuple:
    """
    Bin 인덱스를 연령대 범위로 변환합니다.
    
    Args:
        bin_idx: Bin 인덱스
        bins: Age bins 리스트
    
    Returns:
        (min_age, max_age) 튜플
    """
    if 0 <= bin_idx < len(bins):
        return bins[bin_idx]
    return bins[-1]


if __name__ == "__main__":
    # Age Head 테스트
    input_dim = 1280
    num_bins = 10
    
    age_head = AgeHead(
        input_dim=input_dim,
        num_bins=num_bins,
        hidden_dim=512,
        dropout=0.5
    )
    
    dummy_features = torch.randn(2, input_dim)
    output = age_head.predict(dummy_features)
    
    print(f"Age Head")
    print(f"Input shape: {dummy_features.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Predicted bins: {output['predicted_bin']}")
    
    # Age bins 생성 테스트
    bins = create_age_bins(num_bins=10, min_age=0, max_age=100)
    print(f"\nAge bins: {bins}")
    print(f"Age 25 -> Bin: {age_to_bin(25, bins)}")
    print(f"Bin 2 -> Age range: {bin_to_age_range(2, bins)}")

