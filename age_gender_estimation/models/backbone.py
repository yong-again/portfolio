"""
Backbone Network for Feature Extraction

EfficientNet 또는 ResNet 계열을 Backbone으로 사용하여 얼굴 이미지로부터 feature를 추출합니다.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


def build_backbone(
    name: str = "efficientnet-b0",
    pretrained: bool = True,
    num_classes: int = 0  # 0이면 classifier 제거
) -> nn.Module:
    """
    Backbone 네트워크를 생성합니다.
    
    Args:
        name: Backbone 모델 이름
            - EfficientNet: "efficientnet-b0", "efficientnet-b1", ...
            - ResNet: "resnet18", "resnet34", "resnet50", ...
        pretrained: ImageNet 사전 학습 가중치 사용 여부
        num_classes: 출력 클래스 수 (0이면 feature extractor만 반환)
    
    Returns:
        Backbone 네트워크 모듈
    """
    # timm 라이브러리를 사용하여 모델 생성
    model = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=num_classes,
        global_pool='avg'  # Global Average Pooling 사용
    )
    
    return model


def get_backbone_feature_dim(backbone_name: str) -> int:
    """
    Backbone의 출력 feature dimension을 반환합니다.
    
    Args:
        backbone_name: Backbone 모델 이름
    
    Returns:
        Feature dimension
    """
    # timm 모델의 feature dimension 매핑
    feature_dims = {
        "efficientnet-b0": 1280,
        "efficientnet-b1": 1280,
        "efficientnet-b2": 1408,
        "efficientnet-b3": 1536,
        "efficientnet-b4": 1792,
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048,
    }
    
    if backbone_name in feature_dims:
        return feature_dims[backbone_name]
    
    # 기본값 (대부분의 EfficientNet-B0/B1)
    return 1280


class FeatureExtractor(nn.Module):
    """
    Feature Extractor Wrapper
    
    Backbone을 래핑하여 feature extraction을 수행합니다.
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet-b0",
        pretrained: bool = True,
        freeze: bool = False
    ):
        """
        Args:
            backbone_name: Backbone 모델 이름
            pretrained: 사전 학습 가중치 사용 여부
            freeze: Backbone 파라미터 고정 여부
        """
        super(FeatureExtractor, self).__init__()
        
        # Backbone 생성 (classifier 제거)
        self.backbone = build_backbone(
            name=backbone_name,
            pretrained=pretrained,
            num_classes=0  # Feature extractor만 사용
        )
        
        # Feature dimension
        self.feature_dim = get_backbone_feature_dim(backbone_name)
        
        # Backbone 고정
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
        
        Returns:
            Feature vector [B, feature_dim]
        """
        # Backbone forward
        features = self.backbone(x)
        
        # Global Average Pooling이 이미 적용되어 있으므로
        # [B, feature_dim] 형태로 반환됨
        return features
    
    def get_feature_dim(self) -> int:
        """Feature dimension 반환"""
        return self.feature_dim


if __name__ == "__main__":
    # Backbone 테스트
    backbone = FeatureExtractor(
        backbone_name="efficientnet-b0",
        pretrained=True,
        freeze=False
    )
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = backbone(dummy_input)
    
    print(f"Backbone: EfficientNet-B0")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Feature dimension: {backbone.get_feature_dim()}")

