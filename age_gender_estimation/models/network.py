"""
Complete Network Architecture

Backbone + Age Head + Gender Head를 결합한 전체 네트워크입니다.
Multi-task learning을 통해 두 작업을 동시에 수행합니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .backbone import FeatureExtractor
from .age_head import AgeHead
from .gender_head import GenderHead


class AgeGenderNetwork(nn.Module):
    """
    Age & Gender Estimation Network
    
    Shared Backbone + Multi-Head 구조로 Age와 Gender를 동시에 예측합니다.
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet-b0",
        backbone_pretrained: bool = True,
        freeze_backbone: bool = False,
        age_num_classes: int = 101,  # 0~100세
        age_hidden_dim: int = 512,
        gender_num_classes: int = 2,
        gender_hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        """
        Args:
            backbone_name: Backbone 모델 이름
            backbone_pretrained: Backbone 사전 학습 가중치 사용 여부
            freeze_backbone: Backbone 파라미터 고정 여부
            age_num_classes: Age 클래스 수 (0~100세 = 101)
            age_hidden_dim: Age head hidden dimension
            gender_num_classes: Gender 클래스 수 (2)
            gender_hidden_dim: Gender head hidden dimension
            dropout: Dropout 확률
        """
        super(AgeGenderNetwork, self).__init__()
        
        # Backbone (Feature Extractor)
        self.backbone = FeatureExtractor(
            backbone_name=backbone_name,
            pretrained=backbone_pretrained,
            freeze=freeze_backbone
        )
        
        feature_dim = self.backbone.get_feature_dim()
        
        # Age Head
        self.age_head = AgeHead(
            input_dim=feature_dim,
            num_classes=age_num_classes,
            hidden_dim=age_hidden_dim,
            dropout=dropout
        )
        
        # Gender Head
        self.gender_head = GenderHead(
            input_dim=feature_dim,
            num_classes=gender_num_classes,
            hidden_dim=gender_hidden_dim,
            dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
            return_features: 중간 feature도 반환할지 여부
        
        Returns:
            Dictionary containing:
                - 'age_logits': Age logits [B, num_classes] (0~100세)
                - 'gender_logits': Gender logits [B, num_classes]
                - 'features': (optional) Backbone features [B, feature_dim]
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Age prediction
        age_logits = self.age_head(features)
        
        # Gender prediction
        gender_logits = self.gender_head(features)
        
        output = {
            'age_logits': age_logits,
            'gender_logits': gender_logits
        }
        
        if return_features:
            output['features'] = features
        
        return output
    
    def predict(self, x: torch.Tensor) -> Dict:
        """
        예측 결과를 반환합니다.
        
        Args:
            x: 입력 이미지 [B, 3, H, W]
        
        Returns:
            Dictionary containing:
                - 'age': Age prediction results
                    - 'logits': [B, num_classes] (0~100세)
                    - 'probs': [B, num_classes]
                    - 'predicted_age': [B] (0~100)
                - 'gender': Gender prediction results
                    - 'logits': [B, num_classes]
                    - 'probs': [B, num_classes]
                    - 'predicted_class': [B]
                    - 'confidence': [B]
        """
        # Feature extraction
        features = self.backbone(x)
        
        # Age prediction
        age_output = self.age_head.predict(features)
        
        # Gender prediction
        gender_output = self.gender_head.predict(features)
        
        return {
            'age': age_output,
            'gender': gender_output
        }
    
    def get_backbone(self) -> nn.Module:
        """Backbone 모듈 반환"""
        return self.backbone
    
    def get_age_head(self) -> nn.Module:
        """Age head 모듈 반환"""
        return self.age_head
    
    def get_gender_head(self) -> nn.Module:
        """Gender head 모듈 반환"""
        return self.gender_head


def build_network(config: Dict) -> AgeGenderNetwork:
    """
    설정 파일로부터 네트워크를 생성하는 편의 함수
    
    Args:
        config: 설정 딕셔너리 (YAML 파일에서 로드)
    
    Returns:
        AgeGenderNetwork 인스턴스
    """
    model_config = config['model']
    backbone_config = model_config['backbone']
    age_config = model_config['age']
    gender_config = model_config['gender']
    
    network = AgeGenderNetwork(
        backbone_name=backbone_config['name'],
        backbone_pretrained=backbone_config['pretrained'],
        freeze_backbone=backbone_config.get('freeze_backbone', False),
        age_num_classes=age_config['num_classes'],
        age_hidden_dim=age_config['hidden_dim'],
        gender_num_classes=gender_config['num_classes'],
        gender_hidden_dim=gender_config['hidden_dim'],
        dropout=model_config.get('dropout', 0.5)
    )
    
    return network


if __name__ == "__main__":
    # Network 테스트
    network = AgeGenderNetwork(
        backbone_name="efficientnet-b0",
        backbone_pretrained=False,  # 테스트용
        age_num_classes=101,  # 0~100세
        gender_num_classes=2
    )
    
    dummy_input = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    output = network(dummy_input)
    print("Forward pass output:")
    print(f"  Age logits shape: {output['age_logits'].shape}")
    print(f"  Gender logits shape: {output['gender_logits'].shape}")
    
    # Predict
    predictions = network.predict(dummy_input)
    print("\nPredictions:")
    print(f"  Age predicted ages: {predictions['age']['predicted_age']}")
    print(f"  Gender predicted classes: {predictions['gender']['predicted_class']}")
    print(f"  Gender confidence: {predictions['gender']['confidence']}")

