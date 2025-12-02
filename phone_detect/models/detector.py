"""
Phone Detector Model Architecture

이 모듈은 스마트폰 영역을 검출하기 위한 Object Detection 모델을 정의합니다.
EfficientNet 또는 ResNet을 Backbone으로 사용하며, FPN을 통해 multi-scale feature를 추출합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import List, Tuple, Dict, Optional


class FPN(nn.Module):
    """
    Feature Pyramid Network (FPN)
    
    Backbone에서 추출한 multi-scale feature를 결합하여
    다양한 크기의 객체 검출에 효과적인 feature map을 생성합니다.
    """
    
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        """
        Args:
            in_channels_list: Backbone 각 레벨의 출력 채널 수 리스트
            out_channels: FPN 출력 채널 수
        """
        super(FPN, self).__init__()
        self.out_channels = out_channels
        
        # Lateral connections (1x1 conv로 채널 수 통일)
        self.lateral_convs = nn.ModuleList()
        # Output convs (최종 feature map 생성)
        self.fpn_convs = nn.ModuleList()
        
        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: Backbone에서 추출한 feature map 리스트 (낮은 레벨부터 높은 레벨 순서)
        
        Returns:
            FPN으로 처리된 feature map 리스트
        """
        # Top-down pathway
        # 가장 높은 레벨부터 시작
        laterals = [self.lateral_convs[i](features[i]) for i in range(len(features))]
        
        # Top-down으로 feature 결합
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[2:],
                mode='nearest'
            )
        
        # 최종 feature map 생성
        fpn_outputs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals))]
        
        return fpn_outputs


class DetectionHead(nn.Module):
    """
    Object Detection Head
    
    FPN에서 나온 feature map으로부터 bounding box와 class를 예측합니다.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 3
    ):
        """
        Args:
            in_channels: 입력 feature map의 채널 수
            num_classes: 검출할 클래스 수 (background 포함하지 않음)
            num_anchors: 각 위치당 앵커 개수
        """
        super(DetectionHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * num_classes, 1)
        )
        
        # Localization head (bbox regression)
        self.loc_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_anchors * 4, 1)  # 4 = (x, y, w, h)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature map [B, C, H, W]
        
        Returns:
            cls_pred: Classification predictions [B, num_anchors * num_classes, H, W]
            loc_pred: Localization predictions [B, num_anchors * 4, H, W]
        """
        cls_pred = self.cls_head(x)
        loc_pred = self.loc_head(x)
        
        return cls_pred, loc_pred


class PhoneDetector(nn.Module):
    """
    Phone Detection Model
    
    전체 Detection 모델을 구성합니다.
    Backbone -> FPN -> Detection Head 구조를 가집니다.
    """
    
    def __init__(
        self,
        backbone_name: str = "efficientnet-b3",
        backbone_pretrained: bool = True,
        num_classes: int = 1,
        use_fpn: bool = True,
        fpn_channels: int = 256,
        num_anchors: int = 3,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Args:
            backbone_name: Backbone 모델 이름 (timm에서 지원하는 모델)
            backbone_pretrained: 사전 학습된 가중치 사용 여부
            num_classes: 검출할 클래스 수
            use_fpn: FPN 사용 여부
            fpn_channels: FPN 출력 채널 수
            num_anchors: 각 스케일당 앵커 개수
            input_size: 입력 이미지 크기 (width, height)
        """
        super(PhoneDetector, self).__init__()
        
        self.num_classes = num_classes
        self.use_fpn = use_fpn
        self.num_anchors = num_anchors
        self.input_size = input_size
        
        # Backbone 생성
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=backbone_pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)  # 여러 레벨의 feature 추출
        )
        
        # Backbone 출력 채널 수 확인
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, input_size[1], input_size[0])
            backbone_features = self.backbone(dummy_input)
            backbone_channels = [f.shape[1] for f in backbone_features]
        
        # FPN
        if use_fpn:
            self.fpn = FPN(backbone_channels, fpn_channels)
            head_in_channels = fpn_channels
        else:
            self.fpn = None
            # FPN을 사용하지 않으면 마지막 레벨의 feature만 사용
            head_in_channels = backbone_channels[-1]
        
        # Detection Head (각 FPN 레벨마다 하나씩)
        if use_fpn:
            num_heads = len(backbone_features)
        else:
            num_heads = 1
        
        self.heads = nn.ModuleList([
            DetectionHead(head_in_channels, num_classes, num_anchors)
            for _ in range(num_heads)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: 입력 이미지 [B, 3, H, W]
            return_features: 중간 feature map도 반환할지 여부
        
        Returns:
            Dictionary containing:
                - 'cls_preds': Classification predictions 리스트
                - 'loc_preds': Localization predictions 리스트
                - 'features': (optional) 중간 feature map들
        """
        # Backbone forward
        backbone_features = self.backbone(x)
        
        # FPN forward
        if self.use_fpn:
            features = self.fpn(backbone_features)
        else:
            features = [backbone_features[-1]]
        
        # Detection Head forward
        cls_preds = []
        loc_preds = []
        
        for feature, head in zip(features, self.heads):
            cls_pred, loc_pred = head(feature)
            cls_preds.append(cls_pred)
            loc_preds.append(loc_pred)
        
        output = {
            'cls_preds': cls_preds,
            'loc_preds': loc_preds
        }
        
        if return_features:
            output['features'] = features
        
        return output
    
    def get_anchors(
        self,
        feature_sizes: List[Tuple[int, int]],
        anchor_scales: List[int],
        anchor_ratios: List[float]
    ) -> torch.Tensor:
        """
        Anchor 생성 (학습 시 사용)
        
        Args:
            feature_sizes: 각 FPN 레벨의 feature map 크기 [(H1, W1), (H2, W2), ...]
            anchor_scales: 각 레벨의 앵커 스케일
            anchor_ratios: 앵커 비율
        
        Returns:
            모든 앵커 좌표 [N, 4] (x_center, y_center, width, height)
        """
        anchors = []
        
        for level, (h, w) in enumerate(feature_sizes):
            scale = anchor_scales[level]
            
            # 각 위치마다 앵커 생성
            for y in range(h):
                for x in range(w):
                    # Feature map 좌표를 원본 이미지 좌표로 변환
                    x_center = (x + 0.5) / w
                    y_center = (y + 0.5) / h
                    
                    for ratio in anchor_ratios:
                        anchor_w = scale * (ratio ** 0.5) / self.input_size[0]
                        anchor_h = scale / (ratio ** 0.5) / self.input_size[1]
                        
                        anchors.append([x_center, y_center, anchor_w, anchor_h])
        
        return torch.tensor(anchors, dtype=torch.float32)


def build_detector(config: Dict) -> PhoneDetector:
    """
    설정 파일로부터 Detector 모델을 생성하는 편의 함수
    
    Args:
        config: 설정 딕셔너리 (YAML 파일에서 로드)
    
    Returns:
        PhoneDetector 인스턴스
    """
    model_config = config['model']
    training_config = config.get('training', {})
    
    model = PhoneDetector(
        backbone_name=model_config['backbone'],
        backbone_pretrained=model_config['backbone_pretrained'],
        num_classes=config['data']['num_classes'],
        use_fpn=model_config['use_fpn'],
        fpn_channels=model_config['fpn_channels'],
        num_anchors=model_config['num_anchors'],
        input_size=(
            model_config['input_size']['width'],
            model_config['input_size']['height']
        )
    )
    
    return model


if __name__ == "__main__":
    # 모델 테스트
    model = PhoneDetector(
        backbone_name="efficientnet-b3",
        num_classes=1,
        input_size=(640, 640)
    )
    
    dummy_input = torch.randn(2, 3, 640, 640)
    output = model(dummy_input)
    
    print("Model output keys:", output.keys())
    print("Number of detection levels:", len(output['cls_preds']))
    print("Cls pred shape:", output['cls_preds'][0].shape)
    print("Loc pred shape:", output['loc_preds'][0].shape)

