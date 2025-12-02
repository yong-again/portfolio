"""
Loss Functions for Object Detection

Classification Loss (Focal Loss)와 Localization Loss (IoU Loss)를 정의합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for Classification
    
    클래스 불균형 문제를 해결하기 위한 Focal Loss입니다.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: 클래스 가중치 (보통 0.25)
            gamma: Focusing parameter (보통 2.0)
            reduction: 'mean', 'sum', 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: 예측 로짓 [N, num_classes]
            targets: 정답 클래스 인덱스 [N] 또는 one-hot [N, num_classes]
        
        Returns:
            Focal loss 값
        """
        # Cross entropy loss 계산
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Probability 계산
        p_t = torch.exp(-ce_loss)
        
        # Focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    두 bounding box 간의 IoU (Intersection over Union)를 계산합니다.
    
    Args:
        box1: [N, 4] (x_center, y_center, width, height) - 정규화된 좌표
        box2: [M, 4] (x_center, y_center, width, height) - 정규화된 좌표
    
    Returns:
        IoU 값 [N, M]
    """
    # 좌표 변환: center -> corner
    def center_to_corner(box):
        x_center, y_center, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    box1_corner = center_to_corner(box1)
    box2_corner = center_to_corner(box2)
    
    # Intersection 계산
    inter_x1 = torch.max(box1_corner[:, 0:1], box2_corner[:, 0].unsqueeze(0))
    inter_y1 = torch.max(box1_corner[:, 1:2], box2_corner[:, 1].unsqueeze(0))
    inter_x2 = torch.min(box1_corner[:, 2:3], box2_corner[:, 2].unsqueeze(0))
    inter_y2 = torch.min(box1_corner[:, 3:4], box2_corner[:, 3].unsqueeze(0))
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Union 계산
    box1_area = (box1_corner[:, 2] - box1_corner[:, 0]) * (box1_corner[:, 3] - box1_corner[:, 1])
    box2_area = (box2_corner[:, 2] - box2_corner[:, 0]) * (box2_corner[:, 3] - box2_corner[:, 1])
    
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
    
    # IoU 계산
    iou = inter_area / (union_area + 1e-7)
    
    return iou


def calculate_giou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Generalized IoU (GIoU)를 계산합니다.
    
    Args:
        box1: [N, 4] (x_center, y_center, width, height)
        box2: [M, 4] (x_center, y_center, width, height)
    
    Returns:
        GIoU 값 [N, M]
    """
    # IoU 계산
    iou = calculate_iou(box1, box2)
    
    # 최소 외접 박스 (enclosing box) 계산
    def center_to_corner(box):
        x_center, y_center, w, h = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    box1_corner = center_to_corner(box1)
    box2_corner = center_to_corner(box2)
    
    # Enclosing box
    enclose_x1 = torch.min(box1_corner[:, 0:1], box2_corner[:, 0].unsqueeze(0))
    enclose_y1 = torch.min(box1_corner[:, 1:2], box2_corner[:, 1].unsqueeze(0))
    enclose_x2 = torch.max(box1_corner[:, 2:3], box2_corner[:, 2].unsqueeze(0))
    enclose_y2 = torch.max(box1_corner[:, 3:4], box2_corner[:, 3].unsqueeze(0))
    
    enclose_area = torch.clamp(enclose_x2 - enclose_x1, min=0) * torch.clamp(enclose_y2 - enclose_y1, min=0)
    
    # Union 계산
    box1_area = (box1_corner[:, 2] - box1_corner[:, 0]) * (box1_corner[:, 3] - box1_corner[:, 1])
    box2_area = (box2_corner[:, 2] - box2_corner[:, 0]) * (box2_corner[:, 3] - box2_corner[:, 1])
    union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - (iou * union_area)
    
    # GIoU 계산
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)
    
    return giou


class IoULoss(nn.Module):
    """
    IoU-based Localization Loss
    """
    
    def __init__(self, iou_type: str = 'giou', reduction: str = 'mean'):
        """
        Args:
            iou_type: 'iou' 또는 'giou'
            reduction: 'mean', 'sum', 'none'
        """
        super(IoULoss, self).__init__()
        self.iou_type = iou_type
        self.reduction = reduction
    
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: 예측된 bounding box [N, 4] (x_center, y_center, width, height)
            target_boxes: 정답 bounding box [N, 4] (x_center, y_center, width, height)
        
        Returns:
            IoU loss 값
        """
        if self.iou_type == 'giou':
            iou = calculate_giou(pred_boxes.unsqueeze(0), target_boxes.unsqueeze(0))
            iou = iou.squeeze(0).diag()  # 대각선 요소만 (각 예측-정답 쌍)
        else:
            iou = calculate_iou(pred_boxes.unsqueeze(0), target_boxes.unsqueeze(0))
            iou = iou.squeeze(0).diag()
        
        # Loss = 1 - IoU
        loss = 1.0 - iou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DetectionLoss(nn.Module):
    """
    전체 Detection Loss (Classification + Localization)
    """
    
    def __init__(
        self,
        cls_loss_type: str = 'focal',
        loc_loss_type: str = 'iou',
        iou_type: str = 'giou',
        cls_weight: float = 1.0,
        loc_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            cls_loss_type: 'focal' 또는 'cross_entropy'
            loc_loss_type: 'iou' 또는 'smooth_l1'
            iou_type: 'iou' 또는 'giou' (loc_loss_type이 'iou'일 경우)
            cls_weight: Classification loss 가중치
            loc_weight: Localization loss 가중치
            focal_alpha: Focal loss alpha
            focal_gamma: Focal loss gamma
        """
        super(DetectionLoss, self).__init__()
        
        # Classification loss
        if cls_loss_type == 'focal':
            self.cls_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss()
        
        # Localization loss
        if loc_loss_type == 'iou':
            self.loc_loss_fn = IoULoss(iou_type=iou_type)
        else:
            self.loc_loss_fn = nn.SmoothL1Loss()
        
        self.cls_weight = cls_weight
        self.loc_weight = loc_weight
    
    def forward(
        self,
        cls_preds: torch.Tensor,
        loc_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        loc_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            cls_preds: Classification 예측 [N, num_classes]
            loc_preds: Localization 예측 [N, 4]
            cls_targets: Classification 정답 [N]
            loc_targets: Localization 정답 [N, 4]
        
        Returns:
            Dictionary containing:
                - 'total_loss': 전체 loss
                - 'cls_loss': Classification loss
                - 'loc_loss': Localization loss
        """
        # Classification loss
        cls_loss = self.cls_loss_fn(cls_preds, cls_targets)
        
        # Localization loss
        loc_loss = self.loc_loss_fn(loc_preds, loc_targets)
        
        # Weighted sum
        total_loss = self.cls_weight * cls_loss + self.loc_weight * loc_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'loc_loss': loc_loss
        }

