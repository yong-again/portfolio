"""
Evaluation Script for Phone Detection

검증 데이터셋으로 모델을 평가하고 메트릭을 계산합니다.
"""

import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader
import sys
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from models.detector import build_detector
from models.utils import load_checkpoint
from preprocess.dataset import PhoneDetectionDataset, collate_fn


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    두 bounding box 간의 IoU를 계산합니다.
    
    Args:
        box1: [x1, y1, x2, y2] 또는 [x_center, y_center, width, height]
        box2: [x1, y1, x2, y2] 또는 [x_center, y_center, width, height]
    
    Returns:
        IoU 값
    """
    # Center format인 경우 corner format으로 변환
    if len(box1) == 4 and box1[2] <= 1.0:  # 정규화된 좌표라고 가정
        x1_1 = box1[0] - box1[2] / 2
        y1_1 = box1[1] - box1[3] / 2
        x2_1 = box1[0] + box1[2] / 2
        y2_1 = box1[1] + box1[3] / 2
    else:
        x1_1, y1_1, x2_1, y2_1 = box1
    
    if len(box2) == 4 and box2[2] <= 1.0:
        x1_2 = box2[0] - box2[2] / 2
        y1_2 = box2[1] - box2[3] / 2
        x2_2 = box2[0] + box2[2] / 2
        y2_2 = box2[1] + box2[3] / 2
    else:
        x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)
    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-7)
    
    return iou


def match_predictions_to_ground_truth(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    예측된 박스를 정답 박스와 매칭합니다.
    
    Args:
        pred_boxes: 예측 박스 [N, 4]
        pred_scores: 예측 점수 [N]
        gt_boxes: 정답 박스 [M, 4]
        iou_threshold: IoU threshold
    
    Returns:
        matched_pred_indices: 매칭된 예측 인덱스
        matched_gt_indices: 매칭된 정답 인덱스
        unmatched_pred_indices: 매칭되지 않은 예측 인덱스
    """
    if len(pred_boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    if len(gt_boxes) == 0:
        return np.array([]), np.array([]), np.arange(len(pred_boxes))
    
    # Score 기준으로 정렬
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # 매칭
    matched_pred = []
    matched_gt = []
    used_gt = set()
    
    for pred_idx, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            matched_pred.append(sorted_indices[pred_idx])
            matched_gt.append(best_gt_idx)
            used_gt.add(best_gt_idx)
    
    unmatched_pred = [i for i in range(len(pred_boxes)) if sorted_indices[i] not in matched_pred]
    
    return np.array(matched_pred), np.array(matched_gt), np.array(unmatched_pred)


def calculate_metrics(
    all_predictions: List[Dict],
    all_ground_truths: List[Dict],
    iou_thresholds: List[float] = [0.5]
) -> Dict[str, float]:
    """
    평가 메트릭을 계산합니다.
    
    Args:
        all_predictions: 모든 예측 결과 리스트
        all_ground_truths: 모든 정답 리스트
        iou_thresholds: IoU threshold 리스트
    
    Returns:
        메트릭 딕셔너리
    """
    metrics = {}
    
    # 각 IoU threshold에 대해 계산
    for iou_thresh in iou_thresholds:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred_dict, gt_dict in zip(all_predictions, all_ground_truths):
            pred_boxes = pred_dict.get('boxes', np.array([]))
            pred_scores = pred_dict.get('scores', np.array([]))
            gt_boxes = gt_dict.get('boxes', np.array([]))
            
            if len(pred_boxes) == 0:
                false_negatives += len(gt_boxes)
                continue
            
            if len(gt_boxes) == 0:
                false_positives += len(pred_boxes)
                continue
            
            # 매칭
            matched_pred, matched_gt, unmatched_pred = match_predictions_to_ground_truth(
                pred_boxes, pred_scores, gt_boxes, iou_thresh
            )
            
            true_positives += len(matched_pred)
            false_positives += len(unmatched_pred)
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        # Precision, Recall, F1 계산
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        
        metrics[f'precision@{iou_thresh:.2f}'] = precision
        metrics[f'recall@{iou_thresh:.2f}'] = recall
        metrics[f'f1@{iou_thresh:.2f}'] = f1
    
    # mAP 계산 (간단화된 버전)
    # 실제로는 더 복잡한 계산이 필요하지만, 여기서는 평균으로 근사
    if len(iou_thresholds) > 0:
        precisions = [metrics.get(f'precision@{t:.2f}', 0.0) for t in iou_thresholds]
        recalls = [metrics.get(f'recall@{t:.2f}', 0.0) for t in iou_thresholds]
        
        metrics['map'] = np.mean(precisions)
        metrics['map50'] = metrics.get('precision@0.50', 0.0)
        metrics['map75'] = metrics.get('precision@0.75', 0.0) if 0.75 in iou_thresholds else 0.0
    
    return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict
) -> Dict[str, float]:
    """
    모델을 평가합니다.
    
    Args:
        model: 평가할 모델
        dataloader: 데이터 로더
        device: 디바이스
        config: 설정 딕셔너리
    
    Returns:
        메트릭 딕셔너리
    """
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            images = batch['images'].to(device)
            bboxes_list = batch['bboxes']
            original_sizes = batch['original_sizes']
            
            # Forward pass
            outputs = model(images)
            
            # 예측 후처리 (간단화된 버전)
            # 실제로는 anchor matching과 NMS가 필요
            batch_size = images.size(0)
            
            for i in range(batch_size):
                # Ground truth
                gt_boxes = bboxes_list[i].cpu().numpy()
                if len(gt_boxes) > 0:
                    gt_boxes_xyxy = gt_boxes[:, 1:5]  # class_id 제외
                else:
                    gt_boxes_xyxy = np.array([])
                
                all_ground_truths.append({
                    'boxes': gt_boxes_xyxy
                })
                
                # Prediction (임시로 더미 데이터)
                # 실제 구현 시 모델 출력을 후처리하여 박스와 점수를 추출해야 함
                all_predictions.append({
                    'boxes': np.array([]),
                    'scores': np.array([])
                })
    
    # 메트릭 계산
    eval_config = config.get('evaluation', {})
    iou_thresholds = eval_config.get('iou_thresholds', [0.5])
    
    metrics = calculate_metrics(all_predictions, all_ground_truths, iou_thresholds)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phone Detection Model')
    parser.add_argument('--config', type=str, default='configs/service_config.yaml',
                       help='Path to config file (default: service_config.yaml)')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save metrics (JSON)')
    
    args = parser.parse_args()
    
    # Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device 설정
    device_config = config['device']
    if device_config['type'] == 'cuda' and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['device_id']}")
    elif device_config['type'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # 모델 로드
    model = build_detector(config)
    model = model.to(device)
    
    checkpoint_info = load_checkpoint(args.weights, model, device=str(device))
    print(f"Loaded model from {args.weights}")
    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Best Score: {checkpoint_info['best_score']:.4f}")
    
    # 데이터셋 로드
    data_path = config['data'][f'{args.split}_path']
    dataset = PhoneDetectionDataset(data_path, config, is_training=False)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn
    )
    
    print(f"Evaluating on {args.split} set: {len(dataset)} samples")
    
    # 평가 수행
    metrics = evaluate_model(model, dataloader, device, config)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"{key:30s}: {value:.4f}")
        else:
            print(f"{key:30s}: {value}")
    
    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()

