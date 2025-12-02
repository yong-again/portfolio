"""
Evaluation Script for Age & Gender Estimation

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
from typing import Dict, List
import json
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from models.network import build_network
from models.utils import load_checkpoint
from models.age_head import bin_to_age_range, create_age_bins
from models.gender_head import class_to_gender
from preprocess.dataset import AgeGenderDataset, collate_fn


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
    
    all_age_preds = []
    all_age_targets = []
    all_gender_preds = []
    all_gender_targets = []
    all_age_probs = []
    all_gender_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch in pbar:
            images = batch['images'].to(device)
            age_bins = batch['age_bins'].to(device)
            genders = batch['genders'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            age_logits = outputs['age_logits']
            gender_logits = outputs['gender_logits']
            
            # Predictions
            age_preds = torch.argmax(age_logits, dim=1)
            gender_preds = torch.argmax(gender_logits, dim=1)
            
            # Probabilities
            age_probs = torch.softmax(age_logits, dim=1)
            gender_probs = torch.softmax(gender_logits, dim=1)
            
            # Collect results
            all_age_preds.extend(age_preds.cpu().numpy())
            all_age_targets.extend(age_bins.cpu().numpy())
            all_gender_preds.extend(gender_preds.cpu().numpy())
            all_gender_targets.extend(genders.cpu().numpy())
            all_age_probs.extend(age_probs.cpu().numpy())
            all_gender_probs.extend(gender_probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_age_preds = np.array(all_age_preds)
    all_age_targets = np.array(all_age_targets)
    all_gender_preds = np.array(all_gender_preds)
    all_gender_targets = np.array(all_gender_targets)
    all_age_probs = np.array(all_age_probs)
    all_gender_probs = np.array(all_gender_probs)
    
    # Age metrics
    age_accuracy = accuracy_score(all_age_targets, all_age_preds)
    
    # Top-3 accuracy for age
    age_top3_correct = 0
    for i in range(len(all_age_targets)):
        top3_preds = np.argsort(all_age_probs[i])[-3:][::-1]
        if all_age_targets[i] in top3_preds:
            age_top3_correct += 1
    age_top3_accuracy = age_top3_correct / len(all_age_targets)
    
    # Gender metrics
    gender_accuracy = accuracy_score(all_gender_targets, all_gender_preds)
    
    # Per-bin accuracy for age
    age_config = config['model']['age']
    if 'bins' in age_config:
        age_bins = age_config['bins']
    else:
        age_bins = create_age_bins(
            num_bins=age_config['num_bins'],
            min_age=age_config['min_age'],
            max_age=age_config['max_age']
        )
    
    per_bin_accuracy = {}
    for bin_idx in range(len(age_bins)):
        bin_mask = all_age_targets == bin_idx
        if np.sum(bin_mask) > 0:
            bin_acc = accuracy_score(
                all_age_targets[bin_mask],
                all_age_preds[bin_mask]
            )
            per_bin_accuracy[f'bin_{bin_idx}'] = bin_acc
    
    metrics = {
        'age_accuracy': age_accuracy,
        'age_top3_accuracy': age_top3_accuracy,
        'gender_accuracy': gender_accuracy,
        'per_bin_accuracy': per_bin_accuracy
    }
    
    # Confusion matrices
    age_cm = confusion_matrix(all_age_targets, all_age_preds)
    gender_cm = confusion_matrix(all_gender_targets, all_gender_preds)
    
    metrics['age_confusion_matrix'] = age_cm.tolist()
    metrics['gender_confusion_matrix'] = gender_cm.tolist()
    
    return metrics, age_cm, gender_cm


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], title: str, save_path: str):
    """
    Confusion matrix를 시각화합니다.
    
    Args:
        cm: Confusion matrix
        labels: 클래스 라벨 리스트
        title: 그래프 제목
        save_path: 저장 경로
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Age & Gender Estimation Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save metrics (JSON)')
    parser.add_argument('--save-confusion-matrix', action='store_true',
                       help='Save confusion matrix plots')
    
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
    model = build_network(config)
    model = model.to(device)
    
    checkpoint_info = load_checkpoint(args.weights, model, device=str(device))
    print(f"Loaded model from {args.weights}")
    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Best Score: {checkpoint_info['best_score']:.4f}")
    
    # 데이터셋 로드
    data_path = config['data'][f'{args.split}_path']
    dataset = AgeGenderDataset(data_path, config, is_training=False)
    
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
    metrics, age_cm, gender_cm = evaluate_model(model, dataloader, device, config)
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Age Accuracy: {metrics['age_accuracy']:.4f}")
    print(f"Age Top-3 Accuracy: {metrics['age_top3_accuracy']:.4f}")
    print(f"Gender Accuracy: {metrics['gender_accuracy']:.4f}")
    
    if metrics['per_bin_accuracy']:
        print("\nPer-bin Age Accuracy:")
        for bin_name, acc in metrics['per_bin_accuracy'].items():
            print(f"  {bin_name}: {acc:.4f}")
    
    # Confusion matrix 저장
    if args.save_confusion_matrix or config['evaluation'].get('save_confusion_matrix', False):
        output_dir = Path(config['paths']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Age bins 라벨 생성
        age_config = config['model']['age']
        if 'bins' in age_config:
            age_bins = age_config['bins']
        else:
            age_bins = create_age_bins(
                num_bins=age_config['num_bins'],
                min_age=age_config['min_age'],
                max_age=age_config['max_age']
            )
        
        age_labels = [f"{b[0]}-{b[1]}" for b in age_bins]
        gender_labels = ['Male', 'Female']
        
        plot_confusion_matrix(
            age_cm,
            age_labels,
            'Age Confusion Matrix',
            str(output_dir / 'age_confusion_matrix.png')
        )
        
        plot_confusion_matrix(
            gender_cm,
            gender_labels,
            'Gender Confusion Matrix',
            str(output_dir / 'gender_confusion_matrix.png')
        )
        
        print(f"\nConfusion matrices saved to {output_dir}")
    
    # 결과 저장
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()

