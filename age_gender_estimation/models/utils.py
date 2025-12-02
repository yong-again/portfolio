"""
Model Utility Functions

모델 관련 유틸리티 함수들 (체크포인트 저장/로드, 모델 초기화 등)
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Any


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    체크포인트 파일에서 모델 가중치와 옵티마이저 상태를 로드합니다.
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        model: 로드할 모델 인스턴스
        optimizer: (optional) 로드할 옵티마이저 인스턴스
        device: 디바이스 ('cuda', 'cpu', 'mps')
    
    Returns:
        체크포인트 정보 딕셔너리 (epoch, best_score, config 등)
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 모델 가중치 로드
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 직접 state_dict가 저장된 경우
        model.load_state_dict(checkpoint)
    
    # 옵티마이저 상태 로드
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    result = {
        'epoch': checkpoint.get('epoch', 0),
        'best_score': checkpoint.get('best_score', 0.0),
        'config': checkpoint.get('config', None),
        'checkpoint_path': str(checkpoint_path)
    }
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {result['epoch']}")
    print(f"  Best Score: {result['best_score']:.4f}")
    
    return result


def save_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    epoch: int,
    best_score: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[Dict] = None,
    is_best: bool = False,
    filename: Optional[str] = None
) -> str:
    """
    체크포인트를 저장합니다.
    
    Args:
        checkpoint_dir: 체크포인트 저장 디렉토리
        model: 저장할 모델 인스턴스
        epoch: 현재 epoch
        best_score: 현재까지의 최고 점수
        optimizer: (optional) 저장할 옵티마이저 인스턴스
        config: (optional) 설정 딕셔너리
        is_best: 현재 모델이 최고 성능인지 여부
        filename: (optional) 저장할 파일명 (지정하지 않으면 자동 생성)
    
    Returns:
        저장된 체크포인트 파일 경로
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_score': best_score,
        'config': config
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # 파일명 생성
    if filename is None:
        if is_best:
            filename = 'best_model.pt'
        else:
            filename = f'checkpoint_epoch_{epoch:04d}.pt'
    
    checkpoint_path = checkpoint_dir / filename
    torch.save(checkpoint, checkpoint_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")
    
    return str(checkpoint_path)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    모델의 파라미터 개수를 계산합니다.
    
    Args:
        model: 파라미터를 계산할 모델
    
    Returns:
        {'total': 총 파라미터 수, 'trainable': 학습 가능한 파라미터 수}
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    모델의 크기를 MB 단위로 계산합니다.
    
    Args:
        model: 크기를 계산할 모델
    
    Returns:
        모델 크기 (MB)
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return size_all_mb

