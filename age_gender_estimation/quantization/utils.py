"""
Quantization Experiment Utilities

공통 유틸리티 모음:
- 모델 생성 팩토리
- 파라미터 수 계산
- 모델 파일 크기 측정
- 레이턴시 벤치마크
"""

import os
import time
import tempfile
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# 상위 패키지의 AgeGenderNetwork 재활용
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.network import AgeGenderNetwork

logger = logging.getLogger(__name__)


# ============================================================
# 모델 팩토리
# ============================================================

def build_model(
    backbone_name: str,
    age_num_classes: int = 101,
    gender_num_classes: int = 2,
    age_hidden_dim: int = 512,
    gender_hidden_dim: int = 256,
    dropout: float = 0.5,
    pretrained: bool = True,
) -> nn.Module:
    """
    backbone 이름만으로 AgeGenderNetwork를 생성하는 팩토리.

    Args:
        backbone_name: timm 모델 이름 (예: 'efficientnet_b3', 'mobilenetv3_small_100')
        age_num_classes: age head 클래스 수
        gender_num_classes: gender head 클래스 수
        age_hidden_dim: age head hidden dim
        gender_hidden_dim: gender head hidden dim
        dropout: dropout 확률
        pretrained: ImageNet 사전학습 여부

    Returns:
        AgeGenderNetwork 인스턴스
    """
    model = AgeGenderNetwork(
        backbone_name=backbone_name,
        backbone_pretrained=pretrained,
        age_num_classes=age_num_classes,
        age_hidden_dim=age_hidden_dim,
        gender_num_classes=gender_num_classes,
        gender_hidden_dim=gender_hidden_dim,
        dropout=dropout,
    )
    return model


# ============================================================
# 파라미터 수 / 모델 크기
# ============================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    모델의 총 파라미터 수를 반환합니다.

    Args:
        model: PyTorch 모델
        trainable_only: True이면 requires_grad=True인 파라미터만 계산

    Returns:
        파라미터 수 (정수)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    모델을 임시 파일로 저장해 실제 파일 크기(MB)를 반환합니다.
    양자화된 모델도 정확히 측정 가능합니다.

    Args:
        model: PyTorch 모델

    Returns:
        파일 크기 (MB)
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        torch.save(model.state_dict(), tmp_path)
        size_bytes = os.path.getsize(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return size_bytes / (1024 ** 2)


# ============================================================
# 레이턴시 측정
# ============================================================

def measure_latency(
    model: nn.Module,
    input_size: Tuple[int, int] = (224, 224),
    batch_size: int = 1,
    num_warmup: int = 20,
    num_runs: int = 200,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    모델 추론 레이턴시를 측정합니다.

    워밍업 후 num_runs 회 측정해 mean/std/min/max를 반환합니다.

    Args:
        model: 평가할 PyTorch 모델
        input_size: 입력 이미지 크기 (H, W)
        batch_size: 배치 크기
        num_warmup: 워밍업 반복 횟수 (결과에 포함되지 않음)
        num_runs: 실제 측정 반복 횟수
        device: 'cpu' 또는 'cuda'

    Returns:
        dict with keys: mean_ms, std_ms, min_ms, max_ms
    """
    model = model.to(device)
    model.eval()

    h, w = input_size
    dummy_input = torch.randn(batch_size, 3, h, w).to(device)

    # 워밍업
    logger.debug(f"Warming up ({num_warmup} runs)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # 실제 측정
    logger.debug(f"Benchmarking ({num_runs} runs)...")
    times_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000)

    arr = np.array(times_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms":  float(np.std(arr)),
        "min_ms":  float(np.min(arr)),
        "max_ms":  float(np.max(arr)),
    }


# ============================================================
# 포맷 / 출력 유틸리티
# ============================================================

def format_params(num_params: int) -> str:
    """파라미터 수를 'XXX.XM' 또는 'XXX.XK' 형식으로 포맷"""
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def print_model_info(label: str, model: nn.Module, latency: Optional[Dict] = None) -> None:
    """모델 기본 정보를 콘솔에 출력합니다."""
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print(f"\n{'─'*50}")
    print(f"  Model   : {label}")
    print(f"  Params  : {format_params(params)} ({params:,})")
    print(f"  Size    : {size_mb:.2f} MB")
    if latency:
        print(f"  Latency : {latency['mean_ms']:.2f} ± {latency['std_ms']:.2f} ms")
    print(f"{'─'*50}")


def set_seed(seed: int = 42) -> None:
    """재현성을 위해 랜덤 시드를 고정합니다."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")
