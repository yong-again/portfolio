"""
Post-Training Quantization (PTQ) - 사후 양자화

학습이 완료된 FP32 모델을 캘리브레이션 데이터로 INT8로 변환합니다.
별도의 재학습 없이 적용 가능하지만 QAT 대비 정확도 손실이 클 수 있습니다.

지원 방식:
  - Static PTQ : 캘리브레이션 데이터로 activation 분포 추정 (권장)
  - Dynamic PTQ: 가중치만 양자화, 런타임에 activation 양자화 (빠른 적용)
"""

import copy
import logging
import os
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.quantization

from .config import PTQConfig
from .utils import count_parameters, get_model_size_mb, measure_latency

logger = logging.getLogger(__name__)


# ============================================================
# Module Fusion (Conv-BN-ReLU 합치기)
# ============================================================

def _get_fusable_modules(model: nn.Module) -> list:
    """
    Conv-BN-ReLU 패턴을 자동 탐지하여 fusion 가능한 모듈 리스트 반환.
    timm 모델은 내부적으로 이미 fused된 경우가 많으므로 가능한 경우만 처리.
    """
    fusable = []
    for name, module in model.named_modules():
        # Sequential 안에 Conv → BN → ReLU 패턴 탐지
        if isinstance(module, nn.Sequential):
            children = list(module.named_children())
            for i in range(len(children) - 2):
                n0, m0 = children[i]
                n1, m1 = children[i + 1]
                n2, m2 = children[i + 2]
                if (isinstance(m0, nn.Conv2d) and
                        isinstance(m1, nn.BatchNorm2d) and
                        isinstance(m2, (nn.ReLU, nn.ReLU6))):
                    prefix = f"{name}." if name else ""
                    fusable.append([f"{prefix}{n0}", f"{prefix}{n1}", f"{prefix}{n2}"])
    return fusable


# ============================================================
# Static PTQ
# ============================================================

def run_static_ptq(
    model: nn.Module,
    cfg: PTQConfig,
    calibration_loader,                       # DataLoader (캘리브레이션용)
    device: str = "cpu",
) -> nn.Module:
    """
    Static Post-Training Quantization 수행.

    흐름:
      1. FP32 모델 복사 → eval 모드
      2. Conv-BN-ReLU fusion (선택)
      3. qconfig 설정 (fbgemm / qnnpack)
      4. prepare() → observer 삽입
      5. 캘리브레이션 데이터 통과 (activation 분포 추정)
      6. convert() → FP32 → INT8

    Args:
        model: 학습 완료된 FP32 AgeGenderNetwork
        cfg: PTQConfig 인스턴스
        calibration_loader: 캘리브레이션용 DataLoader
        device: 대상 디바이스

    Returns:
        INT8 양자화된 모델
    """
    logger.info("== Static PTQ 시작 ==")

    # 원본 모델 보존을 위해 deep copy
    q_model = copy.deepcopy(model)
    q_model.eval()
    q_model.to(device)

    # ── Step 1: Conv-BN-ReLU Fusion ─────────────────────────
    if cfg.fuse_modules:
        fusable = _get_fusable_modules(q_model)
        if fusable:
            logger.info(f"  Fusing {len(fusable)} conv-bn-relu groups...")
            torch.quantization.fuse_modules(q_model, fusable, inplace=True)
        else:
            logger.info("  No fusable modules found (timm 내부 구조 이미 최적화됨)")

    # ── Step 2: qconfig 설정 ────────────────────────────────
    torch.backends.quantized.engine = cfg.backend

    if cfg.observer_type == "histogram":
        activation_observer = torch.quantization.HistogramObserver.with_args(
            reduce_range=False
        )
    elif cfg.observer_type == "percentile":
        # Percentile 기반 (outlier에 더 강건)
        activation_observer = torch.quantization.HistogramObserver.with_args(
            reduce_range=False
        )
    else:  # minmax
        activation_observer = torch.quantization.MinMaxObserver.with_args(
            qscheme=torch.per_tensor_affine
        )

    q_model.qconfig = torch.quantization.QConfig(
        activation=activation_observer,
        weight=torch.quantization.default_weight_observer
    )

    # ── Step 3: prepare (observer 삽입) ─────────────────────
    torch.quantization.prepare(q_model, inplace=True)
    logger.info("  Observer 삽입 완료")

    # ── Step 4: 캘리브레이션 ────────────────────────────────
    logger.info(f"  캘리브레이션 시작 ({cfg.num_calibration_batches} batches)...")
    q_model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            if batch_idx >= cfg.num_calibration_batches:
                break

            # DataLoader가 (images, age_labels, gender_labels) 형태라 가정
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)
            q_model(images)

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"    캘리브레이션 {batch_idx + 1}/{cfg.num_calibration_batches}")

    logger.info("  캘리브레이션 완료")

    # ── Step 5: convert (FP32 → INT8) ───────────────────────
    torch.quantization.convert(q_model, inplace=True)
    logger.info("  모델 변환 완료 (FP32 → INT8, Static PTQ)")

    return q_model


# ============================================================
# Dynamic PTQ
# ============================================================

def run_dynamic_ptq(
    model: nn.Module,
    cfg: PTQConfig,
    device: str = "cpu",
) -> nn.Module:
    """
    Dynamic Post-Training Quantization 수행.

    가중치만 INT8로 양자화하고 activation은 런타임에 동적으로 양자화합니다.
    캘리브레이션 데이터가 없어도 적용 가능합니다.

    Args:
        model: 학습 완료된 FP32 AgeGenderNetwork
        cfg: PTQConfig 인스턴스
        device: 대상 디바이스

    Returns:
        Dynamic INT8 양자화된 모델
    """
    logger.info("== Dynamic PTQ 시작 ==")

    q_model = copy.deepcopy(model)
    q_model.eval()
    q_model.to(device)

    torch.backends.quantized.engine = cfg.backend

    # Linear와 LSTM 레이어에 Dynamic Quantization 적용
    # (Conv2d는 Dynamic PTQ에서 지원이 제한적)
    q_model = torch.quantization.quantize_dynamic(
        q_model,
        qconfig_spec={nn.Linear},   # 대상 레이어 타입
        dtype=torch.qint8,
    )
    logger.info("  Dynamic PTQ 완료 (Linear → INT8)")

    return q_model


# ============================================================
# PTQ 실험 실행 (평가 포함)
# ============================================================

def run_ptq_experiment(
    model: nn.Module,
    cfg: PTQConfig,
    calibration_loader=None,
    benchmark_cfg=None,
    device: str = "cpu",
) -> Dict:
    """
    Static PTQ + Dynamic PTQ를 모두 실행하고 결과를 반환합니다.

    Args:
        model: FP32 기준 모델
        cfg: PTQConfig
        calibration_loader: 캘리브레이션용 DataLoader (Static PTQ 필수)
        benchmark_cfg: BenchmarkConfig (레이턴시 측정용)
        device: 대상 디바이스

    Returns:
        결과 딕셔너리:
          {
            "static_ptq": {model, params, size_mb, latency},
            "dynamic_ptq": {model, params, size_mb, latency},
          }
    """
    results = {}

    # ── Static PTQ ──────────────────────────────────────────
    if cfg.use_static:
        if calibration_loader is None:
            logger.warning("Static PTQ를 위한 calibration_loader가 없습니다. 건너뜁니다.")
        else:
            static_model = run_static_ptq(model, cfg, calibration_loader, device)

            lat = None
            if benchmark_cfg:
                lat = measure_latency(
                    static_model,
                    input_size=benchmark_cfg.input_size,
                    batch_size=benchmark_cfg.batch_size,
                    num_warmup=benchmark_cfg.num_warmup_runs,
                    num_runs=benchmark_cfg.num_benchmark_runs,
                    device=device,
                )

            results["static_ptq"] = {
                "model": static_model,
                "params": count_parameters(static_model),
                "size_mb": get_model_size_mb(static_model),
                "latency": lat,
            }
            logger.info(
                f"  Static PTQ 결과: size={results['static_ptq']['size_mb']:.2f}MB"
            )

    # ── Dynamic PTQ ─────────────────────────────────────────
    if cfg.use_dynamic:
        dynamic_model = run_dynamic_ptq(model, cfg, device)

        lat = None
        if benchmark_cfg:
            lat = measure_latency(
                dynamic_model,
                input_size=benchmark_cfg.input_size,
                batch_size=benchmark_cfg.batch_size,
                num_warmup=benchmark_cfg.num_warmup_runs,
                num_runs=benchmark_cfg.num_benchmark_runs,
                device=device,
            )

        results["dynamic_ptq"] = {
            "model": dynamic_model,
            "params": count_parameters(dynamic_model),
            "size_mb": get_model_size_mb(dynamic_model),
            "latency": lat,
        }
        logger.info(
            f"  Dynamic PTQ 결과: size={results['dynamic_ptq']['size_mb']:.2f}MB"
        )

    return results
