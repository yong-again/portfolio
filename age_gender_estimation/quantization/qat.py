"""
Quantization-Aware Training (QAT) - 사전 양자화

FP32 모델에 Fake-quantize 노드를 삽입한 후 소수 epoch fine-tuning하여
양자화 오차를 학습 과정에서 보상합니다. PTQ 대비 정확도 손실이 적습니다.

핵심 흐름:
  1. FP32 모델 로드
  2. prepare_qat() → Fake-quantize 노드 삽입
  3. Fine-tuning (낮은 LR, 소수 epoch)
     - 지정 epoch 이후 observer 고정 (freeze_observer_epoch)
     - 지정 epoch 이후 BN eval 모드 고정 (freeze_bn_epoch)
  4. convert() → INT8 모델
"""

import copy
import logging
import os
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.quantization

from .config import QATConfig, BenchmarkConfig
from .utils import count_parameters, get_model_size_mb, measure_latency

logger = logging.getLogger(__name__)


# ============================================================
# 손실 함수 (Age + Gender 멀티태스크)
# ============================================================

def _compute_loss(
    outputs: Dict[str, torch.Tensor],
    age_labels: torch.Tensor,
    gender_labels: torch.Tensor,
    age_weight: float = 1.0,
    gender_weight: float = 1.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Age cross-entropy + Gender cross-entropy 가중합 계산.

    Returns:
        (total_loss, age_loss_val, gender_loss_val)
    """
    criterion = nn.CrossEntropyLoss()

    age_loss = criterion(outputs["age_logits"], age_labels)
    gender_loss = criterion(outputs["gender_logits"], gender_labels)
    total = age_weight * age_loss + gender_weight * gender_loss

    return total, age_loss.item(), gender_loss.item()


# ============================================================
# QAT 학습 1 epoch
# ============================================================

def _train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    cfg: QATConfig,
) -> Dict[str, float]:
    """1 epoch QAT fine-tuning 수행"""
    model.train()

    # Observer 고정 (freeze_observer_epoch 이후)
    if epoch >= cfg.freeze_observer_epoch:
        model.apply(torch.quantization.disable_observer)
        if epoch == cfg.freeze_observer_epoch:
            logger.info(f"  [Epoch {epoch}] Observer 고정 (통계 수집 종료)")

    # BN eval 모드 고정 (freeze_bn_epoch 이후)
    if epoch >= cfg.freeze_bn_epoch:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        if epoch == cfg.freeze_bn_epoch:
            logger.info(f"  [Epoch {epoch}] BatchNorm eval 모드 고정")

    total_loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            images, age_labels, gender_labels = batch[0], batch[1], batch[2]
        else:
            # 레이블 없는 경우 더미 레이블 (dry-run 대비)
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            age_labels = torch.zeros(images.size(0), dtype=torch.long)
            gender_labels = torch.zeros(images.size(0), dtype=torch.long)

        images = images.to(device)
        age_labels = age_labels.to(device)
        gender_labels = gender_labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss, _, _ = _compute_loss(outputs, age_labels, gender_labels)
        loss.backward()
        optimizer.step()

        total_loss_sum += loss.item()
        n_batches += 1

    return {"loss": total_loss_sum / max(n_batches, 1)}


# ============================================================
# QAT 메인 함수
# ============================================================

def run_qat(
    model: nn.Module,
    cfg: QATConfig,
    train_loader,
    device: str = "cpu",
    benchmark_cfg: Optional[BenchmarkConfig] = None,
) -> Tuple[nn.Module, Dict]:
    """
    Quantization-Aware Training 수행.

    Args:
        model: 학습 완료된 FP32 AgeGenderNetwork
        cfg: QATConfig 인스턴스
        train_loader: QAT fine-tuning용 DataLoader
        device: 대상 디바이스 ('cpu' 권장, QAT convert는 CPU 필요)
        benchmark_cfg: BenchmarkConfig (레이턴시 측정용, Optional)

    Returns:
        (int8_model, result_dict)
        result_dict = {
            "params": int,
            "size_mb": float,
            "latency": dict or None,
            "epoch_losses": list[float],
        }
    """
    logger.info("== QAT 시작 ==")

    # 원본 보존 deep copy
    q_model = copy.deepcopy(model)
    q_model.train()
    q_model.to(device)

    # ── Step 1: qconfig 설정 ────────────────────────────────
    torch.backends.quantized.engine = cfg.backend
    q_model.qconfig = torch.quantization.get_default_qat_qconfig(cfg.backend)
    logger.info(f"  Backend: {cfg.backend}")

    # ── Step 2: prepare_qat (Fake-quantize 노드 삽입) ───────
    torch.quantization.prepare_qat(q_model, inplace=True)
    logger.info("  Fake-quantize 노드 삽입 완료")

    # ── Step 3: Optimizer 설정 ──────────────────────────────
    if cfg.optimizer == "adamw":
        optimizer = optim.AdamW(
            q_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer == "adam":
        optimizer = optim.Adam(
            q_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            q_model.parameters(), lr=cfg.lr,
            momentum=0.9, weight_decay=cfg.weight_decay
        )

    # ── Step 4: LR Scheduler ────────────────────────────────
    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )
    elif cfg.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(cfg.epochs // 3, 1), gamma=0.3
        )
    else:
        scheduler = None

    # ── Step 5: Fine-tuning ─────────────────────────────────
    epoch_losses = []
    logger.info(f"  Fine-tuning 시작 (lr={cfg.lr}, epochs={cfg.epochs})")

    for epoch in range(cfg.epochs):
        metrics = _train_one_epoch(q_model, train_loader, optimizer, device, epoch, cfg)
        epoch_losses.append(metrics["loss"])

        if scheduler:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"  [Epoch {epoch + 1:>3}/{cfg.epochs}] "
            f"loss={metrics['loss']:.4f}  lr={current_lr:.2e}"
        )

    # ── Step 6: convert (FP32 + Fake-Q → INT8) ─────────────
    q_model.eval()
    q_model.to("cpu")   # convert는 반드시 CPU 에서 수행
    torch.quantization.convert(q_model, inplace=True)
    logger.info("  모델 변환 완료 (FP32 → INT8, QAT)")

    # ── Step 7: 벤치마크 ────────────────────────────────────
    lat = None
    if benchmark_cfg:
        lat = measure_latency(
            q_model,
            input_size=benchmark_cfg.input_size,
            batch_size=benchmark_cfg.batch_size,
            num_warmup=benchmark_cfg.num_warmup_runs,
            num_runs=benchmark_cfg.num_benchmark_runs,
            device="cpu",
        )

    result = {
        "params": count_parameters(q_model),
        "size_mb": get_model_size_mb(q_model),
        "latency": lat,
        "epoch_losses": epoch_losses,
    }

    logger.info(
        f"  QAT 완료: size={result['size_mb']:.2f}MB  "
        f"final_loss={epoch_losses[-1]:.4f}"
    )

    return q_model, result
