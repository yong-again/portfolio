"""
Quantization Benchmark

모든 모델 변형을 통일된 인터페이스로 실행하여 성능을 비교합니다.

비교 대상:
  1. Large FP32    : 대형 모델 (양자화 전 기준선)
  2. Static PTQ    : 대형 모델 + Static 사후 양자화
  3. Dynamic PTQ   : 대형 모델 + Dynamic 사후 양자화
  4. QAT           : 대형 모델 + 사전 양자화 (학습 데이터 필요)
  5. Small FP32    : 각 경량 모델 (양자화 없는 기준선)

측정 지표:
  - 파라미터 수 (M)
  - 모델 파일 크기 (MB)
  - 평균 추론 레이턴시 (ms) ± std
  - 대형 FP32 대비 압축률 / 속도 향상
"""

import logging
import os
from typing import Dict, List, Optional

import torch
import pandas as pd

from .config import ExperimentConfig
from .utils import (
    build_model,
    count_parameters,
    format_params,
    get_model_size_mb,
    measure_latency,
    set_seed,
)
from .ptq import run_ptq_experiment
from .qat import run_qat

logger = logging.getLogger(__name__)


# ============================================================
# 단일 모델 결과 수집
# ============================================================

def _profile_model(
    label: str,
    model: torch.nn.Module,
    bench_cfg,
    device: str,
) -> Dict:
    """단일 모델의 파라미터 수 / 크기 / 레이턴시를 측정하여 dict 반환"""
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    latency = measure_latency(
        model,
        input_size=bench_cfg.input_size,
        batch_size=bench_cfg.batch_size,
        num_warmup=bench_cfg.num_warmup_runs,
        num_runs=bench_cfg.num_benchmark_runs,
        device=device,
    )
    return {
        "label": label,
        "params": params,
        "params_str": format_params(params),
        "size_mb": round(size_mb, 2),
        "latency_mean_ms": round(latency["mean_ms"], 3),
        "latency_std_ms": round(latency["std_ms"], 3),
        "latency_min_ms": round(latency["min_ms"], 3),
    }


# ============================================================
# 비교 테이블 출력
# ============================================================

def _print_comparison_table(rows: List[Dict], baseline_size_mb: float, baseline_ms: float) -> None:
    """결과를 정렬된 비교 테이블로 출력합니다."""
    # 압축률 / 속도 향상 계산
    for r in rows:
        r["compression_ratio"] = round(baseline_size_mb / r["size_mb"], 2) if r["size_mb"] > 0 else 0
        r["speedup"] = round(baseline_ms / r["latency_mean_ms"], 2) if r["latency_mean_ms"] > 0 else 0

    print("\n" + "=" * 95)
    print("  Quantization Experiment Results")
    print("=" * 95)
    header = (
        f"{'Model':<30} {'Params':>10} {'Size(MB)':>10} "
        f"{'Latency(ms)':>14} {'Compress':>10} {'Speedup':>8}"
    )
    print(header)
    print("-" * 95)
    for r in rows:
        lat_str = f"{r['latency_mean_ms']:.2f} ± {r['latency_std_ms']:.2f}"
        print(
            f"  {r['label']:<28} {r['params_str']:>10} {r['size_mb']:>10.2f} "
            f"{lat_str:>14} {r['compression_ratio']:>9.2f}x {r['speedup']:>7.2f}x"
        )
    print("=" * 95)
    print(f"  * Compression/Speedup 기준: {rows[0]['label']} (FP32 대형 모델)")
    print()


# ============================================================
# 메인 벤치마크 함수
# ============================================================

def run_benchmark(
    cfg: ExperimentConfig,
    calibration_loader=None,
    qat_train_loader=None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    전체 양자화 실험 벤치마크를 실행합니다.

    Args:
        cfg: ExperimentConfig 인스턴스
        calibration_loader: Static PTQ 캘리브레이션용 DataLoader (없으면 Static PTQ 건너뜀)
        qat_train_loader: QAT fine-tuning용 DataLoader (없으면 QAT 건너뜀)
        dry_run: True이면 더미 입력으로 import/shape만 확인

    Returns:
        결과 DataFrame (레이블, 파라미터 수, 크기, 레이턴시 등)
    """
    set_seed(cfg.seed)
    device = cfg.benchmark.device
    bench_cfg = cfg.benchmark
    model_cfg = cfg.model

    rows: List[Dict] = []

    # ── 1. Large FP32 기준선 ────────────────────────────────
    logger.info(f"\n▶ [1/N] Large FP32 기준선: {model_cfg.large_backbone}")
    large_model = build_model(
        backbone_name=model_cfg.large_backbone,
        age_num_classes=model_cfg.age_num_classes,
        gender_num_classes=model_cfg.gender_num_classes,
        age_hidden_dim=model_cfg.age_hidden_dim,
        gender_hidden_dim=model_cfg.gender_hidden_dim,
        dropout=model_cfg.dropout,
        pretrained=model_cfg.pretrained,
    )
    large_model.eval()

    large_row = _profile_model(
        label=f"{model_cfg.large_backbone} (FP32)",
        model=large_model,
        bench_cfg=bench_cfg,
        device=device,
    )
    rows.append(large_row)
    baseline_size = large_row["size_mb"]
    baseline_ms   = large_row["latency_mean_ms"]
    logger.info(f"   완료: {large_row['size_mb']:.2f} MB, {large_row['latency_mean_ms']:.2f} ms")

    # ── 2. Static / Dynamic PTQ ─────────────────────────────
    logger.info(f"\n▶ [2/N] PTQ 실험: {model_cfg.large_backbone}")
    ptq_results = run_ptq_experiment(
        model=large_model,
        cfg=cfg.ptq,
        calibration_loader=calibration_loader,
        benchmark_cfg=bench_cfg,
        device=device,
    )

    for ptq_key, ptq_label in [
        ("static_ptq",  f"{model_cfg.large_backbone} (Static PTQ INT8)"),
        ("dynamic_ptq", f"{model_cfg.large_backbone} (Dynamic PTQ INT8)"),
    ]:
        if ptq_key in ptq_results:
            r = ptq_results[ptq_key]
            lat = r["latency"] or {"mean_ms": 0, "std_ms": 0, "min_ms": 0}
            rows.append({
                "label":             ptq_label,
                "params":            r["params"],
                "params_str":        format_params(r["params"]),
                "size_mb":           round(r["size_mb"], 2),
                "latency_mean_ms":   round(lat["mean_ms"], 3),
                "latency_std_ms":    round(lat["std_ms"], 3),
                "latency_min_ms":    round(lat["min_ms"], 3),
            })

    # ── 3. QAT ──────────────────────────────────────────────
    if qat_train_loader is not None:
        logger.info(f"\n▶ [3/N] QAT 실험: {model_cfg.large_backbone}")
        # QAT용 새 FP32 모델
        qat_base = build_model(
            backbone_name=model_cfg.large_backbone,
            age_num_classes=model_cfg.age_num_classes,
            gender_num_classes=model_cfg.gender_num_classes,
            age_hidden_dim=model_cfg.age_hidden_dim,
            gender_hidden_dim=model_cfg.gender_hidden_dim,
            dropout=model_cfg.dropout,
            pretrained=model_cfg.pretrained,
        )
        qat_model, qat_info = run_qat(
            model=qat_base,
            cfg=cfg.qat,
            train_loader=qat_train_loader,
            device=device,
            benchmark_cfg=bench_cfg,
        )
        lat = qat_info["latency"] or {"mean_ms": 0, "std_ms": 0, "min_ms": 0}
        rows.append({
            "label":           f"{model_cfg.large_backbone} (QAT INT8)",
            "params":          qat_info["params"],
            "params_str":      format_params(qat_info["params"]),
            "size_mb":         round(qat_info["size_mb"], 2),
            "latency_mean_ms": round(lat["mean_ms"], 3),
            "latency_std_ms":  round(lat["std_ms"], 3),
            "latency_min_ms":  round(lat["min_ms"], 3),
        })
    else:
        logger.info("  QAT 건너뜀 (qat_train_loader 없음)")

    # ── 4. 경량 모델 FP32 기준선 ────────────────────────────
    for i, small_name in enumerate(model_cfg.small_backbones, start=1):
        logger.info(f"\n▶ [경량 {i}] Small FP32: {small_name}")
        small_model = build_model(
            backbone_name=small_name,
            age_num_classes=model_cfg.age_num_classes,
            gender_num_classes=model_cfg.gender_num_classes,
            age_hidden_dim=model_cfg.age_hidden_dim,
            gender_hidden_dim=model_cfg.gender_hidden_dim,
            dropout=model_cfg.dropout,
            pretrained=model_cfg.pretrained,
        )
        small_model.eval()
        row = _profile_model(
            label=f"{small_name} (FP32)",
            model=small_model,
            bench_cfg=bench_cfg,
            device=device,
        )
        rows.append(row)
        logger.info(f"   완료: {row['size_mb']:.2f} MB, {row['latency_mean_ms']:.2f} ms")
        del small_model

    # ── 5. 결과 출력 및 저장 ────────────────────────────────
    _print_comparison_table(rows, baseline_size, baseline_ms)

    df = pd.DataFrame(rows)

    # 압축률 / 속도 향상 열 추가
    df["compression_ratio"] = (baseline_size / df["size_mb"]).round(2)
    df["speedup"] = (baseline_ms / df["latency_mean_ms"]).round(2)

    # CSV 저장
    os.makedirs(os.path.dirname(bench_cfg.output_csv) or ".", exist_ok=True)
    df.to_csv(bench_cfg.output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"결과 저장: {bench_cfg.output_csv}")

    return df
