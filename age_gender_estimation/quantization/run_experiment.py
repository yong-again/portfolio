"""
실험 메인 진입점 (CLI)

사용법:
    # 벤치마크만 (더미 입력, 데이터 불필요)
    python -m quantization.run_experiment --mode benchmark --dry-run

    # PTQ만 (캘리브레이션 데이터 필요)
    python -m quantization.run_experiment --mode ptq --data data/val

    # QAT만 (학습 데이터 필요)
    python -m quantization.run_experiment --mode qat --data data/train

    # 전체 실험
    python -m quantization.run_experiment --mode all --data data/val

하이퍼파라미터 오버라이드 예시:
    python -m quantization.run_experiment --mode benchmark \\
        --large-backbone efficientnet_b4 \\
        --qat-lr 5e-5 \\
        --qat-epochs 5 \\
        --calib-batches 50
"""

import argparse
import logging
import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

# 상위 패키지 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantization.config import ExperimentConfig
from quantization.benchmark import run_benchmark
from quantization.utils import set_seed


# ============================================================
# 더미 DataLoader (dry-run 전용)
# ============================================================

def _make_dummy_loader(
    batch_size: int = 4,
    n_batches: int = 5,
    input_size: tuple = (224, 224),
    age_classes: int = 101,
    gender_classes: int = 2,
) -> DataLoader:
    """
    실제 데이터 없이 형상 확인용 더미 DataLoader를 생성합니다.
    dry-run / import 테스트에서 사용됩니다.
    """
    h, w = input_size
    n = batch_size * n_batches
    images = torch.randn(n, 3, h, w)
    age_labels = torch.randint(0, age_classes, (n,))
    gender_labels = torch.randint(0, gender_classes, (n,))
    dataset = TensorDataset(images, age_labels, gender_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================
# 실제 DataLoader (데이터 경로 지정 시)
# ============================================================

def _build_data_loaders(
    data_dir: str,
    cfg: ExperimentConfig,
) -> tuple:
    """
    data_dir 경로에서 캘리브레이션/QAT 데이터 로더를 빌드합니다.
    기존 프로젝트의 데이터셋 클래스가 없을 경우 더미를 반환합니다.

    Returns:
        (calibration_loader, qat_loader)
    """
    try:
        # 기존 프로젝트 데이터셋 클래스 재활용 시도
        from preprocess.transforms import get_train_transforms, get_val_transforms
        from torch.utils.data import Dataset

        # ── 간단한 ImageFolder 기반 래퍼 ───────────────────
        from torchvision import datasets, transforms as T

        val_tf = get_val_transforms(
            input_size=cfg.benchmark.input_size[0]
        ) if hasattr(get_val_transforms, "__call__") else T.Compose([
            T.Resize(cfg.benchmark.input_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # 실제 데이터가 있으면 ImageFolder로 로드
        if os.path.isdir(data_dir):
            dataset = datasets.ImageFolder(data_dir, transform=val_tf)
            calib_loader = DataLoader(
                dataset,
                batch_size=cfg.ptq.calibration_batch_size,
                shuffle=True,
                num_workers=2,
            )
            qat_loader = DataLoader(
                dataset,
                batch_size=cfg.qat.batch_size,
                shuffle=True,
                num_workers=2,
            )
            logging.getLogger(__name__).info(
                f"데이터 로드 완료: {len(dataset)} samples from {data_dir}"
            )
            return calib_loader, qat_loader

    except Exception as e:
        logging.getLogger(__name__).warning(
            f"실제 데이터 로드 실패 ({e}). 더미 데이터를 사용합니다."
        )

    # 폴백: 더미 데이터
    calib_loader = _make_dummy_loader(
        batch_size=cfg.ptq.calibration_batch_size,
        n_batches=cfg.ptq.num_calibration_batches,
        input_size=cfg.benchmark.input_size,
    )
    qat_loader = _make_dummy_loader(
        batch_size=cfg.qat.batch_size,
        n_batches=20,
        input_size=cfg.benchmark.input_size,
    )
    return calib_loader, qat_loader


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="양자화 실험: 대형 모델 PTQ/QAT vs 경량 모델 비교",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--mode",
        choices=["ptq", "qat", "benchmark", "all"],
        default="benchmark",
        help="실행할 실험 모드 (default: benchmark)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="데이터 경로 (없으면 더미 데이터 사용)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="더미 입력으로 파이프라인 형상 확인만 수행",
    )

    # ── 모델 설정 오버라이드 ────────────────────────────────
    parser.add_argument(
        "--large-backbone",
        type=str,
        default=None,
        help="양자화 대상 대형 backbone (예: efficientnet_b3)",
    )

    # ── PTQ 오버라이드 ─────────────────────────────────────
    parser.add_argument(
        "--calib-batches",
        type=int,
        default=None,
        help="PTQ 캘리브레이션 배치 수",
    )
    parser.add_argument(
        "--ptq-backend",
        type=str,
        choices=["fbgemm", "qnnpack"],
        default=None,
        help="PTQ 백엔드 (x86: fbgemm, ARM: qnnpack)",
    )

    # ── QAT 오버라이드 ─────────────────────────────────────
    parser.add_argument(
        "--qat-lr",
        type=float,
        default=None,
        help="QAT fine-tuning 학습률",
    )
    parser.add_argument(
        "--qat-epochs",
        type=int,
        default=None,
        help="QAT fine-tuning epoch 수",
    )

    # ── 벤치마크 오버라이드 ────────────────────────────────
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="벤치마크 배치 크기",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="벤치마크 디바이스 (cpu / cuda)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="결과 CSV 저장 경로",
    )

    return parser.parse_args()


def apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    """CLI args를 ExperimentConfig에 반영합니다."""
    if args.large_backbone:
        cfg.model.large_backbone = args.large_backbone
    if args.calib_batches is not None:
        cfg.ptq.num_calibration_batches = args.calib_batches
    if args.ptq_backend:
        cfg.ptq.backend = args.ptq_backend
        cfg.qat.backend = args.ptq_backend   # 일치 권장
    if args.qat_lr is not None:
        cfg.qat.lr = args.qat_lr
    if args.qat_epochs is not None:
        cfg.qat.epochs = args.qat_epochs
    if args.batch_size is not None:
        cfg.benchmark.batch_size = args.batch_size
    if args.device:
        cfg.benchmark.device = args.device
    if args.output_csv:
        cfg.benchmark.output_csv = args.output_csv

    # dry-run: 캘리브레이션 배치·QAT epoch 최소화
    if args.dry_run:
        cfg.ptq.num_calibration_batches = 2
        cfg.qat.epochs = 1
        cfg.benchmark.num_warmup_runs = 3
        cfg.benchmark.num_benchmark_runs = 5


def main() -> None:
    args = parse_args()

    # 로깅 기본 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 설정 생성 및 오버라이드 적용
    cfg = ExperimentConfig()
    apply_overrides(cfg, args)
    print(cfg.summary())

    set_seed(cfg.seed)

    # 데이터 로더 준비
    if args.dry_run:
        calib_loader = _make_dummy_loader(
            n_batches=cfg.ptq.num_calibration_batches,
            input_size=cfg.benchmark.input_size,
        )
        qat_loader = _make_dummy_loader(
            n_batches=10,
            input_size=cfg.benchmark.input_size,
        )
    elif args.data:
        calib_loader, qat_loader = _build_data_loaders(args.data, cfg)
    else:
        calib_loader = None
        qat_loader   = None

    # mode에 따른 실행
    mode = args.mode

    # PTQ/QAT/benchmark 모두 benchmark 함수에서 통합 처리
    run_benchmark(
        cfg=cfg,
        calibration_loader=calib_loader if mode in ("ptq", "benchmark", "all") else None,
        qat_train_loader=qat_loader if mode in ("qat", "all") else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
