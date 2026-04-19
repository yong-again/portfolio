"""
Quantization Experiment Config

모든 하이퍼파라미터를 Python dataclass로 관리합니다.
각 파라미터에는 중요도(1~3점)와 이유가 주석으로 표시됩니다.

[중요도: 3/3] : 실험 결과에 가장 큰 영향. 반드시 이해 후 조정 필요
[중요도: 2/3] : 중요하지만 기본값으로도 합리적인 결과 도출 가능
[중요도: 1/3] : 부차적 설정. 재현성/편의를 위한 값
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================
# 모델 설정
# ============================================================

@dataclass
class ModelConfig:
    """비교 실험에 사용할 모델 정의"""

    # [중요도: 3/3] 양자화 대상 대형 모델
    # 이유: B0 대비 파라미터 수가 약 3배 이상 차이나야 양자화 이득이 명확히 드러남.
    #       EfficientNet-B3 (12M params)은 경량 모델 B0(5.3M)과 의미 있는 비교 가능.
    large_backbone: str = "efficientnet_b3"

    # [중요도: 2/3] 비교 대상 경량 모델 목록
    # 이유: 양자화한 대형 모델이 "이미 작은" 모델보다 얼마나 유리/불리한지 비교.
    #       MobileNetV3는 모바일 최적화 아키텍처로 대표적 경량 baseline.
    small_backbones: List[str] = field(default_factory=lambda: [
        "efficientnet_b0",       # 가장 작은 EfficientNet (동일 계열 비교)
        "mobilenetv3_small_100", # 모바일 최적화 경량 모델
    ])

    # [중요도: 1/3] Age head 클래스 수 (0~100세 = 101)
    age_num_classes: int = 101

    # [중요도: 1/3] Gender head 클래스 수 (male/female = 2)
    gender_num_classes: int = 2

    # [중요도: 1/3] Age head hidden dimension
    age_hidden_dim: int = 512

    # [중요도: 1/3] Gender head hidden dimension
    gender_hidden_dim: int = 256

    # [중요도: 1/3] Dropout (FP32 기준선 모델 전용)
    dropout: float = 0.5

    # [중요도: 1/3] ImageNet 사전학습 가중치 사용 여부
    # 이유: pretrained=True 시 양자화 후 성능 저하가 더 잘 관찰됨 (수렴된 가중치 기준)
    pretrained: bool = True


# ============================================================
# Post-Training Quantization (사후 양자화)
# ============================================================

@dataclass
class PTQConfig:
    """Post-Training Quantization 하이퍼파라미터"""

    # [중요도: 3/3] Quantization backend
    # 이유: 플랫폼에 따라 지원 연산자가 다름.
    #   - "fbgemm" : x86 CPU (Intel/AMD) 환경, 가장 일반적
    #   - "qnnpack": ARM CPU (모바일, Raspberry Pi, Apple Silicon)
    #   잘못 선택하면 실제 속도 이득 없이 정확도만 떨어짐.
    backend: str = "fbgemm"

    # [중요도: 3/3] 캘리브레이션에 사용할 배치 수 (Static PTQ 전용)
    # 이유: 너무 적으면 outlier activation을 포착 못해 scale/zero_point 추정 불량 →
    #       정확도 급락. 너무 많으면 시간 낭비.
    #       일반적으로 100~500 이미지(배치 × 배치크기) 수준이 권장됨.
    num_calibration_batches: int = 100

    # [중요도: 2/3] Static vs Dynamic PTQ 선택
    # 이유: Static은 캘리브레이션 필요하지만 정확도·속도 모두 우수.
    #       Dynamic은 캘리브레이션 불필요(데이터 없어도 가능)하나 속도 이득 적음.
    #       두 방식을 모두 실험해 트레이드오프 확인.
    use_static: bool = True   # True → Static PTQ
    use_dynamic: bool = True  # True → Dynamic PTQ

    # [중요도: 2/3] Activation observer 종류 (Static PTQ)
    # 이유: MinMax는 단순하지만 outlier에 민감. HistogramObserver는 더 정교한
    #       분포 추정으로 정확도 손실 감소. 실험 비교에 권장.
    #   선택지: "minmax", "histogram", "percentile"
    observer_type: str = "histogram"

    # [중요도: 1/3] 캘리브레이션 배치 크기
    calibration_batch_size: int = 32

    # [중요도: 1/3] 캘리브레이션 데이터 비율 (전체 val set 대비)
    # 이유: 전체 데이터는 필요 없으나 너무 적으면 분포를 대표하지 못함.
    calibration_data_ratio: float = 0.1

    # [중요도: 1/3] Conv-BN-ReLU fusion 여부
    # 이유: fusion 후 양자화 시 별도 BN 연산 제거 → 추론 속도·정확도 향상.
    #       특히 EfficientNet처럼 BN이 많은 구조에서 효과적.
    fuse_modules: bool = True


# ============================================================
# Quantization-Aware Training (사전 양자화)
# ============================================================

@dataclass
class QATConfig:
    """Quantization-Aware Training 하이퍼파라미터"""

    # [중요도: 3/3] QAT fine-tuning 학습률
    # 이유: Fake-quantize 노드가 삽입된 상태에서 너무 높은 lr은 양자화 노이즈와
    #       결합해 학습을 발산시킴. FP32 lr의 1/10 이하 (예: 1e-4)가 안전한 시작점.
    #       이 파라미터가 QAT 성패를 가장 크게 좌우함.
    lr: float = 1e-4

    # [중요도: 3/3] QAT fine-tuning epoch 수
    # 이유: 너무 적으면(1~2 epoch) 양자화 오차 보상 미흡 → PTQ와 차이 없음.
    #       너무 많으면 과적합 위험. 보통 5~15 epoch이 실용적 범위.
    epochs: int = 10

    # [중요도: 2/3] Fake-quantize 적용 시작 epoch (observer update 구간)
    # 이유: 초반 몇 epoch는 실수 범위로 학습해 가중치를 안정시킨 뒤
    #       fake-quantize를 켜야 수렴이 잘 됨.
    freeze_observer_epoch: int = 3   # 이 epoch부터 observer 고정 (통계 수집 종료)
    freeze_bn_epoch: int = 2         # 이 epoch부터 BN를 eval 모드로 고정

    # [중요도: 2/3] Backend (PTQ와 통일 권장)
    # 이유: 동일 backend 사용 시 비교 공정성 확보. PTQConfig.backend와 맞출 것.
    backend: str = "fbgemm"

    # [중요도: 1/3] 배치 크기
    batch_size: int = 32

    # [중요도: 1/3] Weight decay (fine-tuning 단계, 과적합 방지)
    weight_decay: float = 1e-4

    # [중요도: 1/3] 데이터 증강 사용 여부 (QAT fine-tuning 중)
    # 이유: QAT는 재학습이므로 augmentation이 일반화에 도움.
    use_augmentation: bool = True

    # [중요도: 1/3] Optimizer 선택 ('adam', 'sgd', 'adamw')
    optimizer: str = "adamw"

    # [중요도: 1/3] LR Scheduler ('cosine', 'step', 'none')
    scheduler: str = "cosine"


# ============================================================
# 벤치마크 설정
# ============================================================

@dataclass
class BenchmarkConfig:
    """레이턴시·파라미터 수·모델 크기 비교 설정"""

    # [중요도: 3/3] 벤치마크 배치 크기
    # 이유: 배치 크기에 따라 메모리·레이턴시 특성이 크게 달라짐.
    #       모든 모델에 동일한 배치 크기를 적용해야 공정한 비교 가능.
    batch_size: int = 1

    # [중요도: 2/3] 워밍업 반복 횟수
    # 이유: CPU/GPU 캐시, PyTorch JIT 컴파일, 메모리 할당 등이 첫 추론을 왜곡함.
    #       최소 10~20회 워밍업 후 측정해야 안정적인 레이턴시 값 얻음.
    num_warmup_runs: int = 20

    # [중요도: 2/3] 측정 반복 횟수
    # 이유: 단일 측정은 OS 스케줄링 등 외부 요인에 취약. 최소 100회 이상 권장.
    #       평균±표준편차로 리포트해 신뢰 구간 확인.
    num_benchmark_runs: int = 200

    # [중요도: 1/3] 입력 이미지 해상도 (H, W)
    # 이유: EfficientNet-B3는 300×300이 native이나 공정 비교를 위해 통일.
    input_size: tuple = (224, 224)

    # [중요도: 1/3] 측정 디바이스
    # 이유: CPU 벤치마크가 edge deployment 시나리오에 더 적합.
    device: str = "cpu"

    # [중요도: 1/3] 결과 CSV 저장 경로
    output_csv: str = "results/quantization_benchmark.csv"

    # [중요도: 1/3] 모델 저장/로드 경로 (크기 측정용 임시 저장)
    tmp_model_dir: str = "results/tmp_models"


# ============================================================
# 전체 실험 설정 (통합 진입점)
# ============================================================

@dataclass
class ExperimentConfig:
    """
    전체 실험 설정을 통합하는 최상위 dataclass.

    사용 예:
        from quantization.config import ExperimentConfig
        cfg = ExperimentConfig()
        # 개별 수정
        cfg.ptq.num_calibration_batches = 50
        cfg.qat.lr = 5e-5
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    ptq: PTQConfig = field(default_factory=PTQConfig)
    qat: QATConfig = field(default_factory=QATConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    # [중요도: 1/3] 재현성을 위한 랜덤 시드
    seed: int = 42

    # [중요도: 1/3] 로그 레벨 ('DEBUG', 'INFO', 'WARNING')
    log_level: str = "INFO"

    # [중요도: 1/3] 실험 결과 루트 디렉토리
    result_dir: str = "results/quantization"

    def summary(self) -> str:
        """현재 설정 요약 문자열 반환"""
        lines = [
            "=" * 60,
            "  Quantization Experiment Config Summary",
            "=" * 60,
            f"  [Model]",
            f"    Large backbone : {self.model.large_backbone}",
            f"    Small backbones: {self.model.small_backbones}",
            f"  [PTQ]",
            f"    Backend        : {self.ptq.backend}",
            f"    Static PTQ     : {self.ptq.use_static}",
            f"    Dynamic PTQ    : {self.ptq.use_dynamic}",
            f"    Calibration    : {self.ptq.num_calibration_batches} batches",
            f"    Observer       : {self.ptq.observer_type}",
            f"  [QAT]",
            f"    LR             : {self.qat.lr}",
            f"    Epochs         : {self.qat.epochs}",
            f"    Backend        : {self.qat.backend}",
            f"  [Benchmark]",
            f"    Batch size     : {self.benchmark.batch_size}",
            f"    Warmup runs    : {self.benchmark.num_warmup_runs}",
            f"    Benchmark runs : {self.benchmark.num_benchmark_runs}",
            f"    Device         : {self.benchmark.device}",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    cfg = ExperimentConfig()
    print(cfg.summary())
