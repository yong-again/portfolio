"""
YOLO Training Configuration

YOLO 학습을 위한 설정 파일입니다.
모든 YOLO 학습 파라미터를 포함합니다.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Union


@dataclass
class YOLOTrainingConfig:
    """YOLO 학습 설정 클래스"""
    
    # 필수 파라미터
    model: str = "yolo11s.pt"  # 학습에 사용할 모델 파일
    data: Optional[str] = "/workspace/portfolio/age_gender_estimation/data/detection_augmented/dataset.yaml"  # 데이터 세트 구성 파일 경로 (예: coco8.yaml)
    
    # 학습 기본 설정
    epochs: int = 500  # 총 학습 epoch 수
    time: Optional[float] = None  # 최대 학습 시간(시간), epochs를 덮어씀
    patience: int = 50  # 조기 중단을 위한 patience (epochs)
    batch: Union[int, float] = -1  # 배치 크기 (정수, -1=자동 60%, 0.70=70% GPU)
    imgsz: int = 1024  # 학습 대상 이미지 크기
    
    # 저장 및 체크포인트
    save: bool = True  # 학습 체크포인트 및 최종 모델 가중치 저장
    save_period: int = -1  # 모델 체크포인트 저장 빈도(epoch 단위), -1=비활성화
    cache: Union[bool, str] = False  # 데이터 세트 이미지 캐싱 (True/ram, disk, False)
    
    # 하드웨어 설정
    device: Optional[Union[int, str, List[int]]] = None  # 학습 장치 (0, [0,1], cpu, mps, -1=자동)
    workers: int = 8  # 데이터 로딩을 위한 worker 스레드 수
    
    # 프로젝트 및 이름
    project: Optional[str] = None  # 훈련 결과가 저장되는 프로젝트 디렉터리
    name: Optional[str] = "head_detection"  # 훈련 실행 이름
    exist_ok: bool = False  # 기존 프로젝트/이름 디렉터리 덮어쓰기
    
    # 사전 학습
    pretrained: Union[bool, str] = True  # 사전 훈련된 모델에서 훈련 시작 여부
    
    # 최적화
    optimizer: str = 'AdamW'  # 최적화 알고리즘 (SGD, Adam, AdamW, NAdam, RAdam, RMSProp, auto)
    seed: int = 42  # 훈련을 위한 임의 시드
    deterministic: bool = True  # 결정론적 알고리즘 사용 강제
    
    # 데이터 설정
    single_cls: bool = False  # 모든 클래스를 단일 클래스로 취급
    classes: Optional[List[int]] = None  # 학습할 클래스 ID 목록
    rect: bool = False  # 최소 패딩 전략 사용
    multi_scale: bool = False  # 다중 스케일 학습 활성화
    
    # 학습률 스케줄러
    cos_lr: bool = True  # 코사인 학습률 스케줄러 사용
    
    # 데이터 증강
    close_mosaic: int = 10  # 마지막 N epoch에서 모자이크 데이터 증강 비활성화 (0=비활성화)
    
    # 재개
    resume: bool = False  # 마지막 체크포인트부터 훈련 재개
    
    # 정밀도 및 성능
    amp: bool = True  # 자동 혼합 정밀도 (AMP) 훈련 활성화
    fraction: float = 1.0  # 훈련에 사용할 데이터 세트의 비율
    profile: bool = False  # ONNX 및 TensorRT 속도 프로파일링 활성화
    
    # 전이 학습
    freeze: Optional[Union[int, List[int]]] = None  # 고정할 레이어 수 또는 인덱스
    
    # 학습률 하이퍼파라미터
    lr0: float = 0.001  # 초기 학습률 (SGD=1E-2, Adam=1E-3)
    lrf: float = 0.01  # 최종 학습률 비율 (lr0 * lrf)
    momentum: float = 0.937  # SGD 또는 Adam 옵티마이저용 모멘텀 계수
    weight_decay: float = 0.0005  # L2 정규화 항
    
    # 워밍업
    warmup_epochs: float = 3.0  # 학습률 워밍업을 위한 epoch 수
    warmup_momentum: float = 0.8  # 워밍업 단계의 초기 모멘텀
    warmup_bias_lr: float = 0.1  # 워밍업 단계 동안 편향 파라미터에 대한 학습률
    
    # 손실 함수 가중치
    box: float = 8.5  # 박스 손실 요소의 가중치
    cls: float = 0.5  # 분류 손실의 가중치
    dfl: float = 1.5  # 분포 focal loss의 가중치
    pose: float = 12.0  # 포즈 손실의 가중치
    kobj: float = 2.0  # 키포인트 객체성 손실의 가중치
    
    # 기타
    nbs: int = 64  # 손실 정규화를 위한 명목 배치 크기
    overlap_mask: bool = True  # 객체 마스크를 단일 마스크로 병합할지 여부
    mask_ratio: int = 4  # 세분화 마스크의 다운샘플링 비율
    dropout: float = 0.0  # 드롭아웃 비율
    
    # 검증 및 평가
    val: bool = True  # 학습 중 유효성 검사 활성화
    plots: bool = True  # 학습 및 유효성 검사 플롯 생성
    
    # 컴파일
    compile: Union[bool, str] = False  # PyTorch 2.x torch.compile (True, False, "default", "reduce-overhead", "max-autotune-no-cudagraphs")
    
    def to_dict(self) -> dict:
        """설정을 딕셔너리로 변환 (None 값 제외)"""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result
    
    def update(self, **kwargs):
        """설정 업데이트"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")


# 기본 설정 인스턴스
# default_config = YOLOTrainingConfig(
#     model="yolo11n.pt",
#     epochs=100,
#     batch=16,
#     imgsz=640,
#     patience=100,
#     project="runs/detect",
#     name="head_detection",
#     optimizer="auto",
#     cache=False,
#     exist_ok=False,
#)


# 사용 예시:
# 
# 방법 1: config.py 파일에서 직접 설정 수정
# config = YOLOTrainingConfig(
#     data="../data/detection_cleaned_t/dataset.yaml",
#     epochs=500,
#     batch=32,
#     patience=20,
#     optimizer="AdamW",
#     cache=True,
#     exist_ok=True,
#     project="../result",
#     name="head_detection",
# )
#
# 방법 2: train_detector.py에서 설정
# from config import YOLOTrainingConfig
# config = YOLOTrainingConfig(
#     data="path/to/dataset.yaml",
#     epochs=500,
#     batch=32,
#     # ... 기타 설정
# )
# train_head_detector(config)
#
# 방법 3: 기본 설정 사용 후 일부만 수정
# from config import default_config
# config = YOLOTrainingConfig(**default_config.__dict__)
# config.data = "path/to/dataset.yaml"
# config.epochs = 500
# train_head_detector(config)

