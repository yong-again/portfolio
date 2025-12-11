"""
YOLO Head Detection Training Script

Ultralytics YOLO를 사용하여 human head detection 모델을 학습합니다.
config.py 파일을 사용하여 학습 설정을 관리합니다.
"""

from pathlib import Path
from ultralytics import YOLO
from config import YOLOTrainingConfig, default_config


def train_head_detector(config: YOLOTrainingConfig):
    """
    YOLO를 사용하여 head detection 모델을 학습합니다.
    
    Args:
        config: YOLOTrainingConfig 인스턴스
    
    Returns:
        학습된 모델
    """
    # data는 필수이므로 확인
    if config.data is None:
        raise ValueError("'data' parameter is required. Set config.data to dataset YAML file path.")
    
    # 모델 로드
    yolo_model = YOLO(config.model)
    
    # Config를 딕셔너리로 변환 (None 값 제외)
    train_args = config.to_dict()
    
    print("=" * 50)
    print("YOLO Head Detection Training")
    print("=" * 50)
    print(f"Model: {config.model}")
    print(f"Data: {config.data}")
    print(f"Epochs: {config.epochs}")
    print(f"Image size: {config.imgsz}")
    print(f"Batch size: {config.batch}")
    print(f"Device: {config.device}")
    print(f"Optimizer: {config.optimizer}")
    print(f"Patience: {config.patience}")
    print("=" * 50)
    
    # 학습 시작
    results = yolo_model.train(**train_args)
    
    print("\nTraining completed!")
    print(f"Best model saved to: {results.save_dir / 'weights' / 'best.pt'}")
    
    return yolo_model


if __name__ == "__main__":
    """
    사용 방법:
    1. config.py 파일을 수정하여 설정 변경
    2. 또는 여기서 직접 설정 생성
    
    예시:
        config = YOLOTrainingConfig(
            data="../data/detection_cleaned_t/dataset.yaml",
            epochs=500,
            batch=32,
            patience=20,
            optimizer="AdamW",
            cache=True,
            exist_ok=True,
            project="../result",
            name="head_detection",
        )
        train_head_detector(config)
    """
    # 필수 파라미터 설정 (config.py에서 설정하거나 여기서 설정)
    # config.data = "../data/detection_cleaned_t/dataset.yaml"
    
    # 학습 실행
    config = YOLOTrainingConfig()
    train_head_detector(config)

