"""
YOLO Head Detection Training Script

Ultralytics YOLO를 사용하여 human head detection 모델을 학습합니다.
"""

import argparse
from pathlib import Path
import yaml
from ultralytics import YOLO


def train_head_detector(
    data_yaml: str,
    model: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = None,
    project: str = "runs/detect",
    name: str = "head_detection",
    **kwargs
):
    """
    YOLO를 사용하여 head detection 모델을 학습합니다.
    
    Args:
        data_yaml: 데이터셋 설정 YAML 파일 경로 (YOLO format)
        model: 사전 학습된 모델 경로 또는 모델 이름 (yolo11n.pt, yolo11s.pt 등)
        epochs: 학습 epoch 수
        imgsz: 입력 이미지 크기
        batch: 배치 크기
        device: 학습 디바이스 ('cpu', '0', '0,1', 'mps' 등)
        project: 프로젝트 디렉토리
        name: 실험 이름
        **kwargs: 추가 YOLO 학습 인수
    
    Returns:
        학습된 모델
    """
    # 모델 로드
    yolo_model = YOLO(model)
    
    # 학습 인수 설정
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'project': project,
        'name': name,
        **kwargs
    }
    
    if device is not None:
        train_args['device'] = device
    
    print("=" * 50)
    print("YOLO Head Detection Training")
    print("=" * 50)
    print(f"Model: {model}")
    print(f"Data: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # 학습 시작
    results = yolo_model.train(**train_args)
    
    print("\nTraining completed!")
    print(f"Best model saved to: {results.save_dir / 'weights' / 'best.pt'}")
    
    return yolo_model


def main():
    parser = argparse.ArgumentParser(description='Train YOLO Head Detection Model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset YAML file (YOLO format)')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Pretrained model (yolo11n.pt, yolo11s.pt, etc.)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Training device (cpu, 0, 0,1, mps)')
    parser.add_argument('--project', type=str, default='runs/detect',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='head_detection',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # 학습 실행
    train_head_detector(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()

