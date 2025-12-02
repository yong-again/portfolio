"""
YOLO Head Detection Validation Script

학습된 YOLO head detection 모델을 검증합니다.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def validate_head_detector(
    model_path: str,
    data_yaml: str = None,
    imgsz: int = 640,
    batch: int = 16,
    device: str = None,
    **kwargs
):
    """
    YOLO head detection 모델을 검증합니다.
    
    Args:
        model_path: 학습된 모델 가중치 경로
        data_yaml: 데이터셋 설정 YAML 파일 경로 (선택사항)
        imgsz: 검증 이미지 크기
        batch: 배치 크기
        device: 검증 디바이스
        **kwargs: 추가 YOLO 검증 인수
    
    Returns:
        검증 결과
    """
    # 모델 로드
    model = YOLO(model_path)
    
    # 검증 인수 설정
    val_args = {
        'imgsz': imgsz,
        'batch': batch,
        **kwargs
    }
    
    if data_yaml is not None:
        val_args['data'] = data_yaml
    
    if device is not None:
        val_args['device'] = device
    
    print("=" * 50)
    print("YOLO Head Detection Validation")
    print("=" * 50)
    print(f"Model: {model_path}")
    if data_yaml:
        print(f"Data: {data_yaml}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")
    print(f"Device: {device}")
    print("=" * 50)
    
    # 검증 실행
    results = model.val(**val_args)
    
    print("\nValidation completed!")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Validate YOLO Head Detection Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to dataset YAML file (optional)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for validation')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default=None,
                       help='Validation device (cpu, 0, 0,1, mps)')
    
    args = parser.parse_args()
    
    # 검증 실행
    validate_head_detector(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )


if __name__ == "__main__":
    main()

