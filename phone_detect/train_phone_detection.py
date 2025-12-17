"""
Training Script for Phone Region Detection

YOLO를 사용하여 휴대폰 영역 검출 모델을 학습합니다.
전면/후면용 YOLO 모델 학습에 사용됩니다.
"""

import torch
import yaml
import argparse
from pathlib import Path
import logging
from datetime import datetime
from ultralytics import YOLO
import sys

sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(log_dir: str, log_level: str = "INFO"):
    """로깅 설정"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"phone_detection_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train Phone Detection Model (YOLO)')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file (default: train_config.yaml)')
    parser.add_argument('--section', type=str, default='front',
                       choices=['front', 'back'],
                       help='Section to train (front or back)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to dataset YAML file (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--imgsz', type=int, default=None,
                       help='Image size (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 로깅 설정
    logger = setup_logging(
        config['paths']['log_dir'],
        config['logging']['level']
    )
    
    logger.info("=" * 50)
    logger.info(f"Phone Detection Training - {args.section.upper()}")
    logger.info("=" * 50)
    logger.info(f"Config: {config_path}")
    
    # Section 설정 가져오기
    section_config = config['phone_detection'][args.section]
    
    # Device 설정
    device_config = config['device']
    if device_config['type'] == 'cuda' and torch.cuda.is_available():
        device = f"cuda:{device_config['device_id']}"
    elif device_config['type'] == 'mps' and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    logger.info(f"Device: {device}")
    
    # 모델 설정
    model_name = section_config.get('model', 'yolo11n.pt')
    input_size = section_config['input_size']
    imgsz = args.imgsz if args.imgsz is not None else input_size['width']
    
    logger.info(f"Model: {model_name}")
    logger.info(f"Input size: {imgsz}x{imgsz}")
    
    # 데이터셋 설정
    if args.data:
        data_yaml = args.data
    else:
        # 데이터셋 YAML 파일 경로 (YOLO format)
        data_dir = Path(config['data']['train_path'])
        data_yaml = data_dir / f"{args.section}_dataset.yaml"
        
        # YAML 파일이 없으면 생성
        if not data_yaml.exists():
            logger.warning(f"Dataset YAML not found: {data_yaml}")
            logger.info("Creating dataset YAML file...")
            
            # YOLO format 데이터셋 구조 가정
            train_images = data_dir / args.section / 'train' / 'images'
            val_images = data_dir / args.section / 'val' / 'images'
            
            dataset_yaml_content = f"""
# {args.section.upper()} Phone Detection Dataset
path: {data_dir / args.section}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')

# Classes
names:
  0: phone
"""
            data_yaml.parent.mkdir(parents=True, exist_ok=True)
            with open(data_yaml, 'w') as f:
                f.write(dataset_yaml_content)
            logger.info(f"Created dataset YAML: {data_yaml}")
    
    logger.info(f"Dataset YAML: {data_yaml}")
    
    # YOLO 모델 초기화
    model = YOLO(model_name)
    
    # 학습 파라미터
    epochs = args.epochs if args.epochs is not None else config['training']['num_epochs']
    batch_size = args.batch_size if args.batch_size is not None else config['training']['batch_size']
    
    # 출력 경로
    output_dir = Path(config['paths']['weights_dir']) / 'phone_detection' / args.section
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training parameters:")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Image size: {imgsz}")
    logger.info(f"  Output dir: {output_dir}")
    
    # 학습 시작
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=str(output_dir),
            name='train',
            resume=args.resume is not None,
            exist_ok=True,
            verbose=True
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {output_dir / 'train' / 'weights' / 'best.pt'}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

