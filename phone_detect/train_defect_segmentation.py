"""
Training Script for Defect Segmentation

결함 영역 segmentation 모델을 학습합니다.
디스플레이/측면/전면용 defect segmentation 모델 학습에 사용됩니다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import sys
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models.defect_segmenter import DefectSegmenter
from models.utils import save_checkpoint, load_checkpoint, count_parameters
from preprocess.defect_preprocess import DefectPreprocessor


def setup_logging(log_dir: str, log_level: str = "INFO"):
    """로깅 설정"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"defect_segmentation_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class SegmentationDataset(torch.utils.data.Dataset):
    """
    Defect Segmentation을 위한 Dataset 클래스
    
    이미지와 segmentation mask를 로드합니다.
    """
    
    def __init__(
        self,
        data_dir: str,
        config: dict,
        section: str,
        is_training: bool = True
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
            config: 설정 딕셔너리
            section: 섹션 이름 (display, side, front)
            is_training: 학습 모드 여부
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.section = section
        self.is_training = is_training
        
        # 전처리 설정
        self.preprocessor = DefectPreprocessor(config)
        self.section_config = config['defect_segmentation'][section]
        self.input_size = self.section_config['input_size']
        
        # 이미지와 mask 파일 찾기
        images_dir = self.data_dir / 'images'
        masks_dir = self.data_dir / 'masks'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise ValueError(f"Masks directory not found: {masks_dir}")
        
        # 이미지 파일 목록 수집
        image_extensions = ['.jpg', '.jpeg', '.png']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(list(images_dir.glob(f'*{ext}')))
            self.image_paths.extend(list(images_dir.glob(f'*{ext.upper()}')))
        
        self.image_paths = sorted(self.image_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        print(f"Found {len(self.image_paths)} images in {data_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> dict:
        """
        데이터셋에서 하나의 샘플을 가져옵니다.
        
        Returns:
            Dictionary containing:
                - 'image': 전처리된 이미지 텐서 [C, H, W]
                - 'mask': Segmentation mask [H, W] (각 pixel의 등급)
        """
        import cv2
        from PIL import Image
        
        image_path = self.image_paths[idx]
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Mask 로드
        mask_path = self.data_dir / 'masks' / image_path.name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Mask가 없으면 빈 mask 생성
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize
        image = self.preprocessor.resize(image, self.input_size)
        mask = cv2.resize(mask, (self.input_size['width'], self.input_size['height']), 
                         interpolation=cv2.INTER_NEAREST)
        
        # 전처리
        image = self.preprocessor.preprocess(image, section=self.section, apply_clahe_flag=False)
        
        # Augmentation (학습 시에만)
        if self.is_training:
            # 간단한 augmentation (실제로는 albumentations 등 사용 권장)
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
        
        # Tensor 변환
        image_tensor = self.preprocessor.to_tensor(image)
        image_tensor = torch.from_numpy(image_tensor[0]).float()  # [C, H, W]
        
        mask_tensor = torch.from_numpy(mask).long()  # [H, W]
        
        return {
            'image': image_tensor,
            'mask': mask_tensor
        }


def get_optimizer(model: nn.Module, config: dict) -> optim.Optimizer:
    """옵티마이저 생성"""
    opt_config = config['training']['optimizer']
    opt_name = opt_config['name'].lower()
    lr = opt_config['lr']
    weight_decay = opt_config.get('weight_decay', 0.0)
    
    if opt_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        momentum = opt_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momenum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
    
    return optimizer


def get_scheduler(optimizer: optim.Optimizer, config: dict) -> optim.lr_scheduler._LRScheduler:
    """Learning rate scheduler 생성"""
    sched_config = config['training']['scheduler']
    sched_name = sched_config['name'].lower()
    
    if sched_name == 'cosine':
        T_max = sched_config.get('T_max', 100)
        eta_min = sched_config.get('eta_min', 0.0)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif sched_name == 'step':
        step_size = sched_config.get('step_size', 30)
        gamma = sched_config.get('gamma', 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif sched_name == 'plateau':
        factor = sched_config.get('factor', 0.5)
        patience = sched_config.get('patience', 5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    else:
        scheduler = None
    
    return scheduler


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> dict:
    """한 epoch 학습"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        
        # Loss 계산
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 통계
        total_loss += loss.item()
        num_batches += 1
        
        # Progress bar 업데이트
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}'
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {'loss': avg_loss}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    logger: logging.Logger
) -> dict:
    """검증"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Loss 계산
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {'loss': avg_loss}


def main():
    parser = argparse.ArgumentParser(description='Train Defect Segmentation Model')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file (default: train_config.yaml)')
    parser.add_argument('--section', type=str, default='display',
                       choices=['display', 'side', 'front'],
                       help='Section to train')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to dataset directory (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
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
    logger.info(f"Defect Segmentation Training - {args.section.upper()}")
    logger.info("=" * 50)
    logger.info(f"Config: {config_path}")
    
    # Section 설정 가져오기
    section_config = config['defect_segmentation'][args.section]
    
    # Device 설정
    device_config = config['device']
    if device_config['type'] == 'cuda' and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['device_id']}")
    elif device_config['type'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Device: {device}")
    
    # 모델 생성
    model_config = section_config['model']
    model = DefectSegmenter(
        encoder_name=model_config['encoder_name'],
        encoder_weights=model_config['encoder_weights'],
        decoder_name=model_config['decoder_name'],
        classes=model_config['classes']
    )
    model = model.to(device)
    
    # 모델 정보 출력
    params = count_parameters(model)
    logger.info(f"Total parameters: {params['total']:,}")
    logger.info(f"Trainable parameters: {params['trainable']:,}")
    
    # 데이터셋 생성
    if args.data:
        train_data_dir = Path(args.data) / 'train'
        val_data_dir = Path(args.data) / 'val'
    else:
        train_data_dir = Path(config['data']['train_path']) / args.section
        val_data_dir = Path(config['data']['val_path']) / args.section
    
    train_dataset = SegmentationDataset(
        str(train_data_dir),
        config,
        args.section,
        is_training=True
    )
    val_dataset = SegmentationDataset(
        str(val_data_dir),
        config,
        args.section,
        is_training=False
    )
    
    # DataLoader 생성
    batch_size = args.batch_size if args.batch_size is not None else config['training']['batch_size']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Loss 함수
    loss_config = config['training']['loss']
    loss_type = loss_config.get('type', 'cross_entropy')
    
    if loss_type == 'cross_entropy':
        class_weights = loss_config.get('class_weights', None)
        if class_weights:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'dice':
        # Dice Loss는 별도 구현 필요 (예시로 CrossEntropy 사용)
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = get_optimizer(model, config)
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # Resume from checkpoint
    start_epoch = 0
    best_score = float('inf')
    
    if args.resume:
        checkpoint_info = load_checkpoint(args.resume, model, str(device), optimizer)
        start_epoch = checkpoint_info.get('epoch', 0) + 1
        best_score = checkpoint_info.get('best_score', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, best score: {best_score:.4f}")
    
    # 학습 루프
    num_epochs = args.epochs if args.epochs is not None else config['training']['num_epochs']
    val_interval = config['training']['val_interval']
    save_interval = config['training']['save_interval']
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )
        
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}"
        )
        
        # Validation
        if (epoch + 1) % val_interval == 0:
            val_metrics = validate(
                model, val_loader, criterion, device, epoch, logger
            )
            
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            
            # Best model 저장
            current_score = val_metrics['loss']
            if current_score < best_score:
                best_score = current_score
                output_dir = Path(config['paths']['weights_dir']) / 'defect_segmentation' / args.section
                output_dir.mkdir(parents=True, exist_ok=True)
                
                save_checkpoint(
                    str(output_dir),
                    model,
                    epoch,
                    best_score,
                    optimizer,
                    config,
                    is_best=True
                )
                logger.info(f"New best model saved! Score: {best_score:.4f}")
        
        # Checkpoint 저장
        if (epoch + 1) % save_interval == 0:
            output_dir = Path(config['paths']['weights_dir']) / 'defect_segmentation' / args.section
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_checkpoint(
                str(output_dir),
                model,
                epoch,
                best_score,
                optimizer,
                config,
                is_best=False
            )
        
        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()

