"""
Training Script for Age & Gender Estimation

멀티태스크 학습 스크립트입니다. Age와 Gender를 동시에 학습합니다.
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

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from models.network import build_network
from models.utils import save_checkpoint, load_checkpoint, count_parameters
from preprocess.dataset import AgeGenderDataset, collate_fn


def setup_logging(log_dir: str, log_level: str = "INFO"):
    """로깅 설정"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


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
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
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
    criterion_age: nn.Module,
    criterion_gender: nn.Module,
    device: torch.device,
    config: dict,
    epoch: int,
    logger: logging.Logger
) -> dict:
    """한 epoch 학습"""
    model.train()
    total_loss = 0.0
    total_age_loss = 0.0
    total_gender_loss = 0.0
    age_correct = 0
    gender_correct = 0
    total_samples = 0
    num_batches = 0
    
    loss_config = config['training']['loss']
    age_weight = loss_config['age_weight']
    gender_weight = loss_config['gender_weight']
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        ages = batch['ages'].to(device)  # 0~100
        genders = batch['genders'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)  # Age와 Gender logits을 동시에 출력
        
        age_logits = outputs['age_logits']  # [B, 101] (0~100세)
        gender_logits = outputs['gender_logits']  # [B, 2] (Male/Female)
        
        # Loss 계산
        age_loss = criterion_age(age_logits, ages)
        gender_loss = criterion_gender(gender_logits, genders)
        
        # Multi-task loss: 가중합으로 결합
        total_loss_batch = age_weight * age_loss + gender_weight * gender_loss
        
        # Backward pass
        total_loss_batch.backward()
        optimizer.step()
        
        # 통계
        total_loss += total_loss_batch.item()
        total_age_loss += age_loss.item()
        total_gender_loss += gender_loss.item()
        
        # Accuracy 계산
        age_preds = torch.argmax(age_logits, dim=1)  # 0~100
        gender_preds = torch.argmax(gender_logits, dim=1)
        
        age_correct += (age_preds == ages).sum().item()
        gender_correct += (gender_preds == genders).sum().item()
        total_samples += images.size(0)
        num_batches += 1
        
        # Progress bar 업데이트
        pbar.set_postfix({
            'loss': f'{total_loss_batch.item():.4f}',
            'age_acc': f'{age_correct / total_samples:.3f}',
            'gen_acc': f'{gender_correct / total_samples:.3f}'
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    age_acc = age_correct / total_samples if total_samples > 0 else 0.0
    gender_acc = gender_correct / total_samples if total_samples > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'age_loss': total_age_loss / num_batches if num_batches > 0 else 0.0,
        'gender_loss': total_gender_loss / num_batches if num_batches > 0 else 0.0,
        'age_acc': age_acc,
        'gender_acc': gender_acc
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_age: nn.Module,
    criterion_gender: nn.Module,
    device: torch.device,
    config: dict,
    epoch: int,
    logger: logging.Logger
) -> dict:
    """검증"""
    model.eval()
    total_loss = 0.0
    total_age_loss = 0.0
    total_gender_loss = 0.0
    age_correct = 0
    gender_correct = 0
    total_samples = 0
    num_batches = 0
    
    loss_config = config['training']['loss']
    age_weight = loss_config['age_weight']
    gender_weight = loss_config['gender_weight']
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        
        for batch in pbar:
            images = batch['images'].to(device)
            ages = batch['ages'].to(device)  # 0~100
            genders = batch['genders'].to(device)
            
            # Forward pass
            outputs = model(images)  # Age와 Gender logits을 동시에 출력
            
            age_logits = outputs['age_logits']  # [B, 101] (0~100세)
            gender_logits = outputs['gender_logits']  # [B, 2] (Male/Female)
            
            # Loss 계산: 두 작업의 loss를 동시에 계산
            age_loss = criterion_age(age_logits, ages)
            gender_loss = criterion_gender(gender_logits, genders)
            
            # Multi-task loss: 가중합으로 결합
            total_loss_batch = age_weight * age_loss + gender_weight * gender_loss
            
            total_loss += total_loss_batch.item()
            total_age_loss += age_loss.item()
            total_gender_loss += gender_loss.item()
            
            # Accuracy 계산
            age_preds = torch.argmax(age_logits, dim=1)  # 0~100
            gender_preds = torch.argmax(gender_logits, dim=1)
            
            age_correct += (age_preds == ages).sum().item()
            gender_correct += (gender_preds == genders).sum().item()
            total_samples += images.size(0)
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{total_loss_batch.item():.4f}',
                'age_acc': f'{age_correct / total_samples:.3f}',
                'gen_acc': f'{gender_correct / total_samples:.3f}'
            })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    age_acc = age_correct / total_samples if total_samples > 0 else 0.0
    gender_acc = gender_correct / total_samples if total_samples > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'age_loss': total_age_loss / num_batches if num_batches > 0 else 0.0,
        'gender_loss': total_gender_loss / num_batches if num_batches > 0 else 0.0,
        'age_acc': age_acc,
        'gender_acc': gender_acc
    }


def main():
    parser = argparse.ArgumentParser(description='Train Age & Gender Estimation Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    
    args = parser.parse_args()
    
    # Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Config 오버라이드
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.epochs is not None:
        config['training']['num_epochs'] = args.epochs
    
    # 로깅 설정
    logger = setup_logging(
        config['paths']['log_dir'],
        config['logging']['level']
    )
    
    logger.info("=" * 50)
    logger.info("Age & Gender Estimation Training")
    logger.info("=" * 50)
    logger.info(f"Config: {config_path}")
    
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
    model = build_network(config)
    model = model.to(device)
    
    # 모델 정보 출력
    params = count_parameters(model)
    logger.info(f"Total parameters: {params['total']:,}")
    logger.info(f"Trainable parameters: {params['trainable']:,}")
    
    # 데이터셋 생성
    train_dataset = AgeGenderDataset(
        config['data']['train_path'],
        config,
        is_training=True
    )
    val_dataset = AgeGenderDataset(
        config['data']['val_path'],
        config,
        is_training=False
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        collate_fn=collate_fn
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    # Loss 함수
    loss_config = config['training']['loss']
    criterion_age = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = get_optimizer(model, config)
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # Resume from checkpoint
    start_epoch = 0
    best_score = 0.0
    
    if args.resume:
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, str(device))
        start_epoch = checkpoint_info['epoch'] + 1
        best_score = checkpoint_info['best_score']
        logger.info(f"Resumed from epoch {start_epoch}, best score: {best_score:.4f}")
    
    # 학습 루프
    num_epochs = config['training']['num_epochs']
    val_interval = config['training']['val_interval']
    save_interval = config['training']['save_interval']
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion_age, criterion_gender,
            device, config, epoch, logger
        )
        
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Age Acc: {train_metrics['age_acc']:.4f}, "
            f"Gender Acc: {train_metrics['gender_acc']:.4f}"
        )
        
        # Validation
        if (epoch + 1) % val_interval == 0:
            val_metrics = validate(
                model, val_loader, criterion_age, criterion_gender,
                device, config, epoch, logger
            )
            
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Age Acc: {val_metrics['age_acc']:.4f}, "
                f"Gender Acc: {val_metrics['gender_acc']:.4f}"
            )
            
            # Best model 저장 (monitor 기준)
            monitor = config['training']['early_stopping'].get('monitor', 'val_loss')
            if monitor == 'val_loss':
                current_score = -val_metrics['loss']  # Loss가 낮을수록 좋으므로 음수
            elif monitor == 'val_age_acc':
                current_score = val_metrics['age_acc']
            elif monitor == 'val_gender_acc':
                current_score = val_metrics['gender_acc']
            else:
                current_score = -val_metrics['loss']
            
            if current_score > best_score:
                best_score = current_score
                save_checkpoint(
                    config['paths']['checkpoint_dir'],
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
            save_checkpoint(
                config['paths']['checkpoint_dir'],
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

