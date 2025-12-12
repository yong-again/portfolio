"""
YOLO 데이터셋 증강 스크립트

이 스크립트는 YOLO 형식 데이터셋에 데이터 증강을 적용합니다.
모든 YOLO 증강 파라미터를 지원하며 YOLO 형식 어노테이션을 보존하면서
데이터셋의 증강된 버전을 생성합니다.

사용법:
    python augment_dataset.py \
        --input-dir /path/to/dataset \
        --output-dir /path/to/augmented_dataset \
        --splits train val \
        --hsv-h 0.015 \
        --hsv-s 0.7 \
        --hsv-v 0.4 \
        --degrees 10.0 \
        --translate 0.1 \
        --scale 0.5 \
        --shear 5.0 \
        --perspective 0.0001 \
        --flipud 0.0 \
        --fliplr 0.5 \
        --mosaic 0.0 \
        --mixup 0.0 \
        --cutmix 0.0 \
        --erasing 0.4 \
        --num-workers 8 \
        --augment-factor 1
"""

import argparse
import json
import logging
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(
        description='YOLO 형식 데이터셋에 데이터 증강 적용',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 입력/출력
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='입력 데이터셋 디렉토리 (images/ 및 labels/ 하위 디렉토리를 가진 YOLO 형식)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='증강된 데이터셋을 저장할 출력 디렉토리'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val'],
        choices=['train', 'val', 'test'],
        help='증강할 데이터셋 분할'
    )
    
    # HSV 증강
    parser.add_argument(
        '--hsv-h',
        type=float,
        default=0.015,
        help='색조 조정 (0.0 - 1.0)'
    )
    parser.add_argument(
        '--hsv-s',
        type=float,
        default=0.7,
        help='채도 조정 (0.0 - 1.0)'
    )
    parser.add_argument(
        '--hsv-v',
        type=float,
        default=0.4,
        help='명도 조정 (0.0 - 1.0)'
    )
    
    # 기하학적 증강
    parser.add_argument(
        '--degrees',
        type=float,
        default=0.0,
        help='회전 범위 (도 단위, 0.0 - 180.0)'
    )
    parser.add_argument(
        '--translate',
        type=float,
        default=0.1,
        help='이동 범위 (이미지 크기의 비율, 0.0 - 1.0)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='스케일 범위 (>= 0.0)'
    )
    parser.add_argument(
        '--shear',
        type=float,
        default=0.0,
        help='전단 변환 범위 (도 단위, -180 - +180)'
    )
    parser.add_argument(
        '--perspective',
        type=float,
        default=0.0,
        help='원근 변환 (0.0 - 0.001)'
    )
    
    # 뒤집기 증강
    parser.add_argument(
        '--flipud',
        type=float,
        default=0.0,
        help='상하 뒤집기 확률 (0.0 - 1.0)'
    )
    parser.add_argument(
        '--fliplr',
        type=float,
        default=0.5,
        help='좌우 뒤집기 확률 (0.0 - 1.0)'
    )
    parser.add_argument(
        '--bgr',
        type=float,
        default=0.0,
        help='BGR 채널 뒤집기 확률 (0.0 - 1.0)'
    )
    
    # 고급 증강
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='모자이크 증강 확률 (0.0 - 1.0). 참고: 전처리 시가 아닌 학습 중에 적용됩니다.'
    )
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        help='Mixup 증강 확률 (0.0 - 1.0). 참고: 전처리 시가 아닌 학습 중에 적용됩니다.'
    )
    parser.add_argument(
        '--cutmix',
        type=float,
        default=0.0,
        help='CutMix 증강 확률 (0.0 - 1.0). 참고: 전처리 시가 아닌 학습 중에 적용됩니다.'
    )
    parser.add_argument(
        '--erasing',
        type=float,
        default=0.4,
        help='랜덤 지우기 확률 (0.0 - 0.9)'
    )
    
    # Albumentations
    parser.add_argument(
        '--augmentations',
        type=str,
        default=None,
        help='사용자 정의 Albumentations 변환을 포함한 JSON 파일 경로'
    )
    
    # 처리 옵션
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='병렬 처리를 위한 워커 프로세스 수'
    )
    parser.add_argument(
        '--augment-factor',
        type=int,
        default=1,
        help='이미지당 생성할 증강 버전 수'
    )
    parser.add_argument(
        '--copy-original',
        action='store_true',
        help='원본 이미지를 출력 디렉토리에 복사 (증강 버전에 추가하여)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='재현성을 위한 랜덤 시드'
    )
    
    return parser.parse_args()


def load_yolo_annotations(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    YOLO 형식 어노테이션을 로드합니다.
    
    Returns:
        (class_id, x_center, y_center, width, height) 튜플의 리스트
    """
    if not label_path.exists():
        return []
    
    annotations = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 정규화된 좌표 검증
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and \
                   0 < width <= 1 and 0 < height <= 1:
                    annotations.append((class_id, x_center, y_center, width, height))
            except ValueError:
                continue
    
    return annotations


def save_yolo_annotations(label_path: Path, annotations: List[Tuple[int, float, float, float, float]]):
    """YOLO 형식 어노테이션을 저장합니다."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_path, 'w') as f:
        for class_id, x_center, y_center, width, height in annotations:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def yolo_to_albumentations(annotations: List[Tuple[int, float, float, float, float]], 
                           img_width: int, img_height: int) -> List[Dict]:
    """
    YOLO 형식 어노테이션을 Albumentations 형식으로 변환합니다.
    
    Returns:
        'bbox'와 'class_id' 키를 가진 딕셔너리 리스트
    """
    albu_annotations = []
    for class_id, x_center, y_center, width, height in annotations:
        # 정규화된 중심 좌표를 절대 픽셀 좌표로 변환
        x_abs = x_center * img_width
        y_abs = y_center * img_height
        w_abs = width * img_width
        h_abs = height * img_height
        
        # (x_min, y_min, x_max, y_max) 형식으로 변환
        x_min = x_abs - w_abs / 2
        y_min = y_abs - h_abs / 2
        x_max = x_abs + w_abs / 2
        y_max = y_abs + h_abs / 2
        
        # 이미지 경계로 클리핑
        x_min = max(0, min(x_min, img_width))
        y_min = max(0, min(y_min, img_height))
        x_max = max(0, min(x_max, img_width))
        y_max = max(0, min(y_max, img_height))
        
        if x_max > x_min and y_max > y_min:
            albu_annotations.append({
                'bbox': [x_min, y_min, x_max, y_max],
                'class_id': class_id
            })
    
    return albu_annotations


def albumentations_to_yolo(annotations: List[Dict], 
                           img_width: int, img_height: int) -> List[Tuple[int, float, float, float, float]]:
    """
    Albumentations 형식 어노테이션을 YOLO 형식으로 다시 변환합니다.
    
    Returns:
        (class_id, x_center, y_center, width, height) 튜플의 리스트
    """
    yolo_annotations = []
    for ann in annotations:
        bbox = ann['bbox']
        class_id = ann['class_id']
        
        x_min, y_min, x_max, y_max = bbox
        
        # 중심 좌표로 변환하고 정규화
        width_abs = x_max - x_min
        height_abs = y_max - y_min
        x_center_abs = x_min + width_abs / 2
        y_center_abs = y_min + height_abs / 2
        
        # 정규화
        x_center = x_center_abs / img_width
        y_center = y_center_abs / img_height
        width = width_abs / img_width
        height = height_abs / img_height
        
        # 검증
        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and \
           0 < width <= 1 and 0 < height <= 1:
            yolo_annotations.append((class_id, x_center, y_center, width, height))
    
    return yolo_annotations


def create_augmentation_pipeline(args: argparse.Namespace, is_training: bool = True) -> A.Compose:
    """
    인자에 기반하여 Albumentations 증강 파이프라인을 생성합니다.
    
    Args:
        args: 파싱된 명령줄 인자
        is_training: 학습용인지 여부 (적용되는 증강에 영향을 줌)
    
    Returns:
        Albumentations Compose 객체
    """
    transforms = []
    
    # HSV 증강
    if args.hsv_h > 0 or args.hsv_s > 0 or args.hsv_v > 0:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=int(args.hsv_h * 180),
            sat_shift_limit=int(args.hsv_s * 255),
            val_shift_limit=int(args.hsv_v * 255),
            p=1.0 if is_training else 0.0
        ))
    
    # 기하학적 증강
    if args.degrees > 0:
        transforms.append(A.Rotate(
            limit=int(args.degrees),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0 if is_training else 0.0
        ))
    
    if args.translate > 0:
        transforms.append(A.ShiftScaleRotate(
            shift_limit_x=args.translate,
            shift_limit_y=args.translate,
            scale_limit=0,
            rotate_limit=0,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0 if is_training else 0.0
        ))
    
    if args.scale > 0:
        transforms.append(A.RandomScale(
            scale_limit=(0, args.scale),
            p=1.0 if is_training else 0.0
        ))
    
    if args.shear > 0:
        transforms.append(A.Affine(
            shear=args.shear,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0 if is_training else 0.0
        ))
    
    if args.perspective > 0:
        transforms.append(A.Perspective(
            scale=(0, args.perspective),
            pad_mode=cv2.BORDER_CONSTANT,
            pad_val=0,
            p=1.0 if is_training else 0.0
        ))
    
    # 뒤집기 증강
    if args.flipud > 0:
        transforms.append(A.VerticalFlip(p=args.flipud if is_training else 0.0))
    
    if args.fliplr > 0:
        transforms.append(A.HorizontalFlip(p=args.fliplr if is_training else 0.0))
    
    if args.bgr > 0:
        transforms.append(A.ChannelShuffle(p=args.bgr if is_training else 0.0))
    
    # 랜덤 지우기
    if args.erasing > 0:
        transforms.append(A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            p=args.erasing if is_training else 0.0
        ))
    
    # 제공된 경우 JSON에서 사용자 정의 증강 로드
    if args.augmentations:
        aug_path = Path(args.augmentations)
        if aug_path.exists():
            with open(aug_path, 'r') as f:
                custom_augs = json.load(f)
                # 사용자 정의 증강 파싱 (간소화됨 - 전체 Albumentations JSON 파서가 필요함)
                logger.info(f"사용자 정의 증강을 로드했습니다: {aug_path}")
    
    # bbox 파라미터와 함께 Compose 반환
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_id'],
            min_visibility=0.1  # 너무 작아진 박스 필터링
        )
    )


def process_single_image(args_tuple: Tuple) -> Dict:
    """
    단일 이미지에 증강을 적용하여 처리합니다.
    
    Args:
        args_tuple: (image_path, label_path, output_image_dir, output_label_dir, 
                    transform_args, is_training, augment_factor, copy_original, seed_offset) 튜플
    
    Returns:
        처리 통계를 담은 딕셔너리
    """
    image_path, label_path, output_image_dir, output_label_dir, transform_args, \
        is_training, augment_factor, copy_original, seed_offset = args_tuple
    
    # 워커 프로세스에서 변환 생성 (멀티프로세싱 호환성을 위해)
    transform = create_augmentation_pipeline(transform_args, is_training=is_training)
    
    stats = {
        'processed': 0,
        'failed': 0,
        'annotations': 0
    }
    
    try:
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            stats['failed'] = 1
            return stats
        
        img_height, img_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 어노테이션 로드
        annotations = load_yolo_annotations(label_path)
        if not annotations:
            # 어노테이션이 없으면 빈 라벨 파일 생성
            output_label_path = output_label_dir / label_path.name
            save_yolo_annotations(output_label_path, [])
            if copy_original:
                shutil.copy2(image_path, output_image_dir / image_path.name)
            stats['processed'] = 1
            return stats
        
        # Albumentations 형식으로 변환
        albu_annotations = yolo_to_albumentations(annotations, img_width, img_height)
        bboxes = [ann['bbox'] for ann in albu_annotations]
        class_ids = [ann['class_id'] for ann in albu_annotations]
        
        # 증강 적용
        image_stem = image_path.stem
        image_ext = image_path.suffix
        
        for aug_idx in range(augment_factor):
            # 이 증강을 위한 랜덤 시드 설정
            random.seed(seed_offset + aug_idx)
            np.random.seed(seed_offset + aug_idx)
            
            # 변환 적용
            transformed = transform(
                image=image_rgb,
                bboxes=bboxes,
                class_id=class_ids
            )
            
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_ids = transformed['class_id']
            
            # YOLO 형식으로 다시 변환
            new_annotations = []
            for bbox, class_id in zip(transformed_bboxes, transformed_class_ids):
                new_annotations.append({
                    'bbox': bbox,
                    'class_id': class_id
                })
            
            new_height, new_width = transformed_image.shape[:2]
            yolo_annotations = albumentations_to_yolo(new_annotations, new_width, new_height)
            
            # 증강된 이미지 저장
            if aug_idx == 0:
                output_image_name = f"{image_stem}{image_ext}"
            else:
                output_image_name = f"{image_stem}_aug{aug_idx}{image_ext}"
            
            output_image_path = output_image_dir / output_image_name
            output_label_path = output_label_dir / f"{output_image_name.rsplit('.', 1)[0]}.txt"
            
            # OpenCV를 위해 RGB를 BGR로 다시 변환
            image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_image_path), image_bgr)
            
            # 어노테이션 저장
            save_yolo_annotations(output_label_path, yolo_annotations)
            
            stats['processed'] += 1
            stats['annotations'] += len(yolo_annotations)
        
        # 요청된 경우 원본 복사
        if copy_original:
            shutil.copy2(image_path, output_image_dir / image_path.name)
            shutil.copy2(label_path, output_label_dir / label_path.name)
            stats['processed'] += 1
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        stats['failed'] = 1
    
    return stats


def augment_split(input_dir: Path, output_dir: Path, split: str, args: argparse.Namespace):
    """단일 데이터셋 분할에 증강을 적용합니다."""
    split_input_images = input_dir / 'images' / split
    split_input_labels = input_dir / 'labels' / split
    split_output_images = output_dir / 'images' / split
    split_output_labels = output_dir / 'labels' / split
    
    if not split_input_images.exists():
        logger.warning(f"Split {split} images directory not found: {split_input_images}")
        return
    
    # 출력 디렉토리 생성
    split_output_images.mkdir(parents=True, exist_ok=True)
    split_output_labels.mkdir(parents=True, exist_ok=True)
    
    # 모든 이미지 찾기
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(split_input_images.glob(f'*{ext}')))
    
    if not image_paths:
        logger.warning(f"No images found in {split_input_images}")
        return
    
    logger.info(f"분할 '{split}'에서 {len(image_paths)}개의 이미지 처리 중")
    
    # 증강 파이프라인 생성
    is_training = (split == 'train')
    transform = create_augmentation_pipeline(args, is_training=is_training)
    
    # 워커를 위한 인자 준비
    worker_args = []
    for idx, image_path in enumerate(image_paths):
        label_path = split_input_labels / f"{image_path.stem}.txt"
        seed_offset = args.seed + idx * 1000  # 이미지당 고유 시드
        worker_args.append((
            image_path,
            label_path,
            split_output_images,
            split_output_labels,
            args,  # 변환 생성을 위해 args 객체 전달
            is_training,
            args.augment_factor,
            args.copy_original,
            seed_offset
        ))
    
    # 이미지 처리
    total_stats = {'processed': 0, 'failed': 0, 'annotations': 0}
    
    # 진행 표시줄 생성
    pbar = tqdm(
        total=len(worker_args),
        desc=f"Augmenting {split}",
        unit="image",
        ncols=100
    )
    
    if args.num_workers == 1:
        # 단일 프로세스 모드
        for worker_arg in worker_args:
            stats = process_single_image(worker_arg)
            for key in total_stats:
                total_stats[key] += stats[key]
            pbar.update(1)
            pbar.set_postfix({
                'processed': total_stats['processed'],
                'failed': total_stats['failed'],
                'annotations': total_stats['annotations']
            })
    else:
        # 멀티프로세싱 모드
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_single_image, arg): arg for arg in worker_args}
            
            for future in as_completed(futures):
                try:
                    stats = future.result()
                    for key in total_stats:
                        total_stats[key] += stats[key]
                    pbar.update(1)
                    pbar.set_postfix({
                        'processed': total_stats['processed'],
                        'failed': total_stats['failed'],
                        'annotations': total_stats['annotations']
                    })
                except Exception as e:
                    logger.error(f"Error in worker: {e}", exc_info=True)
                    total_stats['failed'] += 1
                    pbar.update(1)
    
    pbar.close()
    
    logger.info(f"분할 '{split}' 완료:")
    logger.info(f"  처리됨: {total_stats['processed']}")
    logger.info(f"  실패: {total_stats['failed']}")
    logger.info(f"  총 어노테이션: {total_stats['annotations']}")


def copy_dataset_yaml(input_dir: Path, output_dir: Path):
    """dataset.yaml을 출력 디렉토리로 복사하고 경로를 업데이트합니다."""
    input_yaml = input_dir / 'dataset.yaml'
    output_yaml = output_dir / 'dataset.yaml'
    
    if input_yaml.exists():
        with open(input_yaml, 'r') as f:
            content = f.read()
        
        # 경로 업데이트
        content = content.replace(f'path: {input_dir}', f'path: {output_dir}')
        
        with open(output_yaml, 'w') as f:
            f.write(content)
        
        logger.info(f"Copied and updated dataset.yaml to {output_yaml}")
    else:
        logger.warning(f"dataset.yaml not found in {input_dir}")


def main():
    """메인 함수."""
    args = parse_args()
    
    # 랜덤 시드 설정
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    logger.info("=" * 60)
    logger.info("YOLO 데이터셋 증강")
    logger.info("=" * 60)
    logger.info(f"입력 디렉토리: {input_dir}")
    logger.info(f"출력 디렉토리: {output_dir}")
    logger.info(f"분할: {args.splits}")
    logger.info(f"증강 배수: {args.augment_factor}")
    logger.info(f"워커 수: {args.num_workers}")
    logger.info("=" * 60)
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 각 분할 처리
    for split in args.splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"분할 처리 중: {split}")
        logger.info(f"{'='*60}")
        augment_split(input_dir, output_dir, split, args)
    
    # dataset.yaml 복사
    copy_dataset_yaml(input_dir, output_dir)
    
    logger.info("=" * 60)
    logger.info("증강 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

