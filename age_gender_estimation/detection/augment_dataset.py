"""
YOLO Dataset Augmentation Script

This script applies data augmentations to YOLO format datasets.
Supports all YOLO augmentation parameters and creates augmented versions
of the dataset while preserving YOLO format annotations.

Usage:
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply data augmentations to YOLO format dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input dataset directory (YOLO format with images/ and labels/ subdirectories)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for augmented dataset'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val'],
        choices=['train', 'val', 'test'],
        help='Dataset splits to augment'
    )
    
    # HSV Augmentations
    parser.add_argument(
        '--hsv-h',
        type=float,
        default=0.015,
        help='Hue adjustment (0.0 - 1.0)'
    )
    parser.add_argument(
        '--hsv-s',
        type=float,
        default=0.7,
        help='Saturation adjustment (0.0 - 1.0)'
    )
    parser.add_argument(
        '--hsv-v',
        type=float,
        default=0.4,
        help='Value (brightness) adjustment (0.0 - 1.0)'
    )
    
    # Geometric Augmentations
    parser.add_argument(
        '--degrees',
        type=float,
        default=0.0,
        help='Rotation range in degrees (0.0 - 180.0)'
    )
    parser.add_argument(
        '--translate',
        type=float,
        default=0.1,
        help='Translation range as fraction of image size (0.0 - 1.0)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Scale range (>= 0.0)'
    )
    parser.add_argument(
        '--shear',
        type=float,
        default=0.0,
        help='Shear range in degrees (-180 - +180)'
    )
    parser.add_argument(
        '--perspective',
        type=float,
        default=0.0,
        help='Perspective transformation (0.0 - 0.001)'
    )
    
    # Flip Augmentations
    parser.add_argument(
        '--flipud',
        type=float,
        default=0.0,
        help='Upside down flip probability (0.0 - 1.0)'
    )
    parser.add_argument(
        '--fliplr',
        type=float,
        default=0.5,
        help='Left-right flip probability (0.0 - 1.0)'
    )
    parser.add_argument(
        '--bgr',
        type=float,
        default=0.0,
        help='BGR channel flip probability (0.0 - 1.0)'
    )
    
    # Advanced Augmentations
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='Mosaic augmentation probability (0.0 - 1.0). Note: Applied during training, not in preprocessing.'
    )
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        help='Mixup augmentation probability (0.0 - 1.0). Note: Applied during training, not in preprocessing.'
    )
    parser.add_argument(
        '--cutmix',
        type=float,
        default=0.0,
        help='CutMix augmentation probability (0.0 - 1.0). Note: Applied during training, not in preprocessing.'
    )
    parser.add_argument(
        '--erasing',
        type=float,
        default=0.4,
        help='Random erasing probability (0.0 - 0.9)'
    )
    
    # Albumentations
    parser.add_argument(
        '--augmentations',
        type=str,
        default=None,
        help='Path to JSON file with custom Albumentations transforms'
    )
    
    # Processing Options
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Number of worker processes for parallel processing'
    )
    parser.add_argument(
        '--augment-factor',
        type=int,
        default=1,
        help='Number of augmented versions to create per image'
    )
    parser.add_argument(
        '--copy-original',
        action='store_true',
        help='Copy original images to output directory (in addition to augmented versions)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def load_yolo_annotations(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Load YOLO format annotations.
    
    Returns:
        List of (class_id, x_center, y_center, width, height) tuples
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
                
                # Validate normalized coordinates
                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and \
                   0 < width <= 1 and 0 < height <= 1:
                    annotations.append((class_id, x_center, y_center, width, height))
            except ValueError:
                continue
    
    return annotations


def save_yolo_annotations(label_path: Path, annotations: List[Tuple[int, float, float, float, float]]):
    """Save YOLO format annotations."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_path, 'w') as f:
        for class_id, x_center, y_center, width, height in annotations:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def yolo_to_albumentations(annotations: List[Tuple[int, float, float, float, float]], 
                           img_width: int, img_height: int) -> List[Dict]:
    """
    Convert YOLO format annotations to Albumentations format.
    
    Returns:
        List of dicts with 'bbox' and 'class_id' keys
    """
    albu_annotations = []
    for class_id, x_center, y_center, width, height in annotations:
        # Convert normalized center coordinates to absolute pixel coordinates
        x_abs = x_center * img_width
        y_abs = y_center * img_height
        w_abs = width * img_width
        h_abs = height * img_height
        
        # Convert to (x_min, y_min, x_max, y_max) format
        x_min = x_abs - w_abs / 2
        y_min = y_abs - h_abs / 2
        x_max = x_abs + w_abs / 2
        y_max = y_abs + h_abs / 2
        
        # Clip to image boundaries
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
    Convert Albumentations format annotations back to YOLO format.
    
    Returns:
        List of (class_id, x_center, y_center, width, height) tuples
    """
    yolo_annotations = []
    for ann in annotations:
        bbox = ann['bbox']
        class_id = ann['class_id']
        
        x_min, y_min, x_max, y_max = bbox
        
        # Convert to center coordinates and normalize
        width_abs = x_max - x_min
        height_abs = y_max - y_min
        x_center_abs = x_min + width_abs / 2
        y_center_abs = y_min + height_abs / 2
        
        # Normalize
        x_center = x_center_abs / img_width
        y_center = y_center_abs / img_height
        width = width_abs / img_width
        height = height_abs / img_height
        
        # Validate
        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and \
           0 < width <= 1 and 0 < height <= 1:
            yolo_annotations.append((class_id, x_center, y_center, width, height))
    
    return yolo_annotations


def create_augmentation_pipeline(args: argparse.Namespace, is_training: bool = True) -> A.Compose:
    """
    Create Albumentations augmentation pipeline based on arguments.
    
    Args:
        args: Parsed command-line arguments
        is_training: Whether this is for training (affects which augmentations are applied)
    
    Returns:
        Albumentations Compose object
    """
    transforms = []
    
    # HSV augmentations
    if args.hsv_h > 0 or args.hsv_s > 0 or args.hsv_v > 0:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=int(args.hsv_h * 180),
            sat_shift_limit=int(args.hsv_s * 255),
            val_shift_limit=int(args.hsv_v * 255),
            p=1.0 if is_training else 0.0
        ))
    
    # Geometric augmentations
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
    
    # Flip augmentations
    if args.flipud > 0:
        transforms.append(A.VerticalFlip(p=args.flipud if is_training else 0.0))
    
    if args.fliplr > 0:
        transforms.append(A.HorizontalFlip(p=args.fliplr if is_training else 0.0))
    
    if args.bgr > 0:
        transforms.append(A.ChannelShuffle(p=args.bgr if is_training else 0.0))
    
    # Random erasing
    if args.erasing > 0:
        transforms.append(A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            p=args.erasing if is_training else 0.0
        ))
    
    # Load custom augmentations from JSON if provided
    if args.augmentations:
        aug_path = Path(args.augmentations)
        if aug_path.exists():
            with open(aug_path, 'r') as f:
                custom_augs = json.load(f)
                # Parse custom augmentations (simplified - would need full Albumentations JSON parser)
                logger.info(f"Loaded custom augmentations from {aug_path}")
    
    # Return Compose with bbox parameters
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_id'],
            min_visibility=0.1  # Filter out boxes that become too small
        )
    )


def process_single_image(args_tuple: Tuple) -> Dict:
    """
    Process a single image with augmentations.
    
    Args:
        args_tuple: Tuple of (image_path, label_path, output_image_dir, output_label_dir, 
                             transform_args, is_training, augment_factor, copy_original, seed_offset)
    
    Returns:
        Dictionary with processing statistics
    """
    image_path, label_path, output_image_dir, output_label_dir, transform_args, \
        is_training, augment_factor, copy_original, seed_offset = args_tuple
    
    # Create transform in worker process (for multiprocessing compatibility)
    transform = create_augmentation_pipeline(transform_args, is_training=is_training)
    
    stats = {
        'processed': 0,
        'failed': 0,
        'annotations': 0
    }
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to load image: {image_path}")
            stats['failed'] = 1
            return stats
        
        img_height, img_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        annotations = load_yolo_annotations(label_path)
        if not annotations:
            # Create empty label file if no annotations
            output_label_path = output_label_dir / label_path.name
            save_yolo_annotations(output_label_path, [])
            if copy_original:
                shutil.copy2(image_path, output_image_dir / image_path.name)
            stats['processed'] = 1
            return stats
        
        # Convert to Albumentations format
        albu_annotations = yolo_to_albumentations(annotations, img_width, img_height)
        bboxes = [ann['bbox'] for ann in albu_annotations]
        class_ids = [ann['class_id'] for ann in albu_annotations]
        
        # Apply augmentations
        image_stem = image_path.stem
        image_ext = image_path.suffix
        
        for aug_idx in range(augment_factor):
            # Set random seed for this augmentation
            random.seed(seed_offset + aug_idx)
            np.random.seed(seed_offset + aug_idx)
            
            # Apply transformation
            transformed = transform(
                image=image_rgb,
                bboxes=bboxes,
                class_id=class_ids
            )
            
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            transformed_class_ids = transformed['class_id']
            
            # Convert back to YOLO format
            new_annotations = []
            for bbox, class_id in zip(transformed_bboxes, transformed_class_ids):
                new_annotations.append({
                    'bbox': bbox,
                    'class_id': class_id
                })
            
            new_height, new_width = transformed_image.shape[:2]
            yolo_annotations = albumentations_to_yolo(new_annotations, new_width, new_height)
            
            # Save augmented image
            if aug_idx == 0:
                output_image_name = f"{image_stem}{image_ext}"
            else:
                output_image_name = f"{image_stem}_aug{aug_idx}{image_ext}"
            
            output_image_path = output_image_dir / output_image_name
            output_label_path = output_label_dir / f"{output_image_name.rsplit('.', 1)[0]}.txt"
            
            # Convert RGB back to BGR for OpenCV
            image_bgr = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_image_path), image_bgr)
            
            # Save annotations
            save_yolo_annotations(output_label_path, yolo_annotations)
            
            stats['processed'] += 1
            stats['annotations'] += len(yolo_annotations)
        
        # Copy original if requested
        if copy_original:
            shutil.copy2(image_path, output_image_dir / image_path.name)
            shutil.copy2(label_path, output_label_dir / label_path.name)
            stats['processed'] += 1
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        stats['failed'] = 1
    
    return stats


def augment_split(input_dir: Path, output_dir: Path, split: str, args: argparse.Namespace):
    """Augment a single dataset split."""
    split_input_images = input_dir / 'images' / split
    split_input_labels = input_dir / 'labels' / split
    split_output_images = output_dir / 'images' / split
    split_output_labels = output_dir / 'labels' / split
    
    if not split_input_images.exists():
        logger.warning(f"Split {split} images directory not found: {split_input_images}")
        return
    
    # Create output directories
    split_output_images.mkdir(parents=True, exist_ok=True)
    split_output_labels.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(split_input_images.glob(f'*{ext}')))
    
    if not image_paths:
        logger.warning(f"No images found in {split_input_images}")
        return
    
    logger.info(f"Processing {len(image_paths)} images from split '{split}'")
    
    # Create augmentation pipeline
    is_training = (split == 'train')
    transform = create_augmentation_pipeline(args, is_training=is_training)
    
    # Prepare arguments for workers
    worker_args = []
    for idx, image_path in enumerate(image_paths):
        label_path = split_input_labels / f"{image_path.stem}.txt"
        seed_offset = args.seed + idx * 1000  # Unique seed per image
        worker_args.append((
            image_path,
            label_path,
            split_output_images,
            split_output_labels,
            args,  # Pass args object for transform creation
            is_training,
            args.augment_factor,
            args.copy_original,
            seed_offset
        ))
    
    # Process images
    total_stats = {'processed': 0, 'failed': 0, 'annotations': 0}
    
    # Create progress bar
    pbar = tqdm(
        total=len(worker_args),
        desc=f"Augmenting {split}",
        unit="image",
        ncols=100
    )
    
    if args.num_workers == 1:
        # Single process mode
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
        # Multiprocessing mode
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
    
    logger.info(f"Split '{split}' completed:")
    logger.info(f"  Processed: {total_stats['processed']}")
    logger.info(f"  Failed: {total_stats['failed']}")
    logger.info(f"  Total annotations: {total_stats['annotations']}")


def copy_dataset_yaml(input_dir: Path, output_dir: Path):
    """Copy dataset.yaml to output directory and update paths."""
    input_yaml = input_dir / 'dataset.yaml'
    output_yaml = output_dir / 'dataset.yaml'
    
    if input_yaml.exists():
        with open(input_yaml, 'r') as f:
            content = f.read()
        
        # Update path
        content = content.replace(f'path: {input_dir}', f'path: {output_dir}')
        
        with open(output_yaml, 'w') as f:
            f.write(content)
        
        logger.info(f"Copied and updated dataset.yaml to {output_yaml}")
    else:
        logger.warning(f"dataset.yaml not found in {input_dir}")


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    logger.info("=" * 60)
    logger.info("YOLO Dataset Augmentation")
    logger.info("=" * 60)
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Augment factor: {args.augment_factor}")
    logger.info(f"Workers: {args.num_workers}")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in args.splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'='*60}")
        augment_split(input_dir, output_dir, split, args)
    
    # Copy dataset.yaml
    copy_dataset_yaml(input_dir, output_dir)
    
    logger.info("=" * 60)
    logger.info("Augmentation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

