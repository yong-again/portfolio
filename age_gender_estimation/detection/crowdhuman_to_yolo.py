#!/usr/bin/env python3
"""
Convert CrowdHuman JSON annotations to YOLO format labels.

This script processes CrowdHuman-style annotations and converts them to
YOLO format with support for train/val/test splits and multiprocessing.
"""

import argparse
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
except ImportError:
    raise ImportError("PIL/Pillow is required. Install with: pip install Pillow")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Convert CrowdHuman JSON annotations to YOLO format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--root-dir',
        type=str,
        required=True,
        help='Output root directory for YOLO dataset'
    )
    
    parser.add_argument(
        '--raw-img-root',
        type=str,
        required=True,
        help='Base directory of original images'
    )
    
    parser.add_argument(
        '--raw-ann-root',
        type=str,
        required=True,
        help='Base directory of JSON annotation files'
    )
    
    for split in ['train', 'val', 'test']:
        parser.add_argument(
            f'--{split}-src-type',
            type=str,
            choices=['dir', 'txt', 'list'],
            help=f'Source type for {split} split (dir/txt/list)'
        )
        parser.add_argument(
            f'--{split}-src',
            type=str,
            help=f'Source path for {split} split'
        )
    
    parser.add_argument(
        '--box-type',
        type=str,
        choices=['vbox', 'fbox', 'hbox'],
        default='fbox',
        help='Bounding box type to use (vbox/fbox/hbox)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of worker processes (1 = single process)'
    )
    
    img_group = parser.add_mutually_exclusive_group()
    img_group.add_argument(
        '--copy-images',
        action='store_true',
        help='Copy images to output directory'
    )
    img_group.add_argument(
        '--symlink-images',
        action='store_true',
        help='Create symlinks for images in output directory'
    )
    
    parser.add_argument(
        '--create-yaml',
        action='store_true',
        help='Create Ultralytics-style dataset.yaml file'
    )
    
    return parser.parse_args()


def resolve_split_images(
    src_type: Optional[str],
    src: Optional[str],
    raw_img_root: Path
) -> List[Path]:
    """
    Resolve image paths for a split based on source type.
    
    Args:
        src_type: Type of source ('dir', 'txt', 'list', or None)
        src: Source path (directory, text file, JSON file, or comma-separated list)
        raw_img_root: Base directory for raw images
    
    Returns:
        List of absolute image paths
    """
    if not src_type or not src:
        return []
    
    image_paths = []
    
    if src_type == 'dir':
        dir_path = Path(src)
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return []
        
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_paths.extend(dir_path.glob(f'*{ext}'))
            image_paths.extend(dir_path.rglob(f'*{ext}'))
    
    elif src_type == 'txt':
        txt_path = Path(src)
        if not txt_path.exists():
            logger.warning(f"Text file not found: {txt_path}")
            return []
        
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path = Path(line)
                if not img_path.is_absolute():
                    img_path = raw_img_root / img_path
                if img_path.exists():
                    image_paths.append(img_path.resolve())
    
    elif src_type == 'list':
        if ',' in src and not Path(src).exists():
            paths_str = src.split(',')
            for path_str in paths_str:
                path_str = path_str.strip()
                if not path_str:
                    continue
                img_path = Path(path_str)
                if not img_path.is_absolute():
                    img_path = raw_img_root / img_path
                if img_path.exists():
                    image_paths.append(img_path.resolve())
        else:
            json_path = Path(src)
            if not json_path.exists():
                logger.warning(f"JSON file not found: {json_path}")
                return []
            
            with open(json_path, 'r') as f:
                paths_list = json.load(f)
                for path_str in paths_list:
                    img_path = Path(path_str)
                    if not img_path.is_absolute():
                        img_path = raw_img_root / img_path
                    if img_path.exists():
                        image_paths.append(img_path.resolve())
    
    return sorted(set(image_paths))


def image_path_to_json_path(image_path: Path, raw_ann_root: Path) -> Path:
    """
    Map an image path to its corresponding JSON annotation file.
    
    Args:
        image_path: Path to the image file
        raw_ann_root: Base directory for JSON annotations
    
    Returns:
        Path to the JSON annotation file
    """
    image_stem = image_path.stem
    
    json_candidates = [
        raw_ann_root / f"{image_stem}.json",
        raw_ann_root / "annotation_train" / f"{image_stem}.json",
        raw_ann_root / "annotation_val" / f"{image_stem}.json",
        raw_ann_root / "annotation_test" / f"{image_stem}.json",
    ]
    
    for json_path in json_candidates:
        if json_path.exists():
            return json_path
    
    stem_underscore = image_stem.replace(',', '_')
    json_candidates_alt = [
        raw_ann_root / f"{stem_underscore}.json",
        raw_ann_root / "annotation_train" / f"{stem_underscore}.json",
        raw_ann_root / "annotation_val" / f"{stem_underscore}.json",
        raw_ann_root / "annotation_test" / f"{stem_underscore}.json",
    ]
    
    for json_path in json_candidates_alt:
        if json_path.exists():
            return json_path
    
    return raw_ann_root / f"{image_stem}.json"


def load_image_size(image_path: Path) -> Tuple[int, int]:
    """
    Load image and return its width and height.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


def load_annotation(json_path: Path) -> Optional[Dict]:
    """
    Load JSON annotation file.
    
    Args:
        json_path: Path to the JSON annotation file
    
    Returns:
        Parsed JSON dictionary or None if file doesn't exist
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load annotation {json_path}: {e}")
        return None


def convert_bbox_xywh_to_yolo(
    x: float, y: float, w: float, h: float,
    img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from absolute pixel coordinates to YOLO format.
    
    Args:
        x, y, w, h: Bounding box coordinates (x, y, width, height)
        img_w, img_h: Image width and height
    
    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0, 1]
    """
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    ww = w / img_w
    hh = h / img_h
    
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    ww = max(0.0, min(1.0, ww))
    hh = max(0.0, min(1.0, hh))
    
    return (xc, yc, ww, hh)


def tag_to_class_id(tag: str) -> Optional[int]:
    """
    Convert tag string to class ID.
    
    Args:
        tag: Tag string ('person' or 'mask')
    
    Returns:
        Class ID (0 for 'person', 1 for 'mask', None for unknown)
    """
    tag_map = {
        'person': 0,
        'mask': 1
    }
    return tag_map.get(tag.lower())


def process_single_image(args_tuple: Tuple) -> Dict:
    """
    Process a single image: load annotation, convert to YOLO format, write label file.
    
    Args:
        args_tuple: Tuple containing:
            - image_path: Path to the image
            - split: Split name ('train', 'val', 'test')
            - root_dir: Output root directory
            - raw_ann_root: Base directory for JSON annotations
            - box_type: Bounding box type ('vbox', 'fbox', 'hbox')
            - copy_images: Whether to copy images
            - symlink_images: Whether to symlink images
    
    Returns:
        Dictionary with processing statistics
    """
    (image_path, split, root_dir, raw_ann_root, box_type,
     copy_images, symlink_images) = args_tuple
    
    stats = {
        'processed': 0,
        'failed': 0,
        'empty': 0,
        'class_counts': {0: 0, 1: 0}
    }
    
    output_img_dir = Path(root_dir) / 'images' / split
    output_label_dir = Path(root_dir) / 'labels' / split
    output_img_path = output_img_dir / image_path.name
    output_label_path = output_label_dir / f"{image_path.stem}.txt"
    
    try:
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            stats['failed'] = 1
            return stats
        
        img_w, img_h = load_image_size(image_path)
        
        json_path = image_path_to_json_path(image_path, raw_ann_root)
        ann_data = load_annotation(json_path)
        
        yolo_lines = []
        
        if ann_data is None:
            logger.warning(f"No annotation found for {image_path}, creating empty label file")
        else:
            gtboxes = ann_data.get('gtboxes', [])
            
            for gtbox in gtboxes:
                extra = gtbox.get('extra', {})
                if extra.get('ignore', 0) == 1:
                    continue
                
                tag = gtbox.get('tag', '')
                class_id = tag_to_class_id(tag)
                if class_id is None:
                    continue
                
                bbox = gtbox.get(box_type)
                if not bbox or len(bbox) != 4:
                    continue
                
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    continue
                
                xc, yc, ww, hh = convert_bbox_xywh_to_yolo(x, y, w, h, img_w, img_h)
                yolo_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")
                stats['class_counts'][class_id] += 1
        
        if copy_images:
            shutil.copy2(image_path, output_img_path)
        elif symlink_images:
            if output_img_path.exists():
                output_img_path.unlink()
            output_img_path.symlink_to(image_path.resolve())
        
        with open(output_label_path, 'w') as f:
            if yolo_lines:
                f.write('\n'.join(yolo_lines))
                f.write('\n')
        
        if len(yolo_lines) == 0:
            stats['empty'] = 1
        else:
            stats['processed'] = 1
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        stats['failed'] = 1
    
    return stats


def create_dataset_yaml(root_dir: Path, output_path: Path):
    """Create Ultralytics-style dataset.yaml file."""
    yaml_content = f"""path: {root_dir.absolute()}
train: images/train
val: images/val
test: images/test

nc: 2
names: ["person", "mask"]
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content)
    logger.info(f"Created dataset.yaml at {output_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    root_dir = Path(args.root_dir)
    raw_img_root = Path(args.raw_img_root)
    raw_ann_root = Path(args.raw_ann_root)
    
    root_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {}
    for split in ['train', 'val', 'test']:
        src_type = getattr(args, f'{split}_src_type')
        src = getattr(args, f'{split}_src')
        image_paths = resolve_split_images(src_type, src, raw_img_root)
        splits[split] = image_paths
        logger.info(f"{split.capitalize()} split: {len(image_paths)} images")
    
    total_images = sum(len(paths) for paths in splits.values())
    if total_images == 0:
        logger.error("No images found in any split. Exiting.")
        return
    
    logger.info(f"Total images to process: {total_images}")
    logger.info(f"Using {args.num_workers} worker(s)")
    logger.info(f"Box type: {args.box_type}")
    
    for split in ['train', 'val', 'test']:
        if splits[split]:
            (root_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (root_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    all_tasks = []
    for split, image_paths in splits.items():
        for image_path in image_paths:
            all_tasks.append((
                image_path,
                split,
                str(root_dir),
                raw_ann_root,
                args.box_type,
                args.copy_images,
                args.symlink_images
            ))
    
    total_stats = {
        'processed': 0,
        'failed': 0,
        'empty': 0,
        'class_counts': {0: 0, 1: 0}
    }
    
    if args.num_workers == 1:
        for task in all_tasks:
            stats = process_single_image(task)
            total_stats['processed'] += stats['processed']
            total_stats['failed'] += stats['failed']
            total_stats['empty'] += stats['empty']
            for class_id in [0, 1]:
                total_stats['class_counts'][class_id] += stats['class_counts'][class_id]
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_task = {
                executor.submit(process_single_image, task): task
                for task in all_tasks
            }
            
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                if completed % 100 == 0:
                    logger.info(f"Progress: {completed}/{total_images}")
                
                try:
                    stats = future.result()
                    total_stats['processed'] += stats['processed']
                    total_stats['failed'] += stats['failed']
                    total_stats['empty'] += stats['empty']
                    for class_id in [0, 1]:
                        total_stats['class_counts'][class_id] += stats['class_counts'][class_id]
                except Exception as e:
                    logger.error(f"Task failed with exception: {e}")
                    total_stats['failed'] += 1
    
    logger.info("=" * 60)
    logger.info("Conversion Summary:")
    logger.info(f"  Successfully processed: {total_stats['processed']}")
    logger.info(f"  Empty labels (no valid boxes): {total_stats['empty']}")
    logger.info(f"  Failed: {total_stats['failed']}")
    logger.info(f"  Class 0 (person): {total_stats['class_counts'][0]}")
    logger.info(f"  Class 1 (mask): {total_stats['class_counts'][1]}")
    logger.info("=" * 60)
    
    if args.create_yaml:
        yaml_path = root_dir / 'dataset.yaml'
        create_dataset_yaml(root_dir, yaml_path)
    
    logger.info("Conversion completed!")


if __name__ == '__main__':
    main()

