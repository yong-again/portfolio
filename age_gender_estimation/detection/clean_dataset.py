"""
CrowdHuman YOLO 데이터셋 정리 스크립트

EDA 결과를 바탕으로 다음 작업을 수행합니다:
1. 중복 이미지 제거
2. 유효하지 않은 바운딩 박스 필터링
3. 저품질 이미지 제거 (블러 임계값)
4. 바운딩 박스를 이미지 경계로 클리핑
5. 이상치 머리 박스 제거 (크기 기반 이상치 탐지)
6. 정리 후 빈 어노테이션 제거
7. 정리 보고서 생성
"""

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import imagehash
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_blur_score(image_path: Path) -> float:
    """Laplacian 분산을 사용하여 블러 점수를 계산합니다."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    except Exception as e:
        logger.warning(f"Error calculating blur for {image_path}: {e}")
        return 0.0


def calculate_image_hash(image_path: Path) -> Optional[str]:
    """중복 탐지를 위한 perceptual hash를 계산합니다."""
    try:
        img = Image.open(image_path)
        img_hash = str(imagehash.phash(img))
        return img_hash
    except Exception as e:
        logger.warning(f"Error calculating hash for {image_path}: {e}")
        return None


def clip_bbox(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """바운딩 박스를 이미지 경계로 클리핑합니다."""
    # 양수 크기 확인
    if w <= 0 or h <= 0:
        return None
    
    # 좌표 클리핑
    x = max(0, min(x, img_w))
    y = max(0, min(y, img_h))
    
    # 경계를 벗어나는 경우 너비와 높이 조정
    if x + w > img_w:
        w = img_w - x
    if y + h > img_h:
        h = img_h - y
    
    # 클리핑 후 유효한 크기 확인
    if w <= 0 or h <= 0:
        return None
    
    return (x, y, w, h)


def validate_bbox(x: float, y: float, w: float, h: float, img_w: int, img_h: int, 
                  min_size: int = 2) -> bool:
    """바운딩 박스를 검증합니다."""
    # 크기 확인
    if w <= 0 or h <= 0:
        return False
    
    # 최소 크기 확인
    if w < min_size or h < min_size:
        return False
    
    # 이미지 밖에 완전히 있는지 확인
    if x >= img_w or y >= img_h or (x + w) <= 0 or (y + h) <= 0:
        return False
    
    return True


def convert_yolo_to_xywh(yolo_line: str, img_w: int, img_h: int) -> Optional[Tuple[float, float, float, float]]:
    """YOLO 형식을 xywh로 변환합니다."""
    try:
        parts = yolo_line.strip().split()
        if len(parts) != 5:
            return None
        
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_w
        y_center = float(parts[2]) * img_h
        width = float(parts[3]) * img_w
        height = float(parts[4]) * img_h
        
        x = x_center - width / 2
        y = y_center - height / 2
        
        return (x, y, width, height)
    except Exception:
        return None


def convert_xywh_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int, class_id: int = 0) -> str:
    """xywh를 YOLO 형식으로 변환합니다."""
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    width = w / img_w
    height = h / img_h
    
    # 값이 [0, 1] 범위에 있는지 확인
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def calculate_outlier_thresholds(values: np.ndarray, iqr_multiplier: float = 1.5) -> Tuple[float, float]:
    """IQR 방법을 사용하여 이상치 임계값을 계산합니다."""
    if len(values) == 0:
        return 0.0, float('inf')
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    
    return lower_bound, upper_bound


def is_head_box_outlier(w: float, h: float, area: float, aspect_ratio: float,
                       size_thresholds: Optional[Dict[str, Tuple[float, float]]] = None) -> bool:
    """크기 통계를 기반으로 머리 박스가 이상치인지 확인합니다."""
    if size_thresholds is None:
        return False
    
    # 너비 확인
    w_lower, w_upper = size_thresholds.get('width', (0, float('inf')))
    if w < w_lower or w > w_upper:
        return True
    
    # 높이 확인
    h_lower, h_upper = size_thresholds.get('height', (0, float('inf')))
    if h < h_lower or h > h_upper:
        return True
    
    # 면적 확인
    area_lower, area_upper = size_thresholds.get('area', (0, float('inf')))
    if area < area_lower or area > area_upper:
        return True
    
    # 종횡비 확인
    ar_lower, ar_upper = size_thresholds.get('aspect_ratio', (0, float('inf')))
    if aspect_ratio < ar_lower or aspect_ratio > ar_upper:
        return True
    
    return False


def collect_head_box_statistics(root_dir: Path, splits: List[str]) -> Dict[str, np.ndarray]:
    """모든 split에 걸쳐 머리 박스(class_id = 0)에 대한 통계를 수집합니다."""
    logger.info("Collecting head box statistics for outlier detection...")
    
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    
    for split in splits:
        images_dir = root_dir / "images" / split
        labels_dir = root_dir / "labels" / split
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Collecting stats from {split}", leave=False):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Load image to get dimensions
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    class_id = int(parts[0])
                    # 머리 박스(class_id = 0)에 대한 통계만 수집
                    if class_id != 0:
                        continue
                    
                    # YOLO 형식을 xywh로 변환
                    bbox = convert_yolo_to_xywh(line, img_w, img_h)
                    if bbox is None:
                        continue
                    
                    x, y, w, h = bbox
                    
                    # 유효한 박스만 수집
                    if w > 0 and h > 0:
                        area = w * h
                        aspect_ratio = w / h if h > 0 else 0
                        
                        widths.append(w)
                        heights.append(h)
                        areas.append(area)
                        aspect_ratios.append(aspect_ratio)
            
            except Exception:
                continue
    
    return {
        'width': np.array(widths),
        'height': np.array(heights),
        'area': np.array(areas),
        'aspect_ratio': np.array(aspect_ratios)
    }


def clean_label_file(label_path: Path, image_path: Path, min_size: int = 2,
                    head_outlier_thresholds: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[List[str], int, int]:
    """단일 라벨 파일을 정리합니다.
    
    Returns:
        (valid_lines, removed_count, removed_outliers_count) 튜플
    """
    if not label_path.exists():
        return [], 0, 0
    
    # 이미지 로드하여 크기 확인
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning(f"Cannot read image: {image_path}")
        return [], 0, 0
    
    img_h, img_w = img.shape[:2]
    
    # 라벨 파일 읽기
    valid_lines = []
    removed_count = 0
    removed_outliers_count = 0
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # class_id 파싱
            parts = line.split()
            if len(parts) != 5:
                removed_count += 1
                continue
            
            class_id = int(parts[0])
            
            # YOLO 형식을 xywh로 변환
            bbox = convert_yolo_to_xywh(line, img_w, img_h)
            if bbox is None:
                removed_count += 1
                continue
            
            x, y, w, h = bbox
            
            # bbox 검증
            if not validate_bbox(x, y, w, h, img_w, img_h, min_size):
                removed_count += 1
                continue
            
            # bbox 클리핑
            clipped = clip_bbox(x, y, w, h, img_w, img_h)
            if clipped is None:
                removed_count += 1
                continue
            
            x, y, w, h = clipped
            
            # 머리 박스 이상치 확인 (class_id = 0)
            if class_id == 0 and head_outlier_thresholds is not None:
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                if is_head_box_outlier(w, h, area, aspect_ratio, head_outlier_thresholds):
                    removed_outliers_count += 1
                    continue
            
            # YOLO 형식으로 다시 변환
            yolo_line = convert_xywh_to_yolo(x, y, w, h, img_w, img_h, class_id)
            valid_lines.append(yolo_line)
    
    except Exception as e:
        logger.warning(f"Error processing label file {label_path}: {e}")
        return [], 0, 0
    
    return valid_lines, removed_count, removed_outliers_count


def find_duplicates(image_dir: Path, split: str, sample_size: Optional[int] = None) -> List[Tuple[Path, Path, str]]:
    """perceptual hashing을 사용하여 중복 이미지를 찾습니다."""
    logger.info(f"Finding duplicates in {split} split...")
    
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    if sample_size:
        image_files = image_files[:sample_size]
    
    hash_to_paths = defaultdict(list)
    duplicates = []
    
    for img_path in tqdm(image_files, desc="Hashing images"):
        img_hash = calculate_image_hash(img_path)
        if img_hash:
            hash_to_paths[img_hash].append(img_path)
    
    # Find duplicates
    for img_hash, paths in hash_to_paths.items():
        if len(paths) > 1:
            # 첫 번째는 유지하고 나머지를 중복으로 표시
            for dup_path in paths[1:]:
                duplicates.append((paths[0], dup_path, img_hash))
    
    return duplicates


def clean_dataset(
    root_dir: Path,
    output_dir: Path,
    min_blur_score: float = 50.0,
    min_bbox_size: int = 2,
    remove_duplicates: bool = True,
    remove_head_outliers: bool = True,
    outlier_iqr_multiplier: float = 1.5,
    splits: List[str] = None
):
    """
    YOLO 데이터셋을 정리합니다.
    
    Args:
        root_dir: YOLO 데이터셋의 루트 디렉토리 (images/ 및 labels/ 포함)
        output_dir: 정리된 데이터셋의 출력 디렉토리
        min_blur_score: 최소 블러 점수 임계값 (낮을수록 더 흐림)
        min_bbox_size: 최소 바운딩 박스 크기 (픽셀 단위)
        remove_duplicates: 중복 이미지 제거 여부
        remove_head_outliers: 이상치 머리 박스 제거 여부
        outlier_iqr_multiplier: 이상치 탐지를 위한 IQR 배수 (기본값: 1.5)
        splits: 정리할 split 목록 (기본값: ['train', 'val', 'test'])
    """
    if splits is None:
        splits = ['train', 'val', 'test']
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # 이상치 탐지를 위한 머리 박스 통계 수집
    head_outlier_thresholds = None
    if remove_head_outliers:
        head_stats = collect_head_box_statistics(root_dir, splits)
        
        if len(head_stats['width']) > 0:
            head_outlier_thresholds = {
                'width': calculate_outlier_thresholds(head_stats['width'], outlier_iqr_multiplier),
                'height': calculate_outlier_thresholds(head_stats['height'], outlier_iqr_multiplier),
                'area': calculate_outlier_thresholds(head_stats['area'], outlier_iqr_multiplier),
                'aspect_ratio': calculate_outlier_thresholds(head_stats['aspect_ratio'], outlier_iqr_multiplier)
            }
            
            logger.info("\nHead box outlier thresholds (IQR method):")
            logger.info(f"  Width: [{head_outlier_thresholds['width'][0]:.2f}, {head_outlier_thresholds['width'][1]:.2f}]")
            logger.info(f"  Height: [{head_outlier_thresholds['height'][0]:.2f}, {head_outlier_thresholds['height'][1]:.2f}]")
            logger.info(f"  Area: [{head_outlier_thresholds['area'][0]:.2f}, {head_outlier_thresholds['area'][1]:.2f}]")
            logger.info(f"  Aspect Ratio: [{head_outlier_thresholds['aspect_ratio'][0]:.3f}, {head_outlier_thresholds['aspect_ratio'][1]:.3f}]")
        else:
            logger.warning("No head boxes found for outlier detection. Skipping outlier removal.")
            remove_head_outliers = False
    
    # 통계
    stats = {
        'total_images': 0,
        'removed_duplicates': 0,
        'removed_low_quality': 0,
        'removed_empty': 0,
        'removed_invalid_bbox': 0,
        'removed_head_outliers': 0,
        'kept_images': 0,
        'total_bboxes_before': 0,
        'total_bboxes_after': 0,
    }
    
    # 각 split 처리
    for split in splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {split} split...")
        logger.info(f"{'='*60}")
        
        images_dir = root_dir / "images" / split
        labels_dir = root_dir / "labels" / split
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"Split {split} not found, skipping...")
            continue
        
        # Find duplicates
        duplicates = []
        if remove_duplicates:
            duplicates = find_duplicates(images_dir, split)
            duplicate_paths = {dup[1] for dup in duplicates}  # Set of duplicate paths to remove
            stats['removed_duplicates'] += len(duplicates)
            logger.info(f"Found {len(duplicates)} duplicate images")
        else:
            duplicate_paths = set()
        
        # 이미지 처리
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        
        for img_path in tqdm(image_files, desc=f"Cleaning {split}"):
            stats['total_images'] += 1
            
            # 중복 건너뛰기
            if img_path in duplicate_paths:
                continue
            
            # 블러 점수 확인
            blur_score = calculate_blur_score(img_path)
            if blur_score < min_blur_score:
                stats['removed_low_quality'] += 1
                logger.debug(f"Removed low quality image: {img_path.name} (blur: {blur_score:.2f})")
                continue
            
            # 해당하는 라벨 파일 가져오기
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # 라벨 파일 정리
            valid_lines, removed_bboxes, removed_outliers = clean_label_file(
                label_path, img_path, min_bbox_size, head_outlier_thresholds
            )
            stats['removed_invalid_bbox'] += removed_bboxes
            if remove_head_outliers:
                stats['removed_head_outliers'] += removed_outliers
            
            # bbox 개수 세기
            if label_path.exists():
                with open(label_path, 'r') as f:
                    stats['total_bboxes_before'] += len([l for l in f.readlines() if l.strip()])
            stats['total_bboxes_after'] += len(valid_lines)
            
            # 유효한 박스가 없으면 건너뛰기
            if len(valid_lines) == 0:
                stats['removed_empty'] += 1
                logger.debug(f"Removed empty annotation: {img_path.name}")
                continue
            
            # 이미지 복사 및 정리된 라벨 작성
            output_img_path = output_dir / "images" / split / img_path.name
            output_label_path = output_dir / "labels" / split / f"{img_path.stem}.txt"
            
            shutil.copy2(img_path, output_img_path)
            
            with open(output_label_path, 'w') as f:
                f.write('\n'.join(valid_lines) + '\n')
            
            stats['kept_images'] += 1
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Cleaning Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Total images processed: {stats['total_images']}")
    logger.info(f"Removed duplicates: {stats['removed_duplicates']}")
    logger.info(f"Removed low quality (blur < {min_blur_score}): {stats['removed_low_quality']}")
    logger.info(f"Removed empty annotations: {stats['removed_empty']}")
    logger.info(f"Removed invalid bboxes: {stats['removed_invalid_bbox']}")
    if remove_head_outliers:
        logger.info(f"Removed head box outliers: {stats['removed_head_outliers']}")
    logger.info(f"Kept images: {stats['kept_images']}")
    logger.info(f"Total bboxes before: {stats['total_bboxes_before']}")
    logger.info(f"Total bboxes after: {stats['total_bboxes_after']}")
    logger.info(f"Bbox removal rate: {(1 - stats['total_bboxes_after']/stats['total_bboxes_before'])*100:.2f}%" if stats['total_bboxes_before'] > 0 else "N/A")
    logger.info(f"\nCleaned dataset saved to: {output_dir}")
    
    # 통계 저장
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(output_dir / "cleaning_stats.csv", index=False)
    logger.info(f"Statistics saved to: {output_dir / 'cleaning_stats.csv'}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Clean YOLO dataset based on EDA findings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        required=True,
        help='Root directory of YOLO dataset (contains images/ and labels/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for cleaned dataset'
    )
    parser.add_argument(
        '--min-blur-score',
        type=float,
        default=50.0,
        help='Minimum blur score threshold (lower = blurrier, remove if below)'
    )
    parser.add_argument(
        '--min-bbox-size',
        type=int,
        default=2,
        help='Minimum bounding box size in pixels'
    )
    parser.add_argument(
        '--no-remove-duplicates',
        action='store_true',
        help='Do not remove duplicate images'
    )
    parser.add_argument(
        '--no-remove-head-outliers',
        action='store_true',
        help='Do not remove outlier head boxes'
    )
    parser.add_argument(
        '--outlier-iqr-multiplier',
        type=float,
        default=1.5,
        help='IQR multiplier for outlier detection (default: 1.5, higher = more lenient)'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help='Dataset splits to clean'
    )
    
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    
    if not root_dir.exists():
        logger.error(f"Root directory does not exist: {root_dir}")
        return
    
    # 데이터셋 정리
    stats = clean_dataset(
        root_dir=root_dir,
        output_dir=output_dir,
        min_blur_score=args.min_blur_score,
        min_bbox_size=args.min_bbox_size,
        remove_duplicates=not args.no_remove_duplicates,
        remove_head_outliers=not args.no_remove_head_outliers,
        outlier_iqr_multiplier=args.outlier_iqr_multiplier,
        splits=args.splits
    )
    
    logger.info("\nCleaning completed successfully!")


if __name__ == "__main__":
    main()