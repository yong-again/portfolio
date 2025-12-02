"""
Dataset Classes for Phone Detection

YOLO format annotation을 읽어서 PyTorch Dataset을 생성합니다.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
import os

from .pipeline import PreprocessingPipeline, load_image


class PhoneDetectionDataset(Dataset):
    """
    Phone Detection을 위한 Dataset 클래스
    
    YOLO format annotation을 읽어서 이미지와 bounding box를 반환합니다.
    """
    
    def __init__(
        self,
        data_dir: str,
        config: Dict[str, Any],
        is_training: bool = True,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png']
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 경로 (images와 annotations 하위 디렉토리 포함)
            config: 설정 딕셔너리
            is_training: 학습 모드 여부
            image_extensions: 지원하는 이미지 확장자 리스트
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.is_training = is_training
        self.image_extensions = image_extensions
        
        # 전처리 파이프라인
        self.transform = PreprocessingPipeline(config, is_training=is_training)
        
        # 이미지와 annotation 파일 찾기
        images_dir = self.data_dir / 'images'
        annotations_dir = self.data_dir / 'annotations'
        
        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")
        
        # 이미지 파일 목록 수집
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
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        데이터셋에서 하나의 샘플을 가져옵니다.
        
        Returns:
            Dictionary containing:
                - 'image': 전처리된 이미지 텐서 [C, H, W]
                - 'bboxes': Bounding boxes [N, 5] (class_id, x, y, w, h) - 정규화된 좌표
                - 'image_path': 원본 이미지 경로
                - 'original_size': 원본 이미지 크기 (width, height)
        """
        image_path = self.image_paths[idx]
        
        # 이미지 로드
        image = load_image(str(image_path))
        original_size = image.size  # (width, height)
        
        # Annotation 로드
        annotation_path = self._get_annotation_path(image_path)
        bboxes = self._load_annotations(annotation_path) if annotation_path.exists() else None
        
        # 전처리
        image_tensor, bboxes = self.transform(image, bboxes)
        
        # Bounding boxes를 텐서로 변환
        if bboxes is not None and len(bboxes) > 0:
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
        else:
            # Bounding box가 없는 경우 빈 텐서
            bboxes_tensor = torch.zeros((0, 5), dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'bboxes': bboxes_tensor,
            'image_path': str(image_path),
            'original_size': original_size
        }
    
    def _get_annotation_path(self, image_path: Path) -> Path:
        """이미지 경로에 대응하는 annotation 파일 경로를 반환합니다."""
        data_dir = self.data_dir
        images_dir = data_dir / 'images'
        annotations_dir = data_dir / 'annotations'
        
        # images 디렉토리 기준으로 상대 경로 계산
        relative_path = image_path.relative_to(images_dir)
        annotation_path = annotations_dir / relative_path.with_suffix('.txt')
        
        return annotation_path
    
    def _load_annotations(self, annotation_path: Path) -> Optional[np.ndarray]:
        """
        YOLO format annotation 파일을 읽습니다.
        
        Format: class_id x_center y_center width height (모두 정규화된 값)
        
        Returns:
            Bounding boxes [N, 5] (class_id, x_center, y_center, width, height)
        """
        if not annotation_path.exists():
            return None
        
        bboxes = []
        
        with open(annotation_path, 'r') as f:
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
                    
                    # 좌표 검증
                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and \
                       0 < width <= 1 and 0 < height <= 1:
                        bboxes.append([class_id, x_center, y_center, width, height])
                except ValueError:
                    continue
        
        if len(bboxes) == 0:
            return None
        
        return np.array(bboxes, dtype=np.float32)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    DataLoader를 위한 collate function
    
    배치 내에서 이미지는 동일한 크기이지만, bounding box 개수는 다를 수 있으므로
    별도로 처리합니다.
    """
    images = torch.stack([item['image'] for item in batch])
    bboxes_list = [item['bboxes'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    original_sizes = [item['original_size'] for item in batch]
    
    return {
        'images': images,
        'bboxes': bboxes_list,  # List of tensors (각각 다른 길이)
        'image_paths': image_paths,
        'original_sizes': original_sizes
    }

