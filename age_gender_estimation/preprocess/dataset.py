"""
Dataset Classes for Age & Gender Estimation

CSV 또는 JSON 형식의 annotation을 읽어서 PyTorch Dataset을 생성합니다.
Age는 0~100세를 1세 단위로 라벨링합니다.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import json
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import os

from .transforms import get_train_transforms, get_val_transforms


class AgeGenderDataset(Dataset):
    """
    Age & Gender Estimation을 위한 Dataset 클래스
    
    CSV 또는 JSON 형식의 annotation을 읽어서 이미지와 라벨을 반환합니다.
    Age는 0~100세를 1세 단위로 라벨링합니다.
    """
    
    def __init__(
        self,
        data_dir: str,
        config: Dict[str, Any],
        is_training: bool = True,
        annotation_file: Optional[str] = None
    ):
        """
        Args:
            data_dir: 데이터 디렉토리 경로
            config: 설정 딕셔너리
            is_training: 학습 모드 여부
            annotation_file: Annotation 파일명 (None이면 자동 탐지)
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.is_training = is_training
        
        # Annotation 형식
        annotation_format = config['data'].get('annotation_format', 'csv')
        default_annotation_file = config['data'].get('annotation_file', 'annotations.csv')
        annotation_file = annotation_file or default_annotation_file
        
        # Annotation 파일 경로
        annotation_path = self.data_dir / annotation_file
        
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        # Annotation 로드
        if annotation_format == 'csv':
            self.annotations = pd.read_csv(annotation_path)
        else:  # json
            with open(annotation_path, 'r') as f:
                data = json.load(f)
                self.annotations = pd.DataFrame(data.get('annotations', []))
        
        if len(self.annotations) == 0:
            raise ValueError(f"No annotations found in {annotation_path}")
        
        # Transform
        if is_training:
            self.transform = get_train_transforms(config)
        else:
            self.transform = get_val_transforms(config)
        
        print(f"Loaded {len(self.annotations)} samples from {data_dir}")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        데이터셋에서 하나의 샘플을 가져옵니다.
        
        Returns:
            Dictionary containing:
                - 'image': 전처리된 이미지 텐서 [C, H, W]
                - 'age': 나이 (0~100, 정수)
                - 'gender': 성별 (0: Male, 1: Female)
                - 'image_path': 원본 이미지 경로
        """
        row = self.annotations.iloc[idx]
        
        # 이미지 경로
        image_path = row.get('image_path', row.get('path', ''))
        if not os.path.isabs(image_path):
            image_path = self.data_dir / 'images' / image_path
        
        # 이미지 로드
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image: {image_path}, Error: {e}")
        
        # 전처리
        image_tensor = self.transform(image)
        
        # 라벨
        age = int(row.get('age', row.get('Age', 0)))
        # Age 범위 제한 (0~100)
        age = max(0, min(100, age))
        
        gender_str = str(row.get('gender', row.get('Gender', '0')))
        
        # Gender 변환
        from models.gender_head import gender_to_class
        try:
            gender = gender_to_class(gender_str)
        except:
            # 숫자로 변환 시도
            gender = int(gender_str) if gender_str.isdigit() else 0
        
        return {
            'image': image_tensor,
            'age': age,  # 0~100
            'gender': gender,
            'image_path': str(image_path)
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    DataLoader를 위한 collate function
    
    배치 내에서 이미지는 동일한 크기이므로 stack 가능합니다.
    """
    images = torch.stack([item['image'] for item in batch])
    ages = torch.tensor([item['age'] for item in batch], dtype=torch.long)  # 0~100
    genders = torch.tensor([item['gender'] for item in batch], dtype=torch.long)
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'images': images,
        'ages': ages,  # 0~100
        'genders': genders,
        'image_paths': image_paths
    }
