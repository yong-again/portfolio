"""
Phone Region Detection using YOLO

YOLO를 사용하여 휴대폰 영역을 검출합니다 (전면, 후면용).
"""

from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Union
from pathlib import Path


class PhoneDetector:
    """
    YOLO 기반 Phone Region Detector
    
    전면/후면 이미지에서 휴대폰 영역을 검출합니다.
    """
    
    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = None
    ):
        """
        Args:
            model_path: 학습된 YOLO 모델 경로 또는 사전 학습 모델명
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            device: 추론 디바이스 (None이면 자동 선택)
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Device 설정
        if device is None:
            if torch.cuda.is_available():
                self.device = '0'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
    
    def detect(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, float]:
        """
        이미지에서 휴대폰 영역을 검출합니다.
        
        Args:
            image: 입력 이미지 (경로, PIL Image, numpy array)
        
        Returns:
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Confidence score
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image_array = np.array(Image.open(image_path).convert('RGB'))
        elif isinstance(image, Image.Image):
            image_array = np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            image_array = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # YOLO 추론
        results = self.model.predict(
            image_array,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # 결과 파싱
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # 가장 높은 confidence의 박스 선택
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            
            best_idx = np.argmax(scores)
            bbox = boxes[best_idx]  # [x1, y1, x2, y2]
            confidence = float(scores[best_idx])
            
            return bbox, confidence
        else:
            # 검출 실패 시 전체 이미지 반환
            h, w = image_array.shape[:2]
            return np.array([0, 0, w, h]), 0.0
    
    def crop_phone_region(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        padding_ratio: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        검출된 휴대폰 영역을 crop합니다.
        
        Args:
            image: 입력 이미지
            padding_ratio: Crop 시 추가할 padding 비율
        
        Returns:
            cropped_image: Crop된 이미지
            bbox: Bounding box [x1, y1, x2, y2]
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
            image_array = np.array(pil_image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
            image_array = np.array(pil_image)
        elif isinstance(image, np.ndarray):
            image_array = image
            pil_image = Image.fromarray(image_array)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # 검출
        bbox, confidence = self.detect(image_array)
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Padding 추가
        if padding_ratio > 0:
            width = x2 - x1
            height = y2 - y1
            pad_w = int(width * padding_ratio)
            pad_h = int(height * padding_ratio)
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(pil_image.width, x2 + pad_w)
            y2 = min(pil_image.height, y2 + pad_h)
        
        # Crop
        cropped = pil_image.crop((x1, y1, x2, y2))
        cropped_array = np.array(cropped)
        
        return cropped_array, np.array([x1, y1, x2, y2])

