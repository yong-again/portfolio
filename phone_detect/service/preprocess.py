"""
Service Preprocessing Module

서비스에서 사용하는 전처리 기능을 제공합니다.
회사 코드의 BasePreProcess를 참고하여 리팩토링되었습니다.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess.defect_preprocess import DefectPreprocessor, get_image_scaler
from preprocess.defect_postprocess import DefectPostprocessor


class ServicePreprocessor:
    """
    Service Preprocessor
    
    서비스에서 사용하는 전처리 기능을 제공합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.defect_preprocessor = DefectPreprocessor(config)
        self.defect_postprocessor = DefectPostprocessor(config)
    
    def preprocess_image(
        self,
        resized_image: np.ndarray,
        bbox_coords: Optional[List[Tuple[int, int, int, int]]] = None,
        input_size: Optional[Dict[str, int]] = None,
        scalers: Optional[List[str]] = None,
        section_code: str = None
    ) -> torch.Tensor:
        """
        이미지를 전처리하여 모델 입력 형식으로 변환합니다.
        
        Args:
            resized_image: 리사이즈된 이미지 [H, W, 3] 또는 이미지 리스트
            bbox_coords: Bounding box 좌표 리스트 [(x1, y1, x2, y2), ...] (선택사항)
            input_size: 모델 입력 크기 {'width': int, 'height': int} (선택사항)
            scalers: Scaler 이름 리스트 (선택사항)
            section_code: 섹션 코드 ('front', 'side', 'display', 'back')
        
        Returns:
            전처리된 텐서 [B, 3, H, W]
        """
        # 섹션 코드가 'display'인 경우, 이미지를 그대로 사용하고 새로운 차원을 추가
        if section_code is not None and 'disp' in section_code.lower():
            cropped_image = resized_image[np.newaxis, :, :, :] if len(resized_image.shape) == 3 else resized_image
        else:
            # 그 외의 경우, 바운딩 박스를 사용하여 이미지를 크롭
            if bbox_coords is None or input_size is None:
                raise ValueError("bbox_coords and input_size are required for non-display sections")
            
            cropped_image = self._get_cropped_image_by_bbox(
                resized_image,
                bbox_coords,
                input_size
            )
        
        # Scaler 적용
        if scalers is None:
            # 설정에서 가져오기
            if section_code:
                section_key = section_code.lower()
                if section_key == 'front':
                    scalers = self.config['defect_segmentation']['front']['preprocessing'].get('scaler', ['normalize'])
                elif section_key == 'side':
                    scalers = self.config['defect_segmentation']['side']['preprocessing'].get('scaler', ['normalize'])
                elif section_key == 'display':
                    scalers = self.config['defect_segmentation']['display']['preprocessing'].get('scaler', ['normalize'])
                else:
                    scalers = ['normalize']
            else:
                scalers = ['normalize']
        
        # Scaler 함수 리스트 생성
        scaler_funcs = [get_image_scaler(scaler_str) for scaler_str in scalers]
        
        # 이미지 스케일링 수행
        scaled_image = self._apply_scalers(cropped_image, scaler_funcs)
        
        # 이미지의 축을 재배열 (배치, 채널, 높이, 너비 순서로 변경)
        transpose_image_arr = scaled_image.transpose(0, 3, 1, 2)
        
        # 이미지 배열을 PyTorch 텐서로 변환
        image_to_tensor = torch.tensor(transpose_image_arr, dtype=torch.float32)
        
        return image_to_tensor
    
    def _get_cropped_image_by_bbox(
        self,
        image_arr: np.ndarray,
        bbox_point_list: List[Tuple[int, int, int, int]],
        input_shape: Dict[str, int]
    ) -> np.ndarray:
        """
        bbox 좌표값을 통해 핸드폰 영역만 crop하고 model input size로 resize
        
        Args:
            image_arr: 원본 image에서 핸드폰 영역만 남은 images [H, W, 3] 또는 [B, H, W, 3]
            bbox_point_list: images에서 핸드폰 객체가 있는 bbox 좌표 값 [(y1, y2, x1, x2), ...]
            input_shape: resize될 target size 값 {'width': int, 'height': int}
        
        Returns:
            resize_list: 핸드폰 영역만 crop되고 resize 된 images [B, H, W, 3]
        """
        import cv2
        
        # 단일 이미지인 경우 배치 차원 추가
        if len(image_arr.shape) == 3:
            image_arr = image_arr[np.newaxis, :, :, :]
        
        # bbox 좌표 형식: (y1, y2, x1, x2) 형식으로 가정
        cropped_list = []
        for idx, point in enumerate(bbox_point_list):
            if len(point) == 4:
                # (y1, y2, x1, x2) 형식
                y1, y2, x1, x2 = point
                img = image_arr[idx] if image_arr.shape[0] > idx else image_arr[0]
                cropped = img[y1:y2, x1:x2]
                cropped_list.append(cropped)
            else:
                raise ValueError(f"Invalid bbox format: {point}")
        
        if input_shape is None:
            return np.array(cropped_list)
        else:
            resize_list = np.array([
                cv2.resize(img.astype(np.uint8), (input_shape["width"], input_shape["height"]))
                for img in cropped_list
            ])
            return resize_list
    
    def _apply_scalers(
        self,
        image_arr: np.ndarray,
        scaler_funcs: List[Optional[Callable]]
    ) -> np.ndarray:
        """
        이미지에 scaler 함수들을 적용합니다.
        
        Args:
            image_arr: 이미지 배열 [B, H, W, 3]
            scaler_funcs: Scaler 함수 리스트
        
        Returns:
            스케일링된 이미지 배열
        """
        if None in scaler_funcs:
            # None이 있으면 원본 반환
            return image_arr
        
        scaled_image_list = []
        for img in image_arr:
            processed_img = img.copy()
            for scaler_func in scaler_funcs:
                if scaler_func is not None:
                    processed_img = scaler_func(processed_img)
            scaled_image_list.append(processed_img)
        
        return np.array(scaled_image_list)

