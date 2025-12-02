"""
Defect Postprocessing Pipeline

결함 segmentation 결과를 후처리합니다.
Temperature scaling, threshold, morphology 등을 포함합니다.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import copy


class DefectPostprocessor:
    """
    결함 segmentation 후처리 클래스
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
    
    def temperature_scaling(
        self,
        defect_mask_arr: np.ndarray,
        T: float = 1.0
    ) -> np.ndarray:
        """
        Temperature scaling을 적용합니다.
        
        Args:
            defect_mask_arr: Segmentation logits [B, classes, H, W]
            T: Temperature parameter
        
        Returns:
            Scaled probabilities [B, classes, H, W]
        """
        scaled_arr = defect_mask_arr / T
        max_vals = np.max(scaled_arr, axis=1, keepdims=True)
        exp_arr = np.exp(scaled_arr - max_vals)
        sum_arr = np.sum(exp_arr, axis=1, keepdims=True)
        probs = exp_arr / sum_arr
        
        return probs
    
    def apply_threshold(
        self,
        defect_probs: np.ndarray,
        thresholds: List[float]
    ) -> np.ndarray:
        """
        Threshold를 적용하여 mask를 생성합니다.
        
        Args:
            defect_probs: Class probabilities [B, classes, H, W]
            thresholds: 각 클래스별 threshold [classes]
        
        Returns:
            Threshold 적용된 mask [B, H, W]
        """
        batch_size, num_classes, height, width = defect_probs.shape
        
        # Threshold 적용
        threshold_mask = np.zeros((batch_size, height, width), dtype=np.uint8)
        
        for class_idx in range(num_classes):
            class_mask = defect_probs[:, class_idx] > thresholds[class_idx]
            threshold_mask[class_mask] = class_idx
        
        return threshold_mask
    
    def morphology(
        self,
        mask_arr: np.ndarray,
        kernel_size_list: List[int],
        iters_list: List[int]
    ) -> np.ndarray:
        """
        Morphology 연산을 적용합니다.
        
        Args:
            mask_arr: Mask 배열 [B, H, W]
            kernel_size_list: 각 클래스별 kernel 크기 리스트
            iters_list: 각 클래스별 반복 횟수 리스트
        
        Returns:
            Morphology 적용된 mask [B, H, W]
        """
        morphed_mask_arr = []
        
        for mask in mask_arr:
            unique_values = np.unique(mask)
            splitted_mask_list = []
            
            for mask_value in unique_values:
                # Kernel 크기 결정
                if isinstance(kernel_size_list, list):
                    kernel_size = kernel_size_list[int(mask_value)] if int(mask_value) < len(kernel_size_list) else kernel_size_list[0]
                else:
                    kernel_size = kernel_size_list
                
                # 반복 횟수 결정
                if isinstance(iters_list, list):
                    iters = iters_list[int(mask_value)] if int(mask_value) < len(iters_list) else iters_list[0]
                else:
                    iters = iters_list
                
                # Kernel 생성
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                
                # 해당 클래스만 추출
                splitted_mask = np.where(mask == mask_value, mask_value, 0).astype(np.uint8)
                
                # Morphology 연산 (Opening)
                splitted_mask = cv2.morphologyEx(
                    splitted_mask,
                    cv2.MORPH_OPEN,
                    kernel,
                    iterations=iters
                )
                
                splitted_mask_list.append(splitted_mask)
            
            # 모든 마스크 결합
            if splitted_mask_list:
                morph_mask = np.maximum.reduce(splitted_mask_list)
            else:
                morph_mask = mask.copy()
            
            morphed_mask_arr.append(morph_mask)
        
        return np.array(morphed_mask_arr)
    
    def remove_small_defects(
        self,
        mask: np.ndarray,
        pixel_count_threshold: int,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        작은 결함을 제거합니다.
        
        Args:
            mask: Defect mask [H, W]
            pixel_count_threshold: 최소 pixel 수 threshold
            target_class: 제거할 클래스 (None이면 모든 클래스)
        
        Returns:
            작은 결함이 제거된 mask
        """
        _mask = mask.copy()
        
        if target_class is not None:
            # 특정 클래스만 처리
            defect_mask = (mask == target_class).astype(np.uint8)
        else:
            # 모든 결함 클래스 처리
            defect_mask = (mask > 0).astype(np.uint8)
        
        # Contour 찾기
        contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 작은 결함 제거
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < pixel_count_threshold:
                # 배경(0)으로 변경
                x, y, w, h = cv2.boundingRect(contour)
                _mask[y:y+h, x:x+w][_mask[y:y+h, x:x+w] == mask[y:y+h, x:x+w].max()] = 0
        
        return _mask
    
    def postprocess(
        self,
        defect_logits: np.ndarray,
        section: str = "display"
    ) -> np.ndarray:
        """
        전체 후처리 파이프라인을 수행합니다.
        
        Args:
            defect_logits: Segmentation logits [B, classes, H, W]
            section: 섹션 이름
        
        Returns:
            후처리된 mask [B, H, W]
        """
        postprocess_config = self.config['defect_segmentation'][section]['postprocessing']
        
        # 1. Temperature scaling
        if postprocess_config.get('temperature_scaling', {}).get('enabled', False):
            T = postprocess_config['temperature_scaling'].get('T', 1.0)
            defect_probs = self.temperature_scaling(defect_logits, T)
        else:
            # Softmax만 적용
            max_vals = np.max(defect_logits, axis=1, keepdims=True)
            exp_arr = np.exp(defect_logits - max_vals)
            sum_arr = np.sum(exp_arr, axis=1, keepdims=True)
            defect_probs = exp_arr / sum_arr
        
        # 2. Threshold 적용
        thresholds = postprocess_config.get('threshold', {})
        threshold_list = [
            thresholds.get('A', 0.0),
            thresholds.get('B', 0.5),
            thresholds.get('C', 0.5),
            thresholds.get('D', 0.5)
        ]
        
        # 클래스 수에 맞게 조정
        num_classes = defect_probs.shape[1]
        threshold_list = threshold_list[:num_classes]
        
        mask = self.apply_threshold(defect_probs, threshold_list)
        
        # 3. Morphology
        morphology_config = postprocess_config.get('morphology', {})
        if morphology_config:
            kernel_size_list = morphology_config.get('kernel_size_list', [3])
            iters_list = morphology_config.get('iters_list', [1])
            mask = self.morphology(mask, kernel_size_list, iters_list)
        
        # 4. 작은 결함 제거
        if postprocess_config.get('small_defect_removal', {}).get('enabled', False):
            pixel_threshold = postprocess_config['small_defect_removal'].get('pixel_count_threshold', 0)
            for i in range(mask.shape[0]):
                mask[i] = self.remove_small_defects(mask[i], pixel_threshold)
        
        return mask

