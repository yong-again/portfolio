"""
Service Postprocessing Module

서비스에서 사용하는 후처리 기능을 제공합니다.
회사 코드의 BasePostProcess를 참고하여 리팩토링되었습니다.
"""

import numpy as np
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess.defect_preprocess import DefectPreprocessor
from preprocess.defect_postprocess import DefectPostprocessor


class ServicePostprocessor:
    """
    Service Postprocessor
    
    서비스에서 사용하는 후처리 기능을 제공합니다.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.defect_preprocessor = DefectPreprocessor(config)
        self.defect_postprocessor = DefectPostprocessor(config)
    
    def post_process(
        self,
        images: np.ndarray,
        config: Dict[str, Any],
        pp_key: str = 'base',
        section: str = 'display'
    ) -> np.ndarray:
        """
        후처리 함수: 이미지 배열에 대해 후처리 작업을 수행합니다.
        
        Args:
            images: Segmentation 결과 [B, classes, H, W] 또는 [B, H, W]
            config: 설정 딕셔너리
            pp_key: Post-processing key ('base' 또는 'flip')
            section: 섹션 이름 ('top', 'back', 'lcd', 'corner', 'display', 'front', 'side')
        
        Returns:
            후처리된 마스크 [B, H, W]
        """
        # 설정 파일에서 섹션별 키를 가져옴
        section_key = config['defect_segmentation'][section]
        post_key = config.get('POSTPROC', {})  # 후처리 설정 (선택사항)
        
        # 1. 온도 스케일링 후처리: 온도 변환을 적용하여 확률을 조정
        if section_key['postprocessing'].get('temperature_scaling', {}).get('enabled', False):
            T = section_key['postprocessing']['temperature_scaling'].get('T', 1.0)
            # images가 logits인 경우 temperature scaling 적용
            if len(images.shape) == 4:  # [B, classes, H, W]
                temper_scaled_images = self.defect_postprocessor.temperature_scaling(images, T)
            else:
                # 이미 probabilities인 경우
                temper_scaled_images = images
        else:
            # Softmax만 적용 (logits인 경우)
            if len(images.shape) == 4:  # [B, classes, H, W]
                # Softmax 적용
                max_vals = np.max(images, axis=1, keepdims=True)
                exp_arr = np.exp(images - max_vals)
                sum_arr = np.sum(exp_arr, axis=1, keepdims=True)
                temper_scaled_images = exp_arr / sum_arr
            else:
                temper_scaled_images = images
        
        # 2. 마스크 확률 값 클리핑: 특정 임계값(threshold) 이하의 확률을 마스크에서 제거
        thresholds = section_key['postprocessing'].get('threshold', {})
        threshold_list = [
            thresholds.get('A', 0.0),
            thresholds.get('B', 0.5),
            thresholds.get('C', 0.5),
            thresholds.get('D', 0.5)
        ]
        
        # 클래스 수에 맞게 조정
        if len(temper_scaled_images.shape) == 4:  # [B, classes, H, W]
            num_classes = temper_scaled_images.shape[1]
        else:
            # 이미 argmax가 적용된 경우
            num_classes = 4
        
        threshold_list = threshold_list[:num_classes]
        # Threshold 적용
        clipping_mask_values = self.defect_postprocessor.apply_threshold(temper_scaled_images, threshold_list)
        
        # 3. 형태학적 연산: 클리핑된 마스크에 형태학적 연산(예: 팽창, 침식)을 적용하여 결함 영역을 정제
        morphology_config = section_key['postprocessing'].get('morphology', {})
        if morphology_config:
            kernel_size_list = morphology_config.get('kernel_size_list', [3])
            iters_list = morphology_config.get('iters_list', [1])
            morphed_mask_arr = self.defect_postprocessor.morphology(
                clipping_mask_values,
                kernel_size_list,
                iters_list
            )
        else:
            morphed_mask_arr = clipping_mask_values
        
        # 섹션별 특화 후처리
        if section == 'top' or section == 'front':
            # 작은 결함 제거
            small_thres = section_key.get('small_thres', section_key['postprocessing'].get('small_defect_removal', {}).get('pixel_count_threshold', 0))
            removed_small_defects = self.defect_postprocessor.remove_small_defects(
                morphed_mask_arr[0] if len(morphed_mask_arr.shape) == 3 else morphed_mask_arr,
                small_thres
            )
            if len(morphed_mask_arr.shape) == 3:
                removed_small_defects = removed_small_defects[np.newaxis, :, :]
            return removed_small_defects
        
        elif section == 'back':
            # 작은 결함 제거
            small_thres = section_key.get('small_thres', section_key['postprocessing'].get('small_defect_removal', {}).get('pixel_count_threshold', 0))
            removed_small_defects = self.defect_postprocessor.remove_small_defects(
                morphed_mask_arr[0] if len(morphed_mask_arr.shape) == 3 else morphed_mask_arr,
                small_thres
            )
            if len(morphed_mask_arr.shape) == 3:
                removed_small_defects = removed_small_defects[np.newaxis, :, :]
            
            # 백 엣지 결함 제거 (설정이 있는 경우)
            if post_key and 'remove_back_edge_defect' in post_key:
                remove_back_edge_defects = self._remove_back_edge_defect(
                    removed_small_defects,
                    **post_key['remove_back_edge_defect']
                )
                return remove_back_edge_defects
            
            return removed_small_defects
        
        elif section == 'lcd' or section == 'display':
            # AIP 블릿 결함 제거 (설정이 있는 경우)
            if post_key and 'remove_aip_bullet' in post_key:
                removed_aip_bullet, _ = self._remove_aip_bullet(
                    mask=morphed_mask_arr[0, :, :] if len(morphed_mask_arr.shape) == 3 else morphed_mask_arr[0],
                    model_name=pp_key,
                    **post_key['remove_aip_bullet']
                )
            else:
                removed_aip_bullet = morphed_mask_arr[0, :, :] if len(morphed_mask_arr.shape) == 3 else morphed_mask_arr[0]
            
            # 작은 블릿 결함 제거 (설정이 있는 경우)
            if post_key and 'remove_small_bullet' in post_key:
                removed_small_bullet_img, _, _, _ = self._remove_small_bullet(
                    mask=removed_aip_bullet,
                    **post_key['remove_small_bullet']
                )
                return np.array([removed_small_bullet_img])
            
            return np.array([removed_aip_bullet])
        
        elif section == 'corner' or section == 'side':
            # 형태학적 연산 결과 그대로 반환
            return morphed_mask_arr
        
        else:
            # 기본 반환
            return morphed_mask_arr
    
    def _remove_back_edge_defect(
        self,
        mask: np.ndarray,
        h_rat: float = 0.1,
        w_rat: float = 0.1,
        bg_mask_val: int = 1
    ) -> np.ndarray:
        """
        후면 좌측 꼬리모양 잔상 파손 삭제
        
        Args:
            mask: 후면 마스크 [B, H, W] 또는 [H, W]
            h_rat: 세로 삭제 영역 비율
            w_rat: 가로 삭제 영역 비율
            bg_mask_val: 핸드폰 영역 마스크 값
        
        Returns:
            후처리된 마스크
        """
        _mask = mask.copy()
        if len(_mask.shape) == 3:
            _, h, w = _mask.shape
        else:
            h, w = _mask.shape
            _mask = _mask[np.newaxis, :, :]
        
        h_thr, w_thr = int(h * h_rat), int(w * w_rat)
        _mask[:, :h_thr, :w_thr] = bg_mask_val
        _mask[:, -h_thr:, :w_thr] = bg_mask_val
        
        if len(mask.shape) == 2:
            return _mask[0]
        return _mask
    
    def _remove_aip_bullet(
        self,
        mask: np.ndarray,
        model_name: str,
        bullet_num: int = 7,
        bullet2lcd_num: int = 1
    ) -> tuple:
        """
        아이폰 총알 파손 영역 모두 제거
        
        Args:
            mask: lcd 예측 마스크 [H, W]
            model_name: 영문 모델명
            bullet_num: 총알파손 마스크 번호
            bullet2lcd_num: 총알파손 제거 이후 마스크 번호
        
        Returns:
            (보정 마스크, 보정 여부)
        """
        is_modified = False
        _mask = mask.copy()
        
        if model_name[:3] == "AIP":
            _mask = np.where(_mask == bullet_num, bullet2lcd_num, _mask)
            is_modified = True
        
        return _mask, is_modified
    
    def _remove_small_bullet(
        self,
        mask: np.ndarray,
        min_bullet_area: int = 100,
        bbox_buffer: int = 5,
        bullet_num: int = 7,
        bullet2lcd_num: int = 1
    ) -> tuple:
        """
        크기가 작은 총알파손 예측 영역 제거
        
        Args:
            mask: lcd 예측 마스크 [H, W]
            min_bullet_area: 최소 총알파손 영역 크기
            bbox_buffer: bbox 버퍼
            bullet_num: 총알파손 마스크 번호
            bullet2lcd_num: 총알파손 제거 이후 마스크 번호
        
        Returns:
            (보정 마스크, 보정 여부, 총알 파손 영역 리스트, bbox 리스트)
        """
        import cv2
        
        is_modified = False
        bullet_areas = []
        bbox_coords = []
        
        _mask = mask.copy()
        unique_nums = np.unique(_mask)
        has_bullet = bullet_num in unique_nums
        
        if has_bullet:
            # 총알파손 영역 외곽선, bbox 검출
            bullet_mask = np.where(_mask == bullet_num, 1, 0).astype(np.uint8).copy()
            cnts, _ = cv2.findContours(bullet_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bbox_coords = [cv2.boundingRect(cnt) for cnt in cnts]
            bbox_coords = [
                (x - bbox_buffer, y - bbox_buffer, w + 2 * bbox_buffer, h + 2 * bbox_buffer)
                for x, y, w, h in bbox_coords
            ]
            
            # 총알 파손 bbox 내 총알파손 영역이 일정값 미만이면 배경으로 변경
            for x, y, w, h in bbox_coords:
                bullet_area = bullet_mask[y:y+h, x:x+w].sum()
                
                if bullet_area < min_bullet_area:
                    _mask[y:y+h, x:x+w][_mask[y:y+h, x:x+w] == bullet_num] = bullet2lcd_num
                    is_modified = True
                
                bullet_areas.append(bullet_area)
        
        return _mask, is_modified, bullet_areas, bbox_coords

