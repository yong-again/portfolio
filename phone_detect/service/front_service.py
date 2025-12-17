"""
Front Defect Detection Service

전면(Front/Top) 영역의 결함 검출 서비스입니다.
회사 코드의 FrontPipeline을 참고하여 리팩토링되었습니다.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.defect_segmenter import DefectSegmenter
from models.phone_segmenter import PhoneSegmenter
from models.utils import load_checkpoint
from preprocess.defect_preprocess import DefectPreprocessor, get_image_scaler
from preprocess.defect_postprocess import DefectPostprocessor
from utils.defect_grade import DefectGradeAnalyzer


class FrontService:
    """
    Front Defect Detection Service
    
    전면 영역의 결함을 검출하는 서비스입니다.
    Phone Segmentation → Defect Segmentation → Post-processing 파이프라인을 수행합니다.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        """
        Args:
            config: 설정 딕셔너리
            device: 디바이스 (cuda, cpu, mps)
        """
        self.config = config
        self.device = device
        
        # 전처리/후처리 초기화
        self.preprocessor = DefectPreprocessor(config)
        self.postprocessor = DefectPostprocessor(config)
        self.grade_analyzer = DefectGradeAnalyzer(config)
        
        # 모델 로드
        self._load_models()
    
    def _load_models(self):
        """모델을 로드합니다."""
        # Phone Segmentation 모델 (전면 영역 검출)
        phone_seg_config = self.config['phone_detection']['front']
        
        if phone_seg_config.get('method') == 'segmentation':
            self.phone_segmenter = PhoneSegmenter(
                encoder_name=phone_seg_config['model']['encoder_name'],
                encoder_weights=phone_seg_config['model']['encoder_weights'],
                classes=phone_seg_config['model']['classes']
            )
            phone_seg_path = phone_seg_config.get('trained_model')
            if phone_seg_path and Path(phone_seg_path).exists():
                load_checkpoint(phone_seg_path, self.phone_segmenter, device=self.device)
            self.phone_segmenter = self.phone_segmenter.to(self.device)
            self.phone_segmenter.eval()
        else:
            # YOLO 사용 시 (추후 구현)
            self.phone_segmenter = None
        
        # Defect Segmentation 모델 (전면 결함 검출)
        defect_config = self.config['defect_segmentation']['front']
        self.defect_segmenter = DefectSegmenter(
            encoder_name=defect_config['model']['encoder_name'],
            encoder_weights=defect_config['model']['encoder_weights'],
            decoder_name=defect_config['model']['decoder_name'],
            classes=defect_config['model']['classes']
        )
        
        defect_path = defect_config.get('trained_model')
        if defect_path and Path(defect_path).exists():
            load_checkpoint(defect_path, self.defect_segmenter, device=self.device)
        self.defect_segmenter = self.defect_segmenter.to(self.device)
        self.defect_segmenter.eval()
    
    def _preprocess_image(
        self,
        resized_image: np.ndarray,
        bbox_coords: List[Tuple[int, int, int, int]],
        input_size: Dict[str, int],
        scalers: List[str]
    ) -> torch.Tensor:
        """
        이미지를 전처리하여 모델 입력 형식으로 변환합니다.
        
        Args:
            resized_image: 리사이즈된 이미지 [H, W, 3] 또는 이미지 리스트
            bbox_coords: Bounding box 좌표 리스트 [(x1, y1, x2, y2), ...]
            input_size: 모델 입력 크기 {'width': int, 'height': int}
            scalers: Scaler 이름 리스트
        
        Returns:
            전처리된 텐서 [B, 3, H, W]
        """
        # Bbox를 사용하여 이미지 크롭
        if isinstance(resized_image, list):
            cropped_images = []
            for img, bbox in zip(resized_image, bbox_coords):
                x1, y1, x2, y2 = bbox
                cropped = img[y1:y2, x1:x2]
                cropped_resized = self.preprocessor.resize(cropped, input_size)
                cropped_images.append(cropped_resized)
            cropped_image = np.array(cropped_images)
        else:
            # 단일 이미지인 경우
            if len(bbox_coords) > 0:
                x1, y1, x2, y2 = bbox_coords[0]
                cropped = resized_image[y1:y2, x1:x2]
            else:
                cropped = resized_image
            cropped_resized = self.preprocessor.resize(cropped, input_size)
            cropped_image = cropped_resized[np.newaxis, :, :, :]
        
        # Scaler 적용
        scaled_image = cropped_image.copy()
        for scaler_str in scalers:
            scaler_func = get_image_scaler(scaler_str)
            if scaler_func is not None:
                # 배치 단위로 scaler 적용
                for i in range(scaled_image.shape[0]):
                    scaled_image[i] = scaler_func(scaled_image[i])
        
        # 텐서 변환 [B, H, W, C] -> [B, C, H, W]
        transpose_image = scaled_image.transpose(0, 3, 1, 2)
        image_tensor = torch.tensor(transpose_image, dtype=torch.float32).to(self.device)
        
        return image_tensor
    
    def _postprocess_mask(
        self,
        defect_mask: np.ndarray,
        config: Dict[str, Any],
        pp_key: str = 'base'
    ) -> np.ndarray:
        """
        결함 마스크를 후처리합니다.
        
        Args:
            defect_mask: Segmentation 결과 [B, classes, H, W] 또는 [B, H, W]
            config: 설정 딕셔너리
            pp_key: Post-processing key ('base' 또는 'flip')
        
        Returns:
            후처리된 마스크 [B, H, W]
        """
        # Temperature scaling, threshold, morphology 적용
        postprocessed = self.postprocessor.postprocess(defect_mask, section='front')
        
        # 전면 영역 특화 후처리
        front_config = config['defect_segmentation']['front']
        
        # 작은 결함 제거
        if front_config['postprocessing'].get('small_defect_removal', {}).get('enabled', False):
            pixel_threshold = front_config['postprocessing']['small_defect_removal'].get('pixel_count_threshold', 0)
            for i in range(postprocessed.shape[0]):
                postprocessed[i] = self.postprocessor.remove_small_defects(
                    postprocessed[i],
                    pixel_threshold
                )
        
        return postprocessed
    
    def _check_b_pixel_count(
        self,
        mask: np.ndarray
    ) -> int:
        """
        B 등급 픽셀 수를 확인합니다.
        
        Args:
            mask: Defect mask [H, W]
        
        Returns:
            B 등급 픽셀 수
        """
        return np.sum(mask == 2)  # B 등급은 2
    
    def _check_b_length_count(
        self,
        mask: np.ndarray,
        bbox_phonearea: Tuple[int, int, int, int],
        defect_input_shape: Dict[str, int],
        pixel_to_length_ratio: float,
        breakage_scratch_length_threshold: float
    ) -> Tuple[np.ndarray, bool]:
        """
        B 등급 결함의 길이를 확인합니다.
        
        Args:
            mask: Defect mask [H, W]
            bbox_phonearea: Phone 영역 bbox (x1, y1, x2, y2)
            defect_input_shape: Defect 모델 입력 크기
            pixel_to_length_ratio: 픽셀 대비 실제 길이 비율 (mm/pixel)
            length_threshold: 길이 임계값 (mm)
        
        Returns:
            (업데이트된 마스크, 길이 임계값 초과 여부)
        """
        # B 등급 픽셀만 추출
        b_mask = (mask == 2).astype(np.uint8)
        
        # Contour 찾기
        import cv2
        contours, _ = cv2.findContours(b_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        updated_mask = mask.copy()
        length_exceeded = False
        
        for contour in contours:
            # Contour의 길이 계산 (픽셀 단위)
            perimeter = cv2.arcLength(contour, closed=True)
            
            # 실제 길이로 변환 (mm)
            actual_length = perimeter * pixel_to_length_ratio
            
            if actual_length >= breakage_scratch_length_threshold:
                # 길이 임계값 초과 시 C 등급으로 변경 (23으로 표시)
                x, y, w, h = cv2.boundingRect(contour)
                updated_mask[y:y+h, x:x+w][mask[y:y+h, x:x+w] == 2] = 23
                length_exceeded = True
        
        return updated_mask, length_exceeded
    
    def infer(
        self,
        input_image: np.ndarray,
        bbox_coords_list: List[Tuple[int, int, int, int]],
        pp_key: str = 'base'
    ) -> Dict[str, Any]:
        """
        전면 영역 결함 검출을 수행합니다.
        
        Args:
            input_image: 입력 이미지 [H, W, 3] 또는 이미지 리스트
            bbox_coords_list: Phone 영역 bbox 좌표 리스트
            pp_key: Post-processing key ('base' 또는 'flip')
        
        Returns:
            검출 결과 딕셔너리
        """
        front_config = self.config['defect_segmentation']['front']
        
        # 설정 파라미터 가져오기
        scaler = front_config['preprocessing'].get('scaler', ['normalize'])
        img_size = front_config['input_size']
        pixel_ratio = front_config.get('pixel_to_length_ratio', 1.0)
        length_threshold = front_config.get('length_count_thres', 20.0)
        pixel_count_thres = front_config.get('pixel_count_thres', 0)
        
        # 1. 전처리
        model_input = self._preprocess_image(
            resized_image=input_image,
            bbox_coords=bbox_coords_list,
            input_size=img_size,
            scalers=scaler
        )
        
        # 2. Defect Segmentation 추론
        self.defect_segmenter.eval()
        with torch.no_grad():
            defect_logits, _ = self.defect_segmenter.predict_mask(model_input, return_probs=True)
            defect_result = defect_logits.cpu().numpy() if isinstance(defect_logits, torch.Tensor) else defect_logits
        
        # 3. 후처리
        post_images = self._postprocess_mask(
            defect_result,
            self.config,
            pp_key=pp_key
        )
        
        # 4. 전면 영역 특화 후처리
        grade_changed_mask = post_images.copy()
        
        # B 픽셀 수 검사
        b_pixel_count = self._check_b_pixel_count(post_images[0] if len(post_images.shape) == 3 else post_images)
        
        if b_pixel_count > pixel_count_thres:
            # B 픽셀 수가 임계값 초과 시 C 등급으로 변경 (23)
            grade_changed_mask[grade_changed_mask == 2] = 23
        else:
            # B 픽셀 수가 임계값 이하인 경우, 길이 검사
            if len(bbox_coords_list) >= 4:
                phone_bbox = bbox_coords_list[0]  # 첫 번째 bbox 사용
            else:
                phone_bbox = (0, 0, img_size['width'], img_size['height'])
            
            mask_for_length = post_images[0] if len(post_images.shape) == 3 else post_images
            grade_changed_mask, _ = self._check_b_length_count(
                mask_for_length,
                phone_bbox,
                img_size,
                pixel_ratio,
                length_threshold
            )
            if len(grade_changed_mask.shape) == 2:
                grade_changed_mask = grade_changed_mask[np.newaxis, :, :]
        
        # 5. 결함 분석
        analysis_result = self.grade_analyzer.analyze_defect_pixels(
            grade_changed_mask[0] if len(grade_changed_mask.shape) == 3 else grade_changed_mask,
            section='front'
        )
        
        # 6. 등급 결정
        grade_result = self.grade_analyzer.determine_defect_grade(
            analysis_result,
            section='front'
        )
        
        # 7. Top N 결함 선정
        top_defects = self.grade_analyzer.select_top_defects(
            analysis_result['defects'],
            top_n=self.config['defect_grading'].get('select_top_n', 2)
        )
        
        return {
            'defect_mask': grade_changed_mask.tolist() if isinstance(grade_changed_mask, np.ndarray) else grade_changed_mask,
            'analysis': analysis_result,
            'grade': grade_result,
            'top_defects': top_defects,
            'b_pixel_count': int(b_pixel_count),
            'pixel_count_threshold_exceeded': b_pixel_count > pixel_count_thres
        }

