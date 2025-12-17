"""
Display Defect Detection Service

디스플레이(Display) 영역의 결함 검출 서비스입니다.
회사 코드의 DisplayPipeline을 참고하여 리팩토링되었습니다.
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


class DisplayService:
    """
    Display Defect Detection Service
    
    디스플레이 영역의 결함을 검출하는 서비스입니다.
    Phone Segmentation → Display Crop → Defect Segmentation → Post-processing 파이프라인을 수행합니다.
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
        # Phone Segmentation 모델 (디스플레이 영역 검출)
        phone_seg_config = self.config['phone_detection']['display']
        
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
            self.phone_segmenter = None
        
        # Defect Segmentation 모델 (디스플레이 결함 검출)
        defect_config = self.config['defect_segmentation']['display']
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
        display_cropped: np.ndarray,
        input_size: Dict[str, int],
        scalers: List[str],
        apply_clahe: bool = False
    ) -> torch.Tensor:
        """
        디스플레이 이미지를 전처리하여 모델 입력 형식으로 변환합니다.
        
        Args:
            display_cropped: 크롭된 디스플레이 이미지 [H, W, 3]
            input_size: 모델 입력 크기 {'width': int, 'height': int}
            scalers: Scaler 이름 리스트
            apply_clahe: CLAHE 적용 여부
        
        Returns:
            전처리된 텐서 [1, 3, H, W]
        """
        # 1. 리사이즈
        resized = self.preprocessor.resize(display_cropped, input_size)
        
        # 2. 전처리 (CLAHE, Scaler 적용)
        preprocessed = self.preprocessor.preprocess(
            resized,
            section='display',
            apply_clahe_flag=apply_clahe
        )
        
        # 3. 텐서 변환 [H, W, C] -> [1, C, H, W]
        image_tensor = self.preprocessor.to_tensor(preprocessed)
        image_tensor = torch.from_numpy(image_tensor).float().to(self.device)
        
        return image_tensor
    
    def _postprocess_mask(
        self,
        defect_mask: np.ndarray,
        config: Dict[str, Any]
    ) -> np.ndarray:
        """
        결함 마스크를 후처리합니다.
        
        Args:
            defect_mask: Segmentation 결과 [B, classes, H, W] 또는 [B, H, W]
            config: 설정 딕셔너리
        
        Returns:
            후처리된 마스크 [B, H, W]
        """
        # Temperature scaling, threshold, morphology 적용
        postprocessed = self.postprocessor.postprocess(defect_mask, section='display')
        
        # 디스플레이 영역 특화 후처리
        display_config = config['defect_segmentation']['display']
        
        # 작은 결함 제거
        if display_config['postprocessing'].get('small_defect_removal', {}).get('enabled', False):
            pixel_threshold = display_config['postprocessing']['small_defect_removal'].get('pixel_count_threshold', 0)
            for i in range(postprocessed.shape[0]):
                postprocessed[i] = self.postprocessor.remove_small_defects(
                    postprocessed[i],
                    pixel_threshold
                )
        
        return postprocessed
    
    def infer(
        self,
        image: np.ndarray,
        threshold: float = 0.5,
        padding_ratio: float = 0.0
    ) -> Dict[str, Any]:
        """
        디스플레이 영역 결함 검출을 수행합니다.
        
        Args:
            image: 입력 이미지 [H, W, 3] (1080x1920)
            threshold: Phone segmentation threshold
            padding_ratio: Padding 비율
        
        Returns:
            검출 결과 딕셔너리
        """
        display_config = self.config['defect_segmentation']['display']
        
        # 설정 파라미터 가져오기
        scaler = display_config['preprocessing'].get('scaler', ['normalize'])
        img_size = display_config['input_size']
        apply_clahe = display_config['preprocessing'].get('clahe', {}).get('enabled', False)
        
        # 1. Phone Segmentation (디스플레이 영역 검출)
        if self.phone_segmenter is None:
            raise ValueError("Phone segmenter is not loaded. Check config['phone_detection']['display']")
        
        display_cropped, display_bbox = self.phone_segmenter.crop_phone_region(
            image,
            threshold=threshold,
            padding_ratio=padding_ratio
        )
        
        # 2. 전처리
        model_input = self._preprocess_image(
            display_cropped=display_cropped,
            input_size=img_size,
            scalers=scaler,
            apply_clahe=apply_clahe
        )
        
        # 3. Defect Segmentation 추론
        self.defect_segmenter.eval()
        with torch.no_grad():
            defect_logits, _ = self.defect_segmenter.predict_mask(model_input, return_probs=True)
            defect_result = defect_logits.cpu().numpy() if isinstance(defect_logits, torch.Tensor) else defect_logits
        
        # 4. 후처리
        post_images = self._postprocess_mask(
            defect_result,
            self.config
        )
        
        # 5. 결함 분석
        analysis_result = self.grade_analyzer.analyze_defect_pixels(
            post_images[0] if len(post_images.shape) == 3 else post_images,
            section='display'
        )
        
        # 6. 등급 결정
        grade_result = self.grade_analyzer.determine_defect_grade(
            analysis_result,
            section='display'
        )
        
        # 7. Top N 결함 선정
        top_defects = self.grade_analyzer.select_top_defects(
            analysis_result['defects'],
            top_n=self.config['defect_grading'].get('select_top_n', 2)
        )
        
        return {
            'display_bbox': display_bbox.tolist() if isinstance(display_bbox, np.ndarray) else display_bbox,
            'defect_mask': post_images[0].tolist() if len(post_images.shape) == 3 else post_images.tolist(),
            'analysis': analysis_result,
            'grade': grade_result,
            'top_defects': top_defects
        }

