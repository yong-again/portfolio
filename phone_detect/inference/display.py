"""
Display Defect Detection Pipeline

디스플레이 결함 검출 파이프라인입니다.
Phone Segmentation → Display Crop → Defect Segmentation → Post-processing → Grade Determination
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.phone_segmenter import PhoneSegmenter
from models.defect_segmenter import DefectSegmenter
from models.utils import load_checkpoint
from preprocess.defect_preprocess import DefectPreprocessor
from preprocess.defect_postprocess import DefectPostprocessor
from utils.defect_grade import DefectGradeAnalyzer


class DisplayPipeline:
    """
    Display Defect Detection Pipeline
    
    디스플레이 영역의 결함을 검출합니다.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cuda"):
        """
        Args:
            config: 설정 딕셔너리
            device: 디바이스
        """
        self.config = config
        self.device = device
        
        # 모델 로드
        self._load_models()
        
        # 전처리/후처리
        self.preprocessor = DefectPreprocessor(config)
        self.postprocessor = DefectPostprocessor(config)
        self.grade_analyzer = DefectGradeAnalyzer(config)
    
    def _load_models(self):
        """모델을 로드합니다."""
        # Phone Segmentation 모델 (디스플레이 영역 검출)
        phone_seg_config = self.config['phone_detection']['display']
        self.phone_segmenter = PhoneSegmenter(
            encoder_name=phone_seg_config['model']['encoder_name'],
            encoder_weights=phone_seg_config['model']['encoder_weights'],
            classes=phone_seg_config['model']['classes']
        )
        
        phone_seg_path = phone_seg_config['trained_model']
        if Path(phone_seg_path).exists():
            load_checkpoint(phone_seg_path, self.phone_segmenter, device=self.device)
        self.phone_segmenter = self.phone_segmenter.to(self.device)
        self.phone_segmenter.eval()
        
        # Defect Segmentation 모델
        defect_config = self.config['defect_segmentation']['display']
        self.defect_segmenter = DefectSegmenter(
            encoder_name=defect_config['model']['encoder_name'],
            encoder_weights=defect_config['model']['encoder_weights'],
            decoder_name=defect_config['model']['decoder_name'],
            classes=defect_config['model']['classes']
        )
        
        defect_path = defect_config['trained_model']
        if Path(defect_path).exists():
            load_checkpoint(defect_path, self.defect_segmenter, device=self.device)
        self.defect_segmenter = self.defect_segmenter.to(self.device)
        self.defect_segmenter.eval()
    
    def infer(
        self,
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        디스플레이 결함 검출을 수행합니다.
        
        Args:
            image: 입력 이미지 [H, W, 3] (1080x1920)
        
        Returns:
            검출 결과 딕셔너리
        """
        # 1. Phone Segmentation (디스플레이 영역 검출)
        display_cropped, display_bbox = self.phone_segmenter.crop_phone_region(
            image,
            threshold=0.5,
            padding_ratio=0.0
        )
        
        # 2. 전처리
        input_size = self.config['defect_segmentation']['display']['input_size']
        resized = self.preprocessor.resize(display_cropped, input_size)
        preprocessed = self.preprocessor.preprocess(resized, section='display', apply_clahe_flag=False)
        image_tensor = self.preprocessor.to_tensor(preprocessed)
        image_tensor = torch.from_numpy(image_tensor).float().to(self.device)
        
        # 3. Defect Segmentation
        with torch.no_grad():
            defect_logits = self.defect_segmenter(image_tensor)
            defect_logits_np = defect_logits.cpu().numpy()
        
        # 4. 후처리
        defect_mask = self.postprocessor.postprocess(defect_logits_np, section='display')
        
        # 5. 결함 분석
        analysis_result = self.grade_analyzer.analyze_defect_pixels(
            defect_mask[0],  # 첫 번째 배치
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
            top_n=self.config['defect_grading']['select_top_n']
        )
        
        return {
            'display_bbox': display_bbox.tolist(),
            'defect_mask': defect_mask[0].tolist(),
            'analysis': analysis_result,
            'grade': grade_result,
            'top_defects': top_defects
        }

