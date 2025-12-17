"""
Back Defect Detection Service

후면(Back) 영역의 결함 검출 서비스입니다.
회사 코드의 BackPipeline을 참고하여 리팩토링되었습니다.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.defect_segmenter import DefectSegmenter
from models.phone_detector import PhoneDetector
from models.utils import load_checkpoint
from preprocess.defect_preprocess import DefectPreprocessor, get_image_scaler
from preprocess.defect_postprocess import DefectPostprocessor
from utils.defect_grade import DefectGradeAnalyzer


class BackService:
    """
    Back Defect Detection Service
    
    후면 영역의 결함을 검출하는 서비스입니다.
    Phone Detection (YOLO) → Defect Segmentation → Post-processing 파이프라인을 수행합니다.
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
        # Phone Detection 모델 (후면 영역 검출) - YOLO 사용
        phone_det_config = self.config['phone_detection']['back']
        
        if phone_det_config.get('method') == 'yolo':
            model_path = phone_det_config.get('trained_model')
            if model_path and Path(model_path).exists():
                self.phone_detector = PhoneDetector(
                    model_path=model_path,
                    conf_threshold=0.25,
                    iou_threshold=0.45,
                    device=self.device
                )
            else:
                # 사전 학습 모델 사용
                self.phone_detector = PhoneDetector(
                    model_path=phone_det_config.get('model', 'yolo11n.pt'),
                    conf_threshold=0.25,
                    iou_threshold=0.45,
                    device=self.device
                )
        else:
            self.phone_detector = None
        
        # Defect Segmentation 모델 (후면 결함 검출)
        defect_config = self.config['defect_segmentation'].get('back')
        if defect_config is None:
            # back 설정이 없으면 front 설정 사용
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
        postprocessed = self.postprocessor.postprocess(defect_mask, section='back')
        
        # 후면 영역 특화 후처리
        back_config = self.config['defect_segmentation'].get('back')
        if back_config is None:
            back_config = self.config['defect_segmentation']['front']
        
        # 작은 결함 제거
        if back_config['postprocessing'].get('small_defect_removal', {}).get('enabled', False):
            pixel_threshold = back_config['postprocessing']['small_defect_removal'].get('pixel_count_threshold', 0)
            for i in range(postprocessed.shape[0]):
                postprocessed[i] = self.postprocessor.remove_small_defects(
                    postprocessed[i],
                    pixel_threshold
                )
        
        # 엣지 결함 제거 (설정이 있는 경우)
        post_key = config.get('POSTPROC', {})
        if post_key and 'remove_back_edge_defect' in post_key:
            postprocessed = self._remove_back_edge_defect(
                postprocessed,
                **post_key['remove_back_edge_defect']
            )
        
        return postprocessed
    
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
        import cv2
        
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
    
    def infer(
        self,
        input_image: np.ndarray,
        bbox_coords_list: Optional[List[Tuple[int, int, int, int]]] = None,
        pp_key: str = 'base'
    ) -> Dict[str, Any]:
        """
        후면 영역 결함 검출을 수행합니다.
        
        Args:
            input_image: 입력 이미지 [H, W, 3] (1080x1920)
            bbox_coords_list: Phone 영역 bbox 좌표 리스트 (None이면 자동 검출)
            pp_key: Post-processing key ('base' 또는 'flip')
        
        Returns:
            검출 결과 딕셔너리
        """
        back_config = self.config['defect_segmentation'].get('back')
        if back_config is None:
            back_config = self.config['defect_segmentation']['front']
        
        # 설정 파라미터 가져오기
        scaler = back_config['preprocessing'].get('scaler', ['normalize'])
        img_size = back_config['input_size']
        
        # 1. Phone Detection (후면 영역 검출)
        if bbox_coords_list is None:
            if self.phone_detector is None:
                raise ValueError("Phone detector is not loaded. Check config['phone_detection']['back']")
            
            # YOLO로 phone 영역 검출
            bbox, confidence = self.phone_detector.detect(input_image)
            # bbox 형식 변환: [x1, y1, x2, y2] -> (y1, y2, x1, x2)
            bbox_coords_list = [(int(bbox[1]), int(bbox[3]), int(bbox[0]), int(bbox[2]))]
        else:
            # bbox가 이미 제공된 경우
            if len(bbox_coords_list) > 0 and len(bbox_coords_list[0]) == 4:
                # (y1, y2, x1, x2) 형식으로 가정
                pass
            else:
                raise ValueError(f"Invalid bbox format: {bbox_coords_list}")
        
        # 2. 전처리
        model_input = self._preprocess_image(
            resized_image=input_image,
            bbox_coords=bbox_coords_list,
            input_size=img_size,
            scalers=scaler
        )
        
        # 3. Defect Segmentation 추론
        self.defect_segmenter.eval()
        with torch.no_grad():
            defect_logits, _ = self.defect_segmenter.predict_mask(model_input, return_probs=True)
            defect_result = defect_logits.cpu().numpy() if isinstance(defect_logits, torch.Tensor) else defect_logits
        
        # 4. 후처리
        post_images = self._postprocess_mask(
            defect_result,
            self.config,
            pp_key=pp_key
        )
        
        # 5. 결함 분석
        analysis_result = self.grade_analyzer.analyze_defect_pixels(
            post_images[0] if len(post_images.shape) == 3 else post_images,
            section='back'
        )
        
        # 6. 등급 결정
        grade_result = self.grade_analyzer.determine_defect_grade(
            analysis_result,
            section='back'
        )
        
        # 7. Top N 결함 선정
        top_defects = self.grade_analyzer.select_top_defects(
            analysis_result['defects'],
            top_n=self.config['defect_grading'].get('select_top_n', 2)
        )
        
        return {
            'phone_bbox': bbox_coords_list[0] if len(bbox_coords_list) > 0 else None,
            'defect_mask': post_images[0].tolist() if len(post_images.shape) == 3 else post_images.tolist(),
            'analysis': analysis_result,
            'grade': grade_result,
            'top_defects': top_defects
        }

