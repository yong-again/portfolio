"""
Kiosk Service for Age & Gender Estimation

키오스크 환경에서 나이/성별 추정 서비스를 제공하는 메인 클래스입니다.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import yaml
import sys
import uuid
from datetime import datetime
import threading
import time

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.network import build_network
from models.utils import load_checkpoint
from models.gender_head import class_to_gender
from preprocess.transforms import get_val_transforms
from detection.predict_detector import HeadDetector

from .camera_handler import CameraHandler, simulate_distance_sensor
from .image_quality import ImageQualityFilter
from .database import DatabaseManager


class KioskService:
    """
    키오스크 서비스 메인 클래스
    
    전체 파이프라인을 관리합니다:
    1. 사용자 접근 감지 (20m 이내)
    2. 멀티스레드로 10장 촬영
    3. 이미지 품질 필터링
    4. Head Detection 및 필터링
    5. Age & Gender 추정
    6. DB 저장
    """
    
    def __init__(
        self,
        config_path: str = "configs/config.yaml",
        detection_model_path: Optional[str] = None,
        age_gender_weights: Optional[str] = None,
        db_path: str = "data/kiosk_results.db",
        camera_id: int = 0,
        enable_multi_person: bool = False,
        detection_probability_threshold: float = 0.5,
        head_detection_size: Tuple[int, int] = (640, 640),
        age_gender_input_size: Tuple[int, int] = (224, 224)
    ):
        """
        Args:
            config_path: 설정 파일 경로
            detection_model_path: Head detection 모델 경로
            age_gender_weights: Age/Gender 모델 가중치 경로
            db_path: 데이터베이스 경로
            camera_id: 카메라 ID
            enable_multi_person: 여러 사람 탐지 활성화 여부
            detection_probability_threshold: Head detection probability threshold
            head_detection_size: Head detection용 resize 크기
            age_gender_input_size: Age/Gender 추정용 입력 크기
        """
        # Config 로드
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Device 설정
        device_config = self.config['device']
        if device_config['type'] == 'cuda' and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{device_config['device_id']}")
        elif device_config['type'] == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"Device: {self.device}")
        
        # Detection 모델 경로
        self.detection_model_path = (
            detection_model_path or 
            self.config['detection']['trained_model']
        )
        
        # Age/Gender 모델 가중치 경로
        self.age_gender_weights = age_gender_weights
        
        # 서비스 설정
        self.enable_multi_person = enable_multi_person
        self.detection_probability_threshold = detection_probability_threshold
        self.head_detection_size = head_detection_size
        self.age_gender_input_size = age_gender_input_size
        
        # 모델 로드
        self.detection_model: Optional[HeadDetector] = None
        self.age_gender_model: Optional[nn.Module] = None
        
        self._load_models()
        
        # 이미지 품질 필터
        self.image_quality_filter = ImageQualityFilter(
            blur_threshold=self.config.get('service', {}).get('blur_threshold', 100.0),
            brightness_min=self.config.get('service', {}).get('brightness_min', 0.1),
            brightness_max=self.config.get('service', {}).get('brightness_max', 0.9),
            contrast_min=self.config.get('service', {}).get('contrast_min', 0.3)
        )
        
        # 카메라 핸들러
        self.camera_handler = CameraHandler(camera_id=camera_id)
        
        # 데이터베이스
        self.db_manager = DatabaseManager(db_path=db_path)
        
        # 서비스 상태
        self.is_running = False
        self.service_thread: Optional[threading.Thread] = None
        
        # Transform
        self.transform = get_val_transforms(self.config)
    
    def _load_models(self):
        """모델 로드"""
        # Head Detection 모델
        if Path(self.detection_model_path).exists():
            self.detection_model = HeadDetector(
                model_path=self.detection_model_path,
                conf_threshold=self.config['detection']['conf_threshold'],
                iou_threshold=self.config['detection']['iou_threshold'],
                device=str(self.device)
            )
            print(f"Loaded head detection model from {self.detection_model_path}")
        else:
            raise FileNotFoundError(
                f"Detection model not found: {self.detection_model_path}"
            )
        
        # Age & Gender 모델
        if self.age_gender_weights and Path(self.age_gender_weights).exists():
            self.age_gender_model = build_network(self.config)
            self.age_gender_model = self.age_gender_model.to(self.device)
            
            checkpoint_info = load_checkpoint(
                self.age_gender_weights, 
                self.age_gender_model, 
                device=str(self.device)
            )
            self.age_gender_model.eval()
            print(f"Loaded age & gender model from {self.age_gender_weights}")
        else:
            raise FileNotFoundError(
                f"Age/Gender model weights not found: {self.age_gender_weights}"
            )
    
    def start_service(self):
        """서비스 시작"""
        if self.is_running:
            return
        
        # 카메라 시작
        if not self.camera_handler.start():
            raise RuntimeError("Failed to start camera")
        
        self.is_running = True
        
        # 서비스 스레드 시작
        self.service_thread = threading.Thread(target=self._service_loop, daemon=True)
        self.service_thread.start()
        
        print("Kiosk service started")
    
    def stop_service(self):
        """서비스 중지"""
        self.is_running = False
        
        if self.service_thread and self.service_thread.is_alive():
            self.service_thread.join(timeout=5.0)
        
        self.camera_handler.stop()
        print("Kiosk service stopped")
    
    def _service_loop(self):
        """서비스 메인 루프"""
        while self.is_running:
            # 20m 이내 접근 감지 (센서에서 값이 바로 넘어옴)
            if simulate_distance_sensor():
                # 촬영 및 처리 시작
                self._process_user_session()
            
            time.sleep(0.5)  # 0.5초마다 체크
    
    def _process_user_session(self):
        """
        사용자 세션 처리
        
        전체 파이프라인을 실행합니다:
        1. 10장 촬영 (멀티스레드)
        2. 이미지 품질 필터링
        3. Head Detection 및 필터링
        4. Age & Gender 추정
        5. DB 저장
        """
        try:
            # 1. 멀티스레드로 10장 촬영
            print("Capturing 10 images...")
            captured_images = self.camera_handler.capture_multiple_simple(
                num_images=10,
                interval=0.1
            )
            
            if len(captured_images) == 0:
                print("No images captured")
                return
            
            print(f"Captured {len(captured_images)} images")
            
            # 2. 이미지 품질 필터링
            print("Filtering images by quality...")
            filtered_images, quality_scores = self.image_quality_filter.filter_images(
                captured_images,
                return_scores=True
            )
            
            if len(filtered_images) == 0:
                print("All images failed quality check")
                return
            
            print(f"Filtered to {len(filtered_images)} good quality images")
            
            # 3. 각 이미지에 대해 멀티스레드로 추론 시작 (1장씩)
            all_results = []
            results_lock = threading.Lock()
            
            def inference_worker(image, idx):
                """개별 이미지 추론 워커"""
                result = self._process_single_image(image, idx)
                if result:
                    with results_lock:
                        all_results.append(result)
            
            # 멀티스레드로 각 이미지 추론
            inference_threads = []
            for idx, image in enumerate(filtered_images):
                thread = threading.Thread(
                    target=inference_worker,
                    args=(image, idx),
                    daemon=True
                )
                thread.start()
                inference_threads.append(thread)
            
            # 모든 추론 스레드 완료 대기
            for thread in inference_threads:
                thread.join()
            
            # 4. DB 저장
            if all_results:
                self._save_results_to_db(all_results)
                print(f"Saved {len(all_results)} results to database")
        
        except Exception as e:
            print(f"Error processing user session: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_single_image(
        self,
        image: np.ndarray,
        image_idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        단일 이미지 처리
        
        Args:
            image: 입력 이미지 (numpy array, RGB)
            image_idx: 이미지 인덱스
        
        Returns:
            처리 결과 딕셔너리 또는 None
        """
        try:
            # PIL Image로 변환
            pil_image = Image.fromarray(image)
            
            # 4. Head Detection을 위한 resize
            detection_image = pil_image.resize(
                self.head_detection_size, 
                Image.Resampling.LANCZOS
            )
            
            # Head Detection
            detection_result = self.detection_model.predict(
                detection_image,
                return_crops=False,
                padding_ratio=self.config['detection']['padding_ratio']
            )
            
            if detection_result['num_detections'] == 0:
                return None
            
            # 5. RMS 계산 후 probability threshold를 넘은 head만 추출
            # RMS = Root Mean Square of detection confidence scores
            valid_heads = []
            
            for box, score in zip(detection_result['boxes'], detection_result['scores']):
                # RMS 계산 (여기서는 confidence score를 RMS로 간주)
                rms_score = np.sqrt(np.mean(np.array(score) ** 2)) if isinstance(score, (list, np.ndarray)) else score
                
                if rms_score >= self.detection_probability_threshold:
                    valid_heads.append({
                        'box': box,
                        'score': float(rms_score),
                        'area': (box[2] - box[0]) * (box[3] - box[1])
                    })
            
            if len(valid_heads) == 0:
                return None
            
            # 6. 가장 큰 head만 필터링 (여러 사람 탐지 on-off)
            if self.enable_multi_person:
                # 여러 사람 모드: 모든 valid head 사용
                selected_heads = valid_heads
            else:
                # 단일 사람 모드: 가장 큰 head만 선택
                selected_heads = [max(valid_heads, key=lambda h: h['area'])]
            
            # 각 head에 대해 Age/Gender 추정
            results = []
            
            for head in selected_heads:
                # 원본 이미지에서 head crop
                x1, y1, x2, y2 = head['box']
                
                # 원본 이미지 크기에 맞게 좌표 조정
                scale_x = pil_image.width / self.head_detection_size[0]
                scale_y = pil_image.height / self.head_detection_size[1]
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Padding 추가
                padding_ratio = self.config['detection']['padding_ratio']
                width = x2 - x1
                height = y2 - y1
                pad_w = width * padding_ratio
                pad_h = height * padding_ratio
                
                x1 = max(0, int(x1 - pad_w))
                y1 = max(0, int(y1 - pad_h))
                x2 = min(pil_image.width, int(x2 + pad_w))
                y2 = min(pil_image.height, int(y2 + pad_h))
                
                # Crop
                head_crop = pil_image.crop((x1, y1, x2, y2))
                
                # 7. Age/Gender 추정을 위한 resize
                head_resized = head_crop.resize(
                    self.age_gender_input_size,
                    Image.Resampling.LANCZOS
                )
                
                # 8. Model input 및 추론
                image_tensor = self.transform(head_resized)
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    predictions = self.age_gender_model.predict(image_tensor)
                
                # 결과 파싱
                age_predicted = predictions['age']['predicted_age'][0].item()
                age_prob = predictions['age']['probs'][0][age_predicted].item()
                
                gender_class = predictions['gender']['predicted_class'][0].item()
                gender_str = class_to_gender(gender_class)
                gender_prob = predictions['gender']['confidence'][0].item()
                
                results.append({
                    'image_idx': image_idx,
                    'head_bbox': [x1, y1, x2, y2],
                    'detection_confidence': head['score'],
                    'age': age_predicted,
                    'age_confidence': float(age_prob),
                    'gender': gender_str,
                    'gender_confidence': float(gender_prob)
                })
            
            # 여러 head가 있는 경우 가장 높은 confidence의 결과 반환
            if results:
                best_result = max(results, key=lambda r: r['detection_confidence'])
                return best_result
            
            return None
        
        except Exception as e:
            print(f"Error processing image {image_idx}: {e}")
            return None
    
    def _save_results_to_db(self, results: List[Dict[str, Any]]):
        """
        결과를 DB에 저장
        
        Args:
            results: 처리 결과 리스트
        """
        for result in results:
            # 고유 이미지 ID 생성
            image_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            self.db_manager.save_result(
                image_id=image_id,
                age=result.get('age'),
                gender=result.get('gender'),
                gender_confidence=result.get('gender_confidence'),
                age_confidence=result.get('age_confidence'),
                head_bbox=result.get('head_bbox'),
                detection_confidence=result.get('detection_confidence'),
                metadata={
                    'image_idx': result.get('image_idx'),
                    'processing_timestamp': datetime.now().isoformat()
                }
            )
    
    def process_single_image_file(
        self,
        image_path: str,
        save_to_db: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        단일 이미지 파일 처리 (테스트용)
        
        Args:
            image_path: 이미지 파일 경로
            save_to_db: DB 저장 여부
        
        Returns:
            처리 결과
        """
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        result = self._process_single_image(image_array, 0)
        
        if result and save_to_db:
            image_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            
            self.db_manager.save_result(
                image_id=image_id,
                age=result.get('age'),
                gender=result.get('gender'),
                gender_confidence=result.get('gender_confidence'),
                age_confidence=result.get('age_confidence'),
                head_bbox=result.get('head_bbox'),
                detection_confidence=result.get('detection_confidence'),
                image_path=image_path
            )
        
        return result
    
if __name__ == "__main__":
    # 테스트 코드
    service = KioskService(
        config_path="configs/config.yaml",
        age_gender_weights="weights/best_model.pt",
        enable_multi_person=False
    )
    
    # 단일 이미지 테스트
    result = service.process_single_image_file(
        "data/sample/test.jpg",
        save_to_db=True
    )
    
    if result:
        print("Processing result:")
        print(f"  Age: {result['age']} (confidence: {result['age_confidence']:.3f})")
        print(f"  Gender: {result['gender']} (confidence: {result['gender_confidence']:.3f})")

