"""
YOLO Head Detection Prediction Script

YOLO를 사용하여 이미지에서 human head를 검출하고 crop합니다.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
import cv2


class HeadDetector:
    """
    YOLO 기반 Human Head Detector
    
    이미지에서 머리를 검출하고, 검출된 영역을 crop하여 반환합니다.
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
            model_path: 학습된 YOLO 모델 경로
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS를 위한 IoU 임계값
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
    
    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        return_crops: bool = True,
        padding_ratio: float = 0.1
    ) -> Dict:
        """
        이미지에서 head를 검출합니다.
        
        Args:
            image: 입력 이미지 (경로, PIL Image, numpy array)
            return_crops: 검출된 head 영역을 crop하여 반환할지 여부
            padding_ratio: Crop 시 추가할 padding 비율 (0.1 = 10%)
        
        Returns:
            다음을 포함하는 딕셔너리:
                - 'boxes': 검출된 바운딩 박스 [N, 4] (x1, y1, x2, y2)
                - 'scores': 신뢰도 점수 [N]
                - 'crops': (선택사항) Crop된 머리 이미지 리스트 [PIL.Image]
                - 'num_detections': 검출 개수
        """
        # 이미지 로드
        if isinstance(image, (str, Path)):
            image_path = str(image)
            pil_image = Image.open(image_path).convert('RGB')
            image_array = np.array(pil_image)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
            image_array = np.array(pil_image)
        elif isinstance(image, np.ndarray):
            image_array = image
            pil_image = Image.fromarray(image_array)
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
        boxes = []
        scores = []
        crops = []
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes_tensor = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] (x1, y1, x2, y2)
            scores_tensor = results[0].boxes.conf.cpu().numpy()  # [N]
            
            boxes = boxes_tensor.tolist()
            scores = scores_tensor.tolist()
            
            # Crop 수행
            if return_crops:
                for box in boxes:
                    x1, y1, x2, y2 = box
                    
                    # Padding 추가
                    width = x2 - x1
                    height = y2 - y1
                    pad_w = width * padding_ratio
                    pad_h = height * padding_ratio
                    
                    x1 = max(0, int(x1 - pad_w))
                    y1 = max(0, int(y1 - pad_h))
                    x2 = min(pil_image.width, int(x2 + pad_w))
                    y2 = min(pil_image.height, int(y2 + pad_h))
                    
                    # Crop
                    crop = pil_image.crop((x1, y1, x2, y2))
                    crops.append(crop)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'crops': crops if return_crops else None,
            'num_detections': len(boxes)
        }
    
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
        return_crops: bool = True,
        padding_ratio: float = 0.1
    ) -> List[Dict]:
        """
        여러 이미지에 대해 배치로 head를 검출합니다.
        
        Args:
            images: 입력 이미지 리스트
            return_crops: 검출된 head 영역을 crop하여 반환할지 여부
            padding_ratio: Crop 시 추가할 padding 비율
        
        Returns:
            검출 결과 리스트
        """
        results = []
        for image in images:
            result = self.predict(image, return_crops, padding_ratio)
            results.append(result)
        
        return results


def predict_heads(
    model_path: str,
    source: Union[str, Path, List],
    output_dir: str = "results/detection",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    save_crops: bool = True,
    save_annotated: bool = True,
    device: str = None
):
    """
    이미지 또는 디렉토리에서 head를 검출합니다.
    
    Args:
        model_path: 학습된 YOLO 모델 경로
        source: 입력 이미지 경로 또는 디렉토리
        output_dir: 결과 저장 디렉토리
        conf_threshold: 신뢰도 임계값
        iou_threshold: IoU 임계값
        save_crops: Crop된 head 이미지 저장 여부
        save_annotated: 검출 결과가 그려진 이미지 저장 여부
        device: 추론 디바이스
    """
    detector = HeadDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        device=device
    )
    
    # 입력 경로 처리
    source_path = Path(source)
    if source_path.is_file():
        image_paths = [source_path]
    elif source_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(list(source_path.glob(f'*{ext}')))
        image_paths = sorted(image_paths)
    else:
        raise ValueError(f"Invalid source path: {source_path}")
    
    print(f"Found {len(image_paths)} image(s)")
    
    # 출력 디렉토리 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = output_dir / "crops" if save_crops else None
    annotated_dir = output_dir / "annotated" if save_annotated else None
    
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
    if annotated_dir:
        annotated_dir.mkdir(parents=True, exist_ok=True)
    
    # 검출 수행
    total_detections = 0
    
    for image_path in image_paths:
        print(f"Processing: {image_path}")
        
        result = detector.predict(image_path, return_crops=save_crops)
        total_detections += result['num_detections']
        
        print(f"  Detected {result['num_detections']} head(s)")
        
        # Crop 저장
        if save_crops and result['crops']:
            for idx, crop in enumerate(result['crops']):
                crop_path = crops_dir / f"{image_path.stem}_head_{idx}.jpg"
                crop.save(crop_path)
        
        # Annotated 이미지 저장
        if save_annotated:
            image = Image.open(image_path).convert('RGB')
            annotated = draw_boxes(image, result['boxes'], result['scores'])
            annotated_path = annotated_dir / f"{image_path.stem}_annotated.jpg"
            annotated.save(annotated_path)
    
    print(f"\nDetection completed!")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average detections per image: {total_detections / len(image_paths):.2f}")
    print(f"  Results saved to: {output_dir}")


def draw_boxes(image: Image.Image, boxes: List[List[float]], scores: List[float]) -> Image.Image:
    """
    이미지에 bounding box를 그립니다.
    
    Args:
        image: PIL Image
        boxes: 바운딩 박스 [N, 4] (x1, y1, x2, y2)
        scores: 신뢰도 점수 [N]
    
    Returns:
        바운딩 박스가 그려진 이미지
    """
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = box
        
        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        
        # 라벨 그리기
        label = f"Head: {score:.2f}"
        draw.text((x1, y1 - 25), label, fill='red', font=font)
    
    return image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO Head Detection Prediction')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--output', type=str, default='results/detection',
                       help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold')
    parser.add_argument('--no-crops', action='store_true',
                       help='Do not save cropped heads')
    parser.add_argument('--no-annotated', action='store_true',
                       help='Do not save annotated images')
    parser.add_argument('--device', type=str, default=None,
                       help='Inference device')
    
    args = parser.parse_args()
    
    predict_heads(
        model_path=args.model,
        source=args.source,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_crops=not args.no_crops,
        save_annotated=not args.no_annotated,
        device=args.device
    )

