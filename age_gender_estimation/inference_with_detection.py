"""
Inference Script with Head Detection

Head Detection → Crop → Age & Gender Estimation 파이프라인을 수행합니다.
"""

import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import sys
from typing import List, Dict, Any

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from models.network import build_network
from models.utils import load_checkpoint
from models.gender_head import class_to_gender
from preprocess.transforms import get_val_transforms
from detection.predict_detector import HeadDetector


def inference_with_detection(
    detection_model: HeadDetector,
    age_gender_model: nn.Module,
    image_path: str,
    config: Dict[str, Any],
    device: torch.device,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Head Detection → Age & Gender Estimation 파이프라인을 수행합니다.
    
    Args:
        detection_model: Head detection 모델
        age_gender_model: Age & Gender estimation 모델
        image_path: 이미지 파일 경로
        config: 설정 딕셔너리
        device: 디바이스
        visualize: 시각화 여부
    
    Returns:
        검출 및 추론 결과 딕셔너리
    """
    # 1. Head Detection
    detection_result = detection_model.predict(
        image_path,
        return_crops=True,
        padding_ratio=config['detection']['padding_ratio']
    )
    
    if detection_result['num_detections'] == 0:
        return {
            'image_path': image_path,
            'num_detections': 0,
            'detections': []
        }
    
    # 2. Age & Gender Estimation for each detected head (동시에 추론)
    transform = get_val_transforms(config)
    age_gender_model.eval()
    
    detections = []
    
    with torch.no_grad():
        for idx, (crop, box, score) in enumerate(zip(
            detection_result['crops'],
            detection_result['boxes'],
            detection_result['scores']
        )):
            # 전처리
            image_tensor = transform(crop)
            image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
            
            # Age & Gender 추론: 나이와 성별을 동시에 예측
            predictions = age_gender_model.predict(image_tensor)
            
            # 결과 파싱
            age_predicted = predictions['age']['predicted_age'][0].item()  # 0~100세
            age_prob = predictions['age']['probs'][0][age_predicted].item()
            
            gender_class = predictions['gender']['predicted_class'][0].item()
            gender_str = class_to_gender(gender_class)
            gender_prob = predictions['gender']['confidence'][0].item()
            
            detections.append({
                'head_index': idx,
                'bbox': box,
                'detection_confidence': float(score),
                'age': {
                    'predicted_age': age_predicted,  # 0~100세
                    'confidence': float(age_prob)
                },
                'gender': {
                    'class': gender_class,
                    'label': gender_str,
                    'confidence': float(gender_prob)
                }
            })
    
    result = {
        'image_path': image_path,
        'num_detections': detection_result['num_detections'],
        'detections': detections
    }
    
    return result


def save_results(
    results: List[Dict[str, Any]],
    output_dir: str,
    output_format: str = 'json'
):
    """
    추론 결과를 저장합니다.
    
    Args:
        results: 검출 및 추론 결과 리스트
        output_dir: 출력 디렉토리
        output_format: 출력 형식 ('json', 'csv')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_format == 'json':
        # JSON 형식으로 저장
        output_file = output_dir / 'results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    elif output_format == 'csv':
        # CSV 형식으로 저장
        import pandas as pd
        
        rows = []
        for result in results:
            for det in result['detections']:
                rows.append({
                    'image_path': result['image_path'],
                    'head_index': det['head_index'],
                    'bbox_x1': det['bbox'][0],
                    'bbox_y1': det['bbox'][1],
                    'bbox_x2': det['bbox'][2],
                    'bbox_y2': det['bbox'][3],
                    'detection_confidence': det['detection_confidence'],
                    'age': det['age']['predicted_age'],  # 0~100세
                    'age_confidence': det['age']['confidence'],
                    'gender': det['gender']['label'],
                    'gender_confidence': det['gender']['confidence']
                })
        
        df = pd.DataFrame(rows)
        output_file = output_dir / 'results.csv'
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


def visualize_result(image: Image.Image, result: Dict[str, Any]) -> Image.Image:
    """
    이미지에 검출 및 추론 결과를 시각화합니다.
    
    Args:
        image: 원본 이미지
        result: 검출 및 추론 결과
    
    Returns:
        시각화된 이미지
    """
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    for det in result['detections']:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        
        # 바운딩 박스 그리기
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # 라벨 정보
        age_info = f"Age: {det['age']['predicted_age']}세"  # 0~100세
        gender_info = f"Gender: {det['gender']['label']}"
        det_conf = f"Det: {det['detection_confidence']:.2f}"
        age_conf = f"Age: {det['age']['confidence']:.2f}"
        gen_conf = f"Gen: {det['gender']['confidence']:.2f}"
        
        # 텍스트 배경
        text_y = y1 - 80
        if text_y < 0:
            text_y = y2 + 5
        
        # 라벨 그리기
        draw.text((x1, text_y), age_info, fill='red', font=font)
        draw.text((x1, text_y + 25), gender_info, fill='blue', font=font)
        draw.text((x1, text_y + 50), f"{det_conf} | {age_conf} | {gen_conf}", fill='green', font=small_font)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='Age & Gender Estimation with Head Detection')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--detection-model', type=str, default=None,
                       help='Path to head detection model (overrides config)')
    parser.add_argument('--age-gender-weights', type=str, required=True,
                       help='Path to age & gender estimation model weights')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image')
    parser.add_argument('--source', type=str, default=None,
                       help='Path to image directory')
    parser.add_argument('--output', type=str, default='results/inference_with_detection',
                       help='Output directory')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--save-images', action='store_true',
                       help='Save visualized images')
    
    args = parser.parse_args()
    
    # Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Detection 모델 경로
    detection_model_path = args.detection_model or config['detection']['trained_model']
    if not Path(detection_model_path).exists():
        raise FileNotFoundError(
            f"Detection model not found: {detection_model_path}\n"
            f"Please train the head detection model first using:\n"
            f"  python detection/train_detector.py --data {config['detection']['data_yaml']}"
        )
    
    # Device 설정
    device_config = config['device']
    if device_config['type'] == 'cuda' and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['device_id']}")
    elif device_config['type'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Head Detection 모델 로드
    detection_model = HeadDetector(
        model_path=detection_model_path,
        conf_threshold=config['detection']['conf_threshold'],
        iou_threshold=config['detection']['iou_threshold'],
        device=str(device)
    )
    print(f"Loaded head detection model from {detection_model_path}")
    
    # Age & Gender 모델 로드
    age_gender_model = build_network(config)
    age_gender_model = age_gender_model.to(device)
    
    checkpoint_info = load_checkpoint(args.age_gender_weights, age_gender_model, device=str(device))
    print(f"Loaded age & gender model from {args.age_gender_weights}")
    print(f"  Epoch: {checkpoint_info['epoch']}")
    print(f"  Best Score: {checkpoint_info['best_score']:.4f}")
    
    # 입력 경로 확인
    if args.image:
        image_paths = [Path(args.image)]
    elif args.source:
        source_path = Path(args.source)
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
    else:
        raise ValueError("Either --image or --source must be provided")
    
    print(f"Found {len(image_paths)} image(s)")
    
    # 추론 수행
    results = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualize = not args.no_visualize
    save_images = args.save_images or config['inference'].get('save_images', False)
    
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        
        result = inference_with_detection(
            detection_model, age_gender_model, str(image_path),
            config, device, visualize
        )
        
        results.append(result)
        
        # 결과 출력 (나이와 성별을 동시에 출력)
        print(f"  Detected {result['num_detections']} head(s)")
        for det in result['detections']:
            print(f"    Head {det['head_index']}: "
                  f"Age {det['age']['predicted_age']}세 "
                  f"[Conf: {det['age']['confidence']:.3f}], "
                  f"Gender {det['gender']['label']} [Conf: {det['gender']['confidence']:.3f}]")
        
        # 시각화 이미지 저장
        if save_images and result['num_detections'] > 0:
            image = Image.open(image_path).convert('RGB')
            vis_image = visualize_result(image, result)
            vis_output_path = output_dir / f"{image_path.stem}_result.jpg"
            vis_image.save(vis_output_path)
            print(f"  Saved visualization to {vis_output_path}")
    
    # 결과 저장
    output_format = config['inference'].get('output_format', 'json')
    save_results(results, str(output_dir), output_format)
    
    # 요약 출력
    total_detections = sum(r['num_detections'] for r in results)
    print(f"\nInference completed!")
    print(f"  Total images: {len(results)}")
    print(f"  Total heads detected: {total_detections}")
    print(f"  Average heads per image: {total_detections / len(results):.2f}")


if __name__ == "__main__":
    main()

