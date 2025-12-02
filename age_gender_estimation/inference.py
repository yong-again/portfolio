"""
Inference Script for Age & Gender Estimation

단일 이미지 또는 폴더 내 모든 이미지에 대해 나이와 성별을 추론합니다.
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
from models.age_head import bin_to_age_range, create_age_bins
from models.gender_head import class_to_gender
from preprocess.transforms import get_val_transforms


def inference_single_image(
    model: nn.Module,
    image_path: str,
    config: Dict[str, Any],
    device: torch.device,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    단일 이미지에 대해 추론을 수행합니다.
    
    Args:
        model: 학습된 모델
        image_path: 이미지 파일 경로
        config: 설정 딕셔너리
        device: 디바이스
        visualize: 시각화 여부
    
    Returns:
        검출 결과 딕셔너리
    """
    # 이미지 로드
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        raise ValueError(f"Failed to load image: {image_path}, Error: {e}")
    
    # 전처리
    transform = get_val_transforms(config)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
    
    # 추론
    model.eval()
    with torch.no_grad():
        predictions = model.predict(image_tensor)
    
    # Age bins 생성
    age_config = config['model']['age']
    if 'bins' in age_config:
        age_bins = age_config['bins']
    else:
        age_bins = create_age_bins(
            num_bins=age_config['num_bins'],
            min_age=age_config['min_age'],
            max_age=age_config['max_age']
        )
    
    # 결과 파싱
    age_bin_idx = predictions['age']['predicted_bin'][0].item()
    age_range = bin_to_age_range(age_bin_idx, age_bins)
    age_prob = predictions['age']['probs'][0][age_bin_idx].item()
    
    gender_class = predictions['gender']['predicted_class'][0].item()
    gender_str = class_to_gender(gender_class)
    gender_prob = predictions['gender']['confidence'][0].item()
    
    result = {
        'image_path': image_path,
        'age': {
            'bin': age_bin_idx,
            'range': age_range,
            'estimated_age': (age_range[0] + age_range[1]) // 2,
            'confidence': age_prob
        },
        'gender': {
            'class': gender_class,
            'label': gender_str,
            'confidence': gender_prob
        }
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
        results: 검출 결과 리스트
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
            rows.append({
                'image_path': result['image_path'],
                'age_bin': result['age']['bin'],
                'age_min': result['age']['range'][0],
                'age_max': result['age']['range'][1],
                'estimated_age': result['age']['estimated_age'],
                'age_confidence': result['age']['confidence'],
                'gender': result['gender']['label'],
                'gender_confidence': result['gender']['confidence']
            })
        
        df = pd.DataFrame(rows)
        output_file = output_dir / 'results.csv'
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


def visualize_result(image: Image.Image, result: Dict[str, Any]) -> Image.Image:
    """
    이미지에 추론 결과를 시각화합니다.
    
    Args:
        image: 원본 이미지
        result: 추론 결과
    
    Returns:
        시각화된 이미지
    """
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(image)
    
    # 텍스트 정보
    age_info = f"Age: {result['age']['estimated_age']} ({result['age']['range'][0]}-{result['age']['range'][1]})"
    gender_info = f"Gender: {result['gender']['label']}"
    age_conf = f"Age Conf: {result['age']['confidence']:.2f}"
    gen_conf = f"Gender Conf: {result['gender']['confidence']:.2f}"
    
    # 텍스트 그리기
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    y_offset = 10
    draw.text((10, y_offset), age_info, fill='red', font=font)
    y_offset += 30
    draw.text((10, y_offset), gender_info, fill='blue', font=font)
    y_offset += 30
    draw.text((10, y_offset), age_conf, fill='green', font=font)
    y_offset += 30
    draw.text((10, y_offset), gen_conf, fill='green', font=font)
    
    return image


def main():
    parser = argparse.ArgumentParser(description='Age & Gender Estimation Inference')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--image', type=str, default=None,
                       help='Path to single image')
    parser.add_argument('--source', type=str, default=None,
                       help='Path to image directory')
    parser.add_argument('--output', type=str, default='results/inference',
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
    
    # Device 설정
    device_config = config['device']
    if device_config['type'] == 'cuda' and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['device_id']}")
    elif device_config['type'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # 모델 로드
    model = build_network(config)
    model = model.to(device)
    
    checkpoint_info = load_checkpoint(args.weights, model, device=str(device))
    print(f"Loaded model from {args.weights}")
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
        
        result = inference_single_image(
            model, str(image_path), config, device, visualize
        )
        
        results.append(result)
        
        # 결과 출력
        print(f"  Age: {result['age']['estimated_age']} ({result['age']['range'][0]}-{result['age']['range'][1]}) "
              f"[Conf: {result['age']['confidence']:.3f}]")
        print(f"  Gender: {result['gender']['label']} [Conf: {result['gender']['confidence']:.3f}]")
        
        # 시각화 이미지 저장
        if save_images:
            image = Image.open(image_path).convert('RGB')
            vis_image = visualize_result(image, result)
            vis_output_path = output_dir / f"{image_path.stem}_result.jpg"
            vis_image.save(vis_output_path)
            print(f"  Saved visualization to {vis_output_path}")
    
    # 결과 저장
    output_format = config['inference'].get('output_format', 'json')
    save_results(results, str(output_dir), output_format)
    
    # 요약 출력
    print(f"\nInference completed!")
    print(f"  Total images: {len(results)}")
    avg_age_conf = np.mean([r['age']['confidence'] for r in results])
    avg_gen_conf = np.mean([r['gender']['confidence'] for r in results])
    print(f"  Average age confidence: {avg_age_conf:.3f}")
    print(f"  Average gender confidence: {avg_gen_conf:.3f}")


if __name__ == "__main__":
    main()

