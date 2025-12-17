"""
Inference Script for Phone Defect Detection

디스플레이 및 측면 결함 검출을 수행합니다.
"""

import torch
import yaml
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image
import sys
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from inference.display import DisplayPipeline
from inference.side import SidePipeline


def load_image(image_path: str) -> np.ndarray:
    """
    이미지를 로드합니다.
    
    Args:
        image_path: 이미지 파일 경로
    
    Returns:
        이미지 배열 [H, W, 3] (RGB)
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_results(
    results: dict,
    output_dir: str,
    image_path: str,
    section: str
):
    """
    추론 결과를 저장합니다.
    
    Args:
        results: 검출 결과 딕셔너리
        output_dir: 출력 디렉토리
        image_path: 원본 이미지 경로
        section: 섹션 이름
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 저장
    image_name = Path(image_path).stem
    output_file = output_dir / f"{image_name}_{section}_result.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Phone Defect Detection Inference')
    parser.add_argument('--config', type=str, default='configs/service_config.yaml',
                       help='Path to config file (default: service_config.yaml)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--section', type=str, default='display',
                       choices=['display', 'side', 'all'],
                       help='Section to process')
    parser.add_argument('--output', type=str, default='results/inference',
                       help='Output directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda, cpu, mps)')
    
    args = parser.parse_args()
    
    # Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device 설정
    if args.device:
        device = args.device
    else:
        device_config = config['device']
        if device_config['type'] == 'cuda' and torch.cuda.is_available():
            device = f"cuda:{device_config['device_id']}"
        elif device_config['type'] == 'mps' and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Device: {device}")
    
    # 이미지 로드
    image = load_image(args.image)
    print(f"Loaded image: {args.image}, Shape: {image.shape}")
    
    # 추론 수행
    if args.section == 'display' or args.section == 'all':
        print("\n" + "=" * 50)
        print("Display Defect Detection")
        print("=" * 50)
        
        display_pipeline = DisplayPipeline(config, device)
        display_result = display_pipeline.infer(image)
        
        print(f"Final Grade: {display_result['grade']['final_grade']}")
        print(f"Top Defects: {len(display_result['top_defects'])}")
        for idx, defect in enumerate(display_result['top_defects']):
            print(f"  Defect {idx+1}: Grade {defect['grade']}, Pixels: {defect['pixel_count']}")
        
        save_results(display_result, args.output, args.image, 'display')
    
    if args.section == 'side' or args.section == 'all':
        print("\n" + "=" * 50)
        print("Side Defect Detection")
        print("=" * 50)
        
        side_pipeline = SidePipeline(config, device)
        side_result = side_pipeline.infer(image)
        
        print(f"Final Grade: {side_result['grade']['final_grade']}")
        print(f"Top Defects: {len(side_result['top_defects'])}")
        for idx, defect in enumerate(side_result['top_defects']):
            print(f"  Defect {idx+1}: Grade {defect['grade']}, Pixels: {defect['pixel_count']}")
        
        save_results(side_result, args.output, args.image, 'side')
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
