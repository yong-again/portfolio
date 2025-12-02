"""
ONNX Export Script

PyTorch 모델을 ONNX 형식으로 변환합니다.
Edge device 배포를 위한 최적화를 지원합니다.
"""

import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import sys
import onnx
import onnxruntime as ort

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from models.network import build_network
from models.utils import load_checkpoint


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: tuple = (1, 3, 224, 224),
    opset_version: int = 11,
    dynamic_axes: bool = False,
    input_names: list = None,
    output_names: list = None,
    optimize: bool = True,
    simplify: bool = True
):
    """
    PyTorch 모델을 ONNX 형식으로 변환합니다.
    
    Args:
        model: PyTorch 모델
        output_path: ONNX 모델 저장 경로
        input_size: 입력 크기 (batch, channels, height, width)
        opset_version: ONNX opset 버전
        dynamic_axes: 동적 배치 크기 지원 여부
        input_names: 입력 이름 리스트
        output_names: 출력 이름 리스트
        optimize: 최적화 여부
        simplify: 모델 단순화 여부
    """
    model.eval()
    
    # 기본값 설정
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['age_output', 'gender_output']
    
    # 더미 입력 생성
    dummy_input = torch.randn(*input_size)
    
    # Dynamic axes 설정
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'},
            output_names[1]: {0: 'batch_size'}
        }
    
    print(f"Exporting model to ONNX...")
    print(f"  Input size: {input_size}")
    print(f"  Opset version: {opset_version}")
    print(f"  Dynamic axes: {dynamic_axes}")
    
    # ONNX 변환
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_dict,
            opset_version=opset_version,
            do_constant_folding=True,
            verbose=False
        )
    
    print(f"Model exported to {output_path}")
    
    # ONNX 모델 검증
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation: PASSED")
    except Exception as e:
        print(f"ONNX model validation: FAILED - {e}")
        return
    
    # 모델 단순화
    if simplify:
        try:
            from onnxsim import simplify
            simplified_model, check = simplify(onnx_model)
            if check:
                onnx.save(simplified_model, output_path)
                print("Model simplified successfully")
        except ImportError:
            print("onnxsim not installed, skipping simplification")
        except Exception as e:
            print(f"Simplification failed: {e}")
    
    # 모델 정보 출력
    print(f"\nModel Information:")
    print(f"  Input: {input_names[0]} {input_size}")
    print(f"  Outputs: {output_names}")
    
    # ONNX Runtime으로 테스트
    try:
        session = ort.InferenceSession(output_path)
        input_name = session.get_inputs()[0].name
        output_names_ort = [output.name for output in session.get_outputs()]
        
        print(f"\nONNX Runtime Test:")
        print(f"  Input name: {input_name}")
        print(f"  Output names: {output_names_ort}")
        
        # 테스트 추론
        test_input = dummy_input.numpy()
        outputs = session.run(output_names_ort, {input_name: test_input})
        
        print(f"  Age output shape: {outputs[0].shape}")
        print(f"  Gender output shape: {outputs[1].shape}")
        print("  ONNX Runtime inference: SUCCESS")
    except Exception as e:
        print(f"  ONNX Runtime test failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch Model to ONNX')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to PyTorch model weights')
    parser.add_argument('--output', type=str, default='model.onnx',
                       help='Output ONNX model path')
    parser.add_argument('--opset-version', type=int, default=None,
                       help='ONNX opset version (overrides config)')
    parser.add_argument('--dynamic-axes', action='store_true',
                       help='Enable dynamic batch size')
    parser.add_argument('--input-size', type=int, nargs=2, default=None,
                       help='Input image size (width height)')
    
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
    
    # 입력 크기 설정
    if args.input_size:
        width, height = args.input_size
    else:
        input_size_config = config['preprocessing']['input_size']
        width = input_size_config['width']
        height = input_size_config['height']
    
    input_size = (1, 3, height, width)
    
    # ONNX Export 설정
    onnx_config = config.get('onnx_export', {})
    opset_version = args.opset_version or onnx_config.get('opset_version', 11)
    dynamic_axes = args.dynamic_axes or onnx_config.get('dynamic_axes', False)
    input_names = onnx_config.get('input_names', ['input'])
    output_names = onnx_config.get('output_names', ['age_output', 'gender_output'])
    optimize = onnx_config.get('optimize', True)
    simplify = onnx_config.get('simplify', True)
    
    # ONNX 변환
    export_to_onnx(
        model=model,
        output_path=args.output,
        input_size=input_size,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
        output_names=output_names,
        optimize=optimize,
        simplify=simplify
    )
    
    print(f"\nExport completed! ONNX model saved to {args.output}")


if __name__ == "__main__":
    main()

