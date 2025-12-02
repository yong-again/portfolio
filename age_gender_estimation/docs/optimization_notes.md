**# Optimization Notes

## 설계 의도

이 프로젝트는 실제 키오스크 환경에서의 Age & Gender Estimation 경험을 바탕으로, 포트폴리오 목적으로 재구성된 프로젝트입니다.

### 핵심 원칙

1. **Multi-Task Learning**: Age와 Gender를 동시에 학습하여 효율성과 성능 향상
2. **Shared Backbone**: 두 작업이 공통 feature를 공유하여 메모리와 계산 효율성 확보
3. **Edge Deployment Ready**: ONNX 변환을 통한 다양한 플랫폼 지원
4. **재현 가능성**: 설정 파일 기반으로 모든 하이퍼파라미터 관리

## 모델 구조 최적화

### Backbone 선택

- **EfficientNet**: 경량화와 성능의 균형, Mobile device 배포에 적합
- **ResNet**: 안정적인 성능, 널리 사용되는 구조
- ImageNet 사전 학습 가중치 활용으로 빠른 수렴

### Head 설계

- **Age Head**: 
  - 연령대를 bin으로 분류 (Multi-class classification)
  - 연속값 회귀 대신 분류로 접근하여 안정적인 학습
  - Bin 개수는 설정 가능

- **Gender Head**:
  - Binary classification (Male / Female)
  - 간단한 구조로 빠른 추론

### Loss Function

- **Age Loss**: CrossEntropy Loss (Multi-class)
- **Gender Loss**: CrossEntropy Loss (Binary)
- **Weighted Combination**: 두 loss의 가중치를 조정하여 균형 유지

## 학습 전략

### Data Augmentation

- Horizontal flip: 얼굴 대칭성 활용
- Color jitter: 조명 변화에 강건성
- Random rotation: 얼굴 각도 변화 대응

### Learning Rate Schedule

- CosineAnnealing: 부드러운 학습률 감소
- Warm-up: 초기 학습 안정화

### Regularization

- Weight decay: 과적합 방지
- Dropout (선택사항): Head 부분에 적용 가능

## 추론 최적화

### Batch Processing

- 배치 단위로 처리하여 GPU 활용률 향상
- 단일 이미지 추론도 배치 크기 1로 처리

### ONNX 변환

- PyTorch 모델을 ONNX로 변환
- ONNX Runtime을 통한 최적화된 추론
- 다양한 플랫폼 지원 (CPU, GPU, Edge devices)

## Edge Deployment 고려사항

### 모델 크기

- EfficientNet-B0/B1: 모바일 디바이스에 적합
- ResNet-18: 경량화된 버전 사용 가능

### 추론 속도

- ONNX Runtime 최적화
- Operator fusion을 통한 연산 효율화
- Batch inference로 처리량 향상

### 메모리 사용량

- 모델 가중치 크기 최소화[
- Feature map 크기 고려
- Quantization으로 메모리 사용량 감소