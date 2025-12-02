# Optimization Notes

## 설계 의도

이 프로젝트는 실제 키오스크 환경에서의 Age & Gender Estimation 경험을 바탕으로, 포트폴리오 목적으로 재구성된 프로젝트입니다.

### 핵심 원칙

1. **Multi-Task Learning**: Age와 Gender를 동시에 학습하여 효율성과 성능 향상
2. **Shared Backbone**: 두 작업이 공통 feature를 공유하여 메모리와 계산 효율성 확보
3. **동시 추론**: 나이와 성별을 하나의 forward pass로 동시에 예측
4. **Edge Deployment Ready**: ONNX 변환을 통한 다양한 플랫폼 지원
5. **재현 가능성**: 설정 파일 기반으로 모든 하이퍼파라미터 관리

## 모델 구조 최적화

### Backbone 선택

- **EfficientNet**: 경량화와 성능의 균형, Mobile device 배포에 적합
- **ResNet**: 안정적인 성능, 널리 사용되는 구조
- ImageNet 사전 학습 가중치 활용으로 빠른 수렴

### Head 설계

- **Age Head**: 
  - 0~100세를 1세 단위로 분류 (총 101 classes)
  - Multi-class classification으로 정확한 나이 예측
  - 연속값 회귀 대신 분류로 접근하여 안정적인 학습

- **Gender Head**:
  - Binary classification (Male / Female)
  - 간단한 구조로 빠른 추론

### Loss Function

- **Age Loss**: CrossEntropy Loss (101 classes, 0~100세)
- **Gender Loss**: CrossEntropy Loss (Binary)
- **Weighted Combination**: 두 loss의 가중치를 조정하여 균형 유지
  - `Total Loss = α × Age Loss + β × Gender Loss`
  - 동시에 backward pass를 수행하여 두 작업을 함께 학습

### Multi-Task Learning

나이와 성별을 동시에 학습하는 방식:

1. **Forward Pass**: 
   - 하나의 이미지 입력으로 Backbone을 통과
   - 공유된 feature를 Age Head와 Gender Head에 동시에 전달
   - Age logits (101 classes)와 Gender logits (2 classes)를 동시에 출력

2. **Loss Calculation**:
   - Age loss와 Gender loss를 각각 계산
   - 가중합으로 결합하여 하나의 total loss 생성

3. **Backward Pass**:
   - Total loss에 대해 한 번의 backward pass 수행
   - Backbone과 두 Head가 동시에 업데이트됨

4. **장점**:
   - 두 작업이 공유 feature를 통해 서로 도움을 줌
   - 단일 forward pass로 두 예측을 동시에 수행하여 효율적
   - 메모리와 계산 비용 절감

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
- Dropout: Head 부분에 적용하여 일반화 성능 향상

## 평가 전략

### 나이 예측 정확도

나이 예측은 정확히 일치하는 것보다 허용 범위 내 정확도가 더 실용적입니다:

- **±1세 범위**: 예측값이 실제값의 ±1세 범위 내에 있는 비율
- **±3세 범위**: 예측값이 실제값의 ±3세 범위 내에 있는 비율
- **±5세 범위**: 예측값이 실제값의 ±5세 범위 내에 있는 비율 (채택)
- **±10세 범위**: 예측값이 실제값의 ±10세 범위 내에 있는 비율
- **±15세 범위**: 예측값이 실제값의 ±15세 범위 내에 있는 비율

실험 결과, **±5세 범위**에서 가장 높은 정확도를 보여 이를 최종 평가 기준으로 채택했습니다.

### 성별 예측 정확도

- Binary classification accuracy
- Confusion matrix를 통한 Male/Female별 성능 분석

## 추론 최적화

### Batch Processing

- 배치 단위로 처리하여 GPU 활용률 향상
- 단일 이미지 추론도 배치 크기 1로 처리
- 나이와 성별을 동시에 예측하여 추론 시간 단축

### ONNX 변환

- PyTorch 모델을 ONNX로 변환
- ONNX Runtime을 통한 최적화된 추론
- 다양한 플랫폼 지원 (CPU, GPU, Edge devices)
- 단일 forward pass로 두 출력을 동시에 얻을 수 있음

## Edge Deployment 고려사항

### 모델 크기

- EfficientNet-B0/B1: 모바일 디바이스에 적합
- Multi-task learning으로 두 모델을 하나로 통합하여 메모리 효율성 향상

### 추론 속도

- ONNX Runtime 최적화
- Operator fusion을 통한 연산 효율화
- Batch inference로 처리량 향상
- 나이와 성별을 동시에 예측하여 추론 횟수 절반으로 감소

### 메모리 사용량

- 모델 가중치 크기 최소화
- Feature map 크기 고려
- Shared backbone으로 메모리 효율성 향상

## 실험 결과

### 나이 예측 성능

- **정확도 (Exact)**: 정확히 일치하는 비율
- **±5세 범위 정확도**: 가장 높은 정확도를 보임 (채택)
- **MAE (Mean Absolute Error)**: 평균 절대 오차

### 성별 예측 성능

- **정확도**: Binary classification accuracy
- 높은 정확도 달성

### Multi-Task Learning 효과

- 두 작업을 동시에 학습함으로써 공유 feature가 더 일반화됨
- 단일 모델로 두 예측을 수행하여 효율성 향상
- 추론 시간과 메모리 사용량 감소
