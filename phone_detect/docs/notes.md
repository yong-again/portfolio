# Development Notes

## 설계 의도

이 프로젝트는 실제 Mintit(주)에서 진행했던 휴대폰 결함 탐지 모델 개발의 구조와 아이디어를 참고하여, 포트폴리오 목적으로 완전히 새로 구현된 프로젝트입니다.


## 아키텍처 설계

### 모델 구조 선택

#### Phone Area Detection/Segmentation

- **전면/후면 검출 (PhoneDetector)**:
  - **모델**: YOLO (Ultralytics)
  - **이유**: 빠른 추론 속도, 높은 정확도의 객체 검출
  - **출력**: Bounding box [x1, y1, x2, y2]

- **측면/디스플레이 영역 검출 (PhoneSegmenter)**:
  - **Encoder**: EfficientNet-B3
  - **Decoder**: U-Net
  - **사전 학습 가중치**: ImageNet
  - **이유**: Pixel-level 정확도가 필요한 영역 검출, 경량화와 성능의 균형
  - **출력**: Binary segmentation mask

#### Defect Segmentation

- **Encoder**: EfficientNet-B4
  - 이유: 경량화와 성능의 균형, ImageNet 사전 학습 가중치 활용 가능
- **Decoder**: U-Net / FPN / DeepLabV3+
  - U-Net: 기본 선택, 다양한 크기의 결함 검출에 효과적
  - FPN: Multi-scale feature fusion
  - DeepLabV3+: Atrous convolution을 통한 정밀한 경계 검출
- **출력**: Multi-class segmentation mask (A, B, C, D 등급)

### 전처리 파이프라인

- 표준화된 전처리 파이프라인으로 일관성 유지
- 학습/추론 시 다른 전처리 적용 (augmentation은 학습 시만)
- 설정 파일로 전처리 옵션 제어

### 후처리 로직

- NMS (Non-Maximum Suppression) 필수
- Confidence threshold 조정 가능
- 좌표 변환 로직 명확히 분리

## 데이터 관리 전략

### 형식 선택

- **YOLO format 우선**: 간단하고 직관적
- **COCO format 지원 가능**: 확장성 고려

### 데이터 증강

- 학습 시에만 적용
- 다양한 조명, 각도, 스케일 변화에 강건한 모델 학습
- 설정 파일로 증강 강도 조절

## 학습 전략

### Loss Function

- Classification Loss: Focal Loss (클래스 불균형 대응)
- Localization Loss: IoU Loss 또는 Smooth L1 Loss
- 두 Loss의 가중치 조정 가능

### Optimizer

- Adam 또는 AdamW
- Learning rate scheduler: CosineAnnealing 또는 StepLR

### Validation

- 정기적인 검증으로 overfitting 모니터링
- Best model 자동 저장

## 추론 최적화

### 성능 고려사항

- Batch inference 지원
- GPU 메모리 효율적 사용
- 추론 시간 측정 기능

### 출력 형식

- JSON 형식으로 결과 저장 (다른 시스템과 연동 용이)
- 시각화 옵션 제공 (바운딩 박스 그리기)
