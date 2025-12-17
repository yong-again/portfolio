# Data Directory

## 개요

이 디렉토리는 학습 및 추론에 사용되는 데이터를 저장하는 공간입니다.

**중요**: 실제 학습 데이터는 포함되어 있지 않습니다. 사용자는 자신의 데이터셋을 준비해야 합니다.

## 디렉토리 구조

```
data/
├── README.md          # 이 파일
└── sample/           # 예시용 샘플 이미지 (선택사항)
    ├── images/       # 원본 이미지 파일
    └── annotations/  # 라벨 파일 (YOLO 또는 COCO 형식)
```

## 데이터 형식

### 이미지

- 지원 형식: `.jpg`, `.jpeg`, `.png`
- 권장 해상도: 1920x1080 이상 (실제 사용 시 모델 입력 크기에 맞게 조정)

### Annotation 형식

#### YOLO Format (권장)

각 이미지에 대응하는 `.txt` 파일이 필요합니다.

파일명: `image_name.txt` (예: `phone_001.jpg` → `phone_001.txt`)

형식:
```
class_id x_center y_center width height
```

- 모든 좌표는 이미지 크기 대비 정규화된 값 (0.0 ~ 1.0)
- `class_id`: 클래스 ID (0 = phone)
- `x_center, y_center`: 바운딩 박스 중심 좌표
- `width, height`: 바운딩 박스 너비와 높이

예시:
```
0 0.5 0.5 0.8 0.9
```

#### COCO Format (선택사항)

단일 JSON 파일로 모든 annotation을 관리합니다.

```json
{
  "images": [...],
  "annotations": [...],
  "categories": [
    {"id": 0, "name": "phone"}
  ]
}
```

## 데이터 준비 방법

### 1. 데이터 수집

- 스마트폰 이미지 수집 (다양한 각도, 조명 조건 권장)
- 각 이미지에 대해 스마트폰 영역을 라벨링

### 2. 데이터 분할

학습/검증/테스트 세트로 분할:

```
data/
├── train/
│   ├── images/
│   └── annotations/
├── val/
│   ├── images/
│   └── annotations/
└── test/
    ├── images/
    └── annotations/
```

### 3. 설정 파일 업데이트

`configs/train_config.yaml`에서 데이터 경로를 설정:

```yaml
data:
  train_path: data/train
  val_path: data/val
  test_path: data/test
```

## 데이터 증강 (Augmentation)

데이터 증강은 학습 시 자동으로 적용됩니다. 설정은 `configs/train_config.yaml`의 `preprocessing.augmentation` 섹션에서 조정할 수 있습니다.

주요 증강 기법:
- Random flip (horizontal, vertical)
- Random rotation
- Color jitter
- Random crop/resize

## 주의사항

1. **데이터 보안**: 실제 상용 데이터는 이 저장소에 포함하지 마세요.
2. **데이터 균형**: 클래스 불균형이 있을 경우 적절한 샘플링 전략을 고려하세요.
3. **데이터 품질**: 라벨링 오류를 최소화하기 위해 검수 과정을 거치세요.

## 샘플 데이터

`sample/` 디렉토리에는 예시용 샘플 이미지가 포함되어 있습니다. 이는 프로젝트 구조 이해를 위한 것이며, 실제 학습에는 사용하지 않습니다.

