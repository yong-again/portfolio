# Head Detection Dataset

## 개요

YOLO format으로 human head detection을 위한 데이터셋을 준비합니다.

## 데이터셋 구조

YOLO format 데이터셋은 다음과 같은 구조를 가집니다:

```
data/detection/
├── dataset.yaml          # 데이터셋 설정 파일
├── train/
│   ├── images/          # 학습 이미지
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/          # 라벨 파일 (YOLO format)
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## YOLO Format Annotation

각 이미지에 대응하는 `.txt` 파일이 필요합니다.

파일명: `image_name.txt` (예: `image1.jpg` → `image1.txt`)

형식:
```
class_id x_center y_center width height
```

- 모든 좌표는 이미지 크기 대비 정규화된 값 (0.0 ~ 1.0)
- `class_id`: 클래스 ID (0 = head)
- `x_center, y_center`: 바운딩 박스 중심 좌표
- `width, height`: 바운딩 박스 너비와 높이

예시:
```
0 0.5 0.5 0.3 0.4
```

## dataset.yaml 파일

`data/detection/dataset.yaml` 파일 예시:

```yaml
# Dataset configuration for YOLO head detection

# 경로 (프로젝트 루트 기준)
path: data/detection
train: train/images
val: val/images
test: test/images  # 선택사항

# 클래스
names:
  0: head
nc: 1
```

## 데이터 준비 방법

### 1. 데이터 수집

- 얼굴/머리가 포함된 이미지 수집
- 각 이미지에 대해 머리 영역을 라벨링

### 2. 라벨링 도구

다음과 같은 도구를 사용하여 라벨링할 수 있습니다:

- **LabelImg**: https://github.com/tzutalin/labelImg
- **CVAT**: https://cvat.org/
- **Roboflow**: https://roboflow.com/

### 3. 데이터 분할

학습/검증/테스트 세트로 분할:

- Train: 80%
- Val: 10%
- Test: 10% (선택사항)

### 4. dataset.yaml 생성

위 예시를 참고하여 `data/detection/dataset.yaml` 파일을 생성합니다.

## 학습 실행

데이터셋 준비가 완료되면 다음 명령으로 학습을 시작할 수 있습니다:

```bash
python detection/train_detector.py \
    --data data/detection/dataset.yaml \
    --model yolo11n.pt \
    --epochs 100 \
    --imgsz 640 \
    --batch 16
```

