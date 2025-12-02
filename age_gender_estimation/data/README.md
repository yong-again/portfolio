# Data Directory

## 개요

이 디렉토리는 학습 및 추론에 사용되는 데이터를 저장하는 공간입니다.

**중요**: 실제 학습 데이터는 포함되어 있지 않습니다. 사용자는 공개 데이터셋을 다운로드하여 사용해야 합니다.

## 디렉토리 구조

```
data/
├── README.md          # 이 파일
└── sample/           # 예시용 샘플 이미지 (선택사항)
    └── images/       # 얼굴 이미지 파일
```

## 데이터 형식

### 이미지

- 지원 형식: `.jpg`, `.jpeg`, `.png`
- 권장 해상도: 224x224 이상 (모델 입력 크기에 맞게 조정됨)
- 얼굴이 중심에 위치하고 정렬된 이미지 권장

### Annotation 형식

#### CSV Format (권장)

각 행은 하나의 이미지와 라벨을 나타냅니다.

파일명: `annotations.csv`

형식:
```csv
image_path,age,gender
data/train/image_001.jpg,25,1
data/train/image_002.jpg,35,0
data/train/image_003.jpg,18,1
```

- `image_path`: 이미지 파일 경로 (상대 경로 또는 절대 경로)
- `age`: 나이 (정수, 0-100) 또는 연령대 인덱스
- `gender`: 성별 (0 = Male, 1 = Female)

#### JSON Format (선택사항)

```json
{
  "annotations": [
    {
      "image_path": "data/train/image_001.jpg",
      "age": 25,
      "gender": 1
    },
    {
      "image_path": "data/train/image_002.jpg",
      "age": 35,
      "gender": 0
    }
  ]
}
```

## 데이터 준비 방법

### 1. 공개 데이터셋 다운로드

#### UTKFace

```bash
# UTKFace 데이터셋 다운로드 (예시)
# 실제 다운로드 방법은 데이터셋 제공자 사이트 참조
```

UTKFace 형식:
- 파일명: `[age]_[gender]_[race]_[date&time].jpg`
- 예시: `25_0_0_20170117150515125.jpg` (25세, 남성, 백인)

#### FairFace

```bash
# FairFace 데이터셋 다운로드
# https://github.com/dchen236/FairFace 참조
```

### 2. 데이터 전처리

1. 얼굴 검출 및 정렬 (선택사항)
   - MTCNN, RetinaFace 등을 사용하여 얼굴 영역 추출
   - 얼굴 정렬 (landmark 기반)

2. 데이터 분할
   - Train / Val / Test 세트로 분할
   - 권장 비율: 80% / 10% / 10%

3. Annotation 파일 생성
   - CSV 또는 JSON 형식으로 라벨 파일 생성

### 3. 디렉토리 구조 설정

```
data/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── annotations.csv
├── val/
│   ├── images/
│   └── annotations.csv
└── test/
    ├── images/
    └── annotations.csv
```

### 4. 설정 파일 업데이트

`configs/config.yaml`에서 데이터 경로를 설정:

```yaml
data:
  train_path: data/train
  val_path: data/val
  test_path: data/test
  annotation_format: csv  # csv 또는 json
```

## Age Bins 설정

나이를 연령대(age bins)로 분류할 경우, `configs/config.yaml`에서 설정:

```yaml
model:
  age:
    bins: [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # 또는
    num_bins: 10
    min_age: 0
    max_age: 100
```

## 데이터 증강 (Augmentation)

학습 시 자동으로 적용되는 증강 기법:

- Random horizontal flip
- Random rotation
- Color jitter
- Random crop/resize

설정은 `configs/config.yaml`의 `preprocessing.augmentation` 섹션에서 조정할 수 있습니다.

## 샘플 데이터

`sample/` 디렉토리에는 예시용 샘플 이미지가 포함되어 있습니다. 이는 프로젝트 구조 이해를 위한 것이며, 실제 학습에는 사용하지 않습니다.

