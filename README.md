# Computer Vision Portfolio

실무 경험을 바탕으로 재구현한 Computer Vision 프로젝트 포트폴리오입니다.

## 📋 포트폴리오 개요

이 저장소는 실제 상용 프로젝트 경험을 바탕으로, 회사 기밀 정보를 제외하고 구조와 아이디어만 참고하여 완전히 새로 구현한 독립 실행 가능한 Computer Vision 프로젝트들을 포함합니다.

**중요**: 모든 프로젝트는 실제 상용 코드, 데이터, 모델 가중치를 포함하지 않으며, 구조와 설계 철학만 참고하여 재구현되었습니다.

## 🚀 프로젝트 목록

### 1. Phone Defect Detection System
**중고 스마트폰 자동 외관 검사 시스템**

- **경로**: `phone_detect/`
- **설명**: 중고 스마트폰의 외관 결함을 자동으로 검출하는 시스템
- **주요 기능**:
  - 휴대폰 영역 검출 (YOLO / Segmentation)
  - 결함 영역 Segmentation (Pixel-level)
  - 결함 등급 분류 (A ~ D 등급)
  - 가장 심한 결함 2개 선정
- **기술 스택**: PyTorch, Ultralytics YOLO, Segmentation Models
- **상세 문서**: [phone_detect/README.md](./phone_detect/README.md)

### 2. Age & Gender Estimation System
**나이/성별 추정 시스템**

- **경로**: `age_gender_estimation/`
- **설명**: 키오스크 환경에서 얼굴 이미지로부터 나이와 성별을 동시에 추정하는 시스템
- **주요 기능**:
  - Human Head Detection (YOLO)
  - Multi-Head Classification (Age + Gender)
  - ONNX Export 지원
  - Edge Inference 최적화
- **기술 스택**: PyTorch, Ultralytics YOLO, EfficientNet
- **상세 문서**: [age_gender_estimation/README.md](./age_gender_estimation/README.md)

## 🛠 기술 스택

### Deep Learning Frameworks
- **PyTorch**: 모델 학습 및 추론
- **Ultralytics YOLO**: Object Detection
- **Segmentation Models PyTorch**: Segmentation 모델

### Computer Vision Libraries
- **OpenCV**: 이미지 처리 및 전처리
- **PIL/Pillow**: 이미지 로딩 및 변환
- **Albumentations**: 데이터 증강 (선택적)

## 📁 프로젝트 구조

<details>
<summary>프로젝트 디렉터리 구조 보기/숨기기</summary>

```
portfolio/
│   
├── phone_detect/                      # 중고 스마트폰 결함 검출 시스템
│   ├── configs/                       # 설정 파일
│   │   ├── service_config.yaml         # 추론/서비스용 설정
│   │   ├── train_config.yaml           # 학습용 설정
│   │   └── README.md                   # 설정 파일 가이드
│   │
│   ├── data/                          # 데이터 디렉터리
│   │   ├── README.md
│   │   └── sample/
│   │
│   ├── docs/                          # 문서
│   │   ├── notes.md
│   │   └── pipeline_diagram.md
│   │
│   ├── inference/                     # 추론 파이프라인
│   │   ├── __init__.py
│   │   ├── display.py                 # 디스플레이 결함 검출
│   │   └── side.py                    # 측면 결함 검출
│   │
│   ├── models/                        # 모델 정의
│   │   ├── __init__.py
│   │   ├── phone_detector.py          # YOLO 기반 휴대폰 검출
│   │   ├── phone_segmenter.py         # Segmentation 기반 휴대폰 검출
│   │   ├── defect_segmenter.py        # 결함 Segmentation 모델
│   │   ├── detector.py
│   │   ├── loss.py
│   │   └── utils.py
│   │
│   ├── preprocess/                    # 전처리/후처리
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── defect_preprocess.py
│   │   ├── defect_postprocess.py
│   │   └── pipeline.py
│   │
│   ├── service/                       # 결함 검출 서비스
│   │   ├── __init__.py
│   │   ├── front_service.py           # 전면 결함 검출 서비스
│   │   ├── side_service.py            # 측면 결함 검출 서비스
│   │   └── README.md                  # 서비스 가이드
│   │
│   ├── utils/                         # 유틸리티
│   │   ├── __init__.py
│   │   └── defect_grade.py            # 결함 등급 결정
│   │
│   ├── weights/                       # 모델 가중치
│   ├── train_phone_detection.py       # 휴대폰 검출 학습
│   ├── train_defect_segmentation.py   # 결함 분할 학습
│   ├── inference.py                   # 추론 스크립트
│   ├── evaluation.py                  # 평가 스크립트
│   ├── README.md
│   └── requirements.txt
│
├── age_gender_estimation/             # 나이/성별 추정 시스템
│   ├── configs/
│   │   └── config.yaml                # 설정 파일
│   │
│   ├── data/                          # 데이터 디렉터리
│   │   ├── detection/                 # Detection 데이터셋
│   │   │   └── README.md
│   │   └── README.md
│   │
│   ├── detection/                     # Head Detection
│   │   ├── architecture/              # YOLO 아키텍처 설정
│   │   │   ├── yolo11n-p2.yaml
│   │   │   └── yolo11s-p2.yaml
│   │   ├── results/                   # 학습 결과
│   │   │   └── head_detection/
│   │   ├── splits/                    # 데이터 분할
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   └── val.txt
│   │   ├── __init__.py
│   │   ├── augment_dataset.py         # 데이터 증강
│   │   ├── config.py
│   │   ├── predict_detector.py        # Detection 추론
│   │   ├── train_detector.py          # YOLO 학습
│   │   └── val_detector.py            # Validation
│   │
│   ├── docs/                          # 문서
│   │   ├── optimization_notes.md
│   │   └── pipeline_diagram.md
│   │
│   ├── models/                        # 모델 정의
│   │   ├── __init__.py
│   │   ├── backbone.py                # EfficientNet Backbone
│   │   ├── age_head.py                # Age Classification Head
│   │   ├── gender_head.py             # Gender Classification Head
│   │   ├── network.py                 # Multi-Head Network
│   │   └── utils.py
│   │
│   ├── preprocess/                    # 전처리
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── transforms.py
│   │
│   ├── service/                       # 키오스크 서비스 모듈
│   │   ├── __init__.py
│   │   ├── kiosk_service.py           # 메인 서비스 클래스
│   │   ├── camera_handler.py          # 카메라 및 멀티스레드 촬영
│   │   ├── image_quality.py           # 이미지 품질 필터링
│   │   ├── database.py                # 데이터베이스 관리
│   │   └── README.md
│   │
│   ├── weights/                       # 모델 가중치
│   ├── train.py                       # Multi-task 학습
│   ├── inference.py                   # 단일 이미지 추론
│   ├── inference_with_detection.py    # 통합 추론
│   ├── evaluation.py                  # 평가 스크립트
│   ├── export_onnx.py                 # ONNX 변환
│   ├── README.md
│   └── requirements.txt
│
└── README.md                          # 메인 README
```

</details>

## 🚀 시작하기

### 사전 요구사항

- Python 3.10+
- PyTorch 1.12+
- CUDA (GPU 사용 시, 선택사항)

## 📊 주요 특징

### Phone Defect Detection
- ✅ 영역별 독립 파이프라인 (전면, 후면, 디스플레이, 측면)
- ✅ Pixel-level 결함 검출
- ✅ 결함 등급 시스템 (A ~ D)
- ✅ Threshold 기반 결함 판정
- ✅ Morphology 후처리

### Age & Gender Estimation
- ✅ Multi-Task Learning (Age + Gender 동시 학습)
- ✅ Human Head Detection 통합
- ✅ ONNX Export 지원
- ✅ Edge Inference 최적화

## 📝 프로젝트별 상세 문서

각 프로젝트의 상세한 설명, 사용법, 파이프라인 다이어그램은 각 프로젝트의 README.md를 참조하세요.

- [Phone Defect Detection README](./phone_detect/README.md)
- [Age & Gender Estimation README](./age_gender_estimation/README.md)

## 🔧 설정 파일

모든 프로젝트는 YAML 형식의 설정 파일을 사용합니다:

### Phone Defect Detection
- `phone_detect/configs/service_config.yaml`: 추론/서비스 실행용 설정
- `phone_detect/configs/train_config.yaml`: 모델 학습용 설정
- 자세한 내용: [phone_detect/configs/README.md](./phone_detect/configs/README.md)

### Age & Gender Estimation
- `age_gender_estimation/configs/config.yaml`: Age & Gender Estimation 설정

주요 설정 항목:
- 모델 구조 (backbone, head 등)
- 학습 파라미터 (batch_size, learning_rate, epochs)
- 데이터 경로
- 전처리/후처리 설정
- 하드웨어 설정

## 📈 평가 및 메트릭

### Phone Defect Detection
- IoU (Intersection over Union)
- Pixel Accuracy
- Grade Accuracy (등급별 정확도)
- Confusion Matrix

### Age & Gender Estimation
- Age Classification Accuracy
- Gender Classification Accuracy
- Mean Absolute Error (MAE) for Age
- Confusion Matrix

## 🎯 실무 경험 반영

이 포트폴리오는 다음 실무 경험을 바탕으로 재구현되었습니다:

1. **중고 스마트폰 자동 외관 검사 시스템**
   - 실제 상용 시스템 개발 및 배포 경험
   - 다양한 영역별 독립 파이프라인 설계
   - 결함 등급 시스템 및 후처리 로직 구현

2. **키오스크 환경 나이/성별 추정 시스템**
   - 실제 키오스크 환경 배포 경험
   - Multi-Task Learning 설계
   - Edge Inference 최적화

## ⚖️ Legal Notice

**중요**: 이 포트폴리오의 모든 프로젝트는 실제 상용 코드, 데이터, 모델 가중치를 포함하지 않습니다. 모든 코드, 모델 구조, 설정은 포트폴리오 목적으로 새로 작성되었으며, 실제 상용 시스템과의 유사성은 구조적 아이디어에 한정됩니다.

- 실제 상용 코드는 포함되지 않았습니다(일부 참조 형식)
- 실제 상용 데이터는 포함되지 않았습니다
- 실제 상용 모델 가중치는 포함되지 않았습니다
- 회사 기밀 정보는 포함되지 않았습니다

---

**Note**: 이 포트폴리오는 실제 상용 프로젝트 경험을 바탕으로 재구현되었으나, 실제 상용 코드/데이터/모델은 포함하지 않습니다.
