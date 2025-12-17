# Configuration Files

설정 파일은 용도에 따라 두 개로 분리되어 있습니다.

## Service Config (`service_config.yaml`)

추론 및 서비스 실행에 필요한 설정입니다.

**사용 대상:**
- `inference.py` - 추론 스크립트
- `evaluation.py` - 평가 스크립트
- `service/front_service.py` - Front 서비스
- `service/side_service.py` - Side 서비스

**주요 설정 항목:**
- `phone_detection`: Phone 영역 검출 모델 설정
- `defect_segmentation`: 결함 분할 모델 설정 및 전처리/후처리 파라미터
- `defect_grading`: 결함 등급 결정 설정
- `device`: 하드웨어 설정
- `inference`: 추론 설정

## Train Config (`train_config.yaml`)

모델 학습에 필요한 설정입니다.

**사용 대상:**
- `train_phone_detection.py` - Phone Detection 모델 학습
- `train_defect_segmentation.py` - Defect Segmentation 모델 학습

**주요 설정 항목:**
- `data`: 데이터 경로 (train, val, test)
- `training`: 학습 하이퍼파라미터 (batch_size, epochs, optimizer, scheduler, loss 등)
- `preprocessing`: 전처리 및 데이터 증강 설정
- `evaluation`: 평가 메트릭 설정
- `device`: 하드웨어 설정

## 사용법

### 추론/서비스 실행

```bash
# 기본값으로 service_config.yaml 사용
python inference.py --image path/to/image.jpg

# 또는 명시적으로 지정
python inference.py --config configs/service_config.yaml --image path/to/image.jpg
```

### 모델 학습

```bash
# 기본값으로 train_config.yaml 사용
python train_defect_segmentation.py --section display

# 또는 명시적으로 지정
python train_defect_segmentation.py --config configs/train_config.yaml --section display
```

## 설정 파일 구조

두 설정 파일은 공통된 구조를 공유합니다:
- `project`: 프로젝트 기본 정보
- `input`: 입력 이미지 크기
- `data`: 데이터 형식 및 클래스 정보
- `phone_detection`: Phone Detection 모델 설정
- `defect_segmentation`: Defect Segmentation 모델 설정
- `device`: 하드웨어 설정
- `paths`: 경로 설정
- `logging`: 로깅 설정

차이점:
- `service_config.yaml`: 추론에 필요한 전처리/후처리 파라미터, 결함 등급 결정 설정 포함
- `train_config.yaml`: 학습에 필요한 하이퍼파라미터, 데이터 증강, 평가 설정 포함
