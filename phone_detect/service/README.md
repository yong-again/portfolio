# Phone Defect Detection Service

전면(Front), 후면(Back), 측면(Side), 디스플레이(Display) 영역의 결함 검출 서비스를 제공하는 모듈입니다.

## 개요

이 서비스 모듈은 실제 상용 프로젝트 경험을 바탕으로 재구현되었습니다. 회사 코드의 FrontPipeline, BackPipeline, SidePipeline, DisplayPipeline을 참고하여 완전히 새로 작성되었으며, 원본 코드와의 유사성은 구조적 아이디어에 한정됩니다.

**중요**: 이 코드는 실제 상용 코드를 포함하지 않으며, 포트폴리오 목적으로 새로 작성되었습니다.

## 주요 기능

### FrontService (전면 결함 검출)

전면(Front/Top) 영역의 결함을 검출하는 서비스입니다.

**주요 파이프라인:**
1. Phone Segmentation: 전면 영역 검출
2. Defect Segmentation: 결함 영역 pixel-level 분류
3. Post-processing:
   - Temperature scaling
   - Threshold 적용
   - Morphology 연산
   - 작은 결함 제거
   - B 등급 픽셀 수 검사
   - B 등급 길이 검사 (임계값 초과 시 C 등급으로 변경)

**특화 후처리:**
- B 등급 픽셀 수가 임계값을 초과하면 C 등급(23)으로 변경
- B 등급 픽셀 수가 임계값 이하인 경우, 결함 길이를 검사하여 임계값 초과 시 C 등급으로 변경

### SideService (측면 결함 검출)

측면(Side/Corner) 영역의 결함을 검출하는 서비스입니다.

**주요 파이프라인:**
1. Phone Segmentation: 측면 영역 검출
2. Defect Segmentation: 결함 영역 pixel-level 분류
3. Post-processing:
   - Temperature scaling
   - Threshold 적용
   - Morphology 연산
   - B 등급 픽셀 수 검사 (임계값 초과 시 C 등급으로 변경)

**특화 후처리:**
- B 등급 픽셀 수가 임계값 이상이면 C 등급(3)으로 변경

### DisplayService (디스플레이 결함 검출)

디스플레이(Display) 영역의 결함을 검출하는 서비스입니다.

**주요 파이프라인:**
1. Phone Segmentation: 디스플레이 영역 검출 및 크롭
2. Defect Segmentation: 결함 영역 pixel-level 분류
3. Post-processing:
   - Temperature scaling
   - Threshold 적용
   - Morphology 연산
   - 작은 결함 제거

**특화 전처리:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization) 지원
- 디스플레이 영역 자동 크롭

### BackService (후면 결함 검출)

후면(Back) 영역의 결함을 검출하는 서비스입니다.

**주요 파이프라인:**
1. Phone Detection: YOLO를 사용한 후면 영역 검출
2. Defect Segmentation: 결함 영역 pixel-level 분류
3. Post-processing:
   - Temperature scaling
   - Threshold 적용
   - Morphology 연산
   - 작은 결함 제거
   - 엣지 결함 제거

**특화 후처리:**
- 작은 결함 제거
- 후면 좌측 꼬리모양 잔상 파손 삭제 (엣지 결함 제거)

### ServicePreprocessor (서비스 전처리)

서비스에서 사용하는 전처리 기능을 제공합니다.

**주요 기능:**
- 이미지 크롭 및 리사이즈
- Scaler 적용 (normalize, histogram normalization 등)
- 섹션별 전처리 파이프라인
- 텐서 변환

**사용 예시:**
```python
from service.preprocess import ServicePreprocessor

preprocessor = ServicePreprocessor(config)
tensor = preprocessor.preprocess_image(
    resized_image=image,
    bbox_coords=[(y1, y2, x1, x2)],
    input_size={'width': 480, 'height': 256},
    scalers=['imagenet_standardize'],
    section_code='front'
)
```

### ServicePostprocessor (서비스 후처리)

서비스에서 사용하는 후처리 기능을 제공합니다.

**주요 기능:**
- Temperature scaling
- Threshold 적용
- Morphology 연산
- 섹션별 특화 후처리:
  - Front/Top: 작은 결함 제거
  - Back: 작은 결함 제거, 엣지 결함 제거
  - Display/LCD: AIP 블릿 제거, 작은 블릿 제거
  - Side/Corner: 기본 후처리

**사용 예시:**
```python
from service.postprocess import ServicePostprocessor

postprocessor = ServicePostprocessor(config)
processed_mask = postprocessor.post_process(
    images=defect_logits,
    config=config,
    pp_key='base',
    section='front'
)
```

## 사용법

### FrontService 사용 예시

```python
import yaml
import numpy as np
from service.front_service import FrontService

# 설정 파일 로드
with open('configs/service_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 서비스 초기화
front_service = FrontService(config, device='cuda')

# 입력 이미지 및 bbox 좌표
input_image = np.array(...)  # [H, W, 3]
bbox_coords_list = [(x1, y1, x2, y2), ...]  # Phone 영역 bbox 리스트

# 추론 수행
result = front_service.infer(
    input_image=input_image,
    bbox_coords_list=bbox_coords_list,
    pp_key='base'  # 'base' 또는 'flip'
)

# 결과 확인
print(f"최종 등급: {result['grade']['final_grade']}")
print(f"B 픽셀 수: {result['b_pixel_count']}")
print(f"Top 결함: {result['top_defects']}")
```

### SideService 사용 예시

```python
import yaml
import numpy as np
from service.side_service import SideService

# 설정 파일 로드
with open('configs/service_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 서비스 초기화
side_service = SideService(config, device='cuda')

# 입력 이미지 및 bbox 좌표
input_image = np.array(...)  # [H, W, 3]
bbox_coords_list = [(x1, y1, x2, y2), ...]  # Phone 영역 bbox 리스트

# 추론 수행
result = side_service.infer(
    input_image=input_image,
    bbox_coords_list=bbox_coords_list,
    pp_key='base'  # 'base' 또는 'flip'
)

# 결과 확인
print(f"최종 등급: {result['grade']['final_grade']}")
print(f"B 픽셀 수: {result['b_pixel_count']}")
print(f"Top 결함: {result['top_defects']}")
```

### DisplayService 사용 예시

```python
import yaml
import numpy as np
from service.display_service import DisplayService

# 설정 파일 로드
with open('configs/service_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 서비스 초기화
display_service = DisplayService(config, device='cuda')

# 입력 이미지
input_image = np.array(...)  # [H, W, 3] (1080x1920)

# 추론 수행
result = display_service.infer(
    image=input_image,
    threshold=0.5,      # Phone segmentation threshold
    padding_ratio=0.0   # Padding 비율
)

# 결과 확인
print(f"디스플레이 영역: {result['display_bbox']}")
print(f"최종 등급: {result['grade']['final_grade']}")
print(f"Top 결함: {result['top_defects']}")
```

## 프로젝트 구조

```
service/
├── __init__.py              # 모듈 초기화
├── front_service.py         # Front 서비스 클래스
├── side_service.py          # Side 서비스 클래스
├── display_service.py       # Display 서비스 클래스
├── back_service.py          # Back 서비스 클래스
├── preprocess.py            # 서비스 전처리 모듈
├── postprocess.py           # 서비스 후처리 모듈
└── README.md                # 이 문서
```

## 설정 파일

서비스는 `configs/service_config.yaml` 파일의 다음 섹션을 사용합니다:

### FrontService 설정

```yaml
phone_detection:
  front:
    method: "segmentation"  # 또는 "yolo"
    model:
      encoder_name: "efficientnet-b3"
      encoder_weights: "imagenet"
      classes: 5
    trained_model: "weights/phone_detection/front_best.pt"
    input_size:
      width: 1920
      height: 1080

defect_segmentation:
  front:
    model:
      encoder_name: "efficientnet-b4"
      encoder_weights: "imagenet"
      decoder_name: "unet"
      classes: 4
    trained_model: "weights/defect_segmentation/front_best.pt"
    input_size:
      width: 480
      height: 256
    preprocessing:
      scaler: ["imagenet_standardize"]
    postprocessing:
      temperature_scaling:
        enabled: true
        T: 1
      threshold:
        A: 0.0
        B: 0.0
        C: 0.9999
        D: 0.99
      morphology:
        kernel_size_list: 3
        iters_list: 1
      small_defect_removal:
        enabled: true
        pixel_count_threshold: 200
    pixel_to_length_ratio: 1.0  # mm/pixel
    length_count_thres: 20.0    # mm
    pixel_count_thres: 0         # B 등급 픽셀 수 임계값
```

### SideService 설정

```yaml
phone_detection:
  side:
    method: "segmentation"
    model:
      encoder_name: "efficientnet-b3"
      encoder_weights: "imagenet"
      classes: 5
    trained_model: "weights/phone_detection/side_best.pt"
    input_size:
      width: 1920
      height: 1080

defect_segmentation:
  side:
    model:
      encoder_name: "efficientnet-b4"
      encoder_weights: "imagenet"
      decoder_name: "unet"
      classes: 3  # A, B, C
    trained_model: "weights/defect_segmentation/side_best.pt"
    input_size:
      width: 768
      height: 256
    preprocessing:
      scaler: ["normalize_histogram"]
    postprocessing:
      temperature_scaling:
        enabled: true
        T: 2
      threshold:
        A: 0.0
        B: 0.8
        C: 0.8
      morphology:
        kernel_size_list: [3, 3, 3]
        iters_list: [1, 1, 1]
      pixel_count_threshold: 12500  # B 등급 픽셀 수 임계값
```

### DisplayService 설정

```yaml
phone_detection:
  display:
    method: "segmentation"
    model:
      encoder_name: "efficientnet-b3"
      encoder_weights: "imagenet"
      classes: 2  # 배경 + 디스플레이
    trained_model: "weights/phone_detection/display_best.pt"
    input_size:
      width: 512
      height: 512

defect_segmentation:
  display:
    model:
      encoder_name: "efficientnet-b4"
      encoder_weights: "imagenet"
      decoder_name: "unet"
      classes: 4  # A, B, C, D
    trained_model: "weights/defect_segmentation/display_best.pt"
    input_size:
      width: 928
      height: 512
    preprocessing:
      scaler: ["normalize"]
      clahe:
        enabled: True
        clip_limit: 2.0
        tile_grid_size: [8, 8]
    postprocessing:
      temperature_scaling:
        enabled: true
        T: 1
      threshold:
        A: 0.0
        B: 0.9
        C: 0.99
        D: 0.99
      morphology:
        kernel_size_list: [3, 3, 3, 3, 3, 3, 3, 1, 1]
        iters_list: [1, 1, 1, 1, 1, 1, 1, 1, 1]
      small_defect_removal:
        enabled: true
        pixel_count_threshold: 200
```

### BackService 설정

```yaml
phone_detection:
  back:
    method: "yolo"
    model: "yolo11n.pt"
    trained_model: "weights/phone_detection/back_best.pt"
    input_size:
      width: 640
      height: 640

defect_segmentation:
  back:
    model:
      encoder_name: "efficientnet-b4"
      encoder_weights: "imagenet"
      decoder_name: "unet"
      classes: 4  # A, B, C, D
    trained_model: "weights/defect_segmentation/back_best.pt"
    input_size:
      width: 480
      height: 256
    preprocessing:
      scaler: ["imagenet_standardize"]
    postprocessing:
      temperature_scaling:
        enabled: true
        T: 1
      threshold:
        A: 0.0
        B: 0.0
        C: 0.9999
        D: 0.99
      morphology:
        kernel_size_list: 3
        iters_list: 1
      small_defect_removal:
        enabled: true
        pixel_count_threshold: 200

# 후처리 설정 (선택사항)
POSTPROC:
  remove_back_edge_defect:
    h_rat: 0.1  # 세로 삭제 영역 비율
    w_rat: 0.1  # 가로 삭제 영역 비율
    bg_mask_val: 1  # 핸드폰 영역 마스크 값
```

## 반환 결과 형식

### FrontService.infer() 반환값

```python
{
    'defect_mask': List[List[List[int]]],  # 후처리된 결함 마스크
    'analysis': {
        'grade_counts': Dict[str, int],     # 등급별 픽셀 수
        'grade_areas': Dict[str, int],      # 등급별 영역 수
        'defects': List[Dict]               # 각 결함 영역 정보
    },
    'grade': {
        'final_grade': str,                 # 최종 등급 (A, B, C, D)
        'grade_details': Dict[str, Dict],  # 등급별 상세 정보
        'all_defects': List[Dict]           # 모든 결함 정보
    },
    'top_defects': List[Dict],             # 가장 심한 결함 N개
    'b_pixel_count': int,                  # B 등급 픽셀 수
    'pixel_count_threshold_exceeded': bool # 픽셀 수 임계값 초과 여부
}
```

### SideService.infer() 반환값

```python
{
    'defect_mask': List[List[List[int]]],  # 후처리된 결함 마스크
    'analysis': {
        'grade_counts': Dict[str, int],     # 등급별 픽셀 수
        'grade_areas': Dict[str, int],      # 등급별 영역 수
        'defects': List[Dict]               # 각 결함 영역 정보
    },
    'grade': {
        'final_grade': str,                 # 최종 등급 (A, B, C)
        'grade_details': Dict[str, Dict],  # 등급별 상세 정보
        'all_defects': List[Dict]           # 모든 결함 정보
    },
    'top_defects': List[Dict],             # 가장 심한 결함 N개
    'b_pixel_count': int,                  # B 등급 픽셀 수
    'pixel_count_threshold_exceeded': bool # 픽셀 수 임계값 초과 여부
}
```

### DisplayService.infer() 반환값

```python
{
    'display_bbox': List[int],               # 디스플레이 영역 bbox [x1, y1, x2, y2]
    'defect_mask': List[List[int]],         # 후처리된 결함 마스크 [H, W]
    'analysis': {
        'grade_counts': Dict[str, int],     # 등급별 픽셀 수
        'grade_areas': Dict[str, int],      # 등급별 영역 수
        'defects': List[Dict]               # 각 결함 영역 정보
    },
    'grade': {
        'final_grade': str,                 # 최종 등급 (A, B, C, D)
        'grade_details': Dict[str, Dict],  # 등급별 상세 정보
        'all_defects': List[Dict]           # 모든 결함 정보
    },
    'top_defects': List[Dict]              # 가장 심한 결함 N개
}
```

### BackService.infer() 반환값

```python
{
    'phone_bbox': Tuple[int, int, int, int],  # Phone 영역 bbox (y1, y2, x1, x2) 또는 None
    'defect_mask': List[List[int]],           # 후처리된 결함 마스크 [H, W]
    'analysis': {
        'grade_counts': Dict[str, int],        # 등급별 픽셀 수
        'grade_areas': Dict[str, int],         # 등급별 영역 수
        'defects': List[Dict]                  # 각 결함 영역 정보
    },
    'grade': {
        'final_grade': str,                    # 최종 등급 (A, B, C, D)
        'grade_details': Dict[str, Dict],      # 등급별 상세 정보
        'all_defects': List[Dict]              # 모든 결함 정보
    },
    'top_defects': List[Dict]                  # 가장 심한 결함 N개
}
```

## 참고사항

1. **Post-processing Key (pp_key)**:
   - `'base'`: 일반 모델용 후처리 파라미터
   - `'flip'`: Galaxy Flip 모델용 후처리 파라미터

2. **등급 매핑**:
   - A: 0 (정상)
   - B: 1 또는 2 (경미한 결함)
   - C: 2 또는 3 (중간 결함)
   - D: 3 (심한 결함)
   - 특수 등급: 23 (Front에서 B→C 변경된 경우)

3. **B 등급 처리**:
   - Front: 픽셀 수 또는 길이 임계값 초과 시 C 등급(23)으로 변경
   - Side: 픽셀 수 임계값 초과 시 C 등급(3)으로 변경

## Legal Notice

**중요**: 이 서비스 코드는 실제 상용 코드를 포함하지 않습니다. 모든 코드는 포트폴리오 목적으로 새로 작성되었으며, 실제 상용 시스템과의 유사성은 구조적 아이디어에 한정됩니다.

- 실제 상용 코드는 포함되지 않았습니다
- 실제 상용 데이터는 포함되지 않았습니다
- 실제 상용 모델 가중치는 포함되지 않았습니다
- 회사 기밀 정보는 포함되지 않았습니다

