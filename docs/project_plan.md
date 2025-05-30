# 차량 분류 AI 프로젝트 v3 계획서

## 프로젝트 개요
- **목표**: 396개 차종 분류 AI 모델 개발
- **데이터**: 33,137개 이미지
- **환경**: Apple M4 Pro (14코어, 48GB RAM), macOS
- **접근법**: 5개 모델 앙상블 시스템

## 데이터셋 정보
- **총 이미지 수**: 33,137개
- **클래스 수**: 396개 차종
- **데이터 분할**: 5-Fold Cross Validation
- **이미지 크기**: 다양 (리사이즈 필요)

## 모델 아키텍처

### 앙상블 구성 (5개 모델)
1. **EfficientNetV2-L** (25% 가중치)
   - 효율성과 성능의 완벽한 균형
   - 이미지 크기: 480x480

2. **ConvNeXt Large** (25% 가중치)
   - 최신 CNN 아키텍처의 정점
   - 이미지 크기: 384x384

3. **Swin Transformer Large** (20% 가중치)
   - 윈도우 기반 어텐션 메커니즘
   - 이미지 크기: 384x384

4. **ResNet152** (15% 가중치)
   - 검증된 클래식 아키텍처
   - 이미지 크기: 224x224

5. **Inception-v4** (15% 가중치)
   - 다중 스케일 특징 추출의 대가
   - 이미지 크기: 299x299

## 성능 최적화 전략

### 현재 상황 분석
- **문제**: 학습 속도가 매우 느림 (배치당 142초)
- **원인**: 설정이 너무 공격적일 가능성
- **해결책**: 보수적 접근법 적용

### 보수적 최적화 방안

#### 1. 배치 크기 조정
```yaml
# 현재 설정 (너무 큰 배치 크기)
batch_size: 16

# 보수적 설정 (안정성 우선)
batch_size: 8
```

#### 2. 이미지 크기 축소
```yaml
# 현재 설정
EfficientNetV2-L: 480x480
ConvNeXt Large: 384x384
Swin Large: 384x384

# 보수적 설정
EfficientNetV2-L: 384x384  # 20% 축소
ConvNeXt Large: 320x320    # 17% 축소
Swin Large: 320x320        # 17% 축소
```

#### 3. 학습률 조정
```yaml
# 현재 설정
learning_rate: 1e-4

# 보수적 설정
learning_rate: 5e-5  # 50% 감소
```

#### 4. 데이터 로더 최적화
```yaml
# 현재 설정
num_workers: 2
persistent_workers: False

# 보수적 설정
num_workers: 1  # 메모리 사용량 감소
persistent_workers: True  # 워커 재사용
```

#### 5. 메모리 관리 강화
- 주기적 GPU 메모리 정리
- 가비지 컬렉션 빈도 증가
- 배치 처리 후 즉시 메모리 해제

## 기술 스택
- **딥러닝**: PyTorch, timm
- **데이터 처리**: pandas, numpy, albumentations
- **시각화**: matplotlib, seaborn
- **평가**: scikit-learn
- **환경 관리**: conda/pip

## 프로젝트 구조
```
car_classification/
├── config/
│   └── config.yaml              # 설정 파일
├── data/
│   ├── train_images/           # 학습 이미지
│   ├── test_images/            # 테스트 이미지
│   ├── train.csv              # 학습 레이블
│   └── sample_submission.csv   # 제출 형식
├── src/
│   ├── data/
│   │   ├── dataset.py         # 데이터셋 클래스
│   │   └── transforms.py      # 데이터 변환
│   ├── models/
│   │   ├── __init__.py
│   │   └── model_factory.py   # 모델 팩토리
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # 학습 로직
│   │   └── loss.py           # 손실 함수
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py         # 유틸리티 함수
│       └── metrics.py         # 평가 메트릭
├── scripts/
│   ├── train_ensemble.py      # 앙상블 학습
│   └── ensemble_inference.py  # 앙상블 추론
├── outputs/
│   ├── models/               # 저장된 모델
│   ├── logs/                # 학습 로그
│   └── submissions/         # 제출 파일
└── docs/
    └── project_plan.md      # 프로젝트 계획서
```

## 학습 전략

### 1단계: 개별 모델 학습
```bash
# 각 모델별로 5-Fold 학습
python scripts/train_ensemble.py --fold 0
python scripts/train_ensemble.py --fold 1
python scripts/train_ensemble.py --fold 2
python scripts/train_ensemble.py --fold 3
python scripts/train_ensemble.py --fold 4
```

### 2단계: 전체 Fold 학습
```bash
# 모든 Fold 한번에 학습
python scripts/train_ensemble.py --all_folds
```

### 3단계: 앙상블 추론
```bash
# 앙상블 예측 수행
python scripts/ensemble_inference.py \
    --test_dir data/test_images \
    --output_path outputs/submissions/ensemble_submission.csv
```

## 예상 성능

### 개별 모델 성능 (Log Loss)
- EfficientNetV2-L: 0.8-1.2
- ConvNeXt Large: 1.0-1.4
- Swin Large: 1.2-1.6
- ResNet152: 1.4-1.8
- Inception-v4: 1.5-1.9

### 앙상블 성능 (Log Loss)
- **1-Fold 앙상블**: 0.7-1.0 (상위 20-30%)
- **5-Fold 앙상블**: 0.5-0.8 (상위 5-15%)
- **목표 성능**: 0.08 이하 (1등 목표)

## 최적화 계획

### 하드웨어 활용
- **CPU**: M4 Pro 14코어 (멀티프로세싱)
- **메모리**: 48GB RAM (대용량 배치)
- **GPU**: Apple Silicon MPS (가속)

### 소프트웨어 최적화
- Mixed Precision Training
- Gradient Accumulation
- Model Parallelism (필요시)
- Data Pipeline 최적화

## 일정 계획

### 1주차: 환경 설정 및 데이터 준비
- [x] 개발 환경 구축
- [x] 데이터 전처리
- [x] 기본 모델 구현

### 2주차: 개별 모델 학습
- [x] EfficientNet 계열 학습
- [x] ConvNeXt 학습
- [x] Swin Transformer 학습
- [x] ResNet 학습
- [x] Inception 학습

### 3주차: 앙상블 시스템 구축
- [x] 앙상블 학습 스크립트
- [x] 앙상블 추론 시스템
- [x] 성능 평가 및 튜닝

### 4주차: 최종 최적화 및 제출
- [ ] 하이퍼파라미터 튜닝
- [ ] 최종 모델 학습
- [ ] 제출 파일 생성

## 리스크 관리

### 기술적 리스크
- **메모리 부족**: 배치 크기 조정으로 해결
- **학습 시간**: 효율적인 모델 선택
- **과적합**: 강력한 정규화 적용

### 일정 리스크
- **학습 지연**: 병렬 학습으로 단축
- **디버깅 시간**: 충분한 테스트 코드

## 성공 지표
- **1차 목표**: Log Loss 1.0 이하
- **2차 목표**: Log Loss 0.5 이하  
- **최종 목표**: Log Loss 0.08 이하 (1등)

## 현재 진행 상황

### 완료된 작업
- [x] 프로젝트 구조 설계
- [x] 데이터 전처리 파이프라인
- [x] 5개 모델 아키텍처 구현
- [x] 앙상블 학습 시스템
- [x] 앙상블 추론 시스템
- [x] 오류 처리 및 안정화
- [x] GitHub 저장소 설정

### 현재 작업
- [x] 성능 최적화 (보수적 접근)
- [x] 학습 속도 개선
- [x] 메모리 사용량 최적화

### 다음 작업
- [ ] 최적화된 설정으로 학습 실행
- [ ] 성능 모니터링 및 조정
- [ ] 최종 앙상블 모델 완성

## 참고 자료
- [EfficientNet 논문](https://arxiv.org/abs/1905.11946)
- [ConvNeXt 논문](https://arxiv.org/abs/2201.03545)
- [Swin Transformer 논문](https://arxiv.org/abs/2103.14030)
- [ResNet 논문](https://arxiv.org/abs/1512.03385)
- [Inception 논문](https://arxiv.org/abs/1602.07261)