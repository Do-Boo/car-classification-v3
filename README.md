# 차량 분류 AI 프로젝트 v2

> 396개 차종을 분류하는 고성능 앙상블 모델

## 프로젝트 개요

- **목표**: 396개 차종 분류 (33,137개 학습 이미지)
- **모델**: 5개 최강 모델 앙상블 (EfficientNetV2-L, ConvNeXt Large, Swin Transformer Large, ResNet152, Inception-v4)
- **예상 성능**: Log Loss 0.5-0.8 (5-Fold 앙상블)
- **환경**: Apple Silicon 최적화

## 빠른 시작

### 1. 환경 설정
```bash
git clone https://github.com/Do-Boo/car-classification-v2.git
cd car-classification-v2
chmod +x setup.sh
./setup.sh
```

### 2. 데이터 준비
```bash
# 데이터 디렉토리 생성
mkdir -p data/train data/test

# 대회 데이터를 다음 구조로 배치:
# data/
# ├── train/          # 학습 이미지들
# ├── test/           # 테스트 이미지들
# ├── test.csv        # 테스트 메타데이터
# └── sample_submission.csv
```

### 3. 앙상블 학습 실행
```bash
# 단일 Fold 학습
python scripts/train_ensemble.py --fold 0

# 전체 5-Fold 학습
python scripts/train_ensemble.py --all_folds
```

### 4. 추론 실행
```bash
python scripts/ensemble_inference.py --fold 0
```

## 프로젝트 구조

```
car-classification-v2/
├── src/                    # 소스 코드
│   ├── models/            # 모델 정의
│   ├── data/              # 데이터 로더
│   ├── training/          # 학습 관련
│   └── utils/             # 유틸리티
├── scripts/               # 실행 스크립트
│   ├── train_ensemble.py  # 앙상블 학습
│   └── ensemble_inference.py # 앙상블 추론
├── config/                # 설정 파일
├── docs/                  # 문서
└── requirements.txt       # 의존성
```

## 모델 구성

| 모델 | 가중치 | 이미지 크기 | 배치 크기 | 설명 |
|------|--------|-------------|-----------|------|
| EfficientNetV2-L | 25% | 384 | 24 | 효율성과 성능의 균형 |
| ConvNeXt Large | 25% | 384 | 20 | 최신 CNN 아키텍처 |
| Swin Transformer Large | 20% | 384 | 18 | 윈도우 기반 어텐션 |
| ResNet152 | 15% | 224 | 32 | 검증된 클래식 아키텍처 |
| Inception-v4 | 15% | 299 | 28 | 다중 스케일 특징 추출 |

## 성능 예상

- **1-Fold 앙상블**: Log Loss 0.7-1.0
- **5-Fold 앙상블**: Log Loss 0.5-0.8
- **예상 순위**: 상위 5-15%

## 시스템 요구사항

- **권장**: Apple M4 Pro (14코어, 48GB RAM)
- **최소**: 16GB RAM, GPU 8GB+
- **OS**: macOS, Linux, Windows
- **Python**: 3.8+

## 주요 기능

- Apple Silicon 최적화
- 자동 메모리 관리
- 안전한 중단/재시작
- 실시간 메트릭 모니터링
- K-Fold 교차 검증
- TTA (Test Time Augmentation)

## 사용법

### 설정 파일 수정
```yaml
# config/config.yaml
data:
  num_classes: 396
  img_size: 384

training:
  epochs: 100
  batch_size: 24
  learning_rate: 0.005
```

### 커스텀 모델 추가
```python
# scripts/train_ensemble.py
ENSEMBLE_MODELS = {
    "your_model": {
        "backbone": "your_backbone",
        "img_size": 224,
        "batch_size": 32,
        "weight": 0.2
    }
}
```

## 문제 해결

### 메모리 부족
```bash
# 배치 크기 줄이기
# config/config.yaml에서 batch_size 값 감소
```

### 학습 중단
```bash
# 안전한 중단: Ctrl+C
# 재시작: 동일한 명령어로 체크포인트부터 재개
```

## 라이선스

MIT License

## 기여

1. Fork the Project
2. Create your Feature Branch
3. Commit your Changes
4. Push to the Branch
5. Open a Pull Request

## 연락처

- GitHub: [@Do-Boo](https://github.com/Do-Boo)
- Repository: [car-classification-v2](https://github.com/Do-Boo/car-classification-v2)

---

이 프로젝트가 도움이 되었다면 Star를 눌러주세요!
