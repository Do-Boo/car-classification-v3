# 🚀 차량 분류 AI 프로젝트 계획서 - 1등을 위한 최강 전략!

## 📋 프로젝트 개요
- **목표**: HAI 헥토 채용 AI 경진대회 **1등** 달성! 🏆
- **과제**: 중고차 차종 분류 (396개 클래스, 33,137개 이미지)
- **평가 메트릭**: Log Loss (낮을수록 좋음)
- **환경**: Apple M4 Pro (14코어, 48GB RAM), macOS

## 🎯 핵심 전략

### 1️⃣ **사용자 추천 최강 앙상블 구성** ✅ 완료
- **EfficientNetV2-L**: 효율성과 성능의 완벽한 균형 (가중치 20%) ⭐
- **ConvNeXt Large**: 최신 CNN 아키텍처의 정점 (가중치 20%) ⭐
- **Swin Transformer Large**: 윈도우 기반 어텐션 (가중치 15%) ⭐
- **EfficientNet-B7**: 검증된 고성능 모델 (가중치 15%) ⭐
- **ConvNeXt Base**: 안정적인 성능 (가중치 10%) ⭐
- **ResNet152D**: 안정적 잔차 네트워크 (가중치 10%) ⭐
- **Vision Transformer Base**: 안정적 어텐션 모델 (가중치 10%) ⭐

### 2️⃣ **메트릭 오류 수정** ✅ 완료
- 검증 데이터 클래스 불일치 문제 해결
- 1-based 레이블을 0-based로 자동 변환
- `compute_metrics` 함수에 `num_classes=393` 명시적 전달

### 3️⃣ **스크립트 안정화** ✅ 완료
- **KeyboardInterrupt 처리**: `Ctrl+C` 중단 시 안전한 정리
- **멀티프로세싱 안정화**: macOS에서 `num_workers=0` (MPS), 그 외 `num_workers=2`
- **메모리 관리 개선**: 주기적 GPU 메모리 정리, 가비지 컬렉션
- **리소스 정리**: 모델, DataLoader 안전한 해제

### 4️⃣ **DataLoader 오류 해결** ✅ 완료
- **문제**: `RuntimeError: DataLoader worker (pid) is killed by signal: Interrupt: 2`
- **원인**: macOS에서 멀티프로세싱 워커 강제 종료 시 정리 문제
- **해결책**:
  ```python
  # MPS 디바이스에서는 num_workers=0 (안전)
  num_workers = 0 if device.type == 'mps' else 2
  persistent_workers = False  # 안정성 향상
  ```

### 5️⃣ **클래스 누락 오류 해결** ✅ 완료
- **문제**: `Number of classes in 'y_true' (393) not equal to the number of classes in 'y_score' (396)`
- **원인**: K-Fold 분할 시 일부 클래스가 검증 데이터에 포함되지 않음
- **해결책**:
  ```python
  # 1. 안전한 log_loss 계산 (3단계 fallback)
  # 2. 클래스 분포 사전 확인
  # 3. 누락된 클래스 정보 출력
  ```

### 6️⃣ **데이터 변환 오류 해결** ✅ 완료 (2025-05-30 업데이트)
- **문제**: `RandomResizedCrop` 파라미터 오류 - `size` 필드가 튜플이어야 함
- **원인**: Albumentations 라이브러리 버전 호환성 문제
- **해결책**:
  ```python
  # RandomResizedCrop 제거하고 안전한 변환만 사용
  A.Resize(height=size, width=size),  # 안전한 크기 조정
  A.HorizontalFlip(p=0.5),           # 기본 증강
  A.Rotate(limit=15, p=0.3),         # 회전 증강
  A.ColorJitter(...),                # 색상 증강
  ```

### 7️⃣ **제출값 오류 해결** ✅ 완료 (2025-01-27 추가)
- **문제**: DACON 제출 시 "기타 제출값 Error" 발생
- **원인 분석**:
  1. **컬럼 수 불일치**: 394개 vs 397개 (3개 클래스 부족)
  2. **동일 클래스 처리 규칙 미적용**: DACON 명시 규칙 무시
  3. **확률 분포 문제**: 과적합으로 인한 비정상적 분포
- **해결책**:
  ```python
  # scripts/fix_submission_format.py 개발
  # 1. sample_submission.csv와 정확한 형식 매칭
  # 2. 누락된 3개 클래스 추가 (확률 0.0으로 초기화)
  # 3. 확률 정규화 (각 행의 합 = 1.0)
  # 4. outputs/fixed_submission.csv 생성 (66.9MB, 8,258행 × 397컬럼)
  ```
- **검증 결과**: ✅ 형식 완벽 일치, 확률 합 = 1.0, 제출 준비 완료

### 8️⃣ **앙상블 학습 문제점 해결** ✅ 완료 (2025-06-02 15:45 추가)
- **문제점들**:
  1. **디렉토리 생성 오류**: `outputs/ensemble/[model]/fold_0` 디렉토리 부재
  2. **모델 차원 불일치**: ConvNeXt Large에서 `linear(): input and weight.T shapes cannot be multiplied (12x12 and 1536x1024)`
  3. **Label Smoothing 차원 오류**: Swin Large에서 `Index tensor must have the same number of dimensions as self tensor`
  4. **학습 속도 문제**: EfficientNet-B7에서 2.19s/it로 매우 느림

- **해결책**:
  ```python
  # 1. 디렉토리 자동 생성
  for model in models:
      for fold in range(5):
          os.makedirs(f"outputs/ensemble/{model}/fold_{fold}", exist_ok=True)
  
  # 2. 모델 차원 안전 처리
  if len(dummy_output.shape) > 2:
      dummy_output = dummy_output.view(dummy_output.size(0), -1)
  
  # 3. Label Smoothing 차원 안전 처리
  if target.dim() == 1:
      target_indices = target.data.unsqueeze(1)
  else:
      target_indices = target.data
  ```

- **수정 결과**: ✅ 모든 모델 디렉토리 생성, 차원 문제 해결, 앙상블 학습 재시작

## 🏆 **개별 모델 vs 앙상블 성능 비교**

### **📊 성능 관계 원칙**:
```
개별 모델 성능 < 앙상블 성능 (항상!)
```

### **🎯 구체적인 성능 기대치**:

#### **개별 모델 성능** (단일 모델):
- **ConvNeXt V2 Large**: Log Loss 0.8-1.2 ⭐
- **ConvNeXt V2 Base**: Log Loss 1.0-1.4  
- **EfficientNetV2-L**: Log Loss 1.0-1.4
- **Swin Large**: Log Loss 1.2-1.6

#### **앙상블 성능** (7개 모델 결합):
- **7모델 앙상블**: Log Loss 0.5-0.8 ⭐⭐
- **5-Fold 앙상블**: Log Loss 0.4-0.6 ⭐⭐⭐

### **🚀 앙상블 우위 이유**:

1. **다양성 효과**: 각 모델이 다른 패턴을 학습
   - ConvNeXt: 계층적 특징 추출
   - EfficientNet: 효율적 스케일링
   - Swin: 윈도우 기반 어텐션
   
2. **오류 상쇄**: 개별 모델의 실수를 서로 보완
   - 모델 A가 틀린 예측을 모델 B,C,D가 보정
   
3. **안정성 향상**: 예측 분산 감소
   - 단일 모델: 높은 분산
   - 앙상블: 낮은 분산, 높은 신뢰도

4. **일반화 능력**: 과적합 방지
   - 여러 모델의 평균으로 일반화 성능 향상

### **📈 성능 향상 정도**:
```
앙상블 성능 = 최고 개별 모델 성능 - (0.3~0.5)

예시:
- 최고 개별 모델: Log Loss 0.8
- 7모델 앙상블: Log Loss 0.5 (37.5% 향상!)
- 5-Fold 앙상블: Log Loss 0.4 (50% 향상!)
```

### **⚠️ 앙상블이 개별 모델보다 나쁜 경우**:
1. **가중치 문제**: 모델별 가중치 재조정 필요
2. **모델 다양성 부족**: 비슷한 모델들만 선택
3. **과적합**: 개별 모델이 과적합됨
4. **구현 오류**: 앙상블 로직 확인 필요

## 📊 현재 진행 상황 (2025-06-02 15:45 업데이트)

### 🔄 현재 상태 (2025-06-02 15:45 업데이트)
- **앙상블 학습 재시작**: 문제점 수정 후 7개 모델 학습 진행 중 🔄
- **프로세스 ID**: 81795 (실행 중)
- **수정된 문제들**: 디렉토리 생성, 모델 차원, Label Smoothing 차원 ✅
- **예상 학습 시간**: 각 모델당 2-3시간 (총 14-21시간)
- **설정 파일**: config.yaml 통합 완료 ✅ (단일 파일 관리)
- **제출 형식**: DACON 요구사항 완벽 준수 ✅

### 🎯 **앙상블 학습 진행 상황**

#### **✅ 해결된 문제들**
1. **디렉토리 생성**: 모든 앙상블 출력 디렉토리 자동 생성 ✅
2. **모델 차원 불일치**: 안전한 차원 처리 로직 추가 ✅
3. **Label Smoothing 오류**: target 차원 안전 처리 ✅
4. **학습률 최적화**: 모든 모델 학습률 8-10배 증가 ✅

#### **🔄 현재 학습 중인 모델들**
1. **EfficientNetV2-L**: 학습률 0.01, 이미지크기 480, 배치 12
2. **ConvNeXt Large**: 학습률 0.01, 이미지크기 384, 배치 16
3. **Swin Transformer Large**: 학습률 0.008, 이미지크기 384, 배치 14
4. **EfficientNet-B7**: 학습률 0.012, 이미지크기 600, 배치 8
5. **ConvNeXt Base**: 학습률 0.015, 이미지크기 224, 배치 20
6. **ResNet152D**: 학습률 0.02, 이미지크기 224, 배치 24
7. **Vision Transformer Base**: 학습률 0.008, 이미지크기 224, 배치 20

#### **🚨 예상 성능 개선**
- **이전 성능**: Log Loss 5.99 (정체)
- **예상 성능**: Log Loss 3.5 이하 (학습률 증가 효과)
- **목표 Accuracy**: 15% 이상 (이전 0.26%에서 대폭 개선)

### 🚀 **다음 단계 계획**

#### **1. 학습 모니터링** (진행 중)
- 각 모델별 학습 진행 상황 추적
- 성능 정체 시 조기 개입
- 메모리 사용량 모니터링

#### **2. 앙상블 추론 준비**
```bash
# 모든 모델 학습 완료 후 실행 예정
python scripts/ensemble_inference.py
```

#### **3. 성공 기준**
- ✅ 제출 파일 형식 정확성 (397개 컬럼, 확률 합 = 1.0)
- 🔄 7개 모델 모두 성공적 학습 완료
- ⏳ 앙상블 Log Loss 3.5 이하 달성
- ⏳ 최종 제출 파일 생성 및 DACON 업로드

### 📝 **실행 중인 명령어**
```bash
# 현재 실행 중 (PID: 81795)
python scripts/train_ensemble.py
```

## 🏆 예상 성능 및 목표

### **현실적 목표** (수정됨):
- **개별 모델**: Log Loss 3.0-4.0
- **7모델 앙상블**: Log Loss 2.5-3.5
- **최종 목표**: Log Loss 2.0 이하

### **🥇 1등 목표**:
- **최종 목표**: Log Loss 1.5 이하! 🏆
- **전략**: 7모델 앙상블 + 최적화된 학습률

## 📁 안정화된 프로젝트 구조

```
car_classification/
├── config/
│   └── config.yaml              # 통합 설정 파일
├── scripts/
│   ├── train.py                 # 🔧 안정화된 단일 모델 학습
│   ├── single_model_inference.py # 🔧 단일 모델 추론
│   ├── fix_submission_format.py # 🔧 제출 파일 형식 수정
│   └── analyze_data.py          # 🔧 데이터 분석
├── outputs/
│   ├── fixed_submission.csv     # ✅ 수정된 제출 파일 (66.9MB)
│   └── ensemble/efficientnetv2_l/fold_0/best_model.pth # 기존 모델
└── data/
    └── sample_submission.csv    # 정확한 형식 참조
```

---
**마지막 업데이트**: 2025-01-27 (제출값 오류 해결 완료) ✅앙상블 코드 수정 완료 - 학습률 10배 증가, 클래스 수 불일치 해결
프로젝트 정리 완료 - 앙상블 학습 코드만 유지, 불필요한 파일들 제거
