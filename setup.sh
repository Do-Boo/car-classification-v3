#!/bin/bash

# 차량 분류 AI 프로젝트 - 전체 설정 스크립트

echo "🚀 차량 분류 AI 프로젝트 설정 시작..."

# 프로젝트 루트 디렉토리
PROJECT_ROOT="/Users/doyoukim/Desktop/car_classification"
cd "$PROJECT_ROOT"

# 필요한 디렉토리 생성
echo "📁 디렉토리 구조 생성 중..."
mkdir -p config
mkdir -p src/{data,models,training,utils,inference}
mkdir -p scripts
mkdir -p notebooks
mkdir -p outputs/{models,submissions,logs,data}
mkdir -p data/{train,test}

# __init__.py 파일 생성
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/utils/__init__.py
touch src/inference/__init__.py

echo "✅ 프로젝트 구조 생성 완료!"

# 기본 패키지 설치
echo "📦 기본 패키지 설치 중..."
pip install torch torchvision timm pandas numpy scikit-learn tqdm albumentations opencv-python PyYAML Pillow

# 로깅 시스템 설치
echo "📊 로깅 시스템 설치 중..."
pip install matplotlib seaborn tensorboard

echo "✅ 기본 패키지 설치 완료!"

# WandB 설치 옵션
echo ""
read -p "🌐 WandB를 설치하시겠습니까? (온라인 로깅, 모바일 접근 가능) [y/N]: " install_wandb
if [[ "$install_wandb" =~ ^[Yy]$ ]]; then
    echo "📱 WandB 설치 중..."
    pip install wandb
    echo "✅ WandB 설치 완료!"
    echo ""
    echo "🔑 WandB 로그인 방법:"
    echo "   1. wandb login 명령어 실행"
    echo "   2. https://wandb.ai/authorize 에서 API 키 복사"
    echo "   3. API 키 입력"
    echo "   4. 이제 스마트폰에서도 실험 결과 확인 가능!"
else
    echo "⚠️ WandB 미설치 - 오프라인 모드로만 작동합니다"
fi

echo ""
echo "========================================="
echo "🎯 설정 완료! 다음 단계:"
echo ""
echo "1. 📂 데이터 준비:"
echo "   - data/train/ 에 학습 이미지 배치"
echo "   - data/test/ 에 테스트 이미지 배치"
echo "   - data/test.csv 파일 배치"
echo ""
echo "2. 🏃‍♂️ 실행 방법:"
echo "   - 전처리: python src/data/preprocessing.py"
echo "   - 학습: python scripts/train.py --config config/config.yaml --fold 0"
echo ""
echo "3. 📊 결과 확인:"
echo "   - 로컬: outputs/ 폴더 확인"
echo "   - TensorBoard: tensorboard --logdir outputs/[실험명]/tensorboard"
if [[ "$install_wandb" =~ ^[Yy]$ ]]; then
    echo "   - WandB: 브라우저 또는 모바일 앱에서 확인"
fi
echo ""
echo "🎉 해피 코딩!"
echo "========================================="
