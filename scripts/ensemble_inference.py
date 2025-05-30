#!/usr/bin/env python3
"""
🏆 차량 분류 앙상블 추론 스크립트 - 사용자 추천 5개 모델 구성
"""

import os
import sys
import yaml
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.models.backbone import get_model
from src.data.dataset import CarDataset, get_valid_transforms

def load_ensemble_models(ensemble_results_path, device):
    """사용자 추천 앙상블 모델들 로드"""
    with open(ensemble_results_path, 'r') as f:
        ensemble_results = json.load(f)
    
    models = {}
    
    print("🔄 앙상블 모델 로드 중...")
    for model_name, info in ensemble_results.items():
        model_path = info['model_path']
        weight = info['weight']
        description = info.get('description', 'No description')
        
        if not os.path.exists(model_path):
            print(f"⚠️ 모델 파일이 없습니다: {model_path}")
            continue
        
        try:
            # 체크포인트 로드
            checkpoint = torch.load(model_path, map_location=device)
            config = checkpoint['config']
            
            # 모델 생성 및 가중치 로드
            model = get_model(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            models[model_name] = {
                'model': model,
                'weight': weight,
                'config': config,
                'val_loss': info['val_loss'],
                'description': description
            }
            
            print(f"✅ {model_name}: {description}")
            print(f"   Weight: {weight*100}%, Val Loss: {info['val_loss']:.4f}")
            
        except Exception as e:
            print(f"❌ {model_name} 모델 로드 실패: {e}")
            continue
    
    return models

def create_test_dataloader(test_df, img_size, batch_size=32):
    """테스트 데이터로더 생성 (모델별 이미지 크기 대응)"""
    test_transform = get_valid_transforms(img_size)
    test_dataset = CarDataset(test_df, transform=test_transform, mode='test')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    
    return test_loader

def predict_single_model(model, dataloader, device, model_name):
    """단일 모델 예측"""
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting {model_name}"):
            images = batch[0].to(device)
            outputs = model(images)
            batch_pred = F.softmax(outputs, dim=1)
            all_predictions.append(batch_pred.cpu().numpy())
    
    return np.vstack(all_predictions)

def ensemble_predict(models, test_df, device):
    """사용자 추천 5개 모델 앙상블 예측"""
    print("🚀 사용자 추천 5개 모델 앙상블 예측 시작!")
    print(f"📊 테스트 이미지 수: {len(test_df)}")
    
    all_model_predictions = []
    weights = []
    
    for model_name, model_info in models.items():
        print(f"\n📊 {model_name} 예측 중...")
        print(f"📝 {model_info['description']}")
        
        model = model_info['model']
        weight = model_info['weight']
        config = model_info['config']
        
        # 해당 모델의 이미지 크기로 데이터로더 생성
        img_size = config['data']['img_size']
        print(f"🔧 이미지 크기: {img_size}x{img_size}")
        
        test_loader = create_test_dataloader(test_df, img_size, batch_size=32)
        
        # 예측 수행
        predictions = predict_single_model(model, test_loader, device, model_name)
        
        all_model_predictions.append(predictions)
        weights.append(weight)
        
        print(f"✅ {model_name} 예측 완료: {predictions.shape}")
        print(f"🎯 가중치: {weight*100}%")
    
    # 가중 평균으로 앙상블
    print("\n🔄 앙상블 결합 중...")
    weights = np.array(weights)
    weights = weights / weights.sum()  # 정규화
    
    print("📊 최종 가중치:")
    for i, (model_name, weight) in enumerate(zip(models.keys(), weights)):
        print(f"  • {model_name}: {weight*100:.1f}%")
    
    ensemble_predictions = np.zeros_like(all_model_predictions[0])
    for pred, weight in zip(all_model_predictions, weights):
        ensemble_predictions += pred * weight
    
    # 최종 예측 클래스
    ensemble_classes = np.argmax(ensemble_predictions, axis=1)
    
    print(f"\n🎉 앙상블 예측 완료: {ensemble_predictions.shape}")
    
    return ensemble_predictions, ensemble_classes

def create_submission(test_df, predictions, class_info, output_path):
    """제출 파일 생성"""
    # 클래스 인덱스를 클래스명으로 변환
    label_to_class = class_info['label_to_class']
    predicted_classes = [label_to_class[str(pred)] for pred in predictions]
    
    # 제출 DataFrame 생성
    submission_df = pd.DataFrame({
        'img_path': test_df['img_path'].apply(lambda x: os.path.basename(x)),
        'class': predicted_classes
    })
    
    # CSV 저장
    submission_df.to_csv(output_path, index=False)
    print(f"💾 제출 파일 저장: {output_path}")
    
    # 예측 분포 출력
    class_counts = submission_df['class'].value_counts()
    print(f"\n📊 예측 클래스 분포 (상위 10개):")
    print(class_counts.head(10))
    
    print(f"\n📈 총 예측된 클래스 수: {len(class_counts)}/396")

def main():
    parser = argparse.ArgumentParser(description='Ensemble Inference for Car Classification - User Recommended 5 Models')
    parser.add_argument('--ensemble_results', type=str, required=True)
    parser.add_argument('--test_csv', type=str, default='data/test.csv')
    parser.add_argument('--class_info', type=str, default='outputs/data/class_info.json')
    parser.add_argument('--output', type=str, default='outputs/user_ensemble_submission.csv')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    
    print("🏆 사용자 추천 5개 모델 앙상블 추론 시작!")
    print("🎯 모델 구성: EfficientNetV2-L + ConvNeXt + Swin + ResNet152 + Inception-v4")
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Apple Silicon GPU (MPS) 사용")
    else:
        device = torch.device('cpu')
        print("⚠️ CPU 사용")
    
    # 앙상블 결과 파일 경로 설정
    if not os.path.exists(args.ensemble_results):
        ensemble_results_path = f"outputs/ensemble/ensemble_results_fold_{args.fold}.json"
        if os.path.exists(ensemble_results_path):
            args.ensemble_results = ensemble_results_path
        else:
            print(f"❌ 앙상블 결과 파일을 찾을 수 없습니다: {args.ensemble_results}")
            return
    
    # 데이터 로드
    print("\n📊 데이터 로드 중...")
    
    # 테스트 데이터
    if os.path.exists(args.test_csv):
        test_df = pd.read_csv(args.test_csv)
    else:
        # 테스트 디렉토리에서 이미지 파일 수집
        test_dir = "data/test"
        if os.path.exists(test_dir):
            test_files = []
            for file in os.listdir(test_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_files.append(os.path.join(test_dir, file))
            test_df = pd.DataFrame({'img_path': test_files})
        else:
            print(f"❌ 테스트 데이터를 찾을 수 없습니다: {args.test_csv}")
            return
    
    # 클래스 정보
    with open(args.class_info, 'r') as f:
        class_info = json.load(f)
    
    print(f"📊 테스트 이미지 수: {len(test_df)}")
    print(f"📊 클래스 수: {class_info['num_classes']}")
    
    # 앙상블 모델 로드
    print(f"\n🔄 앙상블 모델 로드 중...")
    models = load_ensemble_models(args.ensemble_results, device)
    
    if not models:
        print("❌ 로드된 모델이 없습니다.")
        return
    
    print(f"\n✅ {len(models)}개 모델 로드 완료")
    
    # 앙상블 예측
    ensemble_proba, ensemble_pred = ensemble_predict(models, test_df, device)
    
    # 제출 파일 생성
    print("\n📝 제출 파일 생성 중...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    create_submission(test_df, ensemble_pred, class_info, args.output)
    
    print("\n🎉 사용자 추천 5개 모델 앙상블 추론 완료!")
    print("🏆 EfficientNetV2-L + ConvNeXt + Swin + ResNet152 + Inception-v4 = 최강 앙상블!")

if __name__ == "__main__":
    main() 