#!/usr/bin/env python3
"""
단일 모델 추론 스크립트 - 기존 학습된 모델 성능 확인
"""

import os
import sys
import yaml
import json
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

def load_model_and_config(model_path, device):
    """모델과 설정 로드"""
    print(f"🔄 모델 로드 중: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # 모델 생성 및 가중치 로드
    model = get_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ 모델 로드 완료")
    print(f"📊 모델 정보:")
    print(f"   - 백본: {config['model']['backbone']}")
    print(f"   - 이미지 크기: {config['data']['img_size']}")
    print(f"   - 클래스 수: {config['data']['num_classes']}")
    
    if 'val_loss' in checkpoint:
        print(f"   - 검증 Loss: {checkpoint['val_loss']:.4f}")
    if 'val_log_loss' in checkpoint:
        print(f"   - 검증 Log Loss: {checkpoint['val_log_loss']:.4f}")
    
    return model, config

def create_test_dataloader(test_df, config):
    """테스트 데이터로더 생성"""
    img_size = config['data']['img_size']
    
    # 이미지 경로 수정
    test_df = test_df.copy()
    test_df['img_path'] = test_df['img_path'].apply(lambda x: x.replace('./test/', 'data/test/'))
    
    test_transform = get_valid_transforms(img_size)
    test_dataset = CarDataset(test_df, transform=test_transform, mode='test')
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"📊 테스트 데이터: {len(test_dataset)}개 이미지")
    print(f"🔧 배치 크기: 16, 이미지 크기: {img_size}x{img_size}")
    
    return test_loader

def predict_model(model, dataloader, device):
    """모델 예측 실행"""
    model.eval()
    all_predictions = []
    all_paths = []
    
    print("🔄 예측 실행 중...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            images, paths = batch
            images = images.to(device)
            
            outputs = model(images)
            predictions = F.softmax(outputs, dim=1)
            
            all_predictions.append(predictions.cpu().numpy())
            all_paths.extend(paths)
    
    predictions = np.vstack(all_predictions)
    
    print(f"✅ 예측 완료: {predictions.shape}")
    
    return predictions, all_paths

def create_submission(test_df, predictions, output_path):
    """제출 파일 생성"""
    print("📝 제출 파일 생성 중...")
    
    # sample_submission.csv에서 실제 클래스명 가져오기
    sample_submission_path = "data/sample_submission.csv"
    class_names = None
    
    try:
        if os.path.exists(sample_submission_path):
            # 헤더만 읽기
            sample_df = pd.read_csv(sample_submission_path, nrows=0)  # 헤더만 읽기
            class_names = list(sample_df.columns[1:])  # ID 컬럼 제외
            print(f"✅ sample_submission.csv에서 클래스명 로드: {len(class_names)}개 클래스")
            print(f"🔍 첫 5개 클래스: {class_names[:5]}")
        else:
            print(f"⚠️ sample_submission.csv 파일이 없습니다: {sample_submission_path}")
    except Exception as e:
        print(f"⚠️ sample_submission.csv 로드 오류: {e}")
    
    # 클래스 정보 파일에서 백업으로 시도
    if class_names is None:
        class_info_path = "outputs/data/class_info.json"
        try:
            if os.path.exists(class_info_path):
                with open(class_info_path, 'r') as f:
                    class_info = json.load(f)
                class_names = class_info['class_names']
                print(f"✅ class_info.json에서 클래스명 로드: {len(class_names)}개 클래스")
        except Exception as e:
            print(f"⚠️ class_info.json 로드 오류: {e}")
    
    # 예측 결과와 클래스명 수 맞추기
    if class_names is not None and len(class_names) != predictions.shape[1]:
        print(f"⚠️ 클래스 수 불일치: 모델={predictions.shape[1]}, sample_submission={len(class_names)}")
        
        if len(class_names) > predictions.shape[1]:
            # sample_submission에 더 많은 클래스가 있는 경우, 모델 예측 수에 맞춰 조정
            class_names = class_names[:predictions.shape[1]]
            print(f"✅ 클래스명을 모델 예측 수에 맞춰 조정: {len(class_names)}개")
        else:
            # 모델이 더 많은 클래스를 예측하는 경우, 부족한 클래스명 추가
            for i in range(len(class_names), predictions.shape[1]):
                class_names.append(f"class_{i}")
            print(f"✅ 부족한 클래스명 추가: {len(class_names)}개")
    elif class_names is None:
        # 기본 클래스명 생성 (최후 수단)
        class_names = [f"class_{i}" for i in range(predictions.shape[1])]
        print(f"⚠️ 기본 클래스명 사용: {len(class_names)}개 클래스")
    
    # 제출 DataFrame 생성 (성능 최적화)
    print("📊 DataFrame 생성 중...")
    
    # ID 컬럼과 예측 결과를 한 번에 결합
    data_dict = {'ID': test_df['ID'].values}
    
    # 클래스별 확률을 딕셔너리에 추가
    for i, class_name in enumerate(class_names):
        if i < predictions.shape[1]:
            data_dict[class_name] = predictions[:, i]
    
    # 한 번에 DataFrame 생성
    submission = pd.DataFrame(data_dict)
    
    # 저장
    submission.to_csv(output_path, index=False)
    
    print(f"✅ 제출 파일 저장: {output_path}")
    print(f"📊 제출 파일 shape: {submission.shape}")
    print(f"🔍 확률 합 검증: {submission.iloc[:, 1:].sum(axis=1).mean():.6f}")
    
    # 예측 분포 확인
    predicted_classes = np.argmax(predictions, axis=1)
    unique_classes, counts = np.unique(predicted_classes, return_counts=True)
    
    print(f"\n📈 예측 분포 (상위 10개 클래스):")
    sorted_indices = np.argsort(counts)[::-1][:10]
    for idx in sorted_indices:
        class_idx = unique_classes[idx]
        count = counts[idx]
        percentage = count / len(predicted_classes) * 100
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"class_{class_idx}"
        print(f"  {class_name}: {count}개 ({percentage:.1f}%)")
    
    # 헤더 확인
    print(f"\n📋 제출 파일 헤더 (처음 5개):")
    print(f"  {', '.join(submission.columns[:6])}")
    
    return submission

def main():
    """메인 함수"""
    print("="*60)
    print("🚗 차량 분류 단일 모델 추론")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🔧 디바이스: {device}")
    
    # 모델 경로
    model_path = "outputs/ensemble/efficientnetv2_l/fold_0/best_model.pth"
    
    try:
        # 1. 모델 로드
        model, config = load_model_and_config(model_path, device)
        
        # 2. 테스트 데이터 로드
        print(f"\n📂 테스트 데이터 로드 중...")
        test_df = pd.read_csv("data/test.csv")
        print(f"📊 테스트 이미지 수: {len(test_df)}")
        
        # 3. 데이터로더 생성
        test_loader = create_test_dataloader(test_df, config)
        
        # 4. 예측 실행
        print(f"\n🚀 예측 시작...")
        predictions, paths = predict_model(model, test_loader, device)
        
        # 5. 제출 파일 생성
        output_path = "outputs/single_model_submission.csv"
        create_submission(test_df, predictions, output_path)
        
        print(f"\n🎉 추론 완료!")
        print(f"📁 결과 파일: {output_path}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 