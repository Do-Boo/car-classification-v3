#!/usr/bin/env python3
"""
🚀 차량 분류 앙상블 학습 스크립트 - 사용자 추천 모델 구성 (안정화 버전)
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import signal
import gc
import copy

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent))

from src.models.backbone import get_model
from src.data.dataset import CarDataset, get_train_transforms, get_valid_transforms
from src.utils.metrics import compute_metrics
from src.training.losses import get_loss_fn

# 전역 변수로 정리 플래그 설정
cleanup_flag = False

def signal_handler(signum, frame):
    """KeyboardInterrupt 처리"""
    global cleanup_flag
    print("\n🛑 학습 중단 신호를 받았습니다. 안전하게 정리 중...")
    cleanup_flag = True

# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)

# 앙상블 모델 설정 (메모리 최적화 버전 - 속도 우선)
ENSEMBLE_MODELS = {
    "efficientnetv2_l": {  # 메모리 최적화
        "backbone": "tf_efficientnetv2_l.in21k_ft_in1k",
        "img_size": 480,  # 384 → 480으로 증가 (성능 향상)
        "batch_size": 12,  # 4 → 12로 3배 증가 (속도 향상)
        "learning_rate": 0.01,
        "weight": 0.20,  # 20%
        "description": "EfficientNetV2-L: 고성능 안정 모델"
    },
    "convnext_large": {  # 메모리 최적화
        "backbone": "convnext_large.fb_in22k_ft_in1k_384",
        "img_size": 384,  # 320 → 384로 증가
        "batch_size": 16,  # 4 → 16으로 4배 증가 (속도 향상)
        "learning_rate": 0.01,
        "weight": 0.20,  # 20%
        "description": "ConvNeXt Large: CNN 아키텍처 최적화"
    },
    "swin_large": {  # 메모리 최적화
        "backbone": "swin_large_patch4_window12_384.ms_in22k_ft_in1k",
        "img_size": 384,  # 320 → 384로 증가
        "batch_size": 14,  # 4 → 14로 3.5배 증가
        "learning_rate": 0.008,
        "weight": 0.15,  # 15%
        "description": "Swin Transformer Large: 안정적 어텐션"
    },
    "efficientnet_b7": {  # 메모리 최적화
        "backbone": "tf_efficientnet_b7.ns_jft_in1k",
        "img_size": 600,  # 512 → 600으로 증가 (원래 최적 크기)
        "batch_size": 8,  # 3 → 8로 2.7배 증가
        "learning_rate": 0.012,
        "weight": 0.15,  # 15%
        "description": "EfficientNet-B7: 검증된 고성능 모델"
    },
    "convnext_base": {  # 메모리 최적화
        "backbone": "convnext_base.fb_in22k_ft_in1k_384", 
        "img_size": 320,  # 256 → 320으로 증가
        "batch_size": 20,  # 6 → 20으로 3.3배 증가
        "learning_rate": 0.015,
        "weight": 0.10,  # 10%
        "description": "ConvNeXt Base: 안정적인 성능"
    },
    "resnet152d": {  # 메모리 최적화
        "backbone": "resnet152d.ra2_in1k",
        "img_size": 256,  # 224 → 256으로 증가
        "batch_size": 24,  # 8 → 24로 3배 증가 (가장 가벼운 모델)
        "learning_rate": 0.02,
        "weight": 0.10,  # 10%
        "description": "ResNet152D: 안정적 잔차 네트워크"
    },
    "vit_base": {  # 메모리 최적화
        "backbone": "vit_base_patch16_384.augreg_in21k_ft_in1k",
        "img_size": 384,  # 320 → 384로 증가
        "batch_size": 16,  # 4 → 16으로 4배 증가
        "learning_rate": 0.008,
        "weight": 0.10,  # 10%
        "description": "Vision Transformer Base: 안정적 어텐션 모델"
    }
}

def create_model_config(base_config, model_info, model_name):
    """모델별 설정 생성 (깊은 복사 사용)"""
    config = copy.deepcopy(base_config)  # 깊은 복사로 중첩 딕셔너리 안전하게 복사
    config['model']['backbone'] = model_info['backbone']
    config['data']['img_size'] = model_info['img_size']
    config['training']['batch_size'] = model_info['batch_size']
    config['training']['learning_rate'] = model_info['learning_rate']
    return config

def cleanup_resources(model=None, train_loader=None, val_loader=None):
    """리소스 정리"""
    try:
        if model is not None:
            del model
        if train_loader is not None:
            del train_loader
        if val_loader is not None:
            del val_loader
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # 가비지 컬렉션
        gc.collect()
        print("🧹 리소스 정리 완료")
    except Exception as e:
        print(f"⚠️ 리소스 정리 중 오류: {e}")

def find_last_checkpoint(save_dir, fold):
    """마지막 체크포인트 찾기"""
    checkpoint_dir = f"{save_dir}/fold_{fold}"
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # best_model.pth가 있으면 해당 에포크부터 재시작
    best_model_path = f"{checkpoint_dir}/best_model.pth"
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        return best_model_path, checkpoint.get('epoch', 0) + 1
    
    return None, 0

def train_single_model(config, model_name, fold, train_df):
    """단일 모델 학습 (안정화 버전)"""
    global cleanup_flag
    
    model_info = ENSEMBLE_MODELS[model_name]
    print(f"\n🚀 {model_name.upper()} 모델 학습 시작 (Fold {fold})")
    print(f"📝 {model_info['description']}")
    print(f"🔧 설정: {model_info['backbone']}, 이미지크기={model_info['img_size']}, 배치={model_info['batch_size']}")
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Apple Silicon GPU (MPS) 사용")
    else:
        device = torch.device('cpu')
        print("⚠️ CPU 사용")
    
    # 출력 디렉토리 생성
    save_dir = f"outputs/ensemble/{model_name}"
    os.makedirs(f"{save_dir}/fold_{fold}", exist_ok=True)
    
    # K-Fold 분할 (개선된 버전)
    print(f"📊 전체 데이터 클래스 분포 확인...")
    unique_classes = train_df['label'].unique()
    print(f"📊 전체 클래스 수: {len(unique_classes)} (범위: {unique_classes.min()}-{unique_classes.max()})")
    
    # 클래스별 샘플 수 확인
    class_counts = train_df['label'].value_counts().sort_index()
    min_samples = class_counts.min()
    print(f"📊 클래스별 최소 샘플 수: {min_samples}")
    
    if min_samples < 5:
        print(f"⚠️ 일부 클래스의 샘플 수가 매우 적습니다 (최소: {min_samples})")
        print("⚠️ 이로 인해 일부 fold에서 클래스가 누락될 수 있습니다.")
    
    # StratifiedKFold with improved parameters
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 각 fold의 클래스 분포 미리 확인
    print(f"🔍 Fold {fold} 클래스 분포 확인 중...")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        if fold_idx == fold:
            val_classes = train_df.iloc[val_idx]['label'].unique()
            train_classes = train_df.iloc[train_idx]['label'].unique()
            
            print(f"📊 Fold {fold} - 학습 데이터 클래스 수: {len(train_classes)}")
            print(f"📊 Fold {fold} - 검증 데이터 클래스 수: {len(val_classes)}")
            
            missing_in_val = set(unique_classes) - set(val_classes)
            missing_in_train = set(unique_classes) - set(train_classes)
            
            if missing_in_val:
                print(f"⚠️ 검증 데이터에 누락된 클래스 수: {len(missing_in_val)}")
                print(f"⚠️ 누락된 클래스 예시: {sorted(list(missing_in_val))[:10]}")
            
            if missing_in_train:
                print(f"⚠️ 학습 데이터에 누락된 클래스 수: {len(missing_in_train)}")
            
            break
    
    model = None
    train_loader = None
    val_loader = None
    
    try:
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
            if fold_idx != fold:
                continue
                
            # 중단 신호 확인
            if cleanup_flag:
                print("🛑 학습 중단됨")
                return None, float('inf')
                
            # 데이터셋 분할
            train_data = train_df.iloc[train_idx].reset_index(drop=True)
            val_data = train_df.iloc[val_idx].reset_index(drop=True)
            
            print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")
            
            # 변환 설정
            img_size = config['data']['img_size']
            train_transform = get_train_transforms(img_size)
            val_transform = get_valid_transforms(img_size)
            
            # 데이터셋 및 로더
            train_dataset = CarDataset(train_data, transform=train_transform, mode='train')
            val_dataset = CarDataset(val_data, transform=val_transform, mode='train')
            
            # 멀티프로세싱 최적화 (14코어 CPU 활용)
            num_workers = 2  # 6 → 2로 감소 (파일 디스크립터 절약)
            use_pin_memory = True  # MPS에서도 pin_memory 활성화
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                drop_last=True,
                persistent_workers=True  # False → True (워커 재사용)
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
                persistent_workers=True  # False → True (워커 재사용)
            )
            
            # 모델 생성
            model = get_model(config).to(device)
            criterion = get_loss_fn(config)
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=config['training']['learning_rate'], 
                weight_decay=0.05
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10,  # 첫 번째 재시작까지의 에포크 수
                T_mult=2,  # 재시작 주기 배수
                eta_min=1e-6  # 최소 학습률
            )
            
            # 학습 루프
            start_epoch = 0
            checkpoint_path, start_epoch = find_last_checkpoint(save_dir, fold)

            if checkpoint_path:
                try:
                    print(f"🔄 체크포인트에서 재시작: {checkpoint_path}")
                    print(f"🔄 시작 에포크: {start_epoch}")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    best_val_loss = checkpoint.get('val_loss', float('inf'))
                    print(f"🔄 이전 최고 성능: {best_val_loss:.4f}")
                except Exception as e:
                    print(f"⚠️ 체크포인트 로딩 실패: {e}")
                    print("🔄 처음부터 새로 시작합니다.")
                    start_epoch = 0
                    best_val_loss = float('inf')
            else:
                print(f"🆕 새로운 모델 학습 시작")
                best_val_loss = float('inf')
            
            best_model_path = f"{save_dir}/fold_{fold}/best_model.pth"
            
            for epoch in range(start_epoch, 100):  # 20 -> 100으로 증가 (0.08점 목표 달성용)
                # 중단 신호 확인
                if cleanup_flag:
                    print("🛑 학습 중단됨")
                    break
                    
                print(f"\n=== Epoch {epoch+1}/100 ===")
                
                # 학습
                model.train()
                train_loss = 0.0
                
                try:
                    for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc='Training')):
                        # 중단 신호 확인
                        if cleanup_flag:
                            print("🛑 배치 처리 중단됨")
                            break
                            
                        images, targets = images.to(device), targets.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        
                        # 메모리 정리 (매 10 배치마다 - 더 자주)
                        if batch_idx % 10 == 0:
                            if torch.backends.mps.is_available():
                                torch.mps.empty_cache()
                            # CPU 메모리도 정리
                            gc.collect()
                    
                    if cleanup_flag:
                        break
                        
                    train_loss /= len(train_loader)
                    
                    # 검증
                    model.eval()
                    val_loss = 0.0
                    all_preds = []
                    all_targets = []
                    
                    with torch.no_grad():
                        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc='Validation')):
                            # 중단 신호 확인
                            if cleanup_flag:
                                break
                                
                            images, targets = images.to(device), targets.to(device)
                            
                            outputs = model(images)
                            loss = criterion(outputs, targets)
                            val_loss += loss.item()
                            
                            probs = torch.softmax(outputs, dim=1)
                            all_preds.append(probs.cpu().numpy())
                            all_targets.append(targets.cpu().numpy())
                            
                            # 검증 중에도 메모리 정리 (매 5 배치마다)
                            if batch_idx % 5 == 0:
                                if torch.backends.mps.is_available():
                                    torch.mps.empty_cache()
                                gc.collect()
                    
                    if cleanup_flag:
                        break
                        
                    val_loss /= len(val_loader)
                    all_preds = np.vstack(all_preds)
                    all_targets = np.concatenate(all_targets)
                    
                    # 메트릭 계산
                    metrics = compute_metrics(all_targets, all_preds, num_classes=config.get("model", {}).get("num_classes", 393))
                    
                    print(f"📊 Train Loss: {train_loss:.4f}")
                    print(f"📊 Val Loss: {val_loss:.4f}, Val Log Loss: {metrics['log_loss']:.4f}")
                    print(f"📊 Val Accuracy: {metrics['accuracy']:.2f}%")
                    
                    # 스케줄러 업데이트
                    scheduler.step()
                    
                    # 최고 모델 저장
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'val_loss': val_loss,
                            'val_log_loss': metrics['log_loss'],
                            'config': config
                        }, best_model_path)
                        print(f"✅ 최고 모델 저장: {best_model_path}")
                        
                except KeyboardInterrupt:
                    print("❌ KeyboardInterrupt 감지됨")
                    cleanup_flag = True
                    break
                except Exception as e:
                    print(f"❌ 학습 중 오류 발생: {e}")
                    print(f"❌ 오류 타입: {type(e).__name__}")
                    import traceback
                    print(f"❌ 상세 오류:\n{traceback.format_exc()}")
                    break
            
            if not cleanup_flag:
                print(f"\n🎉 {model_name} Fold {fold} 학습 완료!")
                print(f"🏆 최고 검증 Loss: {best_val_loss:.4f}")
            
            return best_model_path, best_val_loss
            
    except Exception as e:
        print(f"❌ {model_name} 학습 중 오류: {e}")
        print(f"❌ 오류 타입: {type(e).__name__}")
        import traceback
        print(f"❌ 상세 오류:\n{traceback.format_exc()}")
        return None, float('inf')
    finally:
        # 리소스 정리
        cleanup_resources(model, train_loader, val_loader)

def train_ensemble(base_config_path, fold=0):
    """앙상블 모델들 학습 (안정화 버전)"""
    global cleanup_flag
    
    print("🚀 차량 분류 앙상블 학습 시작!")
    print("🏆 사용자 추천 7개 모델 구성:")
    
    for model_name, info in ENSEMBLE_MODELS.items():
        print(f"  • {model_name}: {info['description']} (가중치: {info['weight']*100}%)")
    
    # 기본 설정 로드
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # 데이터 로드
    train_df_path = os.path.join(base_config['logging']['save_dir'], 'data', 'train_df.csv')
    
    if not os.path.exists(train_df_path):
        print("❌ 학습 데이터가 준비되지 않았습니다.")
        return
    
    train_df = pd.read_csv(train_df_path)
    print(f"📊 데이터 로드 완료: {len(train_df)}개 이미지")
    
    # 앙상블 결과 저장
    ensemble_results = {}
    
    # 각 모델 학습
    for model_name, model_info in ENSEMBLE_MODELS.items():
        if cleanup_flag:
            print("🛑 앙상블 학습 중단됨")
            break
            
        try:
            print(f"\n{'='*50}")
            print(f"🎯 {model_name.upper()} 학습 시작")
            print(f"{'='*50}")
            
            # 모델별 설정 생성
            model_config = create_model_config(base_config, model_info, model_name)
            
            # 모델 학습
            model_path, val_loss = train_single_model(
                model_config, model_name, fold, train_df
            )
            
            # 성공한 모델만 결과에 추가
            if model_path is not None and not cleanup_flag and val_loss != float('inf'):
                ensemble_results[model_name] = {
                    'val_loss': val_loss,
                    'weight': model_info['weight'],
                    'model_path': model_path,
                    'backbone': model_info['backbone'],
                    'description': model_info['description']
                }
                
                print(f"✅ {model_name} 학습 완료: Val Loss = {val_loss:.4f}")
            else:
                print(f"❌ {model_name} 학습 실패 또는 중단됨 - 앙상블에서 제외")
                # 실패한 모델은 ensemble_results에 추가하지 않음
            
        except Exception as e:
            print(f"❌ {model_name} 학습 실패: {e}")
            print(f"❌ {model_name}을 앙상블에서 제외합니다.")
            continue
    
    # 앙상블 결과 저장
    if ensemble_results and not cleanup_flag:
        ensemble_dir = "outputs/ensemble"
        os.makedirs(ensemble_dir, exist_ok=True)
        
        with open(f"{ensemble_dir}/ensemble_results_fold_{fold}.json", 'w') as f:
            json.dump(ensemble_results, f, indent=2)
        
        print("\n🎉 앙상블 학습 완료!")
        print("📊 앙상블 결과:")
        
        total_weight = 0
        for model_name, info in ensemble_results.items():
            print(f"  • {model_name}: Loss={info['val_loss']:.4f}, Weight={info['weight']*100}%")
            total_weight += info['weight']
        
        print(f"\n🎯 총 가중치: {total_weight*100}% (100%가 되어야 함)")
    else:
        print("\n🛑 앙상블 학습이 중단되었거나 실패했습니다.")

def main():
    parser = argparse.ArgumentParser(description='Train Ensemble Car Classification Models - User Recommended (Stable)')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--all_folds', action='store_true')
    args = parser.parse_args()
    
    try:
        if args.all_folds:
            print("🚀 모든 Fold에 대해 앙상블 학습 시작!")
            for fold in range(5):
                if cleanup_flag:
                    break
                print(f"\n{'='*20} FOLD {fold} {'='*20}")
                train_ensemble(args.config, fold)
        else:
            train_ensemble(args.config, args.fold)
    except KeyboardInterrupt:
        print("\n🛑 프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류: {e}")
    finally:
        print("\n🧹 최종 정리 중...")
        cleanup_resources()
        print("✅ 프로그램 종료")

if __name__ == "__main__":
    main() 