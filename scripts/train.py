"""
메인 학습 스크립트 - 간소화 및 안정화 버전
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import signal
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import CarDataset, get_train_transforms, get_valid_transforms
from src.models.backbone import get_model
from src.training.losses import get_loss_fn
from src.utils.metrics import compute_metrics

# 전역 변수로 정리 플래그 설정
cleanup_flag = False

def signal_handler(signum, frame):
    """KeyboardInterrupt 처리"""
    global cleanup_flag
    print("\n🛑 학습 중단 신호를 받았습니다. 안전하게 정리 중...")
    cleanup_flag = True

# 시그널 핸들러 등록
signal.signal(signal.SIGINT, signal_handler)

def get_optimizer(model, config):
    """옵티마이저 생성"""
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer, config):
    """스케줄러 생성"""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

def train_epoch(model, loader, criterion, optimizer, device):
    """1 에폭 학습"""
    global cleanup_flag
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, targets) in enumerate(pbar):
        # 중단 신호 확인
        if cleanup_flag:
            print("🛑 학습 중단됨")
            break
            
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 메모리 정리 (매 100 배치마다)
        if batch_idx % 100 == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    """검증"""
    global cleanup_flag
    
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation'):
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
    
    if cleanup_flag:
        return {'loss': float('inf'), 'accuracy': 0.0, 'log_loss': float('inf')}
    
    val_loss /= len(loader)
    all_preds = np.vstack(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # 메트릭 계산
    metrics = compute_metrics(all_targets, all_preds, num_classes=396)
    
    return {
        'loss': val_loss,
        'accuracy': metrics['accuracy'],
        'log_loss': metrics['log_loss']
    }

def train_fold(config, fold, train_df):
    """단일 Fold 학습"""
    global cleanup_flag
    
    print(f"\n🚀 Fold {fold} 학습 시작")
    
    # 디바이스 설정
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Apple Silicon GPU (MPS) 사용")
    else:
        device = torch.device('cpu')
        print("⚠️ CPU 사용")
    
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
        
        if fold_idx != fold:
            continue
        
        # 중단 신호 확인
        if cleanup_flag:
            print("🛑 학습 중단됨")
            return
        
        # 데이터 분할
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"📊 Train: {len(train_data)}, Val: {len(val_data)}")
        
        # 데이터 변환
        img_size = config['data']['img_size']
        train_transform = get_train_transforms(img_size)
        val_transform = get_valid_transforms(img_size)
        
        # 데이터셋
        train_dataset = CarDataset(train_data, transform=train_transform, mode='train')
        val_dataset = CarDataset(val_data, transform=val_transform, mode='train')
        
        # macOS에서 안정성을 위해 num_workers 조정
        num_workers = 0 if device.type == 'mps' else 2
        use_pin_memory = device.type != 'mps'
        
        # 데이터로더
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            drop_last=True,
            persistent_workers=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=False
        )
        
        # 모델 생성
        model = get_model(config).to(device)
        criterion = get_loss_fn(config)
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)
        
        # 학습 루프
        best_log_loss = float('inf')
        save_dir = config['logging']['save_dir']
        os.makedirs(f"{save_dir}/fold_{fold}", exist_ok=True)
        
        try:
            for epoch in range(config['training']['epochs']):
                # 중단 신호 확인
                if cleanup_flag:
                    print("🛑 학습 중단됨")
                    break
                    
                print(f"\n=== Epoch {epoch+1}/{config['training']['epochs']} ===")
                
                # 학습
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
                
                if cleanup_flag:
                    break
                
                # 검증
                val_metrics = validate(model, val_loader, criterion, device)
                
                if cleanup_flag:
                    break
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
                print(f"Val Log Loss: {val_metrics['log_loss']:.4f}")
                
                # 스케줄러 업데이트
                scheduler.step(val_metrics['loss'])
                
                # 최고 모델 저장
                if val_metrics['log_loss'] < best_log_loss:
                    best_log_loss = val_metrics['log_loss']
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_log_loss': val_metrics['log_loss'],
                        'config': config
                    }, f"{save_dir}/fold_{fold}/best_model.pth")
                    print(f"✅ 최고 모델 저장: Log Loss = {best_log_loss:.4f}")
            
            if not cleanup_flag:
                print(f"\n🎉 Fold {fold} 학습 완료! 최고 Log Loss: {best_log_loss:.4f}")
                
        except KeyboardInterrupt:
            print("🛑 KeyboardInterrupt 감지됨")
            cleanup_flag = True
        except Exception as e:
            print(f"❌ 학습 중 오류: {e}")
        finally:
            # 리소스 정리
            del model, train_loader, val_loader
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
        
        break

def main():
    parser = argparse.ArgumentParser(description='Train Car Classification Model (Stable)')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--fold', type=int, default=0)
    args = parser.parse_args()
    
    try:
        # 설정 로드
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # 데이터 로드
        train_df_path = os.path.join(config['logging']['save_dir'], 'data', 'train_df.csv')
        
        if not os.path.exists(train_df_path):
            print("❌ 학습 데이터가 준비되지 않았습니다.")
            return
        
        train_df = pd.read_csv(train_df_path)
        print(f"📊 데이터 로드 완료: {len(train_df)}개 이미지")
        
        # 학습 시작
        train_fold(config, args.fold, train_df)
        
    except KeyboardInterrupt:
        print("\n🛑 프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 프로그램 실행 중 오류: {e}")
    finally:
        print("\n🧹 최종 정리 중...")
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        print("✅ 프로그램 종료")

if __name__ == "__main__":
    main()
