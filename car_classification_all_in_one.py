#!/usr/bin/env python
"""
차량 분류 AI 프로젝트 - All-in-One 실행 스크립트
이 스크립트 하나로 전체 프로젝트를 실행할 수 있습니다.

사용법:
1. 이 파일을 car_classification_all_in_one.py로 저장
2. 데이터 준비:
   - data/train/ 폴더에 학습 이미지 배치
   - data/test/ 폴더에 테스트 이미지 배치
   - data/test.csv 파일 배치
3. 실행: python car_classification_all_in_one.py

필요한 패키지:
pip install torch torchvision timm pandas numpy scikit-learn tqdm albumentations opencv-python
"""

import os
import sys
import json
import argparse
import warnings
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

warnings.filterwarnings('ignore')

# ============================================
# 설정 (Configuration)
# ============================================

CONFIG = {
    'data': {
        'train_dir': 'data/train',
        'test_dir': 'data/test',
        'test_csv': 'data/test.csv',
        'img_size': 224,
        'num_classes': 396,
        'class_mapping': {
            "K5_하이브리드_3세대_2020_2023": "K5_3세대_하이브리드_2020_2022",
            "디_올_뉴_니로_2022_2025": "디_올뉴니로_2022_2025",
            "박스터_718_2017_2024": "718_박스터_2017_2024"
        }
    },
    'training': {
        'batch_size': 32,
        'epochs': 10,  # 빠른 테스트를 위해 10 에폭으로 설정
        'num_workers': 4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'amp': True,
        'gradient_accumulation': 1,
        'seed': 42
    },
    'model': {
        'backbone': 'efficientnet_b3',
        'pretrained': True,
        'dropout': 0.3
    },
    'optimizer': {
        'name': 'AdamW',
        'lr': 1e-4,
        'weight_decay': 0.01
    },
    'scheduler': {
        'name': 'CosineAnnealingLR',
        'T_max': 10,
        'eta_min': 1e-6
    },
    'loss': {
        'name': 'CrossEntropyLoss',
        'label_smoothing': 0.1
    },
    'cv': {
        'n_splits': 5,
        'fold': 0  # 실행할 fold 번호
    },
    'tta': {
        'enable': True,
        'num_tta': 5
    }
}

# ============================================
# 유틸리티 함수
# ============================================

def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_output_dirs():
    """출력 디렉토리 생성"""
    dirs = ['outputs', 'outputs/models', 'outputs/submissions', 'outputs/data']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

# ============================================
# 데이터 전처리
# ============================================

def create_train_dataframe(train_dir: str, class_mapping: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    """학습 데이터프레임 생성"""
    print("학습 데이터 준비 중...")
    
    data = []
    class_names = sorted(os.listdir(train_dir))
    
    # 클래스명 매핑 적용
    mapped_classes = []
    for cls in class_names:
        mapped_cls = class_mapping.get(cls, cls)
        mapped_classes.append(mapped_cls)
    unique_classes = sorted(list(set(mapped_classes)))
    class_to_label = {cls: idx for idx, cls in enumerate(unique_classes)}
    
    print(f"총 클래스 수: {len(unique_classes)}")
    
    # 이미지 수집
    for class_name in tqdm(class_names, desc="클래스별 이미지 수집"):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        mapped_class = class_mapping.get(class_name, class_name)
        label = class_to_label[mapped_class]
        
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_file)
                data.append({
                    'img_path': img_path,
                    'class_name': mapped_class,
                    'label': label
                })
    
    df = pd.DataFrame(data)
    print(f"총 이미지 수: {len(df)}")
    
    # 클래스 정보
    class_info = {
        'num_classes': len(unique_classes),
        'class_names': unique_classes,
        'class_to_label': class_to_label,
        'label_to_class': {v: k for k, v in class_to_label.items()}
    }
    
    return df, class_info

# ============================================
# Dataset 클래스
# ============================================

class CarDataset(Dataset):
    """차량 이미지 Dataset"""
    
    def __init__(self, df: pd.DataFrame, transform=None, mode='train'):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 이미지 로드
        image = Image.open(row['img_path']).convert('RGB')
        image = np.array(image)
        
        # 변환 적용
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        if self.mode == 'train':
            return image, row['label']
        else:
            return image, row['img_path']

# ============================================
# 데이터 증강 (Augmentation)
# ============================================

def get_train_transforms(img_size=224):
    """학습용 변환"""
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=15, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(img_size=224):
    """검증/테스트용 변환"""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ============================================
# 모델 정의
# ============================================

class CarClassifier(nn.Module):
    """차량 분류 모델"""
    
    def __init__(self, model_name='efficientnet_b3', num_classes=396, dropout=0.3):
        super().__init__()
        
        # 백본 모델
        self.backbone = timm.create_model(model_name, pretrained=True, drop_rate=dropout)
        
        # 분류 헤드
        in_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

# ============================================
# 손실 함수
# ============================================

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Cross Entropy Loss"""
    
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

# ============================================
# 학습 함수
# ============================================

def train_epoch(model, loader, criterion, optimizer, device, use_amp=True):
    """1 에폭 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    scaler = GradScaler() if use_amp else None
    
    for images, targets in tqdm(loader, desc='Training'):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            # 예측 확률
            probs = torch.softmax(outputs, dim=1)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 메트릭 계산
    all_preds = np.vstack(all_preds)
    all_targets = np.concatenate(all_targets)
    
    val_loss = running_loss / len(loader)
    val_acc = accuracy_score(all_targets, np.argmax(all_preds, axis=1)) * 100
    val_log_loss = log_loss(all_targets, all_preds)
    
    return val_loss, val_acc, val_log_loss

# ============================================
# 추론 함수
# ============================================

def predict_with_tta(model, test_loader, device, num_tta=5):
    """Test Time Augmentation을 사용한 예측"""
    model.eval()
    all_preds = []
    all_paths = []
    
    with torch.no_grad():
        for images, paths in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            
            # TTA 적용 (여기서는 간단히 원본만 사용)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_preds.append(probs.cpu().numpy())
            all_paths.extend(paths)
    
    predictions = np.vstack(all_preds)
    return predictions, all_paths

# ============================================
# 메인 실행 함수
# ============================================

def main():
    """메인 실행 함수"""
    print("="*50)
    print("차량 분류 AI 프로젝트 시작")
    print(f"시작 시간: {datetime.now()}")
    print(f"Device: {CONFIG['training']['device']}")
    print("="*50)
    
    # 시드 설정
    set_seed(CONFIG['training']['seed'])
    
    # 디렉토리 생성
    create_output_dirs()
    
    # 디바이스 설정
    device = torch.device(CONFIG['training']['device'])
    
    # ========== 1. 데이터 준비 ==========
    print("\n[1/4] 데이터 준비")
    
    # 학습 데이터 로드
    train_df, class_info = create_train_dataframe(
        CONFIG['data']['train_dir'],
        CONFIG['data']['class_mapping']
    )
    
    # 데이터 저장
    train_df.to_csv('outputs/data/train_df.csv', index=False)
    with open('outputs/data/class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2)
    
    # K-Fold 분할
    skf = StratifiedKFold(n_splits=CONFIG['cv']['n_splits'], shuffle=True, random_state=CONFIG['training']['seed'])
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
        if fold_idx != CONFIG['cv']['fold']:
            continue
        
        print(f"\n[2/4] Fold {fold_idx} 학습 준비")
        
        # 데이터셋 분할
        train_data = train_df.iloc[train_idx].reset_index(drop=True)
        val_data = train_df.iloc[val_idx].reset_index(drop=True)
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
        
        # 변환 설정
        train_transform = get_train_transforms(CONFIG['data']['img_size'])
        val_transform = get_val_transforms(CONFIG['data']['img_size'])
        
        # 데이터셋 생성
        train_dataset = CarDataset(train_data, transform=train_transform)
        val_dataset = CarDataset(val_data, transform=val_transform)
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG['training']['batch_size'],
            shuffle=True,
            num_workers=CONFIG['training']['num_workers'],
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=CONFIG['training']['batch_size'] * 2,
            shuffle=False,
            num_workers=CONFIG['training']['num_workers'],
            pin_memory=True
        )
        
        # ========== 2. 모델 생성 ==========
        print(f"\n[3/4] 모델 생성: {CONFIG['model']['backbone']}")
        
        model = CarClassifier(
            model_name=CONFIG['model']['backbone'],
            num_classes=CONFIG['data']['num_classes'],
            dropout=CONFIG['model']['dropout']
        ).to(device)
        
        # 손실 함수
        if CONFIG['loss']['label_smoothing'] > 0:
            criterion = LabelSmoothingLoss(
                CONFIG['data']['num_classes'],
                CONFIG['loss']['label_smoothing']
            )
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 옵티마이저
        optimizer = optim.AdamW(
            model.parameters(),
            lr=CONFIG['optimizer']['lr'],
            weight_decay=CONFIG['optimizer']['weight_decay']
        )
        
        # 스케줄러
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG['training']['epochs'],
            eta_min=CONFIG['scheduler']['eta_min']
        )
        
        # ========== 3. 학습 ==========
        print(f"\n[4/4] 학습 시작")
        
        best_log_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(CONFIG['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{CONFIG['training']['epochs']}")
            
            # 학습
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                use_amp=CONFIG['training']['amp']
            )
            
            # 검증
            val_loss, val_acc, val_log_loss = validate(
                model, val_loader, criterion, device
            )
            
            # 스케줄러 업데이트
            scheduler.step()
            
            # 결과 출력
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Log Loss: {val_log_loss:.4f}")
            
            # 모델 저장
            if val_log_loss < best_log_loss:
                best_log_loss = val_log_loss
                best_epoch = epoch + 1
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_log_loss': best_log_loss,
                }, f'outputs/models/fold{fold_idx}_best.pth')
                
                print(f"모델 저장! Best Log Loss: {best_log_loss:.4f}")
        
        print(f"\n학습 완료! Best Epoch: {best_epoch}, Best Log Loss: {best_log_loss:.4f}")
        
        # ========== 4. 테스트 추론 ==========
        print("\n테스트 추론 시작...")
        
        # 테스트 데이터 로드
        test_df = pd.read_csv(CONFIG['data']['test_csv'])
        test_df['img_path'] = test_df['img_path'].apply(lambda x: os.path.join(CONFIG['data']['test_dir'], x))
        
        # 테스트 데이터셋
        test_dataset = CarDataset(test_df, transform=val_transform, mode='test')
        test_loader = DataLoader(
            test_dataset,
            batch_size=CONFIG['training']['batch_size'] * 2,
            shuffle=False,
            num_workers=CONFIG['training']['num_workers'],
            pin_memory=True
        )
        
        # 베스트 모델 로드
        checkpoint = torch.load(f'outputs/models/fold{fold_idx}_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 예측
        predictions, _ = predict_with_tta(model, test_loader, device, num_tta=CONFIG['tta']['num_tta'])
        
        # ========== 5. 제출 파일 생성 ==========
        print("\n제출 파일 생성 중...")
        
        # 제출 DataFrame 생성
        submission = pd.DataFrame({
            'ID': test_df['ID'].values
        })
        
        # 클래스별 확률 추가
        for i, class_name in enumerate(class_info['class_names']):
            submission[class_name] = predictions[:, i]
        
        # 동일 클래스 처리
        for old_class, new_class in CONFIG['data']['class_mapping'].items():
            if old_class in submission.columns and new_class in submission.columns:
                # 이미 매핑되어 있으므로 처리 불필요
                pass
        
        # 저장
        submission.to_csv('outputs/submissions/submission.csv', index=False)
        print(f"제출 파일 저장 완료: outputs/submissions/submission.csv")
        
        # 검증
        print(f"제출 파일 shape: {submission.shape}")
        print(f"확률 합 검증: {submission.iloc[:, 1:].sum(axis=1).mean():.6f}")
        
        break  # 지정된 fold만 실행
    
    print("\n="*50)
    print("프로젝트 완료!")
    print(f"종료 시간: {datetime.now()}")
    print("="*50)

if __name__ == "__main__":
    main()