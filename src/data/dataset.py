"""
PyTorch Dataset 클래스 구현
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class CarDataset(Dataset):
    """차량 이미지 분류를 위한 Dataset 클래스"""
    
    def __init__(self, df, transform=None, mode='train'):
        """
        Args:
            df: DataFrame with columns ['img_path', 'label']
            transform: Albumentations transform
            mode: 'train' or 'test'
        """
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
        else:
            # 기본 변환 (텐서 변환만)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
        if self.mode == 'train':
            label = row['label']
            return image, label
        else:
            # 테스트 모드에서는 이미지 경로도 반환
            return image, row.get('img_path', idx)

def get_train_transforms(size=384):
    """학습용 데이터 변환 (안전한 버전)"""
    
    # size가 튜플이나 리스트인 경우 정수로 변환
    if isinstance(size, (list, tuple)):
        size = size[0]
    size = int(size)
        
    return A.Compose([
        # 크기 조정 (안전한 방법)
        A.Resize(height=size, width=size),
        A.RandomCrop(height=size, width=size, p=0.3),
        
        # 기본 증강
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        
        # 색상 증강 (간소화)
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        
        # 정규화
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

def get_valid_transforms(img_size=224):
    """검증용 변환 파이프라인 (안전한 버전)"""
    # img_size가 리스트나 튜플인 경우 첫 번째 값 사용
    if isinstance(img_size, (list, tuple)):
        size = img_size[0]
    else:
        size = img_size
    size = int(size)
        
    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

def get_tta_transforms(img_size=224):
    """Test Time Augmentation을 위한 변환 리스트"""
    # img_size가 리스트나 튜플인 경우 첫 번째 값 사용
    if isinstance(img_size, (list, tuple)):
        size = img_size[0]
    else:
        size = img_size
        
    transforms = []
    
    # 1. 원본
    transforms.append(get_valid_transforms(img_size))
    
    # 2. 좌우 반전
    transforms.append(A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]))
    
    # 3. 밝기 변경
    for brightness in [0.9, 1.1]:
        transforms.append(A.Compose([
            A.Resize(size, size),
            A.ColorJitter(brightness=brightness-1, contrast=0, saturation=0, hue=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
    
    # 4. 작은 회전
    for angle in [-5, 5]:
        transforms.append(A.Compose([
            A.Resize(size, size),
            A.Rotate(limit=angle, border_mode=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
    
    return transforms
