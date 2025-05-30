"""
데이터 전처리 유틸리티
- 학습 데이터 DataFrame 생성
- 클래스 매핑 처리
- 데이터 검증
"""

import os
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import json

def create_train_dataframe(train_dir, class_mapping=None):
    """
    학습 데이터프레임 생성
    
    Args:
        train_dir: 학습 데이터 디렉토리 경로
        class_mapping: 동일 클래스 매핑 딕셔너리
        
    Returns:
        DataFrame: ['img_path', 'class_name', 'label']
    """
    data = []
    class_names = sorted(os.listdir(train_dir))
    
    # 클래스명을 label로 매핑
    if class_mapping:
        # 매핑 적용
        mapped_classes = []
        for cls in class_names:
            mapped_cls = class_mapping.get(cls, cls)
            mapped_classes.append(mapped_cls)
        unique_classes = sorted(list(set(mapped_classes)))
        class_to_label = {cls: idx for idx, cls in enumerate(unique_classes)}
    else:
        unique_classes = class_names
        class_to_label = {cls: idx for idx, cls in enumerate(class_names)}
    
    print(f"총 클래스 수: {len(unique_classes)}")
    
    # 각 클래스별로 이미지 파일 수집
    for class_name in tqdm(class_names, desc="클래스별 이미지 수집"):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # 매핑된 클래스명 가져오기
        mapped_class = class_mapping.get(class_name, class_name) if class_mapping else class_name
        label = class_to_label[mapped_class]
        
        # 이미지 파일 수집
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_dir, img_file)
                data.append({
                    'img_path': img_path,
                    'class_name': mapped_class,
                    'original_class': class_name,
                    'label': label
                })
    
    df = pd.DataFrame(data)
    
    # 클래스별 이미지 수 출력
    class_counts = df['class_name'].value_counts()
    print(f"\n클래스별 이미지 수 (상위 10개):")
    print(class_counts.head(10))
    print(f"\n최대 이미지 수: {class_counts.max()}")
    print(f"최소 이미지 수: {class_counts.min()}")
    print(f"평균 이미지 수: {class_counts.mean():.2f}")
    print(f"불균형 비율 (max/min): {class_counts.max()/class_counts.min():.2f}")
    
    # 클래스 정보 저장
    class_info = {
        'num_classes': len(unique_classes),
        'class_names': unique_classes,
        'class_to_label': class_to_label,
        'label_to_class': {v: k for k, v in class_to_label.items()},
        'class_mapping': class_mapping if class_mapping else {}
    }
    
    return df, class_info

def verify_images(df, remove_corrupted=True):
    """
    이미지 파일 검증
    
    Args:
        df: 이미지 경로가 포함된 DataFrame
        remove_corrupted: 손상된 이미지 제거 여부
        
    Returns:
        DataFrame: 검증된 이미지만 포함된 DataFrame
    """
    print("\n이미지 검증 시작...")
    corrupted = []
    small_images = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="이미지 검증"):
        try:
            img = Image.open(row['img_path'])
            img.verify()  # 이미지 파일 검증
            
            # 이미지 크기 확인
            img = Image.open(row['img_path'])
            if img.size[0] < 50 or img.size[1] < 50:
                small_images.append(idx)
                
        except Exception as e:
            corrupted.append(idx)
    
    print(f"\n손상된 이미지: {len(corrupted)}개")
    print(f"너무 작은 이미지 (<50x50): {len(small_images)}개")
    
    if remove_corrupted:
        remove_indices = list(set(corrupted + small_images))
        df_clean = df.drop(remove_indices).reset_index(drop=True)
        print(f"제거 후 총 이미지 수: {len(df_clean)}")
        return df_clean
    
    return df

def save_data_info(df, class_info, output_dir):
    """
    데이터 정보 저장
    
    Args:
        df: 학습 데이터 DataFrame
        class_info: 클래스 정보 딕셔너리
        output_dir: 저장 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # DataFrame 저장
    df.to_csv(os.path.join(output_dir, 'train_df.csv'), index=False)
    
    # 클래스 정보 저장
    with open(os.path.join(output_dir, 'class_info.json'), 'w', encoding='utf-8') as f:
        json.dump(class_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n데이터 정보 저장 완료: {output_dir}")

if __name__ == "__main__":
    # 설정
    train_dir = "data/train"
    output_dir = "outputs/data"
    
    # 클래스 매핑 정의
    class_mapping = {
        "K5_하이브리드_3세대_2020_2023": "K5_3세대_하이브리드_2020_2022",
        "디_올_뉴_니로_2022_2025": "디_올뉴니로_2022_2025",
        "박스터_718_2017_2024": "718_박스터_2017_2024"
    }
    
    # 데이터프레임 생성
    df, class_info = create_train_dataframe(train_dir, class_mapping)
    
    # 이미지 검증
    df = verify_images(df, remove_corrupted=True)
    
    # 저장
    save_data_info(df, class_info, output_dir)
