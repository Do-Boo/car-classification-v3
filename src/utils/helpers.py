"""
학습에 필요한 헬퍼 함수들
"""

import torch
import numpy as np
from pathlib import Path

class EarlyStopping:
    """Early Stopping 클래스"""
    
    def __init__(self, patience=10, mode='max', delta=0.001, verbose=True):
        """
        Args:
            patience: 개선이 없을 때 기다릴 에폭 수
            mode: 'max' (높을수록 좋음) 또는 'min' (낮을수록 좋음)
            delta: 개선으로 인정할 최소 변화량
            verbose: 로그 출력 여부
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda score, best: score > best + delta
        else:
            self.is_better = lambda score, best: score < best - delta
    
    def __call__(self, score):
        """
        Args:
            score: 현재 점수
            
        Returns:
            bool: Early stopping 여부
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"✅ 성능 개선! Best score: {self.best_score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ 개선 없음 ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered!")
        
        return self.early_stop

def save_checkpoint(model, optimizer, epoch, score, filepath, scheduler=None, **kwargs):
    """
    모델 체크포인트 저장
    
    Args:
        model: PyTorch 모델
        optimizer: 옵티마이저
        epoch: 현재 에폭
        score: 현재 점수
        filepath: 저장 경로
        scheduler: 스케줄러 (선택적)
        **kwargs: 추가 정보
    """
    # 디렉토리 생성
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'score': score,
        'model_name': model.__class__.__name__,
    }
    
    # 스케줄러 상태 저장
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # 추가 정보 저장
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, filepath)
    print(f"💾 체크포인트 저장: {filepath}")

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """
    모델 체크포인트 로드
    
    Args:
        filepath: 체크포인트 파일 경로
        model: PyTorch 모델
        optimizer: 옵티마이저 (선택적)
        scheduler: 스케줄러 (선택적)
        device: 디바이스
        
    Returns:
        dict: 체크포인트 정보
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # 모델 상태 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 옵티마이저 상태 로드
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 스케줄러 상태 로드
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"📂 체크포인트 로드: {filepath}")
    print(f"   - Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   - Score: {checkpoint.get('score', 'Unknown')}")
    
    return checkpoint

def set_seed(seed=42):
    """재현성을 위한 시드 설정"""
    import random
    import os
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"🎲 시드 설정 완료: {seed}")

def count_parameters(model):
    """모델 파라미터 수 계산"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 모델 파라미터:")
    print(f"   - 전체: {total_params:,}")
    print(f"   - 학습 가능: {trainable_params:,}")
    
    return total_params, trainable_params

def get_lr(optimizer):
    """현재 학습률 가져오기"""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_predictions(predictions, ids, filepath, class_names=None):
    """예측 결과 저장"""
    import pandas as pd
    
    # DataFrame 생성
    if class_names is not None:
        df = pd.DataFrame(predictions, columns=class_names)
        df.insert(0, 'ID', ids)
    else:
        df = pd.DataFrame({'ID': ids})
        for i in range(predictions.shape[1]):
            df[f'class_{i}'] = predictions[:, i]
    
    # 저장
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"📄 예측 결과 저장: {filepath}")
    
    return df 