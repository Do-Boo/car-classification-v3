#!/usr/bin/env python3
"""
🚀 SOTA 앙상블 체크포인트 관리 유틸리티
고성능 학습을 위한 체크포인트 저장/로드/관리 시스템

생성일: 2025-06-02 17:04
목표: 안정적인 장시간 학습 지원
"""

import torch
import json
import glob
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, filepath, **kwargs):
    """
    🔄 SOTA 체크포인트 저장 (완전한 학습 상태 보존)
    
    Args:
        model: PyTorch 모델
        optimizer: 옵티마이저
        scheduler: 학습률 스케줄러
        epoch: 현재 에포크
        metrics: 성능 메트릭 딕셔너리
        filepath: 저장 경로
        **kwargs: 추가 정보 (config, model_name 등)
    """
    # 디렉토리 생성
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'model_name': model.__class__.__name__,
        'timestamp': str(Path(__file__).stat().st_mtime),  # 저장 시간
    }
    
    # 추가 정보 저장
    checkpoint.update(kwargs)
    
    # 체크포인트 저장
    torch.save(checkpoint, filepath)
    print(f"💾 체크포인트 저장: {filepath}")
    print(f"   📊 Epoch: {epoch}")
    print(f"   🎯 Metrics: {metrics}")
    
    return checkpoint

def load_checkpoint(filepath, model, optimizer=None, scheduler=None, device='cpu'):
    """
    📂 SOTA 체크포인트 로드 (완전한 학습 상태 복원)
    
    Args:
        filepath: 체크포인트 파일 경로
        model: PyTorch 모델
        optimizer: 옵티마이저 (선택적)
        scheduler: 스케줄러 (선택적)
        device: 디바이스
        
    Returns:
        tuple: (model, optimizer, scheduler, start_epoch)
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"❌ 체크포인트 파일을 찾을 수 없습니다: {filepath}")
    
    print(f"📂 체크포인트 로드 중: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)
    
    # 모델 상태 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 모델 상태 로드 완료")
    
    # 옵티마이저 상태 로드
    start_epoch = checkpoint.get('epoch', 0)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 옵티마이저 상태 로드 완료")
    
    # 스케줄러 상태 로드
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ 스케줄러 상태 로드 완료")
    
    # 메트릭 정보 출력
    metrics = checkpoint.get('metrics', {})
    print(f"📊 이전 성능:")
    for key, value in metrics.items():
        print(f"   • {key}: {value}")
    
    print(f"🔄 Epoch {start_epoch}부터 학습 재개")
    
    return model, optimizer, scheduler, start_epoch

def find_last_checkpoint(save_dir: Path, fold: int) -> Tuple[Optional[str], int]:
    """
    🔍 마지막 체크포인트 자동 탐지 (중단된 학습 자동 재개)
    
    Args:
        save_dir: 체크포인트 저장 디렉토리
        fold: K-Fold 번호
        
    Returns:
        tuple: (checkpoint_path, start_epoch)
    """
    save_dir = Path(save_dir)
    
    # 가능한 체크포인트 패턴들
    patterns = [
        f"best_fold_{fold}.pth",           # 최고 성능 모델
        f"checkpoint_fold_{fold}_*.pth",   # 에포크별 체크포인트
        f"last_fold_{fold}.pth",           # 마지막 체크포인트
    ]
    
    checkpoint_files = []
    
    # 모든 패턴으로 파일 검색
    for pattern in patterns:
        files = list(save_dir.glob(pattern))
        checkpoint_files.extend(files)
    
    if not checkpoint_files:
        print(f"📝 새로운 학습 시작 (Fold {fold})")
        return None, 0
    
    # 가장 최근 파일 선택 (수정 시간 기준)
    latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
    
    try:
        # 체크포인트에서 에포크 정보 추출
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        start_epoch = checkpoint.get('epoch', 0)
        
        print(f"🔄 체크포인트 발견: {latest_checkpoint}")
        print(f"📊 Epoch {start_epoch}부터 재개")
        
        return str(latest_checkpoint), start_epoch
        
    except Exception as e:
        print(f"⚠️ 체크포인트 로드 실패: {e}")
        print(f"📝 새로운 학습 시작")
        return None, 0

def cleanup_old_checkpoints(save_dir: Path, fold: int, keep_best: bool = True, keep_last: int = 3):
    """
    🧹 오래된 체크포인트 정리 (디스크 공간 절약)
    
    Args:
        save_dir: 체크포인트 디렉토리
        fold: K-Fold 번호
        keep_best: 최고 성능 모델 유지 여부
        keep_last: 유지할 최근 체크포인트 수
    """
    save_dir = Path(save_dir)
    
    # 에포크별 체크포인트 파일들 찾기
    pattern = f"checkpoint_fold_{fold}_epoch_*.pth"
    checkpoint_files = list(save_dir.glob(pattern))
    
    if len(checkpoint_files) <= keep_last:
        return  # 정리할 파일이 충분하지 않음
    
    # 수정 시간 기준으로 정렬 (최신 순)
    checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # 삭제할 파일들 (최근 N개 제외)
    files_to_delete = checkpoint_files[keep_last:]
    
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            print(f"🗑️ 오래된 체크포인트 삭제: {file_path.name}")
        except Exception as e:
            print(f"⚠️ 파일 삭제 실패: {file_path.name} - {e}")
    
    print(f"✅ 체크포인트 정리 완료 (최근 {keep_last}개 유지)")

def save_ensemble_checkpoint(ensemble_results: Dict[str, Any], fold: int, output_dir: str = "outputs/ensemble"):
    """
    🏆 앙상블 결과 체크포인트 저장
    
    Args:
        ensemble_results: 앙상블 모델 결과 딕셔너리
        fold: K-Fold 번호
        output_dir: 출력 디렉토리
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 형태로 앙상블 결과 저장
    results_path = output_dir / f"ensemble_results_fold_{fold}.json"
    
    # JSON 직렬화 가능한 형태로 변환
    serializable_results = {}
    for model_name, result in ensemble_results.items():
        serializable_results[model_name] = {
            'model_name': result['model_name'],
            'model_path': result['model_path'],
            'val_loss': float(result['val_loss']),
            'weight': float(result['weight']),
            'description': result['description'],
            # config는 너무 크므로 제외
        }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"🏆 앙상블 결과 저장: {results_path}")
    print(f"📊 모델 수: {len(ensemble_results)}")
    
    return results_path

def load_ensemble_checkpoint(fold: int, output_dir: str = "outputs/ensemble") -> Optional[Dict[str, Any]]:
    """
    📂 앙상블 결과 체크포인트 로드
    
    Args:
        fold: K-Fold 번호
        output_dir: 출력 디렉토리
        
    Returns:
        dict: 앙상블 결과 딕셔너리 또는 None
    """
    results_path = Path(output_dir) / f"ensemble_results_fold_{fold}.json"
    
    if not results_path.exists():
        print(f"📝 앙상블 결과 파일이 없습니다: {results_path}")
        return None
    
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            ensemble_results = json.load(f)
        
        print(f"📂 앙상블 결과 로드: {results_path}")
        print(f"📊 모델 수: {len(ensemble_results)}")
        
        return ensemble_results
        
    except Exception as e:
        print(f"❌ 앙상블 결과 로드 실패: {e}")
        return None

def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    📊 체크포인트 정보 조회
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        
    Returns:
        dict: 체크포인트 정보
    """
    if not Path(checkpoint_path).exists():
        return {"error": "파일이 존재하지 않습니다"}
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            "epoch": checkpoint.get('epoch', 'Unknown'),
            "model_name": checkpoint.get('model_name', 'Unknown'),
            "metrics": checkpoint.get('metrics', {}),
            "file_size": f"{Path(checkpoint_path).stat().st_size / (1024*1024):.1f} MB",
            "timestamp": checkpoint.get('timestamp', 'Unknown'),
        }
        
        return info
        
    except Exception as e:
        return {"error": f"체크포인트 로드 실패: {e}"}

if __name__ == "__main__":
    # 테스트 코드
    print("🧪 체크포인트 유틸리티 테스트")
    
    # 테스트 디렉토리
    test_dir = Path("test_checkpoints")
    test_dir.mkdir(exist_ok=True)
    
    # find_last_checkpoint 테스트
    checkpoint_path, start_epoch = find_last_checkpoint(test_dir, fold=0)
    print(f"결과: {checkpoint_path}, {start_epoch}")
    
    # 테스트 디렉토리 정리
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)
    
    print("✅ 테스트 완료!") 