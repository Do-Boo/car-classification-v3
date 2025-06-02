"""
평가 메트릭 함수들
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score

def compute_metrics(y_true, y_pred_probs, num_classes=393):
    """
    다양한 메트릭 계산
    
    Args:
        y_true: 실제 레이블 (N,)
        y_pred_probs: 예측 확률 (N, num_classes)
        num_classes: 전체 클래스 수
        
    Returns:
        dict: 메트릭 딕셔너리
    """
    # 🔧 레이블이 1-based인 경우 0-based로 변환
    if y_true.min() > 0:
        print(f"⚠️ 레이블이 1-based입니다. 0-based로 변환합니다. (min: {y_true.min()}, max: {y_true.max()})")
        y_true = y_true - 1  # 1-based → 0-based 변환
    
    # 예측 클래스
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 전체 클래스 레이블 생성 (0부터 num_classes-1까지)
    all_labels = list(range(num_classes))
    
    # 🔧 검증 데이터에 누락된 클래스 처리
    unique_true_labels = np.unique(y_true)
    print(f"📊 검증 데이터 클래스 수: {len(unique_true_labels)}/{num_classes}")
    print(f"📊 검증 데이터 클래스 범위: {unique_true_labels.min()}-{unique_true_labels.max()}")
    
    # 🔧 안전한 log_loss 계산
    try:
        # 방법 1: 전체 클래스 레이블 명시
        computed_log_loss = log_loss(y_true, y_pred_probs, labels=all_labels)
        print(f"✅ Log Loss 계산 성공 (방법 1): {computed_log_loss:.6f}")
    except Exception as e:
        print(f"⚠️ 방법 1 실패: {e}")
        try:
            # 방법 2: 검증 데이터에 있는 클래스만 사용
            # 누락된 클래스의 확률을 0으로 설정하고 정규화
            y_pred_probs_safe = y_pred_probs.copy()
            
            # 각 샘플의 확률 합이 1이 되도록 정규화
            row_sums = y_pred_probs_safe.sum(axis=1, keepdims=True)
            y_pred_probs_safe = y_pred_probs_safe / row_sums
            
            # 매우 작은 값으로 클리핑 (log(0) 방지)
            y_pred_probs_safe = np.clip(y_pred_probs_safe, 1e-15, 1 - 1e-15)
            
            # 검증 데이터에 있는 클래스만으로 log_loss 계산
            computed_log_loss = log_loss(y_true, y_pred_probs_safe, labels=all_labels)
            print(f"✅ Log Loss 계산 성공 (방법 2): {computed_log_loss:.6f}")
        except Exception as e2:
            print(f"⚠️ 방법 2도 실패: {e2}")
            # 방법 3: 수동 계산
            y_pred_probs_clipped = np.clip(y_pred_probs, 1e-15, 1 - 1e-15)
            log_probs = np.log(y_pred_probs_clipped)
            computed_log_loss = -np.mean([log_probs[i, y_true[i]] for i in range(len(y_true))])
            print(f"✅ Log Loss 계산 성공 (방법 3 - 수동): {computed_log_loss:.6f}")
    
    # 메트릭 계산
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'log_loss': computed_log_loss,
        'top3_accuracy': top_k_accuracy_score(y_true, y_pred_probs, k=3) * 100,
        'top5_accuracy': top_k_accuracy_score(y_true, y_pred_probs, k=5) * 100,
    }
    
    return metrics

def calculate_class_weights(labels, num_classes):
    """
    클래스 가중치 계산 (불균형 데이터 대응)
    
    Args:
        labels: 레이블 배열
        num_classes: 전체 클래스 수
        
    Returns:
        numpy array: 클래스별 가중치
    """
    # 클래스별 빈도 계산
    class_counts = np.bincount(labels, minlength=num_classes)
    
    # 0인 클래스 처리
    class_counts[class_counts == 0] = 1
    
    # 가중치 계산 (inverse frequency)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    
    # 정규화
    class_weights = class_weights / class_weights.mean()
    
    return class_weights

def multiclass_log_loss_with_mapping(answer_df, submission_df, class_mapping=None):
    """
    대회 평가 함수 (클래스 매핑 포함)
    
    Args:
        answer_df: 정답 DataFrame (ID, label)
        submission_df: 제출 DataFrame (ID, 각 클래스별 확률)
        class_mapping: 동일 클래스 매핑 딕셔너리
        
    Returns:
        float: Log Loss
    """
    # 클래스 리스트
    class_list = sorted(answer_df['label'].unique())
    
    # 매핑 적용
    if class_mapping:
        # 정답 레이블 매핑
        answer_df = answer_df.copy()
        answer_df['label'] = answer_df['label'].map(lambda x: class_mapping.get(x, x))
        
        # 제출 DataFrame 컬럼 매핑
        submission_df = submission_df.copy()
        for old_class, new_class in class_mapping.items():
            if old_class in submission_df.columns and new_class in submission_df.columns:
                # 두 클래스의 확률을 합산
                submission_df[new_class] = submission_df[new_class] + submission_df[old_class]
                submission_df = submission_df.drop(columns=[old_class])
        
        # 클래스 리스트 업데이트
        class_list = sorted(answer_df['label'].unique())
    
    # 검증
    if submission_df.shape[0] != answer_df.shape[0]:
        raise ValueError("submission_df 행 개수가 answer_df와 일치하지 않습니다.")
    
    # ID 정렬
    submission_df = submission_df.sort_values(by='ID').reset_index(drop=True)
    answer_df = answer_df.sort_values(by='ID').reset_index(drop=True)
    
    if not all(answer_df['ID'] == submission_df['ID']):
        raise ValueError("ID가 정렬되지 않았거나 불일치합니다.")
    
    # 누락된 클래스 확인
    missing_cols = [col for col in class_list if col not in submission_df.columns]
    if missing_cols:
        raise ValueError(f"클래스 컬럼 누락: {missing_cols}")
    
    # NaN 확인
    if submission_df[class_list].isnull().any().any():
        raise ValueError("NaN 포함됨")
    
    # 확률 범위 확인
    for col in class_list:
        if not ((submission_df[col] >= 0) & (submission_df[col] <= 1)).all():
            raise ValueError(f"{col}의 확률값이 0~1 범위 초과")
    
    # 정답 인덱스 변환
    true_labels = answer_df['label'].tolist()
    true_idx = [class_list.index(lbl) for lbl in true_labels]
    
    # 확률 정규화 + clip
    probs = submission_df[class_list].values
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_pred = np.clip(probs, 1e-15, 1 - 1e-15)
    
    return log_loss(true_idx, y_pred, labels=list(range(len(class_list))))
