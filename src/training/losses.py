"""
다양한 손실 함수 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LabelSmoothingLoss(nn.Module):
    """Label Smoothing을 적용한 Cross Entropy Loss"""
    
    def __init__(self, classes, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.classes = classes
        self.dim = -1
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            # alpha가 float인 경우 처리
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                # alpha가 tensor인 경우
                if self.alpha.type() != inputs.data.type():
                    self.alpha = self.alpha.type_as(inputs.data)
                alpha_t = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()

class MixUpCrossEntropyLoss(nn.Module):
    """MixUp/CutMix를 위한 Cross Entropy Loss"""
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred, targets_a, targets_b, lam):
        loss_a = F.cross_entropy(pred, targets_a, reduction=self.reduction)
        loss_b = F.cross_entropy(pred, targets_b, reduction=self.reduction)
        return lam * loss_a + (1 - lam) * loss_b

class BiTemperedLogisticLoss(nn.Module):
    """Bi-Tempered Logistic Loss for noise robustness"""
    
    def __init__(self, t1=1.0, t2=1.0, reduction='mean'):
        super().__init__()
        self.t1 = t1
        self.t2 = t2
        self.reduction = reduction
        
    def forward(self, logits, labels):
        # Compute normalization
        if self.t2 == 1.0:
            normalization = torch.logsumexp(logits, dim=1, keepdim=True)
            exp_logits = torch.exp(logits - normalization)
        else:
            # Compute Z(y) for t2 != 1
            normalization = ((1 - self.t2) * logits).logsumexp(dim=1, keepdim=True) / (1 - self.t2)
            exp_logits = torch.exp((logits - normalization) * (1 - self.t2))
            
        # Compute tempered softmax
        if self.t1 == 1.0:
            probabilities = exp_logits / exp_logits.sum(dim=1, keepdim=True)
        else:
            probabilities = torch.pow(exp_logits, 1 / self.t1)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            
        # Compute loss
        labels_one_hot = F.one_hot(labels, num_classes=logits.size(1)).float()
        loss = -torch.sum(labels_one_hot * torch.log(probabilities + 1e-10), dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def get_loss_fn(config, class_weights=None):
    """설정에 따라 손실 함수 반환"""
    # loss 설정이 training 하위에 있는 경우 처리
    if 'loss' in config:
        loss_config = config['loss']
    elif 'training' in config and 'loss' in config['training']:
        loss_config = config['training']['loss']
    else:
        raise ValueError("Loss configuration not found in config")
    
    loss_name = loss_config['type']
    
    if loss_name == 'CrossEntropyLoss':
        label_smoothing = loss_config.get('label_smoothing', 0.0)
        if label_smoothing > 0:
            return LabelSmoothingLoss(
                classes=config['data']['num_classes'],
                smoothing=label_smoothing
            )
        else:
            return nn.CrossEntropyLoss(weight=class_weights)
            
    elif loss_name == 'FocalLoss':
        alpha = loss_config.get('alpha', 1.0)
        gamma = loss_config.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
        
    elif loss_name == 'BiTemperedLoss':
        t1 = loss_config.get('t1', 1.0)
        t2 = loss_config.get('t2', 1.0)
        return BiTemperedLogisticLoss(t1=t1, t2=t2)
        
    elif loss_name == 'LabelSmoothingLoss':
        smoothing = loss_config.get('label_smoothing', 0.1)
        return LabelSmoothingLoss(
            classes=config['data']['num_classes'],
            smoothing=smoothing
        )
        
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
