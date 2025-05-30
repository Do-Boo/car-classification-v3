"""
하이브리드 로깅 시스템
- 오프라인: TensorBoard + 파일 로깅 + 로컬 시각화
- 온라인: WandB + 클라우드 저장 + 모바일 접근
"""

import os
import json
import logging
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import torch

# 선택적 import (설치되지 않아도 에러 없음)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed. Online features will be disabled.")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Local visualization will be limited.")

class HybridLogger:
    """오프라인/온라인 하이브리드 로깅 시스템"""
    
    def __init__(self, config, experiment_name=None):
        self.config = config
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = Path(config['logging']['save_dir']) / self.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 기록 저장용
        self.metrics_history = []
        self.is_online = self._check_internet_connection()
        
        # 로컬 로깅 설정 (항상 작동)
        self._setup_local_logging()
        
        # 온라인 로깅 설정 (인터넷 연결 시에만)
        self._setup_online_logging()
        
        print(f"🔧 로깅 시스템 초기화 완료")
        print(f"📁 로컬 저장 경로: {self.save_dir}")
        print(f"🌐 온라인 모드: {'✅ 활성화' if self.is_online else '❌ 비활성화'}")
    
    def _check_internet_connection(self):
        """인터넷 연결 상태 확인"""
        try:
            import urllib.request
            urllib.request.urlopen('http://www.google.com', timeout=3)
            return True
        except:
            return False
    
    def _setup_local_logging(self):
        """로컬 로깅 설정 (오프라인에서도 작동)"""
        # 1. 파일 로깅
        log_file = self.save_dir / "training.log"
        
        # 기존 핸들러 제거
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['local']['log_level']),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # 2. TensorBoard (로컬 시각화)
        if TENSORBOARD_AVAILABLE and self.config['logging']['local']['use_tensorboard']:
            tb_dir = self.save_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_dir)
            self.logger.info("📊 TensorBoard 활성화")
        else:
            self.tb_writer = None
            
        # 3. 로컬 플롯 저장 디렉토리
        if self.config['logging']['local']['save_plots']:
            self.plots_dir = self.save_dir / "plots"
            self.plots_dir.mkdir(exist_ok=True)
        
        self.logger.info("🏠 로컬 로깅 시스템 준비 완료")
    
    def _setup_online_logging(self):
        """온라인 로깅 설정 (인터넷 연결 시에만)"""
        self.wandb_run = None
        
        if not self.is_online:
            self.logger.info("🌐 인터넷 연결 없음 - 오프라인 모드로 실행")
            return
            
        if not WANDB_AVAILABLE:
            self.logger.warning("📦 WandB 미설치 - 온라인 기능 비활성화")
            return
            
        if not self.config['logging']['online']['use_wandb']:
            self.logger.info("⚙️ WandB 비활성화 설정")
            return
        
        try:
            # WandB 초기화
            self.wandb_run = wandb.init(
                project=self.config['logging']['online']['project_name'],
                entity=self.config['logging']['online']['entity'],
                name=self.experiment_name,
                config=self.config,
                mode="offline" if not self.is_online else "online"
            )
            self.logger.info("🚀 WandB 온라인 로깅 활성화")
            self.logger.info(f"📱 모바일 접근: {self.wandb_run.url}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ WandB 초기화 실패: {e}")
            self.wandb_run = None
    
    def log_metrics(self, metrics, step=None, prefix=""):
        """메트릭 로깅 (로컬 + 온라인)"""
        timestamp = datetime.now()
        
        # 1. 메모리에 저장 (항상)
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "step": step,
            "metrics": metrics
        }
        self.metrics_history.append(log_entry)
        
        # 2. 콘솔 출력 (항상)
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                                 for k, v in metrics.items()])
        self.logger.info(f"Step {step} | {metrics_str}")
        
        # 3. TensorBoard 로깅 (로컬)
        if self.tb_writer and step is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, step)
        
        # 4. WandB 로깅 (온라인)
        if self.wandb_run:
            try:
                wandb_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
                if step is not None:
                    wandb_metrics["step"] = step
                self.wandb_run.log(wandb_metrics)
            except Exception as e:
                self.logger.warning(f"WandB 로깅 실패: {e}")
    
    def log_image(self, image, caption="", step=None):
        """이미지 로깅"""
        # TensorBoard
        if self.tb_writer and step is not None:
            self.tb_writer.add_image(caption, image, step)
        
        # WandB
        if self.wandb_run:
            try:
                self.wandb_run.log({caption: wandb.Image(image)}, step=step)
            except Exception as e:
                self.logger.warning(f"WandB 이미지 로깅 실패: {e}")
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, filename=None):
        """체크포인트 저장 + 로깅"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = self.save_dir / "checkpoints" / filename
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"💾 체크포인트 저장: {checkpoint_path}")
        
        # WandB에 모델 아티팩트 저장
        if self.wandb_run:
            try:
                artifact = wandb.Artifact(f"model_epoch_{epoch}", type="model")
                artifact.add_file(str(checkpoint_path))
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                self.logger.warning(f"WandB 아티팩트 저장 실패: {e}")
    
    def create_plots(self):
        """로컬 플롯 생성 (오프라인에서도 확인 가능)"""
        if not self.config['logging']['local']['save_plots'] or not self.metrics_history:
            return
        
        # 메트릭 히스토리를 DataFrame으로 변환
        df_data = []
        for entry in self.metrics_history:
            row = {"timestamp": entry["timestamp"], "step": entry["step"]}
            row.update(entry["metrics"])
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # 손실 그래프
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['train_loss'], label='Train Loss', alpha=0.8)
            plt.plot(df['step'], df['val_loss'], label='Val Loss', alpha=0.8)
            plt.xlabel('Step')
            plt.ylabel('Loss')
            plt.title('Training Progress - Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir / "loss_curve.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        # 정확도 그래프
        if 'train_acc' in df.columns and 'val_acc' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df['step'], df['train_acc'], label='Train Accuracy', alpha=0.8)
            plt.plot(df['step'], df['val_acc'], label='Val Accuracy', alpha=0.8)
            plt.xlabel('Step')
            plt.ylabel('Accuracy (%)')
            plt.title('Training Progress - Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.plots_dir / "accuracy_curve.png", dpi=150, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"📊 플롯 저장 완료: {self.plots_dir}")
    
    def save_experiment_summary(self):
        """실험 요약 저장"""
        summary = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "start_time": self.metrics_history[0]["timestamp"] if self.metrics_history else None,
            "end_time": datetime.now().isoformat(),
            "total_steps": len(self.metrics_history),
            "online_mode": self.is_online,
            "wandb_url": self.wandb_run.url if self.wandb_run else None
        }
        
        # 최종 메트릭 추가
        if self.metrics_history:
            summary["final_metrics"] = self.metrics_history[-1]["metrics"]
        
        summary_path = self.save_dir / "experiment_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📋 실험 요약 저장: {summary_path}")
    
    def finish(self):
        """로깅 종료"""
        # 플롯 생성
        self.create_plots()
        
        # 실험 요약 저장
        self.save_experiment_summary()
        
        # TensorBoard 종료
        if self.tb_writer:
            self.tb_writer.close()
        
        # WandB 종료
        if self.wandb_run:
            self.wandb_run.finish()
        
        self.logger.info("🎯 로깅 시스템 종료 완료")
        
        # 접근 방법 안내
        print("\n" + "="*60)
        print("📊 실험 결과 확인 방법:")
        print(f"📁 로컬 파일: {self.save_dir}")
        if self.tb_writer:
            print(f"📈 TensorBoard: tensorboard --logdir {self.save_dir}/tensorboard")
        if self.wandb_run and self.is_online:
            print(f"🌐 WandB 대시보드: {self.wandb_run.url}")
            print(f"📱 모바일에서도 위 링크로 접근 가능!")
        print("="*60)

def create_logger(config, experiment_name=None):
    """로거 생성 헬퍼 함수"""
    return HybridLogger(config, experiment_name) 