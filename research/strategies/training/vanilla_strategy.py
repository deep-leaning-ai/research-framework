"""
VanillaTrainingStrategy: 순수 PyTorch 학습 전략 (Lightning 없이)
"""
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ...core.strategies import TrainingStrategy, LoggingStrategy
from ...strategies.task.base import TaskStrategy


class VanillaTrainingStrategy(TrainingStrategy):
    """
    순수 PyTorch 학습 전략
    PyTorch Lightning 없이 직접 학습 루프 구현

    주요 개선사항:
    - TaskStrategy 통합으로 다양한 태스크 지원 (multiclass, binary, regression)
    - LoggingStrategy 통합으로 유연한 로깅
    - GPU 동기화로 정확한 타이밍 측정
    - Learning rate scheduling 지원
    - Early stopping 지원
    - Gradient clipping 지원
    """

    def __init__(self,
                 device: str = None,
                 task_strategy: Optional[TaskStrategy] = None,
                 logging_strategy: Optional[LoggingStrategy] = None):
        """
        Args:
            device: 사용할 디바이스 ('cuda', 'cpu', None=자동선택)
            task_strategy: Task 전략 (None일 경우 CrossEntropyLoss 사용)
            logging_strategy: 로깅 전략 (None일 경우 print 사용)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.task_strategy = task_strategy
        self.logging_strategy = logging_strategy

        self._log(f"[OK] Using device: {self.device}")

    def _log(self, message: str):
        """로깅 헬퍼 메서드"""
        if self.logging_strategy:
            # LoggingStrategy는 dict를 받으므로 메시지를 직접 출력
            print(message)
        else:
            print(message)

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """순수 PyTorch 학습 루프"""

        # 설정 추출
        max_epochs = config.get('max_epochs', 10)
        learning_rate = config.get('learning_rate', 1e-3)
        optimizer_type = config.get('optimizer', 'adam').lower()
        use_scheduler = config.get('use_scheduler', False)
        use_early_stopping = config.get('use_early_stopping', False)
        patience = config.get('patience', 5)
        grad_clip_max_norm = config.get('grad_clip_max_norm', None)

        # 모델을 디바이스로 이동
        model = model.to(self.device)

        # Optimizer 생성
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            momentum = config.get('momentum', 0.9)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Learning rate scheduler (선택적)
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=3, factor=0.5
            )

        # Loss function (TaskStrategy 사용 또는 기본 CrossEntropyLoss)
        if self.task_strategy:
            criterion = self.task_strategy.get_criterion()
            metric_name = self.task_strategy.get_metric_name()
        else:
            # 하위 호환성: TaskStrategy가 없으면 기본값 사용
            criterion = nn.CrossEntropyLoss()
            metric_name = 'accuracy'

        # 학습 히스토리
        history = {
            'train_loss': [],
            f'train_{metric_name}': [],
            'val_loss': [],
            f'val_{metric_name}': []
        }

        best_val_metric = 0.0
        patience_counter = 0
        start_time = time.time()

        self._log("\n" + "=" * 70)
        self._log(f"[TRAIN] Starting Training (Vanilla PyTorch)")
        self._log(f"  Task Strategy: {type(self.task_strategy).__name__ if self.task_strategy else 'Default (CrossEntropy)'}")
        self._log(f"  Primary Metric: {metric_name}")
        self._log(f"  LR Scheduler: {'Enabled' if use_scheduler else 'Disabled'}")
        self._log(f"  Early Stopping: {'Enabled (patience={})'.format(patience) if use_early_stopping else 'Disabled'}")
        self._log(f"  Gradient Clipping: {'Enabled (max_norm={})'.format(grad_clip_max_norm) if grad_clip_max_norm else 'Disabled'}")
        self._log("=" * 70)

        for epoch in range(max_epochs):
            # === Training Phase ===
            model.train()
            train_loss = 0.0
            all_train_outputs = []
            all_train_targets = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                # 레이블 전처리 (TaskStrategy 사용)
                if self.task_strategy:
                    targets = self.task_strategy.prepare_labels(targets)

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward
                loss.backward()

                # Gradient clipping (선택적)
                if grad_clip_max_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)

                optimizer.step()

                # Metrics
                train_loss += loss.item()
                all_train_outputs.append(outputs.detach().cpu())
                all_train_targets.append(targets.detach().cpu())

                # Progress
                if (batch_idx + 1) % 50 == 0:
                    self._log(f"  Epoch [{epoch+1}/{max_epochs}] "
                             f"Batch [{batch_idx+1}/{len(train_loader)}] "
                             f"Loss: {loss.item():.4f}")

            epoch_train_loss = train_loss / len(train_loader)

            # 전체 에폭의 메트릭 계산
            all_train_outputs = torch.cat(all_train_outputs)
            all_train_targets = torch.cat(all_train_targets)

            if self.task_strategy:
                epoch_train_metric = self.task_strategy.calculate_metric(all_train_outputs, all_train_targets)
            else:
                # 기본: accuracy 계산
                _, predicted = all_train_outputs.max(1)
                epoch_train_metric = 100. * predicted.eq(all_train_targets).sum().item() / all_train_targets.size(0)

            # === Validation Phase ===
            model.eval()
            val_loss = 0.0
            all_val_outputs = []
            all_val_targets = []

            with torch.no_grad():
                for inputs, targets in val_loader:
                    # 레이블 전처리
                    if self.task_strategy:
                        targets = self.task_strategy.prepare_labels(targets)

                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    all_val_outputs.append(outputs.cpu())
                    all_val_targets.append(targets.cpu())

            epoch_val_loss = val_loss / len(val_loader)

            # 전체 validation 메트릭 계산
            all_val_outputs = torch.cat(all_val_outputs)
            all_val_targets = torch.cat(all_val_targets)

            if self.task_strategy:
                epoch_val_metric = self.task_strategy.calculate_metric(all_val_outputs, all_val_targets)
            else:
                # 기본: accuracy 계산
                _, predicted = all_val_outputs.max(1)
                epoch_val_metric = 100. * predicted.eq(all_val_targets).sum().item() / all_val_targets.size(0)

            # 히스토리 저장
            history['train_loss'].append(epoch_train_loss)
            history[f'train_{metric_name}'].append(epoch_train_metric)
            history['val_loss'].append(epoch_val_loss)
            history[f'val_{metric_name}'].append(epoch_val_metric)

            # Best model 추적 및 Early Stopping
            if epoch_val_metric > best_val_metric:
                best_val_metric = epoch_val_metric
                patience_counter = 0
            else:
                patience_counter += 1

            # Learning rate scheduler 업데이트
            if scheduler:
                scheduler.step(epoch_val_metric)

            # 결과 출력
            self._log(f"\nEpoch [{epoch+1}/{max_epochs}]:")
            self._log(f"  Train Loss: {epoch_train_loss:.4f} | Train {metric_name}: {epoch_train_metric:.2f}")
            self._log(f"  Val Loss: {epoch_val_loss:.4f} | Val {metric_name}: {epoch_val_metric:.2f}")
            if use_early_stopping:
                self._log(f"  Early Stopping Counter: {patience_counter}/{patience}")
            self._log("-" * 70)

            # Early stopping 체크
            if use_early_stopping and patience_counter >= patience:
                self._log(f"\n[Early Stopping] No improvement for {patience} epochs. Stopping training.")
                break

        # GPU 동기화 (정확한 타이밍 측정)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        training_time = time.time() - start_time

        self._log("\n" + "=" * 70)
        self._log(f"[완료] Training Completed")
        self._log(f"  Total Time: {training_time:.2f}s")
        self._log(f"  Best Val {metric_name}: {best_val_metric:.2f}")
        self._log("=" * 70 + "\n")

        return {
            'training_time': training_time,
            f'best_val_{metric_name}': best_val_metric,
            'final_train_loss': history['train_loss'][-1],
            f'final_train_{metric_name}': history[f'train_{metric_name}'][-1],
            'final_val_loss': history['val_loss'][-1],
            f'final_val_{metric_name}': history[f'val_{metric_name}'][-1],
            'history': history,
            'model': model,
            # 하위 호환성을 위해 'acc' 키도 유지
            'best_val_acc': best_val_metric,
        }

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """테스트 데이터로 모델 평가"""

        model = model.to(self.device)
        model.eval()

        # Loss function (TaskStrategy 사용 또는 기본값)
        if self.task_strategy:
            criterion = self.task_strategy.get_criterion()
            metric_name = self.task_strategy.get_metric_name()
        else:
            criterion = nn.CrossEntropyLoss()
            metric_name = 'accuracy'

        test_loss = 0.0
        all_outputs = []
        all_targets = []

        self._log("\n Evaluating on test set...")

        start_time = time.time()

        with torch.no_grad():
            for inputs, targets in test_loader:
                # 레이블 전처리
                if self.task_strategy:
                    targets = self.task_strategy.prepare_labels(targets)

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # GPU 동기화 (정확한 타이밍 측정)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        inference_time = time.time() - start_time

        test_loss = test_loss / len(test_loader)

        # 전체 메트릭 계산
        all_outputs = torch.cat(all_outputs)
        all_targets = torch.cat(all_targets)

        if self.task_strategy:
            test_metric = self.task_strategy.calculate_metric(all_outputs, all_targets)
        else:
            # 기본: accuracy 계산
            _, predicted = all_outputs.max(1)
            test_metric = 100. * predicted.eq(all_targets).sum().item() / all_targets.size(0)

        # 예측 결과 저장
        if self.task_strategy:
            # TaskStrategy에 따라 predictions 형식이 다를 수 있음
            predictions = all_outputs.numpy()
        else:
            _, predicted = all_outputs.max(1)
            predictions = predicted.numpy()

        labels = all_targets.numpy()

        self._log(f"[OK] Test Loss: {test_loss:.4f}")
        self._log(f"[OK] Test {metric_name}: {test_metric:.2f}")
        self._log(f"[OK] Inference Time: {inference_time:.2f}s\n")

        return {
            'test_loss': test_loss,
            f'test_{metric_name}': test_metric,
            'inference_time': inference_time,
            'predictions': predictions,
            'labels': labels,
            # 하위 호환성을 위해 'acc' 키도 유지
            'test_acc': test_metric,
        }
