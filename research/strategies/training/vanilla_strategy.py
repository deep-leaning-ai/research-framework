"""
VanillaTrainingStrategy: 순수 PyTorch 학습 전략 (Lightning 없이)
"""
import time
from typing import Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ...core.strategies import TrainingStrategy


class VanillaTrainingStrategy(TrainingStrategy):
    """
    순수 PyTorch 학습 전략
    PyTorch Lightning 없이 직접 학습 루프 구현
    """

    def __init__(self, device: str = None):
        """
        Args:
            device: 사용할 디바이스 ('cuda', 'cpu', None=자동선택)
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[OK] Using device: {self.device}")

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

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # 학습 히스토리
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        best_val_acc = 0.0
        start_time = time.time()

        print("\n" + "=" * 70)
        print(f"[TRAIN] Starting Training (Vanilla PyTorch)")
        print("=" * 70)

        for epoch in range(max_epochs):
            # === Training Phase ===
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

                # Progress
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Epoch [{epoch+1}/{max_epochs}] "
                          f"Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")

            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = 100. * train_correct / train_total

            # === Validation Phase ===
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100. * val_correct / val_total

            # 히스토리 저장
            history['train_loss'].append(epoch_train_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_loss'].append(epoch_val_loss)
            history['val_acc'].append(epoch_val_acc)

            # Best model 추적
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc

            # 결과 출력
            print(f"\nEpoch [{epoch+1}/{max_epochs}]:")
            print(f"  Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}%")
            print(f"  Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
            print("-" * 70)

        training_time = time.time() - start_time

        print("\n" + "=" * 70)
        print(f"[완료] Training Completed")
        print(f"  Total Time: {training_time:.2f}s")
        print(f"  Best Val Acc: {best_val_acc:.2f}%")
        print("=" * 70 + "\n")

        return {
            'training_time': training_time,
            'best_val_acc': best_val_acc,
            'final_train_loss': history['train_loss'][-1],
            'final_train_acc': history['train_acc'][-1],
            'final_val_loss': history['val_loss'][-1],
            'final_val_acc': history['val_acc'][-1],
            'history': history,
            'model': model
        }

    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """테스트 데이터로 모델 평가"""

        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        all_predictions = []
        all_labels = []

        print("\n Evaluating on test set...")

        start_time = time.time()

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(targets.cpu().numpy())

        inference_time = time.time() - start_time

        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total

        print(f"[OK] Test Loss: {test_loss:.4f}")
        print(f"[OK] Test Acc: {test_acc:.2f}%")
        print(f"[OK] Inference Time: {inference_time:.2f}s\n")

        return {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'inference_time': inference_time,
            'predictions': all_predictions,
            'labels': all_labels
        }
