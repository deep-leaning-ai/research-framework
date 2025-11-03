"""
실험 실행 및 관리
DIP: TaskStrategy와 MetricCalculator 추상화에 의존
"""

from typing import List, Tuple
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from research.models.simple.base import BaseModel
from research.strategies.task.base import TaskStrategy
from research.metrics.base import MetricCalculator
from research.metrics.tracker import MetricTracker
from research.experiment.result import ExperimentResult
from research.experiment.recorder import ExperimentRecorder


class ExperimentRunner:
    """
    실험 실행 및 관리 클래스

    Task-agnostic 설계로 다양한 ML 태스크 지원
    SOLID 원칙을 준수하여 확장 가능
    """

    def __init__(
        self,
        device: str,
        task_strategy: TaskStrategy,
        metrics: List[MetricCalculator],
        primary_metric: MetricCalculator,
        num_epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 0.001,
    ):
        """
        Args:
            device: 디바이스 ('cuda' 또는 'cpu')
            task_strategy: Task 전략 객체
            metrics: 추적할 메트릭 리스트
            primary_metric: 주요 메트릭 (Early stopping 등에 사용)
            num_epochs: 학습 에폭 수
            batch_size: 배치 크기
            learning_rate: 학습률
        """
        self.device = device
        self.task_strategy = task_strategy
        self.metrics = metrics
        self.primary_metric = primary_metric
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.recorder = ExperimentRecorder()

    def run_single_experiment(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> ExperimentResult:
        """
        단일 모델 실험 수행

        Args:
            model: 학습할 모델
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            test_loader: 테스트 데이터 로더

        Returns:
            실험 결과
        """
        model_name = model.get_model_name()
        print(f"\n{'#'*70}")
        print(f"# {model_name} 실험 시작")
        print(f"# Primary Metric: {self.primary_metric.get_name()}")
        print(f"# Tracking Metrics: {[m.get_name() for m in self.metrics]}")
        print(f"{'#'*70}\n")

        # 모델 분석
        param_count = sum(p.numel() for p in model.parameters())
        print(f"모델 파라미터 수: {param_count:,}")

        # 모델 학습 준비
        model = model.to(self.device)
        criterion = self.task_strategy.get_criterion()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # 메트릭 트래커 초기화
        train_tracker = MetricTracker(self.metrics)
        val_tracker = MetricTracker(self.metrics)
        test_tracker = MetricTracker(self.metrics)

        # 학습 기록
        train_losses = []
        val_losses = []
        test_losses = []
        epoch_times = []

        # 최고 성능 추적
        best_val_metric = (
            float("-inf") if self.primary_metric.is_higher_better() else float("inf")
        )

        # 학습 루프
        for epoch in range(self.num_epochs):
            start_time = time.time()

            # 학습
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, train_tracker
            )

            # 검증
            val_loss = self._evaluate(model, val_loader, criterion, val_tracker)

            # 테스트
            test_loss = self._evaluate(model, test_loader, criterion, test_tracker)

            # GPU 동기화 (정확한 타이밍 측정)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            epoch_time = time.time() - start_time

            # 기록
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            epoch_times.append(epoch_time)

            # Primary metric 업데이트 (validation 기준)
            current_val_metric = val_tracker.get_latest()[
                self.primary_metric.get_name()
            ]
            if self.primary_metric.is_higher_better():
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
            else:
                if current_val_metric < best_val_metric:
                    best_val_metric = current_val_metric

            # 출력
            if (epoch + 1) % 5 == 0 or epoch == 0:
                self._print_epoch_stats(
                    epoch,
                    train_loss,
                    val_loss,
                    test_loss,
                    train_tracker,
                    val_tracker,
                    test_tracker,
                    epoch_time,
                )

        # 추론 시간 측정
        inference_time = self._measure_inference_time(model, test_loader)

        # 과적합 갭 계산 (분류 태스크만)
        final_overfitting_gap = None
        if hasattr(self.task_strategy, "calculate_metric"):
            train_final = train_tracker.get_latest().get(self.primary_metric.get_name())
            val_final = val_tracker.get_latest().get(self.primary_metric.get_name())
            if train_final is not None and val_final is not None:
                final_overfitting_gap = train_final - val_final

        print(f"\n[DONE] 실험 완료!")
        print(
            f"  - Best Validation {self.primary_metric.get_name()}: "
            f"{best_val_metric:{self.primary_metric.get_display_format()}}"
        )
        print(f"  - 평균 추론 시간: {inference_time*1000:.2f}ms")
        if final_overfitting_gap is not None:
            print(f"  - 최종 Overfitting Gap (Train vs Val): {final_overfitting_gap:.2f}")

        # 결과 생성
        result = ExperimentResult(
            model_name=model_name,
            task_type=type(self.task_strategy).__name__,
            parameters=param_count,
            train_metrics=train_tracker.get_all_history(),
            val_metrics=val_tracker.get_all_history(),
            test_metrics=test_tracker.get_all_history(),
            train_loss=train_losses,
            val_loss=val_losses,
            test_loss=test_losses,
            epoch_times=epoch_times,
            inference_time=inference_time,
            primary_metric_name=self.primary_metric.get_name(),
            best_test_metric=best_val_metric,
            final_overfitting_gap=final_overfitting_gap,
        )

        return result

    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        metric_tracker: MetricTracker,
    ) -> float:
        """1 epoch 학습"""
        model.train()
        running_loss = 0.0

        all_outputs = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = self.task_strategy.prepare_labels(labels).to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 배치 단위로 저장 (메트릭 계산용)
            all_outputs.append(outputs.detach())
            all_labels.append(labels.detach())

        # Epoch 끝나고 한번에 메트릭 계산
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        # 원본 레이블로 복원 (메트릭 계산을 위해)
        if hasattr(all_labels, "view") and len(all_labels.shape) > 1:
            all_labels = all_labels.view(-1)

        metric_tracker.update(all_outputs, all_labels)

        epoch_loss = running_loss / len(train_loader)
        return epoch_loss

    def _evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        metric_tracker: MetricTracker,
    ) -> float:
        """모델 평가"""
        model.eval()
        running_loss = 0.0

        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = self.task_strategy.prepare_labels(labels).to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                all_outputs.append(outputs)
                all_labels.append(labels)

        # 메트릭 계산
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        # 원본 레이블로 복원
        if hasattr(all_labels, "view") and len(all_labels.shape) > 1:
            all_labels = all_labels.view(-1)

        metric_tracker.update(all_outputs, all_labels)

        test_loss = running_loss / len(test_loader)
        return test_loss

    def _measure_inference_time(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> float:
        """모델 추론 시간 측정

        Args:
            model: 측정할 모델
            test_loader: 테스트 데이터 로더

        Returns:
            배치당 평균 추론 시간 (초)
        """
        model.eval()

        # 워밍업
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                _ = model(inputs)
                break

        # 실제 측정
        total_time = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)

                # GPU 동기화
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start_time = time.time()
                _ = model(inputs)

                # GPU 동기화
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                total_time += time.time() - start_time
                num_batches += 1

                # 충분한 샘플 측정
                if num_batches >= 10:
                    break

        return total_time / num_batches if num_batches > 0 else 0.0

    def _print_epoch_stats(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        test_loss: float,
        train_tracker: MetricTracker,
        val_tracker: MetricTracker,
        test_tracker: MetricTracker,
        epoch_time: float,
    ):
        """에폭 통계 출력"""
        print(f"Epoch [{epoch+1}/{self.num_epochs}]")
        print(
            f"  Loss: Train={train_loss:.4f}, Val={val_loss:.4f}, Test={test_loss:.4f}"
        )

        latest_train = train_tracker.get_latest()
        latest_val = val_tracker.get_latest()
        latest_test = test_tracker.get_latest()

        for metric in self.metrics:
            metric_name = metric.get_name()
            fmt = metric.get_display_format()
            if (
                metric_name in latest_train
                and metric_name in latest_val
                and metric_name in latest_test
            ):
                print(
                    f"  {metric_name}: "
                    f"Train={latest_train[metric_name]:{fmt}}, "
                    f"Val={latest_val[metric_name]:{fmt}}, "
                    f"Test={latest_test[metric_name]:{fmt}}"
                )

        print(f"  Time: {epoch_time:.2f}s")

    def run_multiple_experiments(
        self,
        models: List[BaseModel],
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ):
        """
        여러 모델 실험 수행

        Args:
            models: 모델 리스트
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            test_loader: 테스트 데이터 로더
        """
        for model in models:
            result = self.run_single_experiment(
                model, train_loader, val_loader, test_loader
            )
            self.recorder.add_result(result)

        # 요약 출력
        self.recorder.print_summary()

    def get_recorder(self) -> ExperimentRecorder:
        """
        실험 기록기 반환

        Returns:
            ExperimentRecorder 객체
        """
        return self.recorder
