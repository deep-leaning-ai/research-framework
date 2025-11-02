"""
Strategy 인터페이스 정의
프레임워크(Lightning, WandB 등)를 쉽게 교체할 수 있도록 추상화
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch.nn as nn
from torch.utils.data import DataLoader


class TrainingStrategy(ABC):
    """학습 전략 인터페이스 (PyTorch Lightning, Vanilla PyTorch 등)"""

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        모델 학습

        Args:
            model: 학습할 모델
            train_loader: 학습 데이터로더
            val_loader: 검증 데이터로더
            config: 학습 설정 (lr, epochs, optimizer 등)

        Returns:
            학습 결과 딕셔너리 (training_time, best_val_acc, history 등)
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader
    ) -> Dict[str, Any]:
        """
        모델 평가

        Args:
            model: 평가할 모델
            test_loader: 테스트 데이터로더

        Returns:
            평가 결과 딕셔너리 (test_acc, test_loss, confusion_matrix 등)
        """
        pass


class LoggingStrategy(ABC):
    """로깅 전략 인터페이스 (WandB, TensorBoard, Simple 등)"""

    @abstractmethod
    def init_run(self, project_name: str, config: Dict[str, Any], **kwargs):
        """
        로깅 세션 초기화

        Args:
            project_name: 프로젝트 이름
            config: 로깅할 하이퍼파라미터
            **kwargs: 추가 설정
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        메트릭 로깅

        Args:
            metrics: 로깅할 메트릭 딕셔너리
            step: 스텝 번호 (에폭, 배치 등)
        """
        pass

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]):
        """
        하이퍼파라미터 로깅

        Args:
            params: 하이퍼파라미터 딕셔너리
        """
        pass

    @abstractmethod
    def log_artifact(self, file_path: str, artifact_type: str = 'file'):
        """
        파일 아티팩트 로깅 (모델 체크포인트, 플롯 등)

        Args:
            file_path: 파일 경로
            artifact_type: 아티팩트 타입
        """
        pass

    @abstractmethod
    def finish(self):
        """로깅 세션 종료"""
        pass


class OptimizationStrategy(ABC):
    """하이퍼파라미터 최적화 전략 인터페이스 (GridSearch, RandomSearch 등)"""

    @abstractmethod
    def search(
        self,
        model_fn: callable,
        train_fn: callable,
        param_space: Dict[str, Any],
        n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        하이퍼파라미터 탐색

        Args:
            model_fn: 모델 생성 함수
            train_fn: 학습 함수
            param_space: 탐색할 파라미터 공간
            n_trials: 탐색 횟수 (RandomSearch용)

        Returns:
            최적 파라미터 및 성능 딕셔너리
        """
        pass

    @abstractmethod
    def get_best_params(self) -> Dict[str, Any]:
        """최적 파라미터 반환"""
        pass

    @abstractmethod
    def get_search_history(self) -> list:
        """탐색 히스토리 반환"""
        pass
