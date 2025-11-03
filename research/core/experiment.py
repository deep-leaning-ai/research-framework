"""
Experiment: 실험 전체를 관리하는 Facade 클래스
복잡한 내부 로직을 숨기고 간단한 API 제공
"""
from typing import Dict, Any, Optional, List
from .base_model import BaseModel
from .strategies import TrainingStrategy, LoggingStrategy
from ..models.pretrained.registry import ModelRegistry


class Experiment:
    """
    실험 오케스트레이터 (Facade Pattern)

    사용법:
        # 설정
        config = {
            'num_classes': 10,
            'learning_rate': 1e-3,
            'max_epochs': 10,
            'batch_size': 32
        }

        # 실험 생성
        exp = Experiment(config)
        exp.setup(
            model_name='resnet50',
            data_module=cifar_dm,
            training_strategy=VanillaTrainingStrategy(),
            logging_strategy=SimpleLoggingStrategy()
        )

        # 실험 실행
        results = exp.run(strategy='fine_tuning')

        # 여러 전략 비교
        comparison = exp.compare_strategies(['feature_extraction', 'fine_tuning'])
    """

    # 클래스 상수 - 기본 설정값
    DEFAULT_MAX_EPOCHS = 100
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_OPTIMIZER = 'adam'
    DEFAULT_NUM_CLASSES = 10
    DEFAULT_IN_CHANNELS = 3

    # Freeze 전략 매핑
    FREEZE_STRATEGIES = {
        'feature_extraction': 'freeze_backbone',
        'fine_tuning': 'unfreeze_all',
        'inference': 'freeze_all'
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 실험 설정 딕셔너리
                - num_classes: 클래스 수
                - in_channels: 입력 채널 수 (기본값 3=RGB, 1=Grayscale/Mel-Spectrogram)
                - learning_rate: 학습률
                - max_epochs: 최대 에폭
                - batch_size: 배치 크기
                - optimizer: 옵티마이저 ('adam', 'adamw', 'sgd')
                - 등...
        """
        self.config = config
        self._model = None  # 캡슐화: 내부 속성으로 변경
        self._model_name = None  # 모델 이름 저장 (리셋용)
        self._initial_state = None  # 초기 상태 저장 (리셋용)
        self.data_module = None
        self.training_strategy = None
        self.logging_strategy = None
        self.results = {}
        self.experiment_history = []

    @property
    def model(self):
        """모델 읽기 전용 접근"""
        return self._model

    def get_model_for_inference(self):
        """추론용 모델 복사본 반환

        Returns:
            평가 모드로 설정된 모델 복사본
        """
        if self._model is None:
            raise RuntimeError("No model available. Call setup() first.")

        import copy
        model_copy = copy.deepcopy(self._model)
        model_copy.model.eval()
        return model_copy

    def setup(
        self,
        model_name: str,
        data_module,
        training_strategy: TrainingStrategy,
        logging_strategy: Optional[LoggingStrategy] = None
    ):
        """
        실험 초기화

        Args:
            model_name: 모델 이름 (ModelRegistry에 등록된 이름)
            data_module: 데이터 모듈 (train/val/test_dataloader 메서드 필요)
            training_strategy: 학습 전략 (VanillaTrainingStrategy 등)
            logging_strategy: 로깅 전략 (None이면 로깅 안함)
        """
        # 모델 생성
        self._model_name = model_name
        self._model = ModelRegistry.create(
            model_name,
            num_classes=self.config.get('num_classes', self.DEFAULT_NUM_CLASSES),
            in_channels=self.config.get('in_channels', self.DEFAULT_IN_CHANNELS)
        )

        # 초기 상태 저장 (deep copy를 위해 state_dict 사용)
        import copy
        self._initial_state = copy.deepcopy(self._model.model.state_dict())

        self.data_module = data_module
        self.training_strategy = training_strategy
        self.logging_strategy = logging_strategy

        # 데이터 준비
        if hasattr(self.data_module, 'prepare_data'):
            self.data_module.prepare_data()

        if hasattr(self.data_module, 'setup'):
            self.data_module.setup()

        print(f"\n[OK] Experiment initialized")
        print(f"  Model: {model_name}")
        print(f"  Training Strategy: {training_strategy.__class__.__name__}")
        print(f"  Logging Strategy: {logging_strategy.__class__.__name__ if logging_strategy else 'None'}")
        print(f"  Config: {self.config}\n")

    def run(self, strategy: str = 'fine_tuning', run_name: Optional[str] = None) -> Dict[str, Any]:
        """
        단일 실험 실행

        Args:
            strategy: 전이학습 전략
                - 'feature_extraction': 백본 동결, 분류기만 학습
                - 'fine_tuning': 전체 네트워크 학습
                - 'inference': 학습 없이 평가만
            run_name: 실험 이름 (로깅용)

        Returns:
            실험 결과 딕셔너리
        """
        if self.model is None:
            raise RuntimeError("Call setup() before run()")

        run_name = run_name or f"{strategy}_experiment"

        # 로깅 초기화
        if self.logging_strategy:
            project_name = self.config.get('project_name', 'dl_research')
            self.logging_strategy.init_run(
                project_name=project_name,
                config={**self.config, 'strategy': strategy},
                run_name=run_name
            )

        # 전략 적용
        print(f"\n{'='*70}")
        print(f"[START] Running Experiment: {run_name}")
        print(f"   Strategy: {strategy}")
        print(f"{'='*70}\n")

        if strategy == 'feature_extraction':
            self.model.freeze_backbone()
        elif strategy == 'fine_tuning':
            self.model.unfreeze_all()
        elif strategy == 'inference':
            self.model.freeze_all()
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'feature_extraction', 'fine_tuning', or 'inference'")

        # 모델 정보 출력
        model_info = self.model.get_model_info()
        print(f" Model Info:")
        print(f"  Total params: {model_info['total_parameters']:,}")
        print(f"  Trainable params: {model_info['trainable_parameters']:,}")
        print(f"  Trainable ratio: {model_info['trainable_ratio']:.2%}\n")

        if self.logging_strategy:
            self.logging_strategy.log_hyperparams(model_info)

        # 학습 (inference가 아닐 경우)
        if strategy != 'inference':
            train_results = self.training_strategy.train(
                model=self.model.model,  # 내부 nn.Module 전달
                train_loader=self.data_module.train_dataloader(),
                val_loader=self.data_module.val_dataloader(),
                config=self.config
            )

            if self.logging_strategy:
                self.logging_strategy.log_metrics(train_results)
        else:
            train_results = {}

        # 평가
        test_results = self.training_strategy.evaluate(
            model=self.model.model,
            test_loader=self.data_module.test_dataloader()
        )

        if self.logging_strategy:
            self.logging_strategy.log_metrics(test_results, step='test')
            self.logging_strategy.finish()

        # 결과 통합
        results = {
            'strategy': strategy,
            'model_info': model_info,
            'train_results': train_results,
            'test_results': test_results
        }

        self.results = results
        self.experiment_history.append(results)

        print(f"\n{'='*70}")
        print(f"[완료] Experiment '{run_name}' completed")
        print(f"{'='*70}\n")

        return results

    def compare_strategies(
        self,
        strategies: List[str],
        reset_model: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        여러 전략 자동 비교

        Args:
            strategies: 비교할 전략 리스트 (예: ['feature_extraction', 'fine_tuning'])
            reset_model: 각 실험 전에 모델 리셋 여부

        Returns:
            전략별 결과 딕셔너리
        """
        comparison = {}

        for strategy in strategies:
            # 모델 리셋 (초기 상태로 복원)
            if reset_model:
                self._reset_model()

            # 실험 실행
            run_name = f"{strategy}_comparison"
            results = self.run(strategy=strategy, run_name=run_name)
            comparison[strategy] = results

        # 비교 결과 출력
        self._print_comparison(comparison)

        return comparison

    def _reset_model(self):
        """모델을 초기 상태로 리셋

        상태 딕셔너리를 사용하여 모델 구조는 유지하면서
        가중치만 초기 상태로 복원합니다.
        """
        if self._initial_state is None:
            raise RuntimeError("No initial state saved. Call setup() first.")

        # 초기 상태로 가중치 복원
        self._model.model.load_state_dict(self._initial_state)
        print(f"[OK] Model reset to initial state")

    def _print_comparison(self, comparison: Dict[str, Dict[str, Any]]):
        """비교 결과 출력"""
        print("\n" + "="*70)
        print(" STRATEGY COMPARISON RESULTS")
        print("="*70)

        for strategy, results in comparison.items():
            print(f"\n[{strategy.upper()}]")
            print(f"  Trainable Params: {results['model_info']['trainable_parameters']:,}")
            print(f"  Trainable Ratio: {results['model_info']['trainable_ratio']:.2%}")

            if 'training_time' in results['train_results']:
                print(f"  Training Time: {results['train_results']['training_time']:.2f}s")

            if 'test_acc' in results['test_results']:
                print(f"  Test Accuracy: {results['test_results']['test_acc']:.2f}%")
            elif 'accuracy' in results['test_results']:
                print(f"  Test Accuracy: {results['test_results']['accuracy']:.2f}%")

        print("\n" + "="*70 + "\n")

    def get_history(self) -> List[Dict[str, Any]]:
        """실험 히스토리 반환"""
        return self.experiment_history

    def get_latest_result(self) -> Dict[str, Any]:
        """최근 실험 결과 반환"""
        return self.results

    def save_results(self, filepath: str):
        """결과를 JSON 파일로 저장"""
        import json

        with open(filepath, 'w') as f:
            # numpy array는 list로 변환
            def convert(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

            json.dump(self.experiment_history, f, indent=2, default=convert)

        print(f"[OK] Results saved to {filepath}")

    def evaluate_pretrained(self, test_loader=None) -> Dict[str, Any]:
        """
        사전학습 모델 평가 (학습 없이 추론만)

        Inference 전용 메서드로, 학습 없이 사전학습된 가중치로만 평가합니다.
        Task 1의 ResNetInferenceModel.evaluate_model()과 동일한 기능을 제공합니다.

        Args:
            test_loader: 테스트 데이터로더 (None이면 data_module에서 자동 가져옴)

        Returns:
            평가 결과 딕셔너리:
                - accuracy: 정확도
                - loss: 손실
                - inference_time: 추론 시간
                - predictions: 예측 레이블 리스트
                - labels: 실제 레이블 리스트
                - confusion_matrix: Confusion matrix
                - classification_report: Classification report (문자열)
                - total_params, trainable_params, trainable_ratio: 모델 파라미터 정보
        """
        if self.model is None:
            raise RuntimeError("Call setup() before evaluate_pretrained()")

        # 모델 동결 (추론 전용)
        self.model.freeze_all()

        # 테스트 데이터로더
        if test_loader is None:
            test_loader = self.data_module.test_dataloader()

        # 클래스 이름 가져오기
        class_names = self.data_module.get_class_names() if hasattr(self.data_module, 'get_class_names') else []

        print(f"\n{'='*70}")
        print(f" Evaluating Pretrained Model (Inference Only)")
        print(f"   Model: {self.model.__class__.__name__}")
        print(f"{'='*70}\n")

        # 평가 수행
        import torch
        import torch.nn as nn
        import time

        device = self.training_strategy.device if hasattr(self.training_strategy, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 직접 평가 수행
        self.model.model.eval()
        self.model.model = self.model.model.to(device)

        total_loss = 0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        # 추론 시간 측정 시작
        start_time = time.time()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model.model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        inference_time = time.time() - start_time
        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'inference_time': inference_time,
            'total_samples': total,
            'correct_predictions': correct
        }

        # 모델 정보 추가
        model_info = self.model.get_model_info()
        results.update(model_info)

        # 결과 출력
        print(f"[OK] Evaluation completed")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Loss: {results['loss']:.4f}")
        print(f"  Inference Time: {results['inference_time']:.2f}s")
        print(f"  Total Params: {results['total_params']:,}")
        print(f"  Trainable Params: {results['trainable_params']:,}\n")

        # 히스토리에 추가
        self.results = {
            'type': 'inference',
            'model_info': model_info,
            'test_results': results
        }
        self.experiment_history.append(self.results)

        return results
