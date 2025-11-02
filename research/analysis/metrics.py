"""
Performance Metrics Calculation Utilities
성능 지표 계산 및 평가 도구
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)
from typing import Dict, Any, List, Tuple, Optional


def compute_confusion_matrix(
    predictions: List[int],
    labels: List[int],
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Confusion matrix 계산

    Args:
        predictions: 예측 레이블 리스트
        labels: 실제 레이블 리스트
        num_classes: 클래스 수 (None이면 자동 감지)

    Returns:
        Confusion matrix (numpy array)
    """
    if num_classes is None:
        num_classes = max(max(predictions), max(labels)) + 1

    cm = confusion_matrix(
        labels,
        predictions,
        labels=list(range(num_classes))
    )

    return cm


def get_classification_report(
    predictions: List[int],
    labels: List[int],
    class_names: List[str],
    output_dict: bool = False
) -> Any:
    """
    Classification report 생성 (precision, recall, F1)

    Args:
        predictions: 예측 레이블 리스트
        labels: 실제 레이블 리스트
        class_names: 클래스 이름 리스트
        output_dict: True면 딕셔너리 반환, False면 문자열 반환

    Returns:
        Classification report (str or dict)
    """
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        output_dict=output_dict,
        zero_division=0
    )

    return report


def calculate_accuracy(predictions: List[int], labels: List[int]) -> float:
    """
    정확도 계산

    Args:
        predictions: 예측 레이블 리스트
        labels: 실제 레이블 리스트

    Returns:
        정확도 (0.0 ~ 1.0)
    """
    return accuracy_score(labels, predictions)


def calculate_per_class_accuracy(
    predictions: List[int],
    labels: List[int],
    class_names: List[str]
) -> Dict[str, float]:
    """
    클래스별 정확도 계산

    Args:
        predictions: 예측 레이블 리스트
        labels: 실제 레이블 리스트
        class_names: 클래스 이름 리스트

    Returns:
        클래스별 정확도 딕셔너리
    """
    cm = compute_confusion_matrix(predictions, labels, len(class_names))
    per_class_acc = {}

    for i, class_name in enumerate(class_names):
        # Diagonal / Row sum
        if cm[i].sum() > 0:
            per_class_acc[class_name] = cm[i, i] / cm[i].sum()
        else:
            per_class_acc[class_name] = 0.0

    return per_class_acc


def calculate_precision_recall_f1(
    predictions: List[int],
    labels: List[int],
    average: str = 'weighted'
) -> Tuple[float, float, float]:
    """
    Precision, Recall, F1-score 계산

    Args:
        predictions: 예측 레이블 리스트
        labels: 실제 레이블 리스트
        average: 평균 방식 ('micro', 'macro', 'weighted')

    Returns:
        (precision, recall, f1_score)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=average,
        zero_division=0
    )

    return precision, recall, f1


def calculate_metrics(
    predictions: List[int],
    labels: List[int],
    class_names: List[str]
) -> Dict[str, Any]:
    """
    종합 메트릭 계산

    Args:
        predictions: 예측 레이블 리스트
        labels: 실제 레이블 리스트
        class_names: 클래스 이름 리스트

    Returns:
        메트릭 딕셔너리:
            - accuracy: 전체 정확도
            - confusion_matrix: Confusion matrix
            - classification_report: Classification report (str)
            - classification_report_dict: Classification report (dict)
            - per_class_accuracy: 클래스별 정확도
            - precision, recall, f1_score: 전체 메트릭
    """
    accuracy = calculate_accuracy(predictions, labels)
    cm = compute_confusion_matrix(predictions, labels, len(class_names))
    report_str = get_classification_report(predictions, labels, class_names, output_dict=False)
    report_dict = get_classification_report(predictions, labels, class_names, output_dict=True)
    per_class_acc = calculate_per_class_accuracy(predictions, labels, class_names)
    precision, recall, f1 = calculate_precision_recall_f1(predictions, labels)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report_str,
        'classification_report_dict': report_dict,
        'per_class_accuracy': per_class_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    모델 파라미터 수 계산

    Args:
        model: PyTorch 모델

    Returns:
        파라미터 정보 딕셔너리:
            - total_params: 전체 파라미터 수
            - trainable_params: 학습 가능 파라미터 수
            - frozen_params: 동결된 파라미터 수
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0.0
    }


def get_model_size_mb(model: nn.Module) -> float:
    """
    모델 크기 계산 (MB)

    Args:
        model: PyTorch 모델

    Returns:
        모델 크기 (MB)
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2

    return size_mb


def measure_inference_time(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    추론 시간 측정

    Args:
        model: PyTorch 모델
        dataloader: 데이터로더
        device: 디바이스 ('cuda' or 'cpu')
        num_batches: 측정할 배치 수 (None이면 전체)

    Returns:
        시간 정보 딕셔너리:
            - total_time: 전체 시간 (초)
            - avg_time_per_batch: 배치당 평균 시간 (초)
            - avg_time_per_sample: 샘플당 평균 시간 (초)
            - num_batches: 측정한 배치 수
            - num_samples: 측정한 샘플 수
    """
    model = model.to(device)
    model.eval()

    total_time = 0.0
    num_samples_processed = 0
    batches_processed = 0

    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break

            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # 시간 측정
            start_time = time.time()
            _ = model(inputs)

            # GPU 동기화 (CUDA 사용 시)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            elapsed_time = time.time() - start_time

            total_time += elapsed_time
            num_samples_processed += batch_size
            batches_processed += 1

    return {
        'total_time': total_time,
        'avg_time_per_batch': total_time / batches_processed if batches_processed > 0 else 0.0,
        'avg_time_per_sample': total_time / num_samples_processed if num_samples_processed > 0 else 0.0,
        'num_batches': batches_processed,
        'num_samples': num_samples_processed
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_names: List[str],
    device: torch.device,
    criterion: Optional[nn.Module] = None
) -> Dict[str, Any]:
    """
    모델 종합 평가 (메트릭 + 추론 시간)

    Args:
        model: PyTorch 모델
        dataloader: 데이터로더
        class_names: 클래스 이름 리스트
        device: 디바이스
        criterion: Loss function (None이면 CrossEntropyLoss 사용)

    Returns:
        평가 결과 딕셔너리
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    # 추론 시간 측정 시작
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            total_loss += loss.item()
            num_batches += 1

    # 추론 시간 측정 종료
    if device.type == 'cuda':
        torch.cuda.synchronize()
    inference_time = time.time() - start_time

    # 메트릭 계산
    metrics = calculate_metrics(all_predictions, all_labels, class_names)

    # Loss 추가
    metrics['loss'] = total_loss / num_batches if num_batches > 0 else 0.0

    # 추론 시간 추가
    metrics['inference_time'] = inference_time
    metrics['predictions'] = all_predictions
    metrics['labels'] = all_labels

    # 파라미터 정보 추가
    param_info = count_parameters(model)
    metrics.update(param_info)

    return metrics
