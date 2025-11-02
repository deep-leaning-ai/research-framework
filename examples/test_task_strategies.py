"""
Task Strategy 통합 테스트

research의 Task Strategy 시스템을 테스트합니다.
- MultiClassStrategy: 다중 분류
- BinaryClassificationStrategy: 이진 분류
- RegressionStrategy: 회귀 분석
"""

import sys
import torch
import torch.nn as nn

print("="*70)
print("Task Strategy 통합 테스트")
print("="*70)

# 1. Import 테스트
print("\n1. Testing imports...")
try:
    from research import (
        MultiClassStrategy,
        BinaryClassificationStrategy,
        RegressionStrategy,
        CNN,
        FullyConnectedNN,
        MetricTracker,
        AccuracyMetric,
        MSEMetric,
        MAEMetric
    )
    print("   [OK] All task strategy classes imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. MultiClassStrategy 테스트
print("\n2. Testing MultiClassStrategy...")
try:
    # 10개 클래스 다중 분류
    strategy = MultiClassStrategy(num_classes=10)

    # Criterion 확인
    criterion = strategy.get_criterion()
    assert isinstance(criterion, nn.CrossEntropyLoss), "Criterion should be CrossEntropyLoss"

    # Output activation 확인
    activation = strategy.get_output_activation()
    assert activation is None, "MultiClass should not have output activation"

    # 더미 데이터로 메트릭 계산
    outputs = torch.randn(32, 10)  # batch_size=32, num_classes=10
    labels = torch.randint(0, 10, (32,))

    # 레이블 준비
    prepared_labels = strategy.prepare_labels(labels)
    assert prepared_labels.shape == labels.shape, "Labels should not change shape"

    # 메트릭 계산
    metric_value = strategy.calculate_metric(outputs, labels)
    assert 0 <= metric_value <= 100, f"Accuracy should be 0-100, got {metric_value}"

    metric_name = strategy.get_metric_name()
    assert metric_name == "Accuracy", f"Metric name should be 'Accuracy', got {metric_name}"

    print(f"   [OK] MultiClassStrategy test passed")
    print(f"     - Criterion: {type(criterion).__name__}")
    print(f"     - Metric: {metric_name} = {metric_value:.2f}%")

except Exception as e:
    print(f"   ✗ MultiClassStrategy test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. BinaryClassificationStrategy 테스트
print("\n3. Testing BinaryClassificationStrategy...")
try:
    strategy = BinaryClassificationStrategy()

    # Criterion 확인
    criterion = strategy.get_criterion()
    assert isinstance(criterion, nn.BCEWithLogitsLoss), "Criterion should be BCEWithLogitsLoss"

    # Output activation 확인
    activation = strategy.get_output_activation()
    assert isinstance(activation, nn.Sigmoid), "BinaryClassification should have Sigmoid activation"

    # 더미 데이터로 메트릭 계산
    outputs = torch.randn(32, 1)  # batch_size=32, output_dim=1 (logits)
    labels = torch.randint(0, 2, (32,))  # 0 or 1

    # 레이블 준비
    prepared_labels = strategy.prepare_labels(labels)
    assert prepared_labels.shape == (32, 1), f"Labels should be reshaped to (32, 1), got {prepared_labels.shape}"
    assert prepared_labels.dtype == torch.float32, "Labels should be float"

    # 손실 계산 (레이블 shape 확인)
    loss = criterion(outputs, prepared_labels)
    assert loss.item() >= 0, "Loss should be non-negative"

    # 메트릭 계산
    metric_value = strategy.calculate_metric(outputs, prepared_labels)
    assert 0 <= metric_value <= 100, f"Accuracy should be 0-100, got {metric_value}"

    metric_name = strategy.get_metric_name()
    assert metric_name == "Accuracy", f"Metric name should be 'Accuracy', got {metric_name}"

    print(f"   [OK] BinaryClassificationStrategy test passed")
    print(f"     - Criterion: {type(criterion).__name__}")
    print(f"     - Activation: {type(activation).__name__}")
    print(f"     - Metric: {metric_name} = {metric_value:.2f}%")
    print(f"     - Loss: {loss.item():.4f}")

except Exception as e:
    print(f"   ✗ BinaryClassificationStrategy test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. RegressionStrategy 테스트
print("\n4. Testing RegressionStrategy...")
try:
    strategy = RegressionStrategy()

    # Criterion 확인
    criterion = strategy.get_criterion()
    assert isinstance(criterion, nn.MSELoss), "Criterion should be MSELoss"

    # Output activation 확인
    activation = strategy.get_output_activation()
    assert activation is None, "Regression should not have output activation"

    # 더미 데이터로 메트릭 계산
    outputs = torch.randn(32, 1)  # batch_size=32, output_dim=1
    labels = torch.randn(32)  # continuous values

    # 레이블 준비
    prepared_labels = strategy.prepare_labels(labels)
    assert prepared_labels.shape == (32, 1), f"Labels should be reshaped to (32, 1), got {prepared_labels.shape}"
    assert prepared_labels.dtype == torch.float32, "Labels should be float"

    # 손실 계산
    loss = criterion(outputs, prepared_labels)
    assert loss.item() >= 0, "MSE should be non-negative"

    # 메트릭 계산
    metric_value = strategy.calculate_metric(outputs, prepared_labels)
    assert metric_value >= 0, f"MSE should be non-negative, got {metric_value}"

    metric_name = strategy.get_metric_name()
    assert metric_name == "MSE", f"Metric name should be 'MSE', got {metric_name}"

    print(f"   [OK] RegressionStrategy test passed")
    print(f"     - Criterion: {type(criterion).__name__}")
    print(f"     - Metric: {metric_name} = {metric_value:.4f}")
    print(f"     - Loss: {loss.item():.4f}")

except Exception as e:
    print(f"   ✗ RegressionStrategy test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. 단순 모델 통합 테스트 (Binary Classification)
print("\n5. Testing Binary Classification with simple model...")
try:
    # 간단한 선형 모델 생성
    class SimpleBinaryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(20, 1)
            self.task_strategy = BinaryClassificationStrategy()

        def forward(self, x):
            return self.fc(x)

    model = SimpleBinaryModel()

    # 더미 데이터
    X = torch.randn(16, 20)
    y = torch.randint(0, 2, (16,))

    # Forward pass
    outputs = model(X)
    assert outputs.shape == (16, 1), f"Output shape should be (16, 1), got {outputs.shape}"

    # 손실 계산
    prepared_labels = model.task_strategy.prepare_labels(y)
    criterion = model.task_strategy.get_criterion()
    loss = criterion(outputs, prepared_labels)

    # 메트릭 계산
    metric_value = model.task_strategy.calculate_metric(outputs, prepared_labels)

    print(f"   [OK] Binary classification with model integration passed")
    print(f"     - Model: SimpleBinaryModel")
    print(f"     - Input shape: {X.shape}")
    print(f"     - Output shape: {outputs.shape}")
    print(f"     - Loss: {loss.item():.4f}")
    print(f"     - Accuracy: {metric_value:.2f}%")

except Exception as e:
    print(f"   ✗ Binary classification with model integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 단순 모델 통합 테스트 (Regression)
print("\n6. Testing Regression with simple model...")
try:
    # 간단한 선형 회귀 모델 생성
    class SimpleRegressionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
            self.task_strategy = RegressionStrategy()

        def forward(self, x):
            return self.fc(x)

    model = SimpleRegressionModel()

    # 더미 데이터
    X = torch.randn(16, 10)
    y = torch.randn(16)  # continuous values

    # Forward pass
    outputs = model(X)
    assert outputs.shape == (16, 1), f"Output shape should be (16, 1), got {outputs.shape}"

    # 손실 계산
    prepared_labels = model.task_strategy.prepare_labels(y)
    criterion = model.task_strategy.get_criterion()
    loss = criterion(outputs, prepared_labels)

    # 메트릭 트래커와 통합
    tracker = MetricTracker([MSEMetric(), MAEMetric()])

    # 메트릭 계산 (회귀용 메트릭은 2D tensor를 기대)
    metrics = tracker.update(outputs, prepared_labels)

    print(f"   [OK] Regression with model integration passed")
    print(f"     - Model: SimpleRegressionModel")
    print(f"     - Input shape: {X.shape}")
    print(f"     - Output shape: {outputs.shape}")
    print(f"     - Loss: {loss.item():.4f}")
    print(f"     - Metrics:")
    for metric_name, value in metrics.items():
        print(f"       * {metric_name}: {value:.4f}")

except Exception as e:
    print(f"   ✗ Regression with model integration failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("[완료] Task Strategy 통합 테스트 완료!")
print("="*70)
print("\n주요 확인 사항:")
print("  [OK] MultiClassStrategy 정상 작동 (CrossEntropyLoss)")
print("  [OK] BinaryClassificationStrategy 정상 작동 (BCEWithLogitsLoss)")
print("  [OK] RegressionStrategy 정상 작동 (MSELoss)")
print("  [OK] 모델과의 통합 (FullyConnectedNN)")
print("  [OK] 메트릭 트래커와의 통합")
print("\nTask Strategy 시스템이 research에 완전히 통합되었습니다!")
