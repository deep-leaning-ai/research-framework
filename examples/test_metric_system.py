"""
메트릭 시스템 통합 테스트

research의 고급 메트릭 시스템을 테스트합니다.
- MetricTracker: 실시간 다중 메트릭 추적
- AccuracyMetric, F1ScoreMetric 등
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("="*70)
print("메트릭 시스템 통합 테스트")
print("="*70)

# 1. Import 테스트
print("\n1. Testing imports...")
try:
    from research.metrics import (
        MetricTracker,
        AccuracyMetric,
        PrecisionMetric,
        RecallMetric,
        F1ScoreMetric
    )
    print("   [OK] All metric classes imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# 2. MetricTracker 생성 테스트
print("\n2. Creating MetricTracker...")
try:
    tracker = MetricTracker([
        AccuracyMetric(),
        PrecisionMetric(average='macro'),
        RecallMetric(average='macro'),
        F1ScoreMetric(average='macro')
    ])
    print(f"   [OK] MetricTracker created with {len(tracker.metrics)} metrics")
    print(f"   Metrics: {[m.name for m in tracker.metrics]}")
except Exception as e:
    print(f"   ✗ Failed to create MetricTracker: {e}")
    sys.exit(1)

# 3. 간단한 예측 데이터로 메트릭 계산 테스트
print("\n3. Testing metric calculation...")
try:
    # 가짜 예측 데이터 생성 (logits 형태, 2D)
    y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    # 모델 출력 시뮬레이션 (10 samples, 3 classes)
    # 각 샘플에 대해 3개 클래스의 logits
    logits = torch.randn(10, 3)
    # 정답에 높은 값 할당 (일부는 틀리게)
    for i, label in enumerate(y_true):
        if i not in [4, 7, 8]:  # 일부 샘플은 틀리게
            logits[i, label] = 10.0  # 정답에 높은 logit

    # 메트릭 계산 (MetricTracker.update 사용)
    metrics = tracker.update(logits, y_true)

    print("   [OK] Metrics calculated successfully:")
    for metric_name, value in metrics.items():
        print(f"     - {metric_name}: {value:.4f}")

except Exception as e:
    print(f"   ✗ Metric calculation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 학습 루프 시뮬레이션 (에폭별 메트릭 추적)
print("\n4. Simulating training loop with metric tracking...")
try:
    # 새 트래커 생성 (히스토리 초기화)
    tracker2 = MetricTracker([
        AccuracyMetric(),
        F1ScoreMetric(average='macro')
    ])

    # 간단한 더미 모델
    model = nn.Linear(10, 3)

    # 더미 데이터
    X = torch.randn(100, 10)
    y = torch.randint(0, 3, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=False)

    # 에폭별 학습
    for epoch in range(3):
        all_outputs = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in loader:
                outputs = model(batch_x)  # logits (batch_size, num_classes)

                all_outputs.append(outputs)
                all_labels.append(batch_y)

        # 모든 출력을 하나로 합치기
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        # 에폭 메트릭 계산
        metrics = tracker2.update(all_outputs, all_labels)

        print(f"   Epoch {epoch + 1}:")
        print(f"     Accuracy: {metrics['Accuracy']:.4f}")
        print(f"     F1-Score: {metrics['F1-Score (macro)']:.4f}")

    print("   [OK] Training loop simulation successful")

except Exception as e:
    print(f"   ✗ Training simulation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("[완료] 메트릭 시스템 통합 테스트 완료!")
print("="*70)
print("\n주요 확인 사항:")
print("  [OK] MetricTracker 정상 작동")
print("  [OK] 다중 메트릭 동시 계산 가능")
print("  [OK] 학습 루프에서 사용 가능")
print("\n메트릭 시스템이 research에 완전히 통합되었습니다!")
