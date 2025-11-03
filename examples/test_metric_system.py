"""
메트릭 시스템 사용 예제

MetricTracker를 사용하여 여러 메트릭을 동시에 추적하고
이동 평균, 최고 성능 등을 분석하는 방법을 보여줍니다.

실행 방법:
    $ python examples/test_metric_system.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from research.metrics import (
    MetricTracker,
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1ScoreMetric,
    MSEMetric,
    MAEMetric,
    R2Metric
)


def simulate_training_epoch(epoch: int) -> tuple:
    """에폭 시뮬레이션 (실제 학습 대신 가상 데이터 생성)"""
    # 에폭이 진행될수록 성능 향상 시뮬레이션
    base_acc = 0.5 + 0.1 * epoch + np.random.uniform(-0.05, 0.05)
    base_acc = min(base_acc, 0.95)  # 최대 95%

    # 가상 예측과 실제 레이블 생성
    batch_size = 100
    num_classes = 10

    # 분류 태스크용
    predictions = torch.randn(batch_size, num_classes)
    predictions = torch.softmax(predictions, dim=1)

    # 정답 레이블 (정확도와 일치하도록 조정)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 일부를 정답으로 만들기
    num_correct = int(batch_size * base_acc)
    pred_labels = torch.argmax(predictions, dim=1)
    pred_labels[:num_correct] = labels[:num_correct]
    predictions = torch.nn.functional.one_hot(pred_labels, num_classes).float()

    # 회귀 태스크용
    reg_predictions = torch.randn(batch_size, 1) * 10
    reg_targets = reg_predictions + torch.randn(batch_size, 1) * 2  # 노이즈 추가

    return predictions, labels, reg_predictions, reg_targets


def main():
    """메트릭 시스템 데모"""
    print("=" * 70)
    print(" Metric System Demonstration")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. 분류 메트릭 추적
    # ========================================================================
    print("[1] Classification Metrics Tracking")
    print("-" * 50)

    # MetricTracker 생성 (window_size로 이동 평균 계산)
    tracker = MetricTracker(
        metrics=[
            AccuracyMetric(),
            PrecisionMetric(average='macro'),
            RecallMetric(average='macro'),
            F1ScoreMetric(average='macro')
        ],
        window_size=3  # 최근 3개 값의 이동 평균
    )

    print("Training simulation (10 epochs):")
    print()

    # 10 에폭 시뮬레이션
    for epoch in range(1, 11):
        predictions, labels, _, _ = simulate_training_epoch(epoch)

        # 메트릭 업데이트
        tracker.update(predictions, labels)

        # 현재 메트릭 출력
        latest = tracker.get_latest()
        print(f"Epoch {epoch:2d}:")
        for metric_name, value in latest.items():
            print(f"  {metric_name:12s}: {value:.4f}")

        # 이동 평균 출력 (3 에폭 이후부터)
        if epoch >= 3:
            moving_avg = tracker.get_moving_average()
            print(f"  Moving Average (last 3 epochs):")
            for metric_name, value in moving_avg.items():
                if value is not None:
                    print(f"    {metric_name:12s}: {value:.4f}")

        print()

    # 최종 요약
    print("\n" + "=" * 70)
    print(" Training Summary")
    print("=" * 70)
    tracker.summary()

    # 최고 성능
    print("\nBest Performance:")
    for metric_name in ['Accuracy', 'F1Score']:
        best = tracker.get_best(metric_name)
        if best is not None:
            print(f"  Best {metric_name}: {best:.4f}")

    # ========================================================================
    # 2. 회귀 메트릭 추적
    # ========================================================================
    print("\n" + "=" * 70)
    print("[2] Regression Metrics Tracking")
    print("-" * 50)

    # 회귀용 MetricTracker
    reg_tracker = MetricTracker(
        metrics=[
            MSEMetric(),
            MAEMetric(),
            R2Metric()
        ],
        window_size=5
    )

    print("\nRegression training simulation (10 epochs):")
    print()

    for epoch in range(1, 11):
        _, _, reg_predictions, reg_targets = simulate_training_epoch(epoch)

        # 에폭이 진행될수록 예측 개선 시뮬레이션
        noise_scale = 2.0 / (1 + epoch * 0.2)
        reg_predictions = reg_targets + torch.randn_like(reg_targets) * noise_scale

        # 메트릭 업데이트
        reg_tracker.update(reg_predictions, reg_targets)

        # 현재 메트릭 출력
        latest = reg_tracker.get_latest()
        print(f"Epoch {epoch:2d}:")
        for metric_name, value in latest.items():
            print(f"  {metric_name:12s}: {value:.4f}")

    # 회귀 요약
    print("\n" + "=" * 70)
    print(" Regression Training Summary")
    print("=" * 70)
    reg_tracker.summary()

    # ========================================================================
    # 3. 히스토리 분석
    # ========================================================================
    print("\n" + "=" * 70)
    print("[3] History Analysis")
    print("-" * 50)

    # 특정 메트릭의 전체 히스토리 가져오기
    acc_history = tracker.get_history('Accuracy')
    if acc_history:
        print("\nAccuracy History:")
        for i, acc in enumerate(acc_history, 1):
            bar = "█" * int(acc * 50)  # 시각적 막대 그래프
            print(f"  Epoch {i:2d}: {bar} {acc:.4f}")

        # 통계 분석
        acc_array = np.array(acc_history)
        print(f"\n  Statistics:")
        print(f"    Mean:   {np.mean(acc_array):.4f}")
        print(f"    Std:    {np.std(acc_array):.4f}")
        print(f"    Min:    {np.min(acc_array):.4f}")
        print(f"    Max:    {np.max(acc_array):.4f}")
        print(f"    Trend:  {'↑' if acc_array[-1] > acc_array[0] else '↓'} "
              f"({(acc_array[-1] - acc_array[0])*100:.2f}% change)")

    # ========================================================================
    # 4. 메트릭 리셋 데모
    # ========================================================================
    print("\n" + "=" * 70)
    print("[4] Metric Reset Demo")
    print("-" * 50)

    print("\nBefore reset:")
    print(f"  Accuracy history length: {len(tracker.get_history('Accuracy'))}")

    # 특정 메트릭만 리셋
    tracker.reset('Accuracy')
    print("\nAfter resetting Accuracy:")
    print(f"  Accuracy history length: {len(tracker.get_history('Accuracy'))}")
    print(f"  Precision history length: {len(tracker.get_history('Precision'))}")

    # 모든 메트릭 리셋
    tracker.reset()
    print("\nAfter resetting all metrics:")
    print(f"  All histories cleared: {all(len(h) == 0 for h in tracker.get_history().values())}")

    # ========================================================================
    # 5. 사용 권장사항
    # ========================================================================
    print("\n" + "=" * 70)
    print(" Usage Recommendations")
    print("=" * 70)
    print("""
    1. Window Size 선택:
       - 작은 값 (3-5): 빠른 변화 감지, 노이즈에 민감
       - 큰 값 (10-20): 안정적인 추세, 느린 반응

    2. 메트릭 선택:
       - 분류: Accuracy + F1Score (불균형 데이터)
       - 회귀: MSE + MAE + R2 (종합적 평가)
       - 이진 분류: Precision + Recall + F1Score

    3. 활용 팁:
       - Early Stopping: get_best()와 현재 값 비교
       - Learning Rate 조정: 이동 평균이 정체될 때
       - 과적합 감지: 훈련/검증 메트릭 차이 추적

    4. 메모리 관리:
       - 긴 학습: 주기적으로 reset() 호출
       - 또는 window_size 설정으로 자동 관리
    """)

    print("\n메트릭 시스템 데모 완료!")


if __name__ == "__main__":
    main()