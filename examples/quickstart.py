"""
KTB ML Framework - 빠른 시작 예제

이 예제는 프레임워크의 전체 워크플로우를 보여줍니다:
1. 메트릭 트래커 생성
2. 간단한 학습 루프 시뮬레이션
3. 실험 결과 기록
4. 모델 비교
5. 시각화 생성
"""

import torch
import torch.nn as nn
import numpy as np

print("="*70)
print("KTB ML Framework - 빠른 시작 예제")
print("="*70)

# Step 1: Import 필요한 컴포넌트
print("\nStep 1: 필요한 컴포넌트 import...")
from research import (
    # 메트릭 시스템
    MetricTracker,
    AccuracyMetric,
    F1ScoreMetric,

    # 실험 관리
    ExperimentRecorder,
    ExperimentResult,

    # 비교 시스템
    ComparisonManager,
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator,

    # 시각화
    ExperimentVisualizer,
)
print("   [OK] Import 완료")

# Step 2: 메트릭 트래커 생성
print("\nStep 2: 메트릭 트래커 생성...")
tracker = MetricTracker([
    AccuracyMetric(),
    F1ScoreMetric(average='macro')
])
print(f"   [OK] 메트릭 트래커 생성 완료 ({len(tracker.metrics)}개 메트릭)")

# Step 3: 실험 시뮬레이션 (3개 모델)
print("\nStep 3: 실험 시뮬레이션 (3개 모델)...")
recorder = ExperimentRecorder()

models_config = [
    {'name': 'Model_A', 'params': 5_000_000, 'hidden_size': 128},
    {'name': 'Model_B', 'params': 10_000_000, 'hidden_size': 256},
    {'name': 'Model_C', 'params': 2_500_000, 'hidden_size': 64},
]

for model_config in models_config:
    print(f"\n   모델: {model_config['name']}")

    # 간단한 더미 모델 생성
    model = nn.Sequential(
        nn.Linear(784, model_config['hidden_size']),
        nn.ReLU(),
        nn.Linear(model_config['hidden_size'], 10)
    )

    # 학습 시뮬레이션
    num_epochs = 5
    train_metrics_history = {'Accuracy': [], 'F1-Score (macro)': []}
    val_metrics_history = {'Accuracy': [], 'F1-Score (macro)': []}
    test_metrics_history = {'Accuracy': [], 'F1-Score (macro)': []}

    train_loss_history = []
    val_loss_history = []
    test_loss_history = []
    epoch_times = []

    for epoch in range(num_epochs):
        # 더미 데이터 생성
        train_outputs = torch.randn(100, 10)
        train_labels = torch.randint(0, 10, (100,))

        val_outputs = torch.randn(50, 10)
        val_labels = torch.randint(0, 10, (50,))

        test_outputs = torch.randn(50, 10)
        test_labels = torch.randint(0, 10, (50,))

        # 메트릭 계산
        train_metrics = {}
        train_metrics['Accuracy'] = AccuracyMetric().calculate(train_outputs, train_labels)
        train_metrics['F1-Score (macro)'] = F1ScoreMetric(average='macro').calculate(train_outputs, train_labels)

        val_metrics = {}
        val_metrics['Accuracy'] = AccuracyMetric().calculate(val_outputs, val_labels)
        val_metrics['F1-Score (macro)'] = F1ScoreMetric(average='macro').calculate(val_outputs, val_labels)

        test_metrics = {}
        test_metrics['Accuracy'] = AccuracyMetric().calculate(test_outputs, test_labels)
        test_metrics['F1-Score (macro)'] = F1ScoreMetric(average='macro').calculate(test_outputs, test_labels)

        # 히스토리에 추가
        for key in train_metrics:
            train_metrics_history[key].append(train_metrics[key])
            val_metrics_history[key].append(val_metrics[key])
            test_metrics_history[key].append(test_metrics[key])

        # 손실 (더미)
        train_loss_history.append(1.5 - epoch * 0.2 + np.random.rand() * 0.1)
        val_loss_history.append(1.6 - epoch * 0.18 + np.random.rand() * 0.1)
        test_loss_history.append(1.65 - epoch * 0.15 + np.random.rand() * 0.1)

        # 에폭 시간 (더미)
        epoch_times.append(1.0 + np.random.rand() * 0.5)

        print(f"     Epoch {epoch + 1}/{num_epochs} - "
              f"Train Acc: {train_metrics['Accuracy']:.2f}%, "
              f"Val Acc: {val_metrics['Accuracy']:.2f}%, "
              f"Test Acc: {test_metrics['Accuracy']:.2f}%")

    # 실험 결과 생성
    result = ExperimentResult(
        model_name=model_config['name'],
        task_type='MultiClassStrategy',
        train_loss=train_loss_history,
        val_loss=val_loss_history,
        test_loss=test_loss_history,
        train_metrics=train_metrics_history,
        val_metrics=val_metrics_history,
        test_metrics=test_metrics_history,
        primary_metric_name='Accuracy',
        best_test_metric=max(test_metrics_history['Accuracy']),
        parameters=model_config['params'],
        epoch_times=epoch_times,
        inference_time=0.01 + np.random.rand() * 0.05
    )

    # 기록기에 추가
    recorder.add_result(result)

    print(f"     [OK] Best Test Accuracy: {result.best_test_metric:.2f}%")

print(f"\n   [OK] 총 {len(recorder.get_all_results())}개 모델 실험 완료")

# Step 4: 실험 결과 요약
print("\nStep 4: 실험 결과 요약...")
recorder.print_summary()

# Step 5: 모델 비교
print("\nStep 5: 모델 비교...")
manager = ComparisonManager()

# 3가지 비교 방식 추가
manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))
manager.add_comparator(EfficiencyComparator('Accuracy'))
manager.add_comparator(SpeedComparator())

# 모든 비교 실행
comparison_results = manager.run_all_comparisons(recorder.get_all_results())

# Step 6: 비교 리포트 저장
print("\nStep 6: 비교 리포트 저장...")
report_path = "quickstart_comparison_report.txt"
manager.export_comparison_report(save_path=report_path)
print(f"   [OK] 리포트 저장: {report_path}")

# Step 7: 시각화 생성
print("\nStep 7: 시각화 생성...")
visualization_path = "quickstart_visualization.png"
ExperimentVisualizer.plot_comparison(
    recorder=recorder,
    save_path=visualization_path
)
print(f"   [OK] 시각화 저장: {visualization_path}")

# Step 8: 최고 모델 찾기
print("\nStep 8: 최고 모델 결정...")
best_model_name = recorder.get_best_model('Accuracy', higher_better=True)
best_result = recorder.get_result(best_model_name)

print(f"""
   [OK] 최고 성능 모델: {best_model_name}
     - Test Accuracy: {best_result.best_test_metric:.2f}%
     - Parameters: {best_result.parameters:,}
     - Inference Time: {best_result.inference_time*1000:.2f}ms
""")

# 요약
print("="*70)
print("[완료] 빠른 시작 예제 완료!")
print("="*70)
print("\n생성된 파일:")
print(f"  - {report_path}")
print(f"  - {visualization_path}")
print("\n주요 단계:")
print("  1. [OK] 메트릭 트래커 생성")
print("  2. [OK] 3개 모델 학습 시뮬레이션")
print("  3. [OK] 실험 결과 기록")
print("  4. [OK] 모델 성능/효율성/속도 비교")
print("  5. [OK] 시각화 생성")
print("  6. [OK] 최고 모델 결정")
print("\n다음 단계:")
print("  - examples/test_*.py 파일들을 참고하여 각 기능 심화 학습")
print("  - README.md의 API 문서 참고")
print("  - 실제 데이터셋으로 전이학습 실험 진행")
print("\nKTB ML Framework를 사용해주셔서 감사합니다!")
print("="*70)
