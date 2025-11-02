"""
시각화 시스템 통합 테스트

research의 시각화 시스템을 테스트합니다.
- ExperimentVisualizer: 8-panel 종합 비교 차트
- plot_confusion_matrix: Confusion matrix 시각화
- plot_comprehensive_comparison: 종합 성능 비교
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI가 없는 환경에서도 작동하도록
import matplotlib.pyplot as plt

print("="*70)
print("시각화 시스템 통합 테스트")
print("="*70)

# 1. Import 테스트
print("\n1. Testing imports...")
try:
    from research import (
        ExperimentVisualizer,
        ExperimentRecorder,
        ExperimentResult,
        plot_confusion_matrix
    )
    print("   [OK] All visualization classes imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. ExperimentRecorder 생성 및 더미 데이터 추가
print("\n2. Creating ExperimentRecorder with dummy data...")
try:
    recorder = ExperimentRecorder()

    # 더미 실험 결과 생성 (2개 모델)
    for model_idx, model_name in enumerate(['ResNet18', 'VGG16']):
        num_epochs = 10

        # 학습 곡선 생성 (점진적으로 향상)
        train_loss = [1.5 - i*0.1 + np.random.rand()*0.05 for i in range(num_epochs)]
        val_loss = [1.6 - i*0.08 + np.random.rand()*0.08 for i in range(num_epochs)]
        test_loss = [1.65 - i*0.07 + np.random.rand()*0.1 for i in range(num_epochs)]

        # 정확도 (높아지는 경향)
        train_acc = [0.4 + i*0.05 + np.random.rand()*0.02 for i in range(num_epochs)]
        val_acc = [0.38 + i*0.048 + np.random.rand()*0.03 for i in range(num_epochs)]
        test_acc = [0.37 + i*0.047 + np.random.rand()*0.03 for i in range(num_epochs)]

        # 메트릭 딕셔너리
        train_metrics = {'Accuracy': train_acc}
        val_metrics = {'Accuracy': val_acc}
        test_metrics = {'Accuracy': test_acc}

        # 에폭 타임
        epoch_times = [2.5 + np.random.rand()*0.5 for _ in range(num_epochs)]

        # ExperimentResult 생성
        result = ExperimentResult(
            model_name=model_name,
            task_type='MultiClassStrategy',
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            primary_metric_name='Accuracy',
            best_test_metric=max(test_acc),
            parameters=11000000 + model_idx * 3000000,  # 11M, 14M
            epoch_times=epoch_times,
            inference_time=0.05 + model_idx * 0.02  # 50ms, 70ms
        )

        recorder.add_result(result)

    print(f"   [OK] ExperimentRecorder created with {len(recorder.get_all_results())} experiments")
    print(f"   Models: {list(recorder.get_all_results().keys())}")

except Exception as e:
    print(f"   ✗ Failed to create ExperimentRecorder: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. ExperimentVisualizer.plot_comparison 테스트 (8-panel chart)
print("\n3. Testing ExperimentVisualizer.plot_comparison (8-panel chart)...")
try:
    save_path = "test_visualization_8panel.png"
    ExperimentVisualizer.plot_comparison(
        recorder=recorder,
        save_path=save_path
    )

    # 파일이 생성되었는지 확인
    import os
    if os.path.exists(save_path):
        print(f"   [OK] 8-panel chart saved to '{save_path}'")
        print(f"   File size: {os.path.getsize(save_path)} bytes")
    else:
        print(f"   ✗ File '{save_path}' was not created")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ plot_comparison failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. plot_confusion_matrix 테스트
print("\n4. Testing plot_confusion_matrix...")
try:
    # 더미 confusion matrix 생성 (3 classes)
    cm = np.array([
        [45, 3, 2],   # True class 0
        [5, 42, 3],   # True class 1
        [2, 4, 44]    # True class 2
    ])

    class_names = ['Class A', 'Class B', 'Class C']

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        title="Test Confusion Matrix",
        ax=ax,
        normalize=True
    )

    save_path = "test_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    import os
    if os.path.exists(save_path):
        print(f"   [OK] Confusion matrix saved to '{save_path}'")
        print(f"   File size: {os.path.getsize(save_path)} bytes")
    else:
        print(f"   ✗ File '{save_path}' was not created")

except Exception as e:
    print(f"   ✗ plot_confusion_matrix failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. ExperimentVisualizer.plot_metric_comparison 테스트
print("\n5. Testing ExperimentVisualizer.plot_metric_comparison...")
try:
    save_path = "test_metric_comparison.png"
    ExperimentVisualizer.plot_metric_comparison(
        recorder=recorder,
        metric_name='Accuracy',
        save_path=save_path
    )

    import os
    if os.path.exists(save_path):
        print(f"   [OK] Metric comparison chart saved to '{save_path}'")
        print(f"   File size: {os.path.getsize(save_path)} bytes")
    else:
        print(f"   ✗ File '{save_path}' was not created")

except Exception as e:
    print(f"   ✗ plot_metric_comparison failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("[완료] 시각화 시스템 통합 테스트 완료!")
print("="*70)
print("\n주요 확인 사항:")
print("  [OK] ExperimentVisualizer 정상 작동")
print("  [OK] 8-panel 종합 비교 차트 생성 가능")
print("  [OK] Confusion matrix 시각화 가능")
print("  [OK] 특정 메트릭 비교 시각화 가능")
print("\n생성된 파일:")
print("  - test_visualization_8panel.png")
print("  - test_confusion_matrix.png")
print("  - test_metric_comparison.png")
print("\n시각화 시스템이 research에 완전히 통합되었습니다!")
