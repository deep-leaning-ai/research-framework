"""
비교 시스템 통합 테스트

research의 모델 비교 시스템을 테스트합니다.
- ComparisonManager: 여러 비교기 통합 관리
- PerformanceComparator: 성능 기반 비교
- EfficiencyComparator: 효율성 기반 비교
- SpeedComparator: 속도 기반 비교
"""

import sys
import numpy as np

print("="*70)
print("비교 시스템 통합 테스트")
print("="*70)

# 1. Import 테스트
print("\n1. Testing imports...")
try:
    from research import (
        ComparisonManager,
        PerformanceComparator,
        EfficiencyComparator,
        SpeedComparator,
        ExperimentResult
    )
    print("   [OK] All comparison classes imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. 더미 실험 결과 생성
print("\n2. Creating dummy experiment results...")
try:
    results = {}

    # 3개 모델의 더미 결과 생성
    models_info = [
        {
            'name': 'ResNet18',
            'params': 11_000_000,
            'inference_time': 0.05,
            'base_accuracy': 0.85
        },
        {
            'name': 'ResNet50',
            'params': 25_000_000,
            'inference_time': 0.12,
            'base_accuracy': 0.88
        },
        {
            'name': 'MobileNet',
            'params': 4_000_000,
            'inference_time': 0.03,
            'base_accuracy': 0.82
        }
    ]

    for model_info in models_info:
        num_epochs = 10

        # 학습 곡선 생성 (점진적으로 향상)
        base_acc = model_info['base_accuracy']
        train_acc = [base_acc - 0.15 + i*0.02 + np.random.rand()*0.01 for i in range(num_epochs)]
        val_acc = [base_acc - 0.10 + i*0.015 + np.random.rand()*0.015 for i in range(num_epochs)]
        test_acc = [base_acc - 0.08 + i*0.012 + np.random.rand()*0.02 for i in range(num_epochs)]

        train_loss = [1.5 - i*0.1 + np.random.rand()*0.05 for i in range(num_epochs)]
        val_loss = [1.6 - i*0.08 + np.random.rand()*0.08 for i in range(num_epochs)]
        test_loss = [1.65 - i*0.07 + np.random.rand()*0.1 for i in range(num_epochs)]

        # 메트릭 딕셔너리
        train_metrics = {'Accuracy': train_acc}
        val_metrics = {'Accuracy': val_acc}
        test_metrics = {'Accuracy': test_acc}

        # 에폭 타임 (파라미터 수에 비례)
        base_time = 2.0 * (model_info['params'] / 10_000_000)
        epoch_times = [base_time + np.random.rand()*0.5 for _ in range(num_epochs)]

        # ExperimentResult 생성
        result = ExperimentResult(
            model_name=model_info['name'],
            task_type='MultiClassStrategy',
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            primary_metric_name='Accuracy',
            best_test_metric=max(test_acc),
            parameters=model_info['params'],
            epoch_times=epoch_times,
            inference_time=model_info['inference_time']
        )

        results[model_info['name']] = result

    print(f"   [OK] Created {len(results)} dummy experiment results")
    print(f"   Models: {list(results.keys())}")
    for name, result in results.items():
        print(f"     - {name}: {result.parameters:,} params, "
              f"best acc={result.best_test_metric:.4f}, "
              f"inference={result.inference_time*1000:.1f}ms")

except Exception as e:
    print(f"   ✗ Failed to create dummy results: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. PerformanceComparator 테스트
print("\n3. Testing PerformanceComparator...")
try:
    comparator = PerformanceComparator(metric_name='Accuracy', higher_better=True)

    comparison_result = comparator.compare(results)

    assert 'ranking' in comparison_result, "Result should contain 'ranking'"
    assert 'best_model' in comparison_result, "Result should contain 'best_model'"
    assert 'best_score' in comparison_result, "Result should contain 'best_score'"
    assert len(comparison_result['ranking']) == 3, "Should have 3 models in ranking"

    print(f"   [OK] PerformanceComparator test passed")
    print(f"     - Best model: {comparison_result['best_model']}")
    print(f"     - Best score: {comparison_result['best_score']:.4f}")
    print(f"     - Ranking:")
    for rank, (model, score) in enumerate(comparison_result['ranking'], 1):
        print(f"       {rank}. {model}: {score:.4f}")

except Exception as e:
    print(f"   ✗ PerformanceComparator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. EfficiencyComparator 테스트
print("\n4. Testing EfficiencyComparator...")
try:
    comparator = EfficiencyComparator(metric_name='Accuracy')

    comparison_result = comparator.compare(results)

    assert 'ranking' in comparison_result, "Result should contain 'ranking'"
    assert 'best_model' in comparison_result, "Result should contain 'best_model'"
    assert len(comparison_result['ranking']) == 3, "Should have 3 models in ranking"

    print(f"   [OK] EfficiencyComparator test passed")
    print(f"     - Most efficient model: {comparison_result['best_model']}")
    print(f"     - Ranking (efficiency = performance / log10(params)):")
    for rank, (model, scores) in enumerate(comparison_result['ranking'], 1):
        print(f"       {rank}. {model}:")
        print(f"          Efficiency: {scores['efficiency']:.4f}")
        print(f"          Performance: {scores['performance']:.4f}")
        print(f"          Parameters: {scores['parameters']:,}")

except Exception as e:
    print(f"   ✗ EfficiencyComparator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. SpeedComparator 테스트
print("\n5. Testing SpeedComparator...")
try:
    comparator = SpeedComparator()

    comparison_result = comparator.compare(results)

    assert 'ranking' in comparison_result, "Result should contain 'ranking'"
    assert 'fastest_model' in comparison_result, "Result should contain 'fastest_model'"
    assert 'fastest_time' in comparison_result, "Result should contain 'fastest_time'"
    assert len(comparison_result['ranking']) == 3, "Should have 3 models in ranking"

    print(f"   [OK] SpeedComparator test passed")
    print(f"     - Fastest model: {comparison_result['fastest_model']}")
    print(f"     - Fastest inference time: {comparison_result['fastest_time']*1000:.2f}ms")
    print(f"     - Ranking:")
    for rank, (model, speeds) in enumerate(comparison_result['ranking'], 1):
        print(f"       {rank}. {model}:")
        print(f"          Inference time: {speeds['inference_time']*1000:.2f}ms")
        print(f"          Avg epoch time: {speeds['avg_epoch_time']:.2f}s")

except Exception as e:
    print(f"   ✗ SpeedComparator test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. ComparisonManager 통합 테스트
print("\n6. Testing ComparisonManager integration...")
try:
    manager = ComparisonManager()

    # 3개의 비교기 추가
    manager.add_comparator(PerformanceComparator('Accuracy', higher_better=True))
    manager.add_comparator(EfficiencyComparator('Accuracy'))
    manager.add_comparator(SpeedComparator())

    # 모든 비교 실행
    all_results = manager.run_all_comparisons(results)

    assert len(all_results) == 3, "Should have 3 comparison results"
    assert 'Performance (Accuracy)' in all_results, "Should have performance comparison"
    assert 'Efficiency (Accuracy)' in all_results, "Should have efficiency comparison"
    assert 'Speed Comparison' in all_results, "Should have speed comparison"

    print(f"\n   [OK] ComparisonManager integration test passed")
    print(f"     - Total comparisons: {len(all_results)}")
    print(f"     - Comparison types: {list(all_results.keys())}")

except Exception as e:
    print(f"   ✗ ComparisonManager integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. ComparisonManager 리포트 저장 테스트
print("\n7. Testing ComparisonManager report export...")
try:
    report_path = "test_comparison_report.txt"
    manager.export_comparison_report(save_path=report_path)

    import os
    if os.path.exists(report_path):
        print(f"   [OK] Comparison report exported successfully")
        print(f"     - File: {report_path}")
        print(f"     - Size: {os.path.getsize(report_path)} bytes")

        # 파일 내용 미리보기
        with open(report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            print(f"     - Preview (first 10 lines):")
            for line in lines:
                print(f"       {line.rstrip()}")
    else:
        print(f"   ✗ Report file was not created")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ Report export test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("[완료] 비교 시스템 통합 테스트 완료!")
print("="*70)
print("\n주요 확인 사항:")
print("  [OK] PerformanceComparator 정상 작동 (메트릭 기준 순위)")
print("  [OK] EfficiencyComparator 정상 작동 (파라미터 대비 효율성)")
print("  [OK] SpeedComparator 정상 작동 (추론/학습 속도)")
print("  [OK] ComparisonManager 통합 관리 정상 작동")
print("  [OK] 리포트 파일 생성 기능 정상 작동")
print(f"\n생성된 파일: {report_path}")
print("\n비교 시스템이 research에 완전히 통합되었습니다!")
