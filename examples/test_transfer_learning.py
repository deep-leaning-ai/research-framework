"""
전이학습 전략 비교 예제

Feature Extraction vs Fine-tuning 전략을 비교합니다.
각 전략의 장단점과 사용 시나리오를 보여줍니다.

실행 방법:
    $ python examples/test_transfer_learning.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from research import (
    Experiment,
    CIFAR10DataModule,
    VanillaTrainingStrategy,
    MultiClassStrategy,
    ComparisonManager,
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator,
    ExperimentRecorder
)


def main():
    """전이학습 전략 비교"""
    print("=" * 70)
    print(" Transfer Learning Strategy Comparison")
    print("=" * 70)
    print()

    # ========================================================================
    # 데이터 준비
    # ========================================================================
    print("[1] Preparing CIFAR-10 dataset...")
    data_module = CIFAR10DataModule(
        data_dir="./data",
        batch_size=64,  # 더 큰 배치 크기
        num_workers=4,
        image_size=224,  # ResNet/VGG 입력 크기
        use_imagenet_norm=True,  # ImageNet 정규화 사용
        persistent_workers=True,  # 성능 최적화
        prefetch_factor=2
    )
    print("✓ Data module ready")
    print()

    # ========================================================================
    # 여러 모델로 실험
    # ========================================================================
    models_to_test = ['resnet18', 'resnet50', 'vgg16']
    all_results = {}

    for model_name in models_to_test:
        print(f"[2] Testing {model_name}...")
        print("-" * 50)

        # 실험 설정
        config = {
            'num_classes': 10,
            'in_channels': 3,
            'learning_rate': 1e-4,
            'max_epochs': 3,  # 데모용 짧은 학습
            'batch_size': 64,
            'optimizer': 'adam'
        }

        # 실험 생성
        exp = Experiment(config)

        # Task와 Training 전략
        task_strategy = MultiClassStrategy(num_classes=10)
        training_strategy = VanillaTrainingStrategy(
            device=None,
            task_strategy=task_strategy
        )

        # 실험 설정
        exp.setup(
            model_name=model_name,
            data_module=data_module,
            training_strategy=training_strategy,
            logging_strategy=None  # 로깅 생략
        )

        # ====================================================================
        # Feature Extraction vs Fine-tuning 비교
        # ====================================================================
        print(f"\n  Comparing strategies for {model_name}:")

        strategies_to_compare = ['feature_extraction', 'fine_tuning']
        comparison = exp.compare_strategies(
            strategies=strategies_to_compare,
            reset_model=True  # 각 실험마다 모델 리셋
        )

        # 결과 저장
        for strategy, result in comparison.items():
            key = f"{model_name}_{strategy}"
            all_results[key] = result

        print()

    # ========================================================================
    # 결과 분석
    # ========================================================================
    print("\n" + "=" * 70)
    print(" Analysis Results")
    print("=" * 70)

    # ExperimentRecorder에 결과 추가
    recorder = ExperimentRecorder()

    for name, result in all_results.items():
        from research.experiment.result import ExperimentResult

        exp_result = ExperimentResult(
            model_name=name,
            task_type="MultiClass",
            parameters=result['model_info']['total_parameters'],
            train_metrics={'accuracy': result['train_results'].get('history', {}).get('train_acc', [])},
            val_metrics={'accuracy': result['train_results'].get('history', {}).get('val_acc', [])},
            test_metrics={'accuracy': [result['test_results'].get('test_acc', 0)]},
            train_loss=result['train_results'].get('history', {}).get('train_loss', []),
            val_loss=result['train_results'].get('history', {}).get('val_loss', []),
            test_loss=[result['test_results'].get('test_loss', 0)],
            epoch_times=result['train_results'].get('epoch_times', []),
            inference_time=result['test_results'].get('inference_time', 0),
            primary_metric_name='accuracy',
            best_test_metric=result['test_results'].get('test_acc', 0),
            final_overfitting_gap=0.0
        )
        recorder.add_result(exp_result)

    # ========================================================================
    # ComparisonManager로 분석
    # ========================================================================
    print("\n[3] Running comparison analysis...")
    manager = ComparisonManager()

    # 여러 기준으로 비교
    manager.add_comparator(PerformanceComparator('accuracy'))
    manager.add_comparator(EfficiencyComparator('accuracy'))
    manager.add_comparator(SpeedComparator())

    # 비교 실행
    all_comparisons = manager.run_all_comparisons(recorder.get_all_results())

    # ========================================================================
    # 주요 발견사항 출력
    # ========================================================================
    print("\n" + "=" * 70)
    print(" Key Findings")
    print("=" * 70)

    # Feature Extraction vs Fine-tuning 분석
    print("\n[Feature Extraction vs Fine-tuning]")
    print("-" * 50)

    for model in models_to_test:
        fe_key = f"{model}_feature_extraction"
        ft_key = f"{model}_fine_tuning"

        if fe_key in all_results and ft_key in all_results:
            fe_result = all_results[fe_key]
            ft_result = all_results[ft_key]

            fe_acc = fe_result['test_results'].get('test_acc', 0)
            ft_acc = ft_result['test_results'].get('test_acc', 0)

            fe_time = fe_result['train_results'].get('training_time', 0)
            ft_time = ft_result['train_results'].get('training_time', 0)

            fe_params = fe_result['model_info']['trainable_parameters']
            ft_params = ft_result['model_info']['trainable_parameters']

            print(f"\n{model}:")
            print(f"  Feature Extraction:")
            print(f"    - Accuracy: {fe_acc:.2f}%")
            print(f"    - Training time: {fe_time:.2f}s")
            print(f"    - Trainable params: {fe_params:,}")

            print(f"  Fine-tuning:")
            print(f"    - Accuracy: {ft_acc:.2f}%")
            print(f"    - Training time: {ft_time:.2f}s")
            print(f"    - Trainable params: {ft_params:,}")

            print(f"  Analysis:")
            if ft_acc > fe_acc:
                improvement = ft_acc - fe_acc
                print(f"    ✓ Fine-tuning improved accuracy by {improvement:.2f}%")
            else:
                print(f"    ✓ Feature extraction achieved comparable results")

            time_ratio = ft_time / fe_time if fe_time > 0 else 0
            print(f"    ✓ Fine-tuning took {time_ratio:.1f}x longer")
            print(f"    ✓ Fine-tuning used {ft_params/fe_params:.1f}x more parameters")

    # ========================================================================
    # 권장사항
    # ========================================================================
    print("\n" + "=" * 70)
    print(" Recommendations")
    print("=" * 70)
    print("""
    1. Feature Extraction 사용 시나리오:
       - 학습 데이터가 적을 때 (< 1000 샘플)
       - 빠른 프로토타이핑이 필요할 때
       - 계산 자원이 제한적일 때
       - 타겟 도메인이 ImageNet과 유사할 때

    2. Fine-tuning 사용 시나리오:
       - 충분한 학습 데이터가 있을 때 (> 5000 샘플)
       - 최고의 성능이 필요할 때
       - 충분한 계산 자원이 있을 때
       - 타겟 도메인이 ImageNet과 다를 때

    3. 하이브리드 접근법:
       - 먼저 Feature Extraction으로 빠르게 학습
       - 그 다음 낮은 학습률로 Fine-tuning
       - 이를 "Progressive Fine-tuning"이라고 함
    """)

    print("\n실험 완료!")


if __name__ == "__main__":
    main()