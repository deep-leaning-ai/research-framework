"""
KTB ML Framework - 빠른 시작 가이드

이 예제는 프레임워크의 전체 워크플로우를 보여줍니다:
1. 데이터 준비
2. 실험 설정
3. 모델 학습
4. 성능 평가
5. 결과 시각화

실행 방법:
    $ python examples/quickstart.py
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from research import (
    Experiment,
    CIFAR10DataModule,
    VanillaTrainingStrategy,
    SimpleLoggingStrategy,
    MultiClassStrategy,
    ExperimentVisualizer,
    ExperimentRecorder
)


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print(" KTB ML Framework - Quick Start Example")
    print("=" * 70)
    print()

    # ========================================================================
    # 1. 데이터 준비
    # ========================================================================
    print("[Step 1] Preparing data...")
    data_module = CIFAR10DataModule(
        data_dir="./data",
        batch_size=32,
        num_workers=4,
        image_size=224,
        use_imagenet_norm=True  # 전이학습용 정규화
    )
    print("✓ CIFAR-10 data module created")
    print()

    # ========================================================================
    # 2. 실험 설정
    # ========================================================================
    print("[Step 2] Configuring experiment...")
    config = {
        'num_classes': 10,
        'in_channels': 3,
        'learning_rate': 1e-4,
        'max_epochs': 2,  # 빠른 데모를 위해 2 에폭만
        'batch_size': 32,
        'optimizer': 'adam'
    }

    exp = Experiment(config)
    print("✓ Experiment configured")
    print(f"  Config: {config}")
    print()

    # ========================================================================
    # 3. 실험 환경 설정
    # ========================================================================
    print("[Step 3] Setting up experiment environment...")

    # Task strategy (다중 분류)
    task_strategy = MultiClassStrategy(num_classes=10)

    # Training strategy
    training_strategy = VanillaTrainingStrategy(
        device=None,  # 자동 선택
        task_strategy=task_strategy
    )

    # Logging strategy (옵션)
    logging_strategy = SimpleLoggingStrategy()

    exp.setup(
        model_name='resnet18',  # 작은 모델로 빠른 테스트
        data_module=data_module,
        training_strategy=training_strategy,
        logging_strategy=logging_strategy
    )
    print("✓ Environment setup complete")
    print()

    # ========================================================================
    # 4. 단일 전략 실행 (Fine-tuning)
    # ========================================================================
    print("[Step 4] Running fine-tuning strategy...")
    result = exp.run(strategy='fine_tuning', run_name='quickstart_demo')

    print("\nResults:")
    print(f"  Model Info:")
    print(f"    - Total params: {result['model_info']['total_parameters']:,}")
    print(f"    - Trainable params: {result['model_info']['trainable_parameters']:,}")

    if 'test_acc' in result['test_results']:
        print(f"  Test Accuracy: {result['test_results']['test_acc']:.2f}%")
    elif 'accuracy' in result['test_results']:
        print(f"  Test Accuracy: {result['test_results']['accuracy']:.2f}%")

    if 'training_time' in result['train_results']:
        print(f"  Training Time: {result['train_results']['training_time']:.2f}s")
    print()

    # ========================================================================
    # 5. 전략 비교 (옵션)
    # ========================================================================
    print("[Step 5] Comparing different strategies...")
    comparison = exp.compare_strategies(
        strategies=['feature_extraction', 'fine_tuning'],
        reset_model=True
    )
    print()

    # ========================================================================
    # 6. 결과 시각화
    # ========================================================================
    print("[Step 6] Visualizing results...")

    # ExperimentRecorder를 사용하여 결과 수집
    recorder = ExperimentRecorder()

    # 각 전략의 결과를 ExperimentResult 형식으로 변환
    for strategy_name, strategy_result in comparison.items():
        from research.experiment.result import ExperimentResult

        # ExperimentResult 객체 생성
        exp_result = ExperimentResult(
            model_name=f"ResNet18_{strategy_name}",
            task_type="MultiClass",
            parameters=strategy_result['model_info']['total_parameters'],
            train_metrics=strategy_result['train_results'].get('history', {}).get('train_acc', []),
            val_metrics=strategy_result['train_results'].get('history', {}).get('val_acc', []),
            test_metrics={'accuracy': [strategy_result['test_results'].get('test_acc', 0)]},
            train_loss=strategy_result['train_results'].get('history', {}).get('train_loss', []),
            val_loss=strategy_result['train_results'].get('history', {}).get('val_loss', []),
            test_loss=[strategy_result['test_results'].get('test_loss', 0)],
            epoch_times=strategy_result['train_results'].get('epoch_times', []),
            inference_time=strategy_result['test_results'].get('inference_time', 0),
            primary_metric_name='accuracy',
            best_test_metric=strategy_result['test_results'].get('test_acc', 0),
            final_overfitting_gap=0.0
        )

        recorder.add_result(exp_result)

    # 시각화 생성
    save_path = 'quickstart_comparison.png'
    ExperimentVisualizer.plot_comparison(
        recorder,
        save_path=save_path
    )
    print(f"✓ Visualization saved to {save_path}")
    print()

    # ========================================================================
    # 7. 실험 결과 저장
    # ========================================================================
    print("[Step 7] Saving experiment results...")
    exp.save_results('quickstart_results.json')
    print("✓ Results saved to quickstart_results.json")
    print()

    print("=" * 70)
    print(" Quick Start Complete!")
    print("=" * 70)
    print("\n다음 단계:")
    print("  1. config의 max_epochs를 늘려서 더 오래 학습해보세요")
    print("  2. 다른 모델 (resnet50, vgg16 등)을 시도해보세요")
    print("  3. learning_rate를 조정해보세요")
    print("  4. WandBLoggingStrategy로 실험을 추적해보세요")


if __name__ == "__main__":
    main()