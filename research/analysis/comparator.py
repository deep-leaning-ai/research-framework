"""
Model Comparator
다중 모델 및 전략 비교 도구
"""
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime


class ModelComparator:
    """
    모델 및 전략 비교 클래스

    사용법:
        comparator = ModelComparator()
        comparator.add_inference_result('ResNet18', inference_results)
        comparator.add_training_result('ResNet18', 'feature_extraction', training_results)

        df = comparator.create_comparison_df()
        report = comparator.generate_text_report()
    """

    def __init__(self):
        """초기화"""
        self.results = {
            'inference': {},      # {model_name: results}
            'training': {}        # {model_name: {strategy: results}}
        }
        self.metadata = {
            'created_at': datetime.now(),
            'num_experiments': 0
        }

    def add_inference_result(self, model_name: str, results: Dict[str, Any]):
        """
        Inference 결과 추가

        Args:
            model_name: 모델 이름 (예: 'ResNet18', 'VGG16')
            results: 추론 결과 딕셔너리
                - accuracy, loss, inference_time
                - total_params, trainable_params
                - confusion_matrix, predictions, labels
        """
        self.results['inference'][model_name] = {
            'type': 'inference',
            'model': model_name,
            'strategy': 'pretrained',
            'accuracy': results.get('accuracy', 0.0),
            'loss': results.get('loss', 0.0),
            'time': results.get('inference_time', 0.0),
            'total_params': results.get('total_params', 0),
            'trainable_params': results.get('trainable_params', 0),
            'trainable_ratio': results.get('trainable_ratio', 0.0),
            'confusion_matrix': results.get('confusion_matrix'),
            'predictions': results.get('predictions'),
            'labels': results.get('labels'),
            'classification_report': results.get('classification_report')
        }

        self.metadata['num_experiments'] += 1
        print(f"[OK] Added inference result for {model_name}")

    def add_training_result(
        self,
        model_name: str,
        strategy: str,
        results: Dict[str, Any]
    ):
        """
        Fine-tuning/Training 결과 추가

        Args:
            model_name: 모델 이름
            strategy: 전략 (예: 'feature_extraction', 'fine_tuning')
            results: 학습 결과 딕셔너리
                - model_info: {total_parameters, trainable_parameters, ...}
                - training_time: 학습 시간
                - test_results: {test_acc, test_loss}
                - train_results: {final_train_acc, final_val_acc, history, ...}
        """
        if model_name not in self.results['training']:
            self.results['training'][model_name] = {}

        # 결과 추출
        model_info = results.get('model_info', {})
        train_results = results.get('train_results', {})
        test_results = results.get('test_results', {})

        self.results['training'][model_name][strategy] = {
            'type': 'training',
            'model': model_name,
            'strategy': strategy,
            'accuracy': test_results.get('test_acc', 0.0),
            'loss': test_results.get('test_loss', 0.0),
            'time': results.get('training_time', train_results.get('training_time', 0.0)),
            'total_params': model_info.get('total_parameters', 0),
            'trainable_params': model_info.get('trainable_parameters', 0),
            'trainable_ratio': model_info.get('trainable_ratio', 0.0),
            'best_val_acc': train_results.get('best_val_acc', train_results.get('final_val_acc', 0.0)),
            'final_train_acc': train_results.get('final_train_acc', 0.0),
            'history': train_results.get('history'),
            'predictions': test_results.get('predictions'),
            'labels': test_results.get('labels')
        }

        self.metadata['num_experiments'] += 1
        print(f"[OK] Added training result for {model_name} - {strategy}")

    def add_result(
        self,
        experiment_name: str,
        results: Dict[str, Any],
        result_type: str = 'auto'
    ):
        """
        일반적인 결과 추가 (자동 감지)

        Args:
            experiment_name: 실험 이름 (예: 'ResNet18-feature_extraction')
            results: 결과 딕셔너리
            result_type: 'inference', 'training', 'auto' (자동 감지)
        """
        # 이름 파싱
        parts = experiment_name.split('-')
        model_name = parts[0]
        strategy = parts[1] if len(parts) > 1 else 'pretrained'

        # 타입 자동 감지
        if result_type == 'auto':
            if 'training_time' in results or 'train_results' in results:
                result_type = 'training'
            else:
                result_type = 'inference'

        # 추가
        if result_type == 'inference':
            self.add_inference_result(model_name, results)
        else:
            self.add_training_result(model_name, strategy, results)

    def create_comparison_df(self) -> pd.DataFrame:
        """
        모든 결과를 DataFrame으로 변환

        Returns:
            비교 DataFrame
                columns: ['Model', 'Strategy', 'Type', 'Accuracy', 'Loss', 'Time',
                         'Total_Params', 'Trainable_Params', 'Trainable_Ratio']
        """
        rows = []

        # Inference 결과
        for model_name, result in self.results['inference'].items():
            rows.append({
                'Model': model_name,
                'Strategy': 'Pretrained',
                'Type': 'Inference',
                'Accuracy': result.get('accuracy', 0.0),
                'Loss': result.get('loss', 0.0),
                'Time': result.get('time', 0.0),
                'Total_Params': result.get('total_params', 0),
                'Trainable_Params': result.get('trainable_params', 0),
                'Trainable_Ratio': result.get('trainable_ratio', 0.0)
            })

        # Training 결과
        for model_name, strategies in self.results['training'].items():
            for strategy, result in strategies.items():
                rows.append({
                    'Model': model_name,
                    'Strategy': strategy.replace('_', ' ').title(),
                    'Type': 'Fine-tuning',
                    'Accuracy': result.get('accuracy', 0.0),
                    'Loss': result.get('loss', 0.0),
                    'Time': result.get('time', 0.0),
                    'Total_Params': result.get('total_params', 0),
                    'Trainable_Params': result.get('trainable_params', 0),
                    'Trainable_Ratio': result.get('trainable_ratio', 0.0)
                })

        df = pd.DataFrame(rows)

        # 정렬 (모델 → 전략)
        if len(df) > 0:
            df = df.sort_values(['Model', 'Strategy'])

        return df

    def create_summary_table(self) -> pd.DataFrame:
        """
        요약 테이블 생성

        Returns:
            요약 DataFrame (Model별 최고 성능)
        """
        df = self.create_comparison_df()

        if len(df) == 0:
            return pd.DataFrame()

        # 모델별 최고 정확도
        summary = df.loc[df.groupby('Model')['Accuracy'].idxmax()]

        return summary[['Model', 'Strategy', 'Accuracy', 'Time', 'Trainable_Params']]

    def compare_strategies(self, model_name: str) -> Dict[str, Any]:
        """
        특정 모델의 전략별 성능 비교

        Args:
            model_name: 모델 이름

        Returns:
            전략별 비교 딕셔너리
        """
        comparison = {}

        # Inference
        if model_name in self.results['inference']:
            comparison['pretrained'] = self.results['inference'][model_name]

        # Training strategies
        if model_name in self.results['training']:
            comparison.update(self.results['training'][model_name])

        return comparison

    def compare_models(self, strategy: str = 'pretrained') -> Dict[str, Any]:
        """
        모델 간 성능 비교 (동일 전략)

        Args:
            strategy: 비교할 전략

        Returns:
            모델별 비교 딕셔너리
        """
        comparison = {}

        if strategy == 'pretrained' or strategy == 'inference':
            # Inference 비교
            comparison = self.results['inference'].copy()
        else:
            # 특정 전략 비교
            for model_name, strategies in self.results['training'].items():
                if strategy in strategies:
                    comparison[model_name] = strategies[strategy]

        return comparison

    def generate_text_report(self) -> str:
        """
        종합 분석 보고서 생성 (Markdown 형식)

        Returns:
            Markdown 형식의 보고서 문자열
        """
        report_lines = []

        report_lines.append("#  딥러닝 모델 성능 분석 보고서\n")
        report_lines.append(f"**생성 시간**: {self.metadata['created_at'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"**총 실험 수**: {self.metadata['num_experiments']}\n")
        report_lines.append("---\n")

        # 1. 전체 비교 테이블
        report_lines.append("\n## 1. 종합 성능 비교\n")
        df = self.create_comparison_df()

        if len(df) > 0:
            report_lines.append(df.to_markdown(index=False))
        else:
            report_lines.append("*No results available.*")

        # 2. 모델별 최고 성능
        report_lines.append("\n\n## 2. 모델별 최고 성능\n")
        summary_df = self.create_summary_table()

        if len(summary_df) > 0:
            report_lines.append(summary_df.to_markdown(index=False))

        # 3. 주요 발견사항
        report_lines.append("\n\n## 3. 주요 발견사항\n")

        if len(df) > 0:
            best_acc_row = df.loc[df['Accuracy'].idxmax()]
            fastest_row = df.loc[df['Time'].idxmin()]

            report_lines.append(f"- **최고 정확도**: {best_acc_row['Model']} ({best_acc_row['Strategy']}) - {best_acc_row['Accuracy']:.4f}\n")
            report_lines.append(f"- **최단 시간**: {fastest_row['Model']} ({fastest_row['Strategy']}) - {fastest_row['Time']:.2f}초\n")

            # Inference vs Fine-tuning 비교
            inference_df = df[df['Type'] == 'Inference']
            finetuning_df = df[df['Type'] == 'Fine-tuning']

            if len(inference_df) > 0 and len(finetuning_df) > 0:
                avg_inference_acc = inference_df['Accuracy'].mean()
                avg_finetuning_acc = finetuning_df['Accuracy'].mean()
                improvement = (avg_finetuning_acc - avg_inference_acc) / avg_inference_acc * 100

                report_lines.append(f"- **Fine-tuning 효과**: 평균 정확도 {improvement:.2f}% 향상\n")
                report_lines.append(f"  - Inference 평균: {avg_inference_acc:.4f}\n")
                report_lines.append(f"  - Fine-tuning 평균: {avg_finetuning_acc:.4f}\n")

        # 4. 전이학습 분석
        report_lines.append("\n\n## 4. 전이학습 이점 분석\n")
        report_lines.append("### 사전학습 모델의 장점\n")
        report_lines.append("- [완료] **빠른 수렴**: ImageNet 사전학습으로 일반적인 특징 이미 학습\n")
        report_lines.append("- [완료] **적은 데이터**: 작은 데이터셋에서도 좋은 성능\n")
        report_lines.append("- [완료] **정규화 효과**: 과적합 방지\n")

        report_lines.append("\n### Fine-tuning 전략 선택 가이드\n")
        report_lines.append("- **Feature Extraction (분류기만 학습)**:\n")
        report_lines.append("  - 데이터가 매우 적을 때 (< 1000 샘플)\n")
        report_lines.append("  - 빠른 실험이 필요할 때\n")
        report_lines.append("  - 도메인이 ImageNet과 유사할 때\n")
        report_lines.append("\n- **Full Fine-tuning (전체 학습)**:\n")
        report_lines.append("  - 충분한 데이터가 있을 때 (> 10000 샘플)\n")
        report_lines.append("  - 최고 성능이 필요할 때\n")
        report_lines.append("  - 도메인이 ImageNet과 다를 때\n")

        # 5. 결론
        report_lines.append("\n\n## 5. 결론 및 권장사항\n")

        if len(df) > 0:
            # 최적 모델 추천
            cost_benefit = df.copy()
            cost_benefit['efficiency'] = cost_benefit['Accuracy'] / (cost_benefit['Time'] + 1)  # +1 to avoid division by zero
            best_efficiency_row = cost_benefit.loc[cost_benefit['efficiency'].idxmax()]

            report_lines.append(f"- **정확도 우선**: {best_acc_row['Model']} - {best_acc_row['Strategy']}\n")
            report_lines.append(f"- **속도 우선**: {fastest_row['Model']} - {fastest_row['Strategy']}\n")
            report_lines.append(f"- **효율성 우선**: {best_efficiency_row['Model']} - {best_efficiency_row['Strategy']}\n")

        report_lines.append("\n---\n")
        report_lines.append("*이 보고서는 KTB DL Research Library로 자동 생성되었습니다.*\n")

        return "\n".join(report_lines)

    def save_report(self, filepath: str):
        """
        보고서를 파일로 저장

        Args:
            filepath: 저장할 파일 경로 (.md 또는 .txt)
        """
        report = self.generate_text_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"[OK] Report saved to {filepath}")

    def get_best_result(self, metric: str = 'accuracy') -> Dict[str, Any]:
        """
        최고 성능 결과 반환

        Args:
            metric: 평가 메트릭 ('accuracy', 'time', etc.)

        Returns:
            최고 성능 결과 딕셔너리
        """
        all_results = []

        # 모든 결과 수집
        for result in self.results['inference'].values():
            all_results.append(result)

        for strategies in self.results['training'].values():
            for result in strategies.values():
                all_results.append(result)

        if len(all_results) == 0:
            return {}

        # 메트릭 기준 정렬
        if metric == 'time':
            # 최소값
            best = min(all_results, key=lambda x: x.get(metric, float('inf')))
        else:
            # 최대값
            best = max(all_results, key=lambda x: x.get(metric, 0.0))

        return best

    def __repr__(self):
        """문자열 표현"""
        num_inference = len(self.results['inference'])
        num_training = sum(len(strategies) for strategies in self.results['training'].values())

        return (
            f"ModelComparator(\n"
            f"  inference_results={num_inference},\n"
            f"  training_results={num_training},\n"
            f"  total_experiments={self.metadata['num_experiments']}\n"
            f")"
        )
