"""
비교 분석 통합 관리
SRP: 여러 비교기를 관리하고 실행하는 역할만 담당
"""

from typing import Dict, Any, List
from research.comparison.base import ModelComparator


class ComparisonManager:
    """
    모델 비교 분석 통합 관리 클래스

    여러 비교 전략을 등록하고 일괄 실행
    """

    def __init__(self):
        """비교 관리자 초기화"""
        self.comparators: List[ModelComparator] = []
        self.comparison_results: Dict[str, Any] = {}

    def add_comparator(self, comparator: ModelComparator):
        """
        비교기 추가

        Args:
            comparator: 추가할 비교기
        """
        self.comparators.append(comparator)

    def compare(
        self, results: Dict[str, "ExperimentResult"]
    ) -> Dict[str, Any]:
        """
        모든 등록된 비교 실행

        Args:
            results: 모델 실험 결과 딕셔너리

        Returns:
            모든 비교 결과를 담은 딕셔너리
        """
        return self.run_all_comparisons(results)

    def run_all_comparisons(
        self, results: Dict[str, "ExperimentResult"]
    ) -> Dict[str, Any]:
        """
        모든 등록된 비교 실행 (compare의 실제 구현)

        Args:
            results: 모델 실험 결과 딕셔너리

        Returns:
            모든 비교 결과를 담은 딕셔너리
        """
        print(f"\n{'='*70}")
        print(f"{'모델 비교 분석 시작':^70}")
        print(f"{'='*70}\n")

        for comparator in self.comparators:
            print(f" {comparator.get_comparison_name()}")
            comparison_result = comparator.compare(results)
            self.comparison_results[comparator.get_comparison_name()] = (
                comparison_result
            )

            # 결과 출력
            self._print_comparison_result(
                comparator.get_comparison_name(), comparison_result
            )
            print()

        return self.comparison_results

    def _print_comparison_result(self, comparison_name: str, result: Dict[str, Any]):
        """
        비교 결과 출력

        Args:
            comparison_name: 비교 이름
            result: 비교 결과
        """
        if "ranking" in result:
            print(f"  Ranking:")
            for rank, (model, score) in enumerate(result["ranking"], 1):
                if isinstance(score, dict):
                    score_str = ", ".join([f"{k}={v:.4f}" for k, v in score.items()])
                    print(f"    {rank}. {model}: {score_str}")
                else:
                    print(f"    {rank}. {model}: {score:.4f}")

        if "best_model" in result and result["best_model"]:
            print(f"  [BEST] Best Model: {result['best_model']}")

    def export_comparison_report(self, save_path: str = "comparison_report.txt"):
        """
        비교 분석 리포트 저장

        Args:
            save_path: 저장 경로
        """
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("모델 비교 분석 리포트\n")
            f.write("=" * 70 + "\n\n")

            for comparison_name, result in self.comparison_results.items():
                f.write(f"\n## {comparison_name}\n")
                f.write("-" * 70 + "\n")

                if "ranking" in result:
                    f.write("Ranking:\n")
                    for rank, (model, score) in enumerate(result["ranking"], 1):
                        if isinstance(score, dict):
                            score_str = ", ".join(
                                [f"{k}={v:.4f}" for k, v in score.items()]
                            )
                            f.write(f"  {rank}. {model}: {score_str}\n")
                        else:
                            f.write(f"  {rank}. {model}: {score}\n")

                if "best_model" in result and result["best_model"]:
                    f.write(f"\n[BEST] Best Model: {result['best_model']}\n")

                f.write("\n")

        print(f"비교 리포트가 '{save_path}'에 저장되었습니다.")

    def get_results(self) -> Dict[str, Any]:
        """
        모든 비교 결과 반환

        Returns:
            비교 결과 딕셔너리
        """
        return self.comparison_results

    def clear(self):
        """비교기 및 결과 초기화"""
        self.comparators.clear()
        self.comparison_results.clear()

    def generate_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        비교 결과를 문자열 리포트로 생성

        Args:
            comparison_results: 비교 결과 딕셔너리

        Returns:
            리포트 문자열
        """
        report = []
        report.append("=" * 70)
        report.append("Model Comparison Report")
        report.append("=" * 70)

        for comparison_name, result in comparison_results.items():
            report.append(f"\n{comparison_name}")
            report.append("-" * 50)

            if "rankings" in result:
                report.append("Rankings:")
                for item in result["rankings"]:
                    if "model" in item and "score" in item:
                        report.append(f"  - {item['model']}: {item['score']:.4f}")

            if "best_model" in result:
                report.append(f"Best Model: {result['best_model']}")

        return "\n".join(report)

    def print_summary(self, comparison_results: Dict[str, Any]):
        """
        비교 결과 요약 출력

        Args:
            comparison_results: 비교 결과 딕셔너리
        """
        print("\n" + "=" * 70)
        print("Comparison Summary")
        print("=" * 70)

        for comparison_name, result in comparison_results.items():
            print(f"\n{comparison_name}:")
            if "best_model" in result:
                print(f"  Best: {result['best_model']}")
