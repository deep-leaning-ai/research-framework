"""
실험 결과 기록 및 관리
SRP: 실험 결과 기록만 담당

개선사항:
- 메모리 관리: 최대 결과 수 제한
- 버전 관리: 동일 모델명 처리
- 자동 저장: 설정 시 자동 저장
"""

import warnings
from typing import Dict, Optional, List
from collections import OrderedDict
from research.experiment.result import ExperimentResult


class ExperimentRecorder:
    """
    실험 결과 기록 및 관리 클래스

    여러 모델의 실험 결과를 저장하고 관리

    개선사항:
    - 최대 결과 수 제한으로 메모리 관리
    - 버전 관리로 동일 모델명 처리
    - 자동 저장 옵션
    """

    # 클래스 상수
    DEFAULT_MAX_RESULTS = 100
    AUTO_SAVE_INTERVAL = 10

    def __init__(
        self,
        max_results: Optional[int] = None,
        auto_save_path: Optional[str] = None,
        allow_duplicate_names: bool = True
    ):
        """
        실험 기록기 초기화

        Args:
            max_results: 최대 결과 수 (None=무제한, 기본값: 100)
            auto_save_path: 자동 저장 경로 (None=자동 저장 안함)
            allow_duplicate_names: 동일 모델명 허용 여부
        """
        self.results: OrderedDict[str, ExperimentResult] = OrderedDict()
        self.max_results = max_results if max_results is not None else self.DEFAULT_MAX_RESULTS
        self.auto_save_path = auto_save_path
        self.allow_duplicate_names = allow_duplicate_names
        self.result_count = 0

    def add_result(self, result: ExperimentResult):
        """
        실험 결과 추가

        Args:
            result: 추가할 실험 결과

        메모리 관리:
        - max_results 도달시 가장 오래된 결과 삭제 (FIFO)
        - 버전 관리: 동일 모델명 처리

        자동 저장:
        - AUTO_SAVE_INTERVAL마다 자동 저장
        """
        model_name = result.model_name

        # 버전 관리: 동일 모델명 처리
        if model_name in self.results:
            if not self.allow_duplicate_names:
                warnings.warn(
                    f"Model '{model_name}' already exists. Overwriting previous result.",
                    UserWarning
                )
            else:
                # 버전 번호 추가
                version = 1
                versioned_name = f"{model_name}_v{version}"
                while versioned_name in self.results:
                    version += 1
                    versioned_name = f"{model_name}_v{version}"
                model_name = versioned_name
                warnings.warn(
                    f"Model '{result.model_name}' already exists. "
                    f"Saving as '{model_name}'.",
                    UserWarning
                )

        # 메모리 관리: 최대 결과 수 제한
        if self.max_results > 0 and len(self.results) >= self.max_results:
            # FIFO: 가장 오래된 결과 삭제
            oldest_key = next(iter(self.results))
            removed_result = self.results.pop(oldest_key)
            warnings.warn(
                f"Max results ({self.max_results}) reached. "
                f"Removing oldest result: '{removed_result.model_name}'",
                UserWarning
            )

        # 결과 추가
        self.results[model_name] = result
        self.result_count += 1

        # 자동 저장
        if self.auto_save_path and self.result_count % self.AUTO_SAVE_INTERVAL == 0:
            self.save_to_file(self.auto_save_path)
            print(f"Auto-saved results to '{self.auto_save_path}'")

    def get_result(self, model_name: str) -> Optional[ExperimentResult]:
        """
        특정 모델의 결과 조회

        Args:
            model_name: 모델 이름

        Returns:
            실험 결과 또는 None
        """
        return self.results.get(model_name)

    def get_all_results(self) -> Dict[str, ExperimentResult]:
        """
        모든 결과 조회

        Returns:
            모든 실험 결과 딕셔너리
        """
        return self.results

    def print_summary(self):
        """결과 요약 출력"""
        print(f"\n{'='*100}")
        print(f"{'실험 결과 요약':^100}")
        print(f"{'='*100}")
        print(
            f"{'모델':<25} | {'파라미터':>15} | {'주요 메트릭':>12} | "
            f"{'최고 성능':>12} | {'추론시간(ms)':>12}"
        )
        print(f"{'-'*100}")

        for model_name, result in self.results.items():
            print(
                f"{model_name:<25} | "
                f"{result.parameters:>15,} | "
                f"{result.primary_metric_name:>12} | "
                f"{result.best_test_metric:>12.2f} | "
                f"{result.inference_time*1000:>12.2f}"
            )

        print(f"{'='*100}\n")

    def clear(self):
        """모든 결과 삭제"""
        self.results.clear()

    def save_to_file(self, filepath: str):
        """
        결과를 파일로 저장

        Args:
            filepath: 저장 경로
        """
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("실험 결과 리포트\n")
            f.write("=" * 100 + "\n\n")

            for model_name, result in self.results.items():
                f.write(f"\n## {model_name}\n")
                f.write("-" * 100 + "\n")
                f.write(result.summary())
                f.write("\n\n")

                # 메트릭 상세 정보
                f.write("Test Metrics:\n")
                for metric_name, values in result.test_metrics.items():
                    if values:
                        best_val = max(values)
                        final_val = values[-1]
                        f.write(
                            f"  {metric_name}: Best={best_val:.4f}, Final={final_val:.4f}\n"
                        )

                f.write("\n")

        print(f"실험 결과가 '{filepath}'에 저장되었습니다.")

    def get_best_model(
        self, metric_name: str, higher_better: bool = True
    ) -> Optional[str]:
        """
        특정 메트릭 기준 최고 모델 반환

        Args:
            metric_name: 메트릭 이름
            higher_better: 높을수록 좋은 메트릭인지

        Returns:
            최고 모델 이름 또는 None
        """
        if not self.results:
            return None

        best_model = None
        best_value = float("-inf") if higher_better else float("inf")

        for model_name, result in self.results.items():
            value = result.get_best_test_metric_for(metric_name, higher_better)
            if value is not None:
                if higher_better:
                    if value > best_value:
                        best_value = value
                        best_model = model_name
                else:
                    if value < best_value:
                        best_value = value
                        best_model = model_name

        return best_model
