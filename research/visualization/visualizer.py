"""
실험 결과 시각화
SRP: 시각화만 담당

"""

from typing import List
import matplotlib.pyplot as plt
import numpy as np
from research.experiment.recorder import ExperimentRecorder


class ExperimentVisualizer:
    """
    실험 결과 시각화 클래스
    다양한 형태의 차트로 실험 결과를 시각화
    """

    @staticmethod
    def plot_comparison(
        recorder: ExperimentRecorder, save_path: str = "experiment_comparison.png"
    ):
        """
        종합 비교 시각화

        Args:
            recorder: 실험 기록기
            save_path: 저장 경로
        """
        results = recorder.get_all_results()

        if not results:
            print("시각화할 결과가 없습니다.")
            return

        # 2x4 레이아웃 (Loss를 Train/Val, Test로 분리)
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        fig.suptitle("Model Comparison Results", fontsize=16, fontweight="bold")

        # 색상 및 스타일 정의
        model_styles = {
            name: {"color": plt.cm.tab10(i), "marker": ["o", "s", "^", "D", "v"][i % 5]}
            for i, name in enumerate(results.keys())
        }

        # 1. Training & Validation Loss (Overfitting Check)
        ax = axes[0, 0]
        for model_name, result in results.items():
            style = model_styles[model_name]
            epochs = range(1, len(result.train_loss) + 1)
            # Train: 점선으로 보조적 표시
            ax.plot(
                epochs,
                result.train_loss,
                label=f"{model_name} (Train)",
                color=style["color"],
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
            )
            # Val: 실선 + 마커로 강조 (주 모니터링 대상)
            ax.plot(
                epochs,
                result.val_loss,
                label=f"{model_name} (Val)",
                color=style["color"],
                linestyle="-",
                alpha=0.9,
                linewidth=2,
                marker=style["marker"],
                markersize=3,
                markevery=max(1, len(epochs) // 10),
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(
            "Training Progress (Overfitting Check)", fontsize=12, fontweight="bold"
        )
        ax.legend(loc="best", fontsize=8, ncol=1)
        ax.grid(True, alpha=0.3)

        # 2. Test Loss (Final Performance)
        ax = axes[0, 1]
        for model_name, result in results.items():
            style = model_styles[model_name]
            epochs = range(1, len(result.test_loss) + 1)
            ax.plot(
                epochs,
                result.test_loss,
                label=f"{model_name}",
                color=style["color"],
                linestyle="-",
                alpha=0.9,
                linewidth=2,
                marker=style["marker"],
                markersize=3,
                markevery=max(1, len(epochs) // 10),
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title("Final Test Performance", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Primary Metric Comparison (Train/Val/Test)
        ax = axes[0, 2]
        primary_metric_name = list(results.values())[0].primary_metric_name

        for model_name, result in results.items():
            if primary_metric_name in result.test_metrics:
                style = model_styles[model_name]
                epochs = range(1, len(result.test_metrics[primary_metric_name]) + 1)

                # Train
                ax.plot(
                    epochs,
                    result.train_metrics[primary_metric_name],
                    color=style["color"],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.5,
                )

                # Validation
                ax.plot(
                    epochs,
                    result.val_metrics[primary_metric_name],
                    label=f"{model_name} (Val)",
                    color=style["color"],
                    linestyle="-.",
                    alpha=0.7,
                    linewidth=2,
                    marker=style["marker"],
                    markersize=4,
                    markevery=5,
                )

                # Test
                ax.plot(
                    epochs,
                    result.test_metrics[primary_metric_name],
                    color=style["color"],
                    linewidth=2,
                    alpha=0.9,
                )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(primary_metric_name, fontsize=11)
        ax.set_title(
            f"{primary_metric_name} (Val shown)", fontsize=12, fontweight="bold"
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. Best Performance Bar Chart
        ax = axes[0, 3]
        model_names = list(results.keys())
        best_scores = [result.best_test_metric for result in results.values()]
        colors = [model_styles[name]["color"] for name in model_names]

        bars = ax.bar(
            range(len(model_names)),
            best_scores,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(f"Best {primary_metric_name}", fontsize=11)
        ax.set_title(
            f"Best {primary_metric_name} Comparison", fontsize=12, fontweight="bold"
        )
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, best_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 5. Parameter Efficiency Scatter
        ax = axes[1, 0]
        params_list = [result.parameters for result in results.values()]
        acc_list = [result.best_test_metric for result in results.values()]
        colors_scatter = [model_styles[name]["color"] for name in model_names]

        for i, name in enumerate(model_names):
            ax.scatter(
                params_list[i],
                acc_list[i],
                color=colors_scatter[i],
                s=200,
                alpha=0.6,
                edgecolors="black",
                linewidths=2,
                label=name,
            )

        ax.set_xlabel("Parameters", fontsize=11)
        ax.set_ylabel(f"Best {primary_metric_name}", fontsize=11)
        ax.set_title("Parameter Efficiency", fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 6. Average Epoch Time
        ax = axes[1, 1]
        avg_times = [np.mean(result.epoch_times) for result in results.values()]

        bars = ax.bar(
            range(len(model_names)),
            avg_times,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Average Time (s)", fontsize=11)
        ax.set_title("Average Training Time per Epoch", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, avg_times):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 7. Inference Time
        ax = axes[1, 2]
        inference_times = [result.inference_time * 1000 for result in results.values()]

        bars = ax.bar(
            range(len(model_names)),
            inference_times,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Time (ms)", fontsize=11)
        ax.set_title("Average Inference Time", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}ms",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 8. Overfitting Gap (Train-Val 차이)
        ax = axes[1, 3]
        overfitting_gaps = []
        for result in results.values():
            if result.final_overfitting_gap is not None:
                overfitting_gaps.append(result.final_overfitting_gap)
            else:
                # Overfitting gap이 없으면 0으로 표시
                overfitting_gaps.append(0)

        bars = ax.bar(
            range(len(model_names)),
            overfitting_gaps,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel("Gap (%)", fontsize=11)
        ax.set_title("Overfitting Gap (Train-Val)", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        for bar, val in zip(bars, overfitting_gaps):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}%",
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n시각화 결과가 '{save_path}'로 저장되었습니다.")
        plt.show()

    @staticmethod
    def plot_metric_comparison(
        recorder: ExperimentRecorder,
        metric_name: str,
        save_path: str = "metric_comparison.png",
    ):
        """
        특정 메트릭에 대한 비교 시각화

        Args:
            recorder: 실험 기록기
            metric_name: 비교할 메트릭 이름
            save_path: 저장 경로
        """
        results = recorder.get_all_results()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Model Comparison - {metric_name}", fontsize=14, fontweight="bold"
        )

        model_styles = {
            name: {"color": plt.cm.tab10(i), "marker": ["o", "s", "^", "D"][i % 4]}
            for i, name in enumerate(results.keys())
        }

        # Train vs Val vs Test over Epochs
        ax = axes[0]
        for model_name, result in results.items():
            if metric_name in result.test_metrics:
                style = model_styles[model_name]
                epochs = range(1, len(result.test_metrics[metric_name]) + 1)

                # Train
                ax.plot(
                    epochs,
                    result.train_metrics[metric_name],
                    label=f"{model_name} (Train)",
                    color=style["color"],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.5,
                )

                # Validation
                ax.plot(
                    epochs,
                    result.val_metrics[metric_name],
                    label=f"{model_name} (Val)",
                    color=style["color"],
                    linestyle="-.",
                    alpha=0.7,
                    linewidth=2,
                )

                # Test
                ax.plot(
                    epochs,
                    result.test_metrics[metric_name],
                    label=f"{model_name} (Test)",
                    color=style["color"],
                    linewidth=2,
                    marker=style["marker"],
                    markersize=4,
                    markevery=5,
                )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f"{metric_name} over Epochs", fontsize=12, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Best Metric Comparison
        ax = axes[1]
        model_names = list(results.keys())
        best_scores = [
            (
                max(results[name].test_metrics[metric_name])
                if metric_name in results[name].test_metrics
                else 0
            )
            for name in model_names
        ]
        colors = [model_styles[name]["color"] for name in model_names]

        bars = ax.bar(
            range(len(model_names)),
            best_scores,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(f"Best {metric_name}", fontsize=11)
        ax.set_title(f"Best {metric_name} Comparison", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars, best_scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n'{metric_name}' 시각화 결과가 '{save_path}'로 저장되었습니다.")
        plt.show()
