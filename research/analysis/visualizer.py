"""
Visualization Utilities
시각화 도구 모음
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import List, Dict, Any, Optional, Tuple


def visualize_samples(
    dataloader: DataLoader,
    class_names: List[str],
    num_samples: int = 16,
    figsize: Tuple[int, int] = (12, 8),
    denormalize: bool = True
):
    """
    데이터셋 샘플 시각화

    Args:
        dataloader: 데이터로더
        class_names: 클래스 이름 리스트
        num_samples: 표시할 샘플 수
        figsize: Figure 크기
        denormalize: ImageNet 정규화 해제 여부
    """
    # ImageNet 정규화 해제
    if denormalize:
        denorm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )
    else:
        denorm = lambda x: x

    # 첫 번째 배치 가져오기
    images, labels = next(iter(dataloader))

    # Grid 크기 계산
    num_rows = int(np.ceil(np.sqrt(num_samples)))
    num_cols = int(np.ceil(num_samples / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.ravel() if num_samples > 1 else [axes]

    for i in range(min(num_samples, len(images))):
        # 이미지 처리
        img = denorm(images[i])
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()

        # 표시
        axes[i].imshow(img)
        axes[i].set_title(f"{class_names[labels[i]]}", fontsize=10)
        axes[i].axis('off')

    # 빈 subplot 제거
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    ax: Optional[plt.Axes] = None,
    normalize: bool = True,
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Confusion matrix 시각화

    Args:
        cm: Confusion matrix (numpy array)
        class_names: 클래스 이름 리스트
        title: 그래프 제목
        ax: Matplotlib axes (None이면 새로 생성)
        normalize: 정규화 여부 (행 합이 1이 되도록)
        cmap: Color map
        figsize: Figure 크기 (ax가 None일 때만 사용)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # 정규화
    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
        vmax = 1.0
    else:
        cm_normalized = cm
        fmt = 'd'
        vmax = None

    # Heatmap
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        vmin=0,
        vmax=vmax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names, rotation=0)

    plt.tight_layout()


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = ['loss', 'acc'],
    figsize: Tuple[int, int] = (14, 5)
):
    """
    학습 곡선 시각화

    Args:
        history: 학습 히스토리 딕셔너리
            {'train_loss': [...], 'val_loss': [...],
             'train_acc': [...], 'val_acc': [...]}
        metrics: 표시할 메트릭 리스트
        figsize: Figure 크기
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            axes[idx].plot(epochs, history[train_key], 'b-o', label='Train', linewidth=2, markersize=5)

        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            axes[idx].plot(epochs, history[val_key], 'r-s', label='Validation', linewidth=2, markersize=5)

        axes[idx].set_title(f'{metric.capitalize()} over Epochs', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch', fontsize=10)
        axes[idx].set_ylabel(metric.capitalize(), fontsize=10)
        axes[idx].legend(loc='best')
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    metrics: List[str] = ['accuracy', 'inference_time', 'total_params'],
    figsize: Tuple[int, int] = (15, 5)
):
    """
    모델 비교 시각화

    Args:
        results_dict: 모델별 결과 딕셔너리
            {'ResNet18': {...}, 'ResNet50': {...}}
        metrics: 비교할 메트릭 리스트
        figsize: Figure 크기
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    model_names = list(results_dict.keys())

    for idx, metric in enumerate(metrics):
        values = [results_dict[model].get(metric, 0) for model in model_names]

        bars = axes[idx].bar(model_names, values, color=sns.color_palette("Set2", len(model_names)))

        # 값 표시
        for bar in bars:
            height = bar.get_height()
            if metric == 'total_params':
                label = f'{height/1e6:.1f}M'
            elif metric in ['accuracy', 'trainable_ratio']:
                label = f'{height:.2%}' if height < 1 else f'{height:.2f}'
            else:
                label = f'{height:.2f}'

            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                label,
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_comprehensive_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    종합 성능 비교 시각화 (2x2 grid)

    Args:
        comparison_df: 비교 DataFrame
            columns: ['Model', 'Strategy', 'Accuracy', 'Time', 'Total_Params', 'Trainable_Ratio', ...]
        figsize: Figure 크기
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Accuracy 비교
    if 'Accuracy' in comparison_df.columns:
        df_sorted = comparison_df.sort_values('Accuracy', ascending=False)
        bars = axes[0, 0].barh(range(len(df_sorted)), df_sorted['Accuracy'], color=sns.color_palette("viridis", len(df_sorted)))

        axes[0, 0].set_yticks(range(len(df_sorted)))
        axes[0, 0].set_yticklabels([f"{row['Model']}-{row.get('Strategy', 'N/A')}" for _, row in df_sorted.iterrows()])
        axes[0, 0].set_xlabel('Accuracy', fontsize=12)
        axes[0, 0].set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, axis='x', alpha=0.3)

        # 값 표시
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            axes[0, 0].text(row['Accuracy'], i, f" {row['Accuracy']:.4f}", va='center', fontsize=9)

    # 2. Time 비교 (log scale)
    if 'Time' in comparison_df.columns:
        df_sorted = comparison_df.sort_values('Time')
        bars = axes[0, 1].barh(range(len(df_sorted)), df_sorted['Time'], color=sns.color_palette("rocket", len(df_sorted)))

        axes[0, 1].set_yticks(range(len(df_sorted)))
        axes[0, 1].set_yticklabels([f"{row['Model']}-{row.get('Strategy', 'N/A')}" for _, row in df_sorted.iterrows()])
        axes[0, 1].set_xlabel('Time (seconds)', fontsize=12)
        axes[0, 1].set_title('Training/Inference Time Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, axis='x', alpha=0.3)

        # 값 표시
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            axes[0, 1].text(row['Time'], i, f" {row['Time']:.2f}s", va='center', fontsize=9)

    # 3. Trainable Parameters 비교
    if 'Trainable_Params' in comparison_df.columns:
        df_sorted = comparison_df.sort_values('Trainable_Params', ascending=False)
        bars = axes[1, 0].barh(range(len(df_sorted)), df_sorted['Trainable_Params'] / 1e6, color=sns.color_palette("mako", len(df_sorted)))

        axes[1, 0].set_yticks(range(len(df_sorted)))
        axes[1, 0].set_yticklabels([f"{row['Model']}-{row.get('Strategy', 'N/A')}" for _, row in df_sorted.iterrows()])
        axes[1, 0].set_xlabel('Trainable Parameters (Millions)', fontsize=12)
        axes[1, 0].set_title('Trainable Parameters Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, axis='x', alpha=0.3)

        # 값 표시
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            axes[1, 0].text(row['Trainable_Params'] / 1e6, i, f" {row['Trainable_Params']/1e6:.2f}M", va='center', fontsize=9)

    # 4. Trainable Ratio 비교
    if 'Trainable_Ratio' in comparison_df.columns:
        df_sorted = comparison_df.sort_values('Trainable_Ratio', ascending=False)
        bars = axes[1, 1].barh(range(len(df_sorted)), df_sorted['Trainable_Ratio'] * 100, color=sns.color_palette("crest", len(df_sorted)))

        axes[1, 1].set_yticks(range(len(df_sorted)))
        axes[1, 1].set_yticklabels([f"{row['Model']}-{row.get('Strategy', 'N/A')}" for _, row in df_sorted.iterrows()])
        axes[1, 1].set_xlabel('Trainable Ratio (%)', fontsize=12)
        axes[1, 1].set_title('Trainable Parameter Ratio', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, axis='x', alpha=0.3)

        # 값 표시
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            axes[1, 1].text(row['Trainable_Ratio'] * 100, i, f" {row['Trainable_Ratio']*100:.2f}%", va='center', fontsize=9)

    plt.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.show()


def plot_accuracy_improvement(
    baseline_results: Dict[str, float],
    finetuned_results: Dict[str, Dict[str, float]],
    figsize: Tuple[int, int] = (14, 6)
) -> pd.DataFrame:
    """
    정확도 개선 시각화 (Baseline vs Fine-tuned)

    Args:
        baseline_results: Baseline 정확도 {'ResNet18': 0.75, 'ResNet50': 0.78}
        finetuned_results: Fine-tuned 정확도
            {'ResNet18': {'classifier_only': 0.85, 'full': 0.88}, ...}
        figsize: Figure 크기

    Returns:
        개선 정보 DataFrame
    """
    improvements = []

    for model_name, baseline_acc in baseline_results.items():
        if model_name in finetuned_results:
            for strategy, ft_acc in finetuned_results[model_name].items():
                absolute_improvement = ft_acc - baseline_acc
                relative_improvement = (absolute_improvement / baseline_acc) * 100 if baseline_acc > 0 else 0

                improvements.append({
                    'Model': model_name,
                    'Strategy': strategy,
                    'Baseline': baseline_acc,
                    'Finetuned': ft_acc,
                    'Absolute_Improvement': absolute_improvement,
                    'Relative_Improvement': relative_improvement
                })

    df = pd.DataFrame(improvements)

    if len(df) == 0:
        print("No fine-tuning results to compare.")
        return df

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Absolute Improvement
    x_labels = [f"{row['Model']}\n{row['Strategy']}" for _, row in df.iterrows()]
    bars1 = axes[0].bar(x_labels, df['Absolute_Improvement'], color=sns.color_palette("Set1", len(df)))

    axes[0].set_title('Absolute Accuracy Improvement', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy Improvement', fontsize=12)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0].grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars1, df['Absolute_Improvement']):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Relative Improvement (%)
    bars2 = axes[1].bar(x_labels, df['Relative_Improvement'], color=sns.color_palette("Set2", len(df)))

    axes[1].set_title('Relative Accuracy Improvement (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Improvement (%)', fontsize=12)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(True, axis='y', alpha=0.3)

    for bar, val in zip(bars2, df['Relative_Improvement']):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()

    return df


def plot_parameter_distribution(
    model_info: Dict[str, int],
    title: str = "Parameter Distribution",
    figsize: Tuple[int, int] = (10, 6)
):
    """
    파라미터 분포 시각화 (Pie chart)

    Args:
        model_info: 파라미터 정보
            {'total_params': 23M, 'trainable_params': 20K, 'frozen_params': 23M}
        title: 그래프 제목
        figsize: Figure 크기
    """
    labels = ['Trainable', 'Frozen']
    sizes = [model_info.get('trainable_params', 0), model_info.get('frozen_params', 0)]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # Trainable 강조

    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('equal')

    # 범례에 절대값 추가
    legend_labels = [
        f'Trainable: {sizes[0]:,} ({sizes[0]/sum(sizes)*100:.2f}%)',
        f'Frozen: {sizes[1]:,} ({sizes[1]/sum(sizes)*100:.2f}%)'
    ]
    ax.legend(legend_labels, loc='best')

    plt.tight_layout()
    plt.show()
