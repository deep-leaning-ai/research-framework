# Comparison 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [ModelComparator 인터페이스](#2-modelcomparator-인터페이스)
3. [PerformanceComparator](#3-performancecomparator)
4. [EfficiencyComparator](#4-efficiencycomparator)
5. [SpeedComparator](#5-speedcomparator)
6. [ComparisonManager](#6-comparisonmanager)
7. [API 명세](#7-api-명세)
8. [테스트 요구사항](#8-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Comparison 시스템은 여러 모델의 실험 결과를 다양한 관점에서 비교하고 순위를 매기는 기능을 제공합니다. Strategy Pattern을 사용하여 비교 기준을 유연하게 확장할 수 있습니다.

### 1.2 설계 원칙

- **전략 패턴**: 비교 알고리즘을 독립적으로 캡슐화
- **개방-폐쇄 원칙**: 새로운 비교기 추가 시 기존 코드 수정 불필요
- **일관성**: 모든 비교기가 동일한 인터페이스 준수
- **하드코딩 지양**: 모든 상수는 클래스 상수로 정의

### 1.3 파일 구조

```
research/comparison/
├── __init__.py           # Comparison exports
├── base.py               # ModelComparator 추상 클래스
├── comparators.py        # 구체적 비교기 구현 (구현 완료)
└── manager.py            # ComparisonManager (구현 완료)
```

---

## 2. ModelComparator 인터페이스

### 2.1 추상 클래스

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from research.experiment import ExperimentResult

class ModelComparator(ABC):
    """모델 비교기의 기본 인터페이스"""

    @abstractmethod
    def compare(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """모델 비교 실행

        Args:
            results: ExperimentResult 리스트

        Returns:
            비교 결과 딕셔너리
        """
        pass

    @abstractmethod
    def get_comparison_name(self) -> str:
        """비교기 이름 반환"""
        pass
```

### 2.2 구현 요구사항

모든 비교기는:
1. `ModelComparator`를 상속
2. `compare()` 메서드에서 순위 결정
3. 일관된 반환 형식 유지
4. Edge case 처리 (빈 결과, 단일 결과)

---

## 3. PerformanceComparator

### 3.1 개요

메트릭 값을 기준으로 모델 성능을 비교합니다. 가장 직관적인 비교 방법입니다.

### 3.2 클래스 구조

```python
class PerformanceComparator(ModelComparator):
    """성능 기반 모델 비교"""

    # 클래스 상수
    DEFAULT_METRIC_NAME = 'accuracy'
    COMPARISON_NAME = 'Performance Comparison'

    def __init__(self,
                 metric_name: str = DEFAULT_METRIC_NAME,
                 higher_better: bool = True):
        """초기화

        Args:
            metric_name: 비교할 메트릭 이름
            higher_better: 높을수록 좋은지 여부
        """
        self.metric_name = metric_name
        self.higher_better = higher_better
```

### 3.3 비교 알고리즘

```python
def compare(self, results: List[ExperimentResult]) -> Dict[str, Any]:
    """성능 비교 알고리즘

    프로세스:
    1. 각 모델의 best metric 추출
    2. higher_better에 따라 정렬
    3. 순위 리스트 생성
    4. 최고 모델 식별

    Returns:
        {
            'ranking': [(model_name, score), ...],  # 순위별 정렬
            'best_model': str,                       # 최고 모델명
            'best_score': float,                     # 최고 점수
            'metric': str                            # 메트릭명
        }
    """
```

### 3.4 사용 예제

```python
comparator = PerformanceComparator(
    metric_name='accuracy',
    higher_better=True
)

result = comparator.compare(experiment_results)
# result['ranking'] = [('resnet50', 0.96), ('resnet18', 0.94), ...]
```

---

## 4. EfficiencyComparator

### 4.1 개요

파라미터 효율성을 기준으로 모델을 비교합니다. 성능 대비 파라미터 수를 고려하여 경량 모델을 식별합니다.

### 4.2 효율성 공식

```
Efficiency = Performance / log₁₀(Parameters + 1)
```

**공식 설명**:
- **Performance**: 메트릭 값 (예: accuracy)
- **log₁₀**: 로그 스케일로 파라미터 영향 감소
- **+1**: log(0) 에러 방지 및 최소값 보장

### 4.3 클래스 구조

```python
class EfficiencyComparator(ModelComparator):
    """파라미터 효율성 비교"""

    # 클래스 상수
    DEFAULT_METRIC_NAME = 'accuracy'
    COMPARISON_NAME = 'Parameter Efficiency'
    LOG_BASE = 10
    EPSILON = 1  # log(params + EPSILON)

    def __init__(self, metric_name: str = DEFAULT_METRIC_NAME):
        """초기화

        Args:
            metric_name: 성능 측정 메트릭
        """
        self.metric_name = metric_name
```

### 4.4 비교 알고리즘

```python
def compare(self, results: List[ExperimentResult]) -> Dict[str, Any]:
    """효율성 비교 알고리즘

    프로세스:
    1. 각 모델에 대해:
       - Best metric 값 추출
       - 파라미터 수 가져오기
       - efficiency = metric / log₁₀(params + 1) 계산
    2. 효율성 기준 내림차순 정렬
    3. 순위 리스트 생성

    Returns:
        {
            'ranking': [(model_name, {
                'efficiency': float,
                'performance': float,
                'parameters': int
            }), ...],
            'best_model': str,
            'metric': str
        }
    """
```

### 4.5 효율성 계산 예시

| 모델 | 파라미터 | Accuracy | log₁₀(params+1) | Efficiency |
|------|---------|----------|-----------------|------------|
| ResNet18 | 11.7M | 0.94 | 7.07 | 0.133 |
| ResNet50 | 25.6M | 0.96 | 7.41 | 0.130 |
| VGG16 | 138.4M | 0.93 | 8.14 | 0.114 |

→ ResNet18이 가장 효율적

### 4.6 한계점

1. **실제 메모리/FLOPs 미반영**
   - 파라미터 수만 고려
   - 실제 연산량 무시

2. **로그 스케일의 임의성**
   - log₁₀ 선택 근거 부족
   - 다른 스케일링 방법 미고려

---

## 5. SpeedComparator

### 5.1 개요

학습 및 추론 속도를 기준으로 모델을 비교합니다. 실시간 처리가 필요한 애플리케이션에 중요합니다.

### 5.2 클래스 구조

```python
class SpeedComparator(ModelComparator):
    """속도 기반 모델 비교"""

    # 클래스 상수
    COMPARISON_NAME = 'Speed Comparison'
    PRIMARY_METRIC = 'inference_time'  # 주요 비교 지표
    SECONDARY_METRIC = 'avg_epoch_time'  # 보조 지표

    def __init__(self):
        """초기화"""
        pass  # 설정 파라미터 없음
```

### 5.3 비교 알고리즘

```python
def compare(self, results: List[ExperimentResult]) -> Dict[str, Any]:
    """속도 비교 알고리즘

    프로세스:
    1. 각 모델의 시간 정보 추출:
       - inference_time (ms)
       - avg_epoch_time (s)
    2. inference_time 기준 오름차순 정렬 (낮을수록 좋음)
    3. 순위 리스트 생성

    Returns:
        {
            'ranking': [(model_name, {
                'inference_time': float,  # ms
                'avg_epoch_time': float   # seconds
            }), ...],
            'fastest_model': str,
            'fastest_time': float  # ms
        }
    """
```

### 5.4 시간 측정 계산

```python
# 평균 에폭 시간
avg_epoch_time = sum(result.epoch_times) / len(result.epoch_times)

# 추론 시간 (이미 측정됨)
inference_time = result.inference_time  # milliseconds
```

### 5.5 비교 예시

| 모델 | 추론 시간 (ms) | 평균 에폭 시간 (s) |
|------|---------------|-------------------|
| ResNet18 | 8.3 | 45.2 |
| ResNet50 | 15.2 | 82.5 |
| VGG16 | 22.1 | 125.3 |

---

## 6. ComparisonManager

### 6.1 개요

여러 비교기를 통합 관리하고 종합 리포트를 생성하는 클래스입니다.

### 6.2 클래스 구조

```python
class ComparisonManager:
    """비교 관리자"""

    # 클래스 상수
    DEFAULT_OUTPUT_FORMAT = 'text'
    REPORT_SEPARATOR = '=' * 60

    def __init__(self):
        """초기화"""
        self.comparators: List[ModelComparator] = []
        self.comparison_results: Dict[str, Any] = {}
```

### 6.3 주요 메서드

#### add_comparator()

```python
def add_comparator(self, comparator: ModelComparator):
    """비교기 추가

    Args:
        comparator: ModelComparator 인스턴스
    """
    self.comparators.append(comparator)
```

#### run_all_comparisons()

```python
def run_all_comparisons(self,
                        results: List[ExperimentResult]) -> Dict[str, Any]:
    """모든 비교 실행

    프로세스:
    1. 등록된 모든 comparator 순회
    2. 각 comparator.compare() 호출
    3. 결과를 comparison_results에 저장
    4. 콘솔에 결과 출력

    Returns:
        모든 비교 결과 딕셔너리
    """
```

#### export_comparison_report()

```python
def export_comparison_report(self, filepath: str):
    """비교 리포트 파일 생성

    포함 내용:
    - 각 비교기별 순위
    - 최고 모델
    - 상세 점수
    - 생성 시간

    Args:
        filepath: 저장할 파일 경로
    """
```

### 6.4 통합 사용 예제

```python
# Manager 생성
manager = ComparisonManager()

# 비교기 추가
manager.add_comparator(PerformanceComparator('accuracy'))
manager.add_comparator(EfficiencyComparator('accuracy'))
manager.add_comparator(SpeedComparator())

# 모든 비교 실행
results = manager.run_all_comparisons(experiment_results)

# 리포트 생성
manager.export_comparison_report('model_comparison.txt')
```

### 6.5 출력 형식

```
============================================================
Performance Comparison
============================================================
Ranking:
1. resnet50: 0.9620
2. resnet18: 0.9450
3. vgg16: 0.9320

Best Model: resnet50
============================================================
Parameter Efficiency
============================================================
Ranking:
1. resnet18: efficiency=0.1330
2. resnet50: efficiency=0.1295
3. vgg16: efficiency=0.1143

Best Model: resnet18
============================================================
```

---

## 7. API 명세

### 7.1 기본 사용법

```python
from research.comparison import (
    PerformanceComparator,
    EfficiencyComparator,
    SpeedComparator,
    ComparisonManager
)

# 1. 개별 비교기 사용
perf_comp = PerformanceComparator('accuracy')
result = perf_comp.compare(experiment_results)
print(f"Best model: {result['best_model']}")

# 2. 효율성 비교
eff_comp = EfficiencyComparator('accuracy')
result = eff_comp.compare(experiment_results)
for model, info in result['ranking']:
    print(f"{model}: efficiency={info['efficiency']:.4f}")

# 3. 속도 비교
speed_comp = SpeedComparator()
result = speed_comp.compare(experiment_results)
print(f"Fastest: {result['fastest_model']} ({result['fastest_time']:.2f}ms)")
```

### 7.2 ComparisonManager 사용

```python
# Manager 설정
manager = ComparisonManager()
manager.add_comparator(PerformanceComparator('accuracy'))
manager.add_comparator(EfficiencyComparator('accuracy'))
manager.add_comparator(SpeedComparator())

# 비교 실행
all_results = manager.run_all_comparisons(experiment_results)

# 특정 비교 결과 접근
perf_result = all_results['Performance Comparison']
eff_result = all_results['Parameter Efficiency']

# 리포트 저장
manager.export_comparison_report('comparison_report.txt')
```

### 7.3 커스텀 비교기 생성

```python
class MemoryEfficiencyComparator(ModelComparator):
    """메모리 효율성 비교 (커스텀)"""

    def get_comparison_name(self) -> str:
        return "Memory Efficiency"

    def compare(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        rankings = []
        for result in results:
            # 메모리 사용량 계산 (파라미터 * 4 bytes)
            memory_mb = result.parameters * 4 / 1024 / 1024
            efficiency = result.best_test_metric / memory_mb
            rankings.append({
                'model': result.model_name,
                'memory_mb': memory_mb,
                'efficiency': efficiency
            })

        # 효율성 기준 정렬
        rankings.sort(key=lambda x: x['efficiency'], reverse=True)

        return {
            'ranking': rankings,
            'best_model': rankings[0]['model'] if rankings else None
        }
```

---

## 8. 테스트 요구사항

### 8.1 단위 테스트

```python
def test_performance_comparator():
    """PerformanceComparator 단위 테스트"""

    # 1. 정상 케이스
    comparator = PerformanceComparator('accuracy')
    results = create_dummy_results()
    comparison = comparator.compare(results)

    assert 'ranking' in comparison
    assert 'best_model' in comparison
    assert len(comparison['ranking']) == len(results)

    # 2. 내림차순 정렬 확인
    scores = [score for _, score in comparison['ranking']]
    assert scores == sorted(scores, reverse=True)
```

### 8.2 Edge Case 테스트

```python
def test_edge_cases():
    """Edge case 처리 테스트"""

    comparator = PerformanceComparator()

    # 1. 빈 결과
    result = comparator.compare([])
    assert result['ranking'] == []
    assert result['best_model'] is None

    # 2. 단일 결과
    single_result = [create_dummy_result()]
    result = comparator.compare(single_result)
    assert len(result['ranking']) == 1

    # 3. 동점 처리
    tied_results = create_tied_results()
    result = comparator.compare(tied_results)
    # 동점시 순서 확인
```

### 8.3 효율성 공식 테스트

```python
def test_efficiency_formula():
    """효율성 계산 검증"""

    comparator = EfficiencyComparator()

    # 수동 계산과 비교
    result = create_result_with_params(
        accuracy=0.95,
        parameters=10_000_000
    )
    efficiency = 0.95 / np.log10(10_000_001)
    calculated = comparator._calculate_efficiency(result)

    assert abs(efficiency - calculated) < 1e-6
```

### 8.4 통합 테스트

```python
def test_comparison_manager():
    """ComparisonManager 통합 테스트"""

    # 1. Manager 설정
    manager = ComparisonManager()
    manager.add_comparator(PerformanceComparator())
    manager.add_comparator(EfficiencyComparator())

    # 2. 비교 실행
    results = create_dummy_results()
    all_comparisons = manager.run_all_comparisons(results)

    # 3. 검증
    assert len(all_comparisons) == 2
    assert 'Performance Comparison' in all_comparisons
    assert 'Parameter Efficiency' in all_comparisons

    # 4. 리포트 생성
    temp_file = 'test_report.txt'
    manager.export_comparison_report(temp_file)
    assert os.path.exists(temp_file)
```

---

## 부록: 개선 제안

### 누락된 비교 차원

1. **메모리 사용량 비교**
   - 실제 GPU 메모리 측정
   - 배치 크기별 메모리 요구사항

2. **에너지 효율성**
   - 전력 소비량 측정
   - Carbon footprint 계산

3. **학습 안정성**
   - 여러 실행의 분산
   - 수렴 속도

4. **Robustness 비교**
   - Adversarial attack 저항성
   - 노이즈 저항성

### 개선 필요사항

1. **통계적 유의성 검증**
   ```python
   def compare_with_significance(self, results, num_runs=5):
       # t-test 또는 ANOVA
       pass
   ```

2. **다중 메트릭 종합**
   ```python
   class MultiMetricComparator:
       def __init__(self, metrics_weights: Dict[str, float]):
           # 가중 평균 계산
           pass
   ```

3. **시각화 통합**
   ```python
   def plot_comparison(self, comparison_result):
       # 비교 결과 차트 생성
       pass
   ```

---

## SOLID 원칙 준수도

| 원칙 | 준수 여부 | 설명 |
|------|----------|------|
| **S**ingle Responsibility | ✅ | 각 비교기는 단일 비교 책임 |
| **O**pen/Closed | ✅ | 새 비교기 추가시 기존 코드 수정 불필요 |
| **L**iskov Substitution | ✅ | 모든 비교기 상호 교체 가능 |
| **I**nterface Segregation | ✅ | 최소 인터페이스 (compare, get_name) |
| **D**ependency Inversion | ✅ | 추상 인터페이스에 의존 |

Comparison 시스템은 SOLID 원칙을 잘 준수하고 있습니다.