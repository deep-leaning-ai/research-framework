# Experiment 관리 시스템 명세서

## 목차

1. [개요](#1-개요)
2. [ExperimentRunner](#2-experimentrunner)
3. [ExperimentResult](#3-experimentresult)
4. [ExperimentRecorder](#4-experimentrecorder)
5. [데이터 플로우](#5-데이터-플로우)
6. [API 명세](#6-api-명세)
7. [테스트 요구사항](#7-테스트-요구사항)

---

## 1. 개요

### 1.1 목적

Experiment 관리 시스템은 머신러닝 실험의 실행, 추적, 기록을 체계적으로 관리합니다. Observer Pattern을 사용하여 실험 결과를 자동으로 수집하고, 다양한 메트릭을 추적합니다.

### 1.2 설계 원칙

- **Observer Pattern**: 실험 결과 자동 수집
- **데이터 캡슐화**: ExperimentResult 데이터 클래스
- **독립성**: 모델과 데이터에 독립적인 실행
- **하드코딩 지양**: 모든 설정은 파라미터화

### 1.3 파일 구조

```
research/experiment/
├── __init__.py        # Experiment exports
├── runner.py          # ExperimentRunner 클래스 (구현 완료)
├── result.py          # ExperimentResult 데이터 클래스 (구현 완료)
└── recorder.py        # ExperimentRecorder 클래스 (구현 완료)
```

---

## 2. ExperimentRunner

### 2.1 개요

실험 실행을 담당하는 핵심 클래스입니다. 학습, 검증, 테스트를 수행하고 메트릭을 추적합니다.

### 2.2 클래스 구조

```python
class ExperimentRunner:
    """실험 실행 및 관리"""

    # 클래스 상수
    DEFAULT_NUM_EPOCHS = 100
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_LEARNING_RATE = 1e-4
    PRINT_INTERVAL = 10  # 에폭 단위 출력 간격

    def __init__(self,
                 device: str = None,
                 task_strategy: TaskStrategy = None,
                 metrics: List[MetricCalculator] = None,
                 primary_metric: MetricCalculator = None,
                 num_epochs: int = DEFAULT_NUM_EPOCHS,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 learning_rate: float = DEFAULT_LEARNING_RATE):
        """초기화

        Args:
            device: 실행 디바이스
            task_strategy: 태스크 전략
            metrics: 추적할 메트릭 리스트
            primary_metric: 주요 성능 지표
            num_epochs: 학습 에폭 수
            batch_size: 배치 크기
            learning_rate: 학습률
        """
```

### 2.3 주요 메서드

#### run_single_experiment()

```python
def run_single_experiment(self,
                         model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         test_loader: DataLoader) -> ExperimentResult:
    """단일 실험 실행

    프로세스:
    1. 모델 분석 (파라미터 수 계산)
    2. Criterion 생성 (TaskStrategy 사용)
    3. MetricTracker 초기화 (train/val/test)
    4. 학습 루프:
       - _train_epoch() 호출
       - _evaluate() 검증/테스트
       - 에폭 시간 측정
       - best_val_metric 추적
    5. 추론 시간 측정
    6. 과적합 갭 계산
    7. ExperimentResult 생성 및 반환

    Returns:
        ExperimentResult: 실험 결과
    """
```

#### _train_epoch()

```python
def _train_epoch(self,
                model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                metric_tracker: MetricTracker) -> float:
    """단일 에폭 학습

    프로세스:
    1. model.train() 설정
    2. 배치 순회:
       - Forward pass
       - 손실 계산
       - Backward pass
       - Optimizer step
    3. 에폭 종료 후 전체 출력/레이블 수집
    4. MetricTracker 업데이트
    5. 평균 손실 반환
    """
```

#### _evaluate()

```python
def _evaluate(self,
             model: nn.Module,
             data_loader: DataLoader,
             criterion: nn.Module,
             metric_tracker: MetricTracker,
             phase: str = "val") -> float:
    """모델 평가

    프로세스:
    1. model.eval() 설정
    2. torch.no_grad() 컨텍스트
    3. 배치별 예측 수행
    4. 전체 출력/레이블 수집
    5. MetricTracker 업데이트
    6. 평균 손실 반환

    Args:
        phase: "val" 또는 "test"
    """
```

### 2.4 MetricTracker 통합

```python
# ExperimentRunner 내부에서 MetricTracker 생성
def run_single_experiment(self, ...):
    # 각 phase별 독립적인 tracker 생성
    train_tracker = MetricTracker(self.metric_names)
    val_tracker = MetricTracker(self.metric_names)
    test_tracker = MetricTracker(self.metric_names)
```

**문제점**: 독립적인 MetricTracker 인스턴스로 인해 전략 간 공유 불가

---

## 3. ExperimentResult

### 3.1 개요

실험 결과를 저장하는 데이터 클래스입니다. 모든 메트릭, 손실, 시간 정보를 포함합니다.

### 3.2 데이터 구조

```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ExperimentResult:
    """실험 결과 데이터 클래스"""

    # 기본 정보
    model_name: str                          # 모델 이름
    task_type: str                           # 태스크 유형
    parameters: int                          # 파라미터 수

    # 메트릭 (각 메트릭별 에폭 히스토리)
    train_metrics: Dict[str, List[float]]   # {"accuracy": [0.6, 0.7, ...]}
    val_metrics: Dict[str, List[float]]     # {"accuracy": [0.65, 0.75, ...]}
    test_metrics: Dict[str, List[float]]    # {"accuracy": [0.9, 0.91, ...]}

    # 손실 히스토리
    train_loss: List[float]                 # [2.3, 1.8, 1.2, ...]
    val_loss: List[float]                   # [2.1, 1.6, 1.1, ...]
    test_loss: List[float]                  # [1.5, 1.4, 1.3, ...]

    # 시간 정보
    epoch_times: List[float]                # 각 에폭 소요 시간
    inference_time: float                   # 추론 시간 (ms)

    # 성능 지표
    primary_metric_name: str                # 주요 메트릭 이름
    best_test_metric: float                 # 최고 테스트 성능
    final_overfitting_gap: Optional[float]  # 과적합 갭

    # 추가 정보
    additional_info: Optional[Dict]         # 기타 정보
```

### 3.3 주요 메서드

```python
def get_final_train_metric(self, metric_name: str) -> float:
    """마지막 학습 메트릭 값 반환"""
    return self.train_metrics[metric_name][-1]

def get_final_val_metric(self, metric_name: str) -> float:
    """마지막 검증 메트릭 값 반환"""
    return self.val_metrics[metric_name][-1]

def get_final_test_metric(self, metric_name: str) -> float:
    """마지막 테스트 메트릭 값 반환"""
    return self.test_metrics[metric_name][-1]

def get_best_test_metric_for(self, metric_name: str) -> float:
    """특정 메트릭의 최고 테스트 값 반환"""
    return max(self.test_metrics[metric_name])

def summary(self) -> str:
    """결과 요약 문자열 반환"""
    return f"""
    Model: {self.model_name}
    Parameters: {self.parameters:,}
    Best {self.primary_metric_name}: {self.best_test_metric:.4f}
    Overfitting Gap: {self.final_overfitting_gap:.4f}
    Inference Time: {self.inference_time:.2f}ms
    """
```

---

## 4. ExperimentRecorder

### 4.1 개요

여러 실험 결과를 수집하고 관리하는 클래스입니다. Observer Pattern을 구현합니다.

### 4.2 클래스 구조

```python
class ExperimentRecorder:
    """실험 결과 수집 및 관리"""

    def __init__(self):
        """초기화"""
        self.results: Dict[str, ExperimentResult] = {}
        # model_name -> ExperimentResult 매핑
```

### 4.3 주요 메서드

#### add_result()

```python
def add_result(self, result: ExperimentResult):
    """실험 결과 추가

    Args:
        result: ExperimentResult 인스턴스
    """
    self.results[result.model_name] = result
```

#### print_summary()

```python
def print_summary(self):
    """실험 결과 요약 테이블 출력

    형식:
    | Model Name | Parameters | Metric | Best | Inference Time |
    |------------|------------|--------|------|----------------|
    | resnet18   | 11,689,512 | accuracy | 0.945 | 8.32ms |
    | resnet50   | 25,557,032 | accuracy | 0.962 | 15.43ms |
    """
```

#### save_to_file()

```python
def save_to_file(self, filepath: str):
    """결과를 텍스트 파일로 저장

    포함 내용:
    - 각 모델별 상세 결과
    - 모든 메트릭의 best/final 값
    - 학습 히스토리
    - 시간 정보
    """
```

#### get_best_model()

```python
def get_best_model(self,
                  metric_name: str,
                  higher_better: bool = True) -> str:
    """특정 메트릭 기준 최고 모델 반환

    Args:
        metric_name: 비교할 메트릭
        higher_better: 높을수록 좋은지 여부

    Returns:
        최고 성능 모델명
    """
```

### 4.4 현재 문제점

1. **메모리 관리 부재**
   - 결과가 계속 누적되어 메모리 증가
   - 정리 메커니즘 없음

2. **영속성 부재**
   - 자동 저장 기능 없음
   - 실험 중단시 결과 손실

3. **버전 관리 부재**
   - 동일 모델명 덮어쓰기
   - 실험 이력 추적 불가

---

## 5. 데이터 플로우

### 5.1 전체 워크플로우

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Model     │───>│ ExperimentRunner │───>│ ExperimentResult │
│ DataLoader  │    │                  │    │                  │
└─────────────┘    └──────────────┘    └─────────────────┘
                            │                      │
                            ↓                      ↓
                    ┌──────────────┐    ┌──────────────────┐
                    │ MetricTracker │    │ ExperimentRecorder│
                    └──────────────┘    └──────────────────┘
```

### 5.2 상세 데이터 흐름

```python
# 1. ExperimentRunner 생성
runner = ExperimentRunner(
    task_strategy=MultiClassStrategy(),
    metrics=[AccuracyMetric(), PrecisionMetric()],
    primary_metric=AccuracyMetric()
)

# 2. 실험 실행
result = runner.run_single_experiment(
    model, train_loader, val_loader, test_loader
)
# 내부 동작:
# - MetricTracker 생성 (train/val/test 별도)
# - 학습 루프 실행
# - 메트릭 추적
# - ExperimentResult 생성

# 3. 결과 기록
recorder = ExperimentRecorder()
recorder.add_result(result)

# 4. 요약 출력
recorder.print_summary()
```

### 5.3 타이밍 측정 문제

**현재 구현**:
```python
# Line 124 in runner.py
epoch_time = time.time() - start_time  # GPU 동기화 없음
```

**문제점**:
- GPU 작업 완료 대기 없이 측정
- 부정확한 타이밍

**개선안**:
```python
if torch.cuda.is_available():
    torch.cuda.synchronize()
epoch_time = time.time() - start_time
```

---

## 6. API 명세

### 6.1 ExperimentRunner 사용

```python
from research.experiment import ExperimentRunner
from research.strategies.task import MultiClassStrategy
from research.metrics import AccuracyMetric, PrecisionMetric

# Runner 생성
runner = ExperimentRunner(
    device='cuda',
    task_strategy=MultiClassStrategy(num_classes=10),
    metrics=[AccuracyMetric(), PrecisionMetric()],
    primary_metric=AccuracyMetric(),
    num_epochs=20,
    learning_rate=1e-4
)

# 단일 실험 실행
result = runner.run_single_experiment(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)

# 다중 실험 실행
models = [model1, model2, model3]
runner.run_multiple_experiments(models, train_loader, val_loader, test_loader)
```

### 6.2 ExperimentRecorder 사용

```python
from research.experiment import ExperimentRecorder

# Recorder 생성
recorder = ExperimentRecorder()

# 결과 추가
recorder.add_result(result1)
recorder.add_result(result2)

# 요약 출력
recorder.print_summary()

# 파일 저장
recorder.save_to_file('experiment_results.txt')

# 최고 모델 찾기
best_model = recorder.get_best_model('accuracy', higher_better=True)
print(f"Best model: {best_model}")

# 모든 결과 가져오기
all_results = recorder.get_all_results()
```

### 6.3 ExperimentResult 접근

```python
# 결과 객체
result = runner.run_single_experiment(...)

# 기본 정보
print(f"Model: {result.model_name}")
print(f"Parameters: {result.parameters:,}")

# 메트릭 접근
final_train_acc = result.get_final_train_metric('accuracy')
best_test_acc = result.get_best_test_metric_for('accuracy')

# 히스토리 접근
train_acc_history = result.train_metrics['accuracy']
val_loss_history = result.val_loss

# 요약
print(result.summary())
```

---

## 7. 테스트 요구사항

### 7.1 단위 테스트

```python
def test_experiment_runner():
    """ExperimentRunner 단위 테스트"""

    # 1. 초기화 테스트
    runner = ExperimentRunner(
        task_strategy=MockTaskStrategy(),
        metrics=[MockMetric()]
    )
    assert runner.device is not None

    # 2. 메트릭 추적 테스트
    # MetricTracker 생성 및 업데이트 확인

    # 3. 타이밍 측정 테스트
    # GPU 동기화 확인
```

### 7.2 통합 테스트

```python
def test_full_experiment_flow():
    """전체 실험 플로우 테스트"""

    # 1. Runner 생성
    runner = ExperimentRunner(...)

    # 2. 실험 실행
    result = runner.run_single_experiment(...)

    # 3. 결과 검증
    assert isinstance(result, ExperimentResult)
    assert result.model_name is not None
    assert len(result.train_metrics) > 0

    # 4. Recorder 테스트
    recorder = ExperimentRecorder()
    recorder.add_result(result)
    assert len(recorder.results) == 1
```

### 7.3 성능 테스트

```python
def test_memory_management():
    """메모리 관리 테스트"""

    recorder = ExperimentRecorder()

    # 많은 결과 추가
    for i in range(1000):
        result = create_dummy_result(f"model_{i}")
        recorder.add_result(result)

    # 메모리 사용량 확인
    memory_usage = get_memory_usage()
    assert memory_usage < MAX_MEMORY_MB
```

### 7.4 엣지 케이스

- 빈 데이터로더 처리
- NaN/Inf 메트릭 처리
- 동일 모델명 중복 추가
- 매우 긴 학습 (체크포인트 필요)

---

## 부록: 개선 로드맵

### 우선순위 P0 (긴급)

1. **GPU 동기화 추가**
   - 정확한 타이밍 측정

2. **메모리 관리**
   - 결과 개수 제한
   - 오래된 결과 자동 삭제

### 우선순위 P1 (중요)

1. **체크포인트 기능**
   ```python
   def save_checkpoint(self, epoch: int):
       checkpoint = {
           'epoch': epoch,
           'model_state': model.state_dict(),
           'optimizer_state': optimizer.state_dict(),
           'metrics': metric_tracker.get_history()
       }
       torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
   ```

2. **실험 재개 기능**
   ```python
   def resume_from_checkpoint(self, checkpoint_path: str):
       checkpoint = torch.load(checkpoint_path)
       model.load_state_dict(checkpoint['model_state'])
       # ...
   ```

### 우선순위 P2 (개선)

1. **분산 학습 지원**
2. **하이퍼파라미터 튜닝 통합**
3. **실시간 모니터링 대시보드**
4. **클라우드 저장소 연동**

---

## SOLID 원칙 준수도

| 원칙 | 준수 여부 | 설명 |
|------|----------|------|
| **S**ingle Responsibility | ⚠️ | Runner가 너무 많은 책임 (학습, 평가, 메트릭, 타이밍) |
| **O**pen/Closed | ✅ | 새로운 메트릭/전략 추가 가능 |
| **L**iskov Substitution | ✅ | 인터페이스 준수 |
| **I**nterface Segregation | ✅ | 최소 인터페이스 |
| **D**ependency Inversion | ✅ | 추상화에 의존 |

ExperimentRunner의 책임을 분리하여 SRP를 개선할 필요가 있습니다.