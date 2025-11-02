# KTB ML Framework - 예제 모음

이 디렉토리에는 KTB ML Framework의 다양한 기능을 시연하는 예제들이 포함되어 있습니다.

## 예제 목록

### 1. 통합 테스트 예제

#### `test_metric_system.py`
메트릭 시스템의 통합 테스트 예제입니다.

**테스트 항목:**
- MetricTracker 생성 및 사용
- 다중 메트릭 동시 계산 (Accuracy, Precision, Recall, F1-Score)
- 학습 루프에서의 실시간 메트릭 추적

**실행 방법:**
```bash
python3 examples/test_metric_system.py
```

**주요 학습 포인트:**
- `MetricTracker`를 사용한 다중 메트릭 관리
- 학습 루프에서 메트릭 업데이트 방법
- 분류 메트릭 사용 방법

---

#### `test_visualization.py`
시각화 시스템의 통합 테스트 예제입니다.

**테스트 항목:**
- ExperimentVisualizer를 사용한 8-panel 종합 차트 생성
- Confusion matrix 시각화
- 특정 메트릭 비교 시각화

**실행 방법:**
```bash
python3 examples/test_visualization.py
```

**생성되는 파일:**
- `test_visualization_8panel.png` - 8패널 종합 비교 차트
- `test_confusion_matrix.png` - Confusion matrix
- `test_metric_comparison.png` - 메트릭 비교 차트

**주요 학습 포인트:**
- `ExperimentVisualizer.plot_comparison()` 사용법
- `plot_confusion_matrix()` 사용법
- 실험 결과 시각화 워크플로우

---

#### `test_task_strategies.py`
Task Strategy 시스템의 통합 테스트 예제입니다.

**테스트 항목:**
- MultiClassStrategy (다중 분류)
- BinaryClassificationStrategy (이진 분류)
- RegressionStrategy (회귀)
- 각 전략의 모델 통합

**실행 방법:**
```bash
python3 examples/test_task_strategies.py
```

**주요 학습 포인트:**
- 다양한 Task Strategy 사용법
- Criterion과 메트릭 계산 방법
- 레이블 준비 (prepare_labels) 방법
- Strategy Pattern의 실제 적용

---

#### `test_comparison_system.py`
모델 비교 시스템의 통합 테스트 예제입니다.

**테스트 항목:**
- PerformanceComparator (성능 기반 비교)
- EfficiencyComparator (효율성 기반 비교)
- SpeedComparator (속도 기반 비교)
- ComparisonManager를 통한 통합 관리

**실행 방법:**
```bash
python3 examples/test_comparison_system.py
```

**생성되는 파일:**
- `test_comparison_report.txt` - 비교 분석 리포트

**주요 학습 포인트:**
- 다양한 관점에서 모델 비교하기
- ComparisonManager를 사용한 일괄 비교
- 비교 결과 리포트 생성

---

### 2. 빠른 시작 예제

#### `quickstart.py`
프레임워크의 전체 워크플로우를 보여주는 종합 예제입니다.

**실행 방법:**
```bash
python3 examples/quickstart.py
```

**다루는 내용:**
- 메트릭 트래커 생성 및 사용
- 간단한 학습 루프 구현
- 실험 결과 기록
- 모델 비교 및 시각화
- 전체 파이프라인 통합

**주요 학습 포인트:**
- 프레임워크의 전체적인 사용 흐름
- 각 컴포넌트 간의 연결 방법
- 실제 프로젝트에 적용할 수 있는 패턴

---

## 모든 예제 실행하기

모든 통합 테스트 예제를 한 번에 실행하려면:

```bash
bash examples/run_all_tests.sh
```

또는 개별적으로:

```bash
for file in examples/test_*.py; do
    echo "Running $file..."
    python3 "$file"
    echo ""
done
```

---

## 예제 실행 요구사항

예제를 실행하기 위해서는 다음이 필요합니다:

```bash
# 기본 설치
pip install -e .

# 또는 모든 의존성 포함
pip install -e ".[all]"
```

---

## 추가 리소스

- **README.md**: 프로젝트 전체 문서
- **QUICKSTART.md**: 빠른 시작 가이드
- **tests/**: pytest 기반 단위 테스트 및 통합 테스트
- **API 문서**: (추후 제공 예정)

---

## 문의 및 이슈

예제 관련 문의 사항이나 버그 발견 시:
- GitHub Issues: https://github.com/ktb-ai/ktb-ml-framework/issues
- Email: ai-research@ktb.com

---

**KTB ML Framework** - Making ML experimentation easier and more structured.
