# KTB ML Framework

통합 머신러닝 실험 프레임워크 - 전이학습과 일반 ML 태스크를 위한 범용 프레임워크

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 개요

**KTB ML Framework**는 기존 `ktb_dl_research`와 `ml_framework`를 통합하여 만든 범용 머신러닝 실험 프레임워크입니다. 전이학습과 일반 ML 태스크를 모두 지원하며, SOLID 원칙과 디자인 패턴을 적용하여 확장성과 유지보수성을 극대화했습니다.

### 주요 특징

- **전이학습 지원**: ResNet, VGG 등 사전학습된 모델을 활용한 Feature Extraction 및 Fine-tuning
- **다양한 태스크**: MultiClass, Binary, Regression 분류 작업 지원
- **고급 메트릭 시스템**: 실시간 다중 메트릭 추적 및 분석
- **모델 비교**: 성능, 효율성, 속도 등 다차원 비교 시스템
- **종합 시각화**: 8-panel 차트를 포함한 풍부한 시각화 도구
- **완전한 하위 호환성**: 기존 ktb_dl_research 코드 100% 호환

## 문서 목차

프로젝트의 전체 문서는 다음과 같이 구성되어 있습니다:

- **[README.md](README.md)** - 프로젝트 개요 및 빠른 시작 (현재 문서)
- **[QUICKSTART.md](QUICKSTART.md)** - 5분 빠른 시작 가이드 및 튜토리얼
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - 아키텍처 설계 및 디자인 패턴
- **[examples/README.md](examples/README.md)** - 실행 가능한 예제 모음
- **[research/visualization/VISUALIZATION_FEATURES.md](research/visualization/VISUALIZATION_FEATURES.md)** - 시각화 API 상세 명세

더 자세한 내용은 각 문서를 참조하세요.

## 설치

### PyPI를 통한 설치 (추후 제공 예정)

```bash
pip install research
```

### 소스코드로부터 설치

```bash
# 저장소 클론
git clone https://github.com/ktb-ai/ktb-ml-framework.git
cd ktb-ml-framework

# 기본 설치
pip install -e .

# 모든 옵션 포함 설치
pip install -e ".[all]"

# 개발 모드 설치
pip install -e ".[dev]"
```

## 빠른 시작

### 전이학습 예제 (ResNet + 멜 스펙트로그램)

```python
import pandas as pd
from research import (
    Experiment,
    VanillaTrainingStrategy,
    SimpleLoggingStrategy
)
from mel_spectrogram_datamodule import MelSpectrogramDataModule

# 1. 데이터 로드
df = pd.read_csv('data/dataframes/df.csv')

# 2. DataModule 생성
mel_dm = MelSpectrogramDataModule(
    data_dir='data/mel_spectrograms/',
    df=df,
    batch_size=32,
    test_size=0.2,
    val_size=0.2
)

# 3. 실험 설정
config = {
    'num_classes': df['label_encoded'].nunique(),
    'learning_rate': 1e-4,
    'max_epochs': 20,
    'batch_size': 32,
    'optimizer': 'adam'
}

# 4. Experiment 생성
exp = Experiment(config)
exp.setup(
    model_name='resnet18',
    data_module=mel_dm,
    training_strategy=VanillaTrainingStrategy(),
    logging_strategy=SimpleLoggingStrategy(),
    in_channels=1  # 1채널 입력 (멜 스펙트로그램)
)

# 5. 전략 비교
comparison = exp.compare_strategies([
    'feature_extraction',  # 백본 동결, 분류기만 학습
    'fine_tuning'          # 전체 네트워크 학습
])
```

### 단순 모델 예제 (CNN + MNIST)

```python
from research import (
    CNN,
    MultiClassStrategy,
    ExperimentRunner,
    MetricTracker,
    AccuracyMetric,
    F1ScoreMetric
)

# 1. 모델 생성
model = CNN(
    input_channels=1,
    output_dim=10,
    task_strategy=MultiClassStrategy()
)

# 2. 메트릭 트래커 생성
tracker = MetricTracker([
    AccuracyMetric(),
    F1ScoreMetric(average='macro')
])

# 3. 실험 실행
runner = ExperimentRunner(
    strategy=MultiClassStrategy(),
    metrics=tracker,
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=10
)

results = runner.run_single_experiment(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)
```

## 사용 가능한 모델

### Pretrained 모델 (전이학습용)

```python
import research as ktb
ktb.list_models()
```

- **ResNet 계열**: resnet18, resnet34, resnet50, resnet101, resnet152
- **VGG 계열**: vgg11, vgg13, vgg16, vgg19 (+ batch normalization 버전)

### Simple 모델 (처음부터 학습)

- **CNN**: Convolutional Neural Network
- **FullyConnectedNN**: Fully Connected Neural Network

## 핵심 기능

### 1. 전이학습 전략

- **Feature Extraction**: 사전학습된 백본 동결, 분류기만 학습
- **Fine-tuning**: 전체 네트워크 학습
- **Inference**: 학습 없이 평가만 수행

### 2. Task 전략

- **MultiClassStrategy**: 다중 분류 (10개 이상 클래스)
- **BinaryClassificationStrategy**: 이진 분류
- **RegressionStrategy**: 회귀 분석

### 3. 메트릭 시스템

실시간 다중 메트릭 추적:
- 분류: Accuracy, Precision, Recall, F1-Score
- 회귀: MSE, MAE, R²

### 4. 비교 시스템

- **PerformanceComparator**: 정확도, 손실 등 성능 비교
- **EfficiencyComparator**: 파라미터 효율성 비교
- **SpeedComparator**: 학습/추론 속도 비교

### 5. 시각화

- Confusion Matrix
- Training Curves (Loss, Accuracy)
- 8-panel 종합 차트
- 정확도 개선 분석

## 아키텍처

프레임워크는 SOLID 원칙과 다음 디자인 패턴을 적용했습니다:

- **Strategy Pattern**: Task 전략, Training 전략
- **Factory + Registry Pattern**: 모델 생성 및 관리
- **Template Method Pattern**: 전이학습 모델 베이스
- **Facade Pattern**: 실험 오케스트레이션

```
research/
├── core/              # 핵심 추상 클래스
├── models/
│   ├── pretrained/    # ResNet, VGG 등
│   └── simple/        # CNN, FullyConnectedNN
├── strategies/
│   ├── training/      # 학습 전략
│   ├── logging/       # 로깅 전략
│   └── task/          # Task 전략
├── metrics/           # 메트릭 시스템
├── experiment/        # 실험 실행 및 기록
├── comparison/        # 모델 비교
├── visualization/     # 시각화 도구
└── analysis/          # 분석 도구
```

더 자세한 아키텍처 정보는 [ARCHITECTURE.md](ARCHITECTURE.md)를 참조하세요.

## 예제

프로젝트에는 다음 예제들이 포함되어 있습니다:

- `train_mel_audio.py`: 멜 스펙트로그램 음성 분류
- `test_library.py`: 라이브러리 기능 테스트
- `task1_notebook_v2.ipynb`: CIFAR-10 분류 노트북

## 기여

기여는 언제나 환영합니다! 다음 절차를 따라주세요:

1. 저장소 Fork
2. Feature 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

### 개발 환경 설정

```bash
# 개발 의존성 설치
pip install -e ".[dev]"

# 코드 포맷팅
black research/
isort research/

# 린팅
flake8 research/
mypy research/

# 테스트 실행
pytest tests/
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 인용

이 프레임워크를 연구에 사용하셨다면, 다음과 같이 인용해주세요:

```bibtex
@software{research,
  title={KTB ML Framework: 통합 머신러닝 실험 프레임워크},
  author={KTB AI Research Team},
  year={2025},
  url={https://github.com/ktb-ai/ktb-ml-framework}
}
```

## 문의

- 이슈 트래커: https://github.com/ktb-ai/ktb-ml-framework/issues
- 이메일: ai-research@ktb.com

## 감사의 말

이 프레임워크는 다음 프로젝트들을 기반으로 만들어졌습니다:
- ktb_dl_research: 전이학습 프레임워크
- ml_framework: SOLID 기반 ML 실험 프레임워크

---

**KTB ML Framework** - Making ML experimentation easier and more structured.
