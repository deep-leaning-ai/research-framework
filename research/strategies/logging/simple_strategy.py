"""
SimpleLoggingStrategy: 기본 print 기반 로깅 (의존성 없음)
"""
from typing import Dict, Any, Optional
from ...core.strategies import LoggingStrategy


class SimpleLoggingStrategy(LoggingStrategy):
    """
    간단한 print 기반 로깅 전략
    외부 의존성 없이 콘솔에 출력
    """

    def __init__(self):
        self.run_name = None
        self.config = {}
        self.metrics_history = []

    def init_run(self, project_name: str, config: Dict[str, Any], **kwargs):
        """로깅 세션 초기화"""
        run_name = kwargs.get('run_name', 'experiment')
        self.run_name = run_name
        self.config = config

        print("\n" + "=" * 70)
        print(f"[START] Starting Experiment: {run_name}")
        print(f" Project: {project_name}")
        print("=" * 70)
        print("\n Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print("=" * 70 + "\n")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """메트릭 로깅"""
        self.metrics_history.append({'step': step, **metrics})

        step_str = f"Step {step}" if step is not None else "Final"
        print(f"\n {step_str} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    def log_hyperparams(self, params: Dict[str, Any]):
        """하이퍼파라미터 로깅"""
        print("\n  Hyperparameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

    def log_artifact(self, file_path: str, artifact_type: str = 'file'):
        """파일 아티팩트 로깅"""
        print(f"\n Artifact saved: {file_path} (type: {artifact_type})")

    def finish(self):
        """로깅 세션 종료"""
        print("\n" + "=" * 70)
        print(f"[완료] Experiment '{self.run_name}' completed")
        print("=" * 70 + "\n")

    def get_history(self):
        """메트릭 히스토리 반환"""
        return self.metrics_history
