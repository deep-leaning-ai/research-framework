"""
WandBLoggingStrategy: Weights & Biases 로깅 전략
"""
from typing import Dict, Any, Optional
from ...core.strategies import LoggingStrategy


class WandBLoggingStrategy(LoggingStrategy):
    """
    Weights & Biases 로깅 전략
    wandb가 설치되어 있어야 사용 가능
    """

    def __init__(self):
        try:
            import wandb
            self.wandb = wandb
            self.available = True
        except ImportError:
            print("Warning: wandb not installed. Install with: pip install wandb")
            self.available = False
            self.wandb = None

        self.run = None

    def init_run(self, project_name: str, config: Dict[str, Any], **kwargs):
        """WandB 실행 초기화"""
        if not self.available:
            print("Error: wandb not available. Falling back to simple logging.")
            return

        run_name = kwargs.get('run_name', None)
        entity = kwargs.get('entity', None)
        tags = kwargs.get('tags', None)
        notes = kwargs.get('notes', None)

        self.run = self.wandb.init(
            project=project_name,
            name=run_name,
            entity=entity,
            config=config,
            tags=tags,
            notes=notes,
            reinit=True
        )

        print(f"[OK] WandB run initialized: {self.run.name}")
        print(f"  URL: {self.run.url}")

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """메트릭 로깅"""
        if not self.available or self.run is None:
            return

        if step is not None:
            self.wandb.log(metrics, step=step)
        else:
            self.wandb.log(metrics)

    def log_hyperparams(self, params: Dict[str, Any]):
        """하이퍼파라미터 로깅 (config 업데이트)"""
        if not self.available or self.run is None:
            return

        self.wandb.config.update(params)

    def log_artifact(self, file_path: str, artifact_type: str = 'file'):
        """파일 아티팩트 로깅"""
        if not self.available or self.run is None:
            return

        artifact = self.wandb.Artifact(
            name=f"artifact_{self.run.id}",
            type=artifact_type
        )
        artifact.add_file(file_path)
        self.wandb.log_artifact(artifact)

    def finish(self):
        """WandB 실행 종료"""
        if not self.available or self.run is None:
            return

        self.wandb.finish()
        print("[OK] WandB run finished")

    def is_available(self) -> bool:
        """WandB 사용 가능 여부 확인"""
        return self.available
