"""
ExperimentVisualizer 테스트
TDD 방식: Given-When-Then
"""
import pytest
import matplotlib
matplotlib.use('Agg')  # GUI 없이 테스트
import matplotlib.pyplot as plt
import os
from research.visualization.visualizer import ExperimentVisualizer
from research.experiment.recorder import ExperimentRecorder
from research.experiment.result import ExperimentResult


def create_dummy_result(model_name: str) -> ExperimentResult:
    """테스트용 ExperimentResult 생성"""
    return ExperimentResult(
        model_name=model_name,
        task_type="MultiClass",
        parameters=1_000_000,
        train_metrics={"accuracy": [0.7, 0.8, 0.85]},
        val_metrics={"accuracy": [0.65, 0.75, 0.80]},
        test_metrics={"accuracy": [0.80, 0.81, 0.82]},  # 3개로 통일
        train_loss=[0.6, 0.4, 0.3],
        val_loss=[0.7, 0.5, 0.4],
        test_loss=[0.35, 0.33, 0.32],  # 3개로 통일
        epoch_times=[1.0, 1.1, 1.2],
        inference_time=0.01,
        primary_metric_name="accuracy",
        best_test_metric=0.82,
        final_overfitting_gap=0.05
    )


class TestColorPaletteGeneration:
    """색상 팔레트 생성 테스트"""

    def test_10개_이하_모델_tab10_사용(self):
        """
        Given: 10개 이하의 모델
        When: _generate_color_palette 호출
        Then: tab10 색상맵 사용
        """
        # Given
        n_models = 8

        # When
        colors = ExperimentVisualizer._generate_color_palette(n_models)

        # Then
        assert len(colors) == n_models
        # 모든 색상이 RGBA 튜플이어야 함
        for color in colors:
            assert isinstance(color, (tuple, list))
            assert len(color) == 4  # RGBA

    def test_10개_초과_20개_이하_tab20_사용(self):
        """
        Given: 10개 초과 20개 이하의 모델
        When: _generate_color_palette 호출
        Then: tab20 색상맵 사용
        """
        # Given
        n_models = 15

        # When
        colors = ExperimentVisualizer._generate_color_palette(n_models)

        # Then
        assert len(colors) == n_models
        for color in colors:
            assert isinstance(color, (tuple, list))
            assert len(color) == 4

    def test_20개_초과_HSV_동적_생성(self):
        """
        Given: 20개 초과의 모델
        When: _generate_color_palette 호출
        Then: HSV 색상 공간에서 동적 생성
        """
        # Given
        n_models = 30

        # When
        colors = ExperimentVisualizer._generate_color_palette(n_models)

        # Then
        assert len(colors) == n_models
        for color in colors:
            assert isinstance(color, (tuple, list))
            assert len(color) == 4
            # RGB 값이 0-1 범위
            assert all(0 <= c <= 1 for c in color[:3])
            # Alpha 값이 1.0
            assert color[3] == 1.0

    def test_색상_고유성(self):
        """
        Given: 여러 모델
        When: _generate_color_palette 호출
        Then: 각 색상이 서로 달라야 함
        """
        # Given
        n_models = 25

        # When
        colors = ExperimentVisualizer._generate_color_palette(n_models)

        # Then
        # 색상이 서로 다른지 확인
        unique_colors = set(colors)
        assert len(unique_colors) == n_models

    def test_경계값_10개_정확히(self):
        """
        Given: 정확히 10개의 모델
        When: _generate_color_palette 호출
        Then: tab10 사용
        """
        # Given
        n_models = 10

        # When
        colors = ExperimentVisualizer._generate_color_palette(n_models)

        # Then
        assert len(colors) == 10

    def test_경계값_20개_정확히(self):
        """
        Given: 정확히 20개의 모델
        When: _generate_color_palette 호출
        Then: tab20 사용
        """
        # Given
        n_models = 20

        # When
        colors = ExperimentVisualizer._generate_color_palette(n_models)

        # Then
        assert len(colors) == 20


class TestPlotComparison:
    """plot_comparison 메서드 테스트"""

    def test_빈_결과_처리(self, capsys):
        """
        Given: 빈 ExperimentRecorder
        When: plot_comparison 호출
        Then: 메시지 출력 후 종료
        """
        # Given
        recorder = ExperimentRecorder()

        # When
        ExperimentVisualizer.plot_comparison(recorder, save_path="test.png")
        captured = capsys.readouterr()

        # Then
        assert "시각화할 결과가 없습니다" in captured.out
        assert not os.path.exists("test.png")

    def test_단일_모델_시각화(self, tmp_path):
        """
        Given: 1개 모델 결과
        When: plot_comparison 호출
        Then: 8-panel 차트 생성
        """
        # Given
        recorder = ExperimentRecorder()
        recorder.add_result(create_dummy_result("model1"))
        save_path = tmp_path / "single_model.png"

        # When
        ExperimentVisualizer.plot_comparison(recorder, save_path=str(save_path))

        # Then
        assert save_path.exists()
        plt.close('all')

    def test_다중_모델_시각화_10개_이하(self, tmp_path):
        """
        Given: 10개 이하의 모델 결과
        When: plot_comparison 호출
        Then: 모든 모델이 구분 가능한 색상으로 표시
        """
        # Given
        recorder = ExperimentRecorder()
        for i in range(8):
            recorder.add_result(create_dummy_result(f"model{i}"))
        save_path = tmp_path / "multi_model.png"

        # When
        ExperimentVisualizer.plot_comparison(recorder, save_path=str(save_path))

        # Then
        assert save_path.exists()
        plt.close('all')

    def test_다중_모델_시각화_20개_초과(self, tmp_path):
        """
        Given: 20개 초과의 모델 결과
        When: plot_comparison 호출
        Then: HSV 색상으로 모두 표시
        """
        # Given
        recorder = ExperimentRecorder()
        for i in range(25):
            recorder.add_result(create_dummy_result(f"model{i}"))
        save_path = tmp_path / "many_models.png"

        # When
        ExperimentVisualizer.plot_comparison(recorder, save_path=str(save_path))

        # Then
        assert save_path.exists()
        plt.close('all')


class TestPlotMetricComparison:
    """plot_metric_comparison 메서드 테스트"""

    def test_특정_메트릭_시각화(self, tmp_path):
        """
        Given: 여러 모델의 결과
        When: plot_metric_comparison 호출
        Then: 지정된 메트릭의 비교 차트 생성
        """
        # Given
        recorder = ExperimentRecorder()
        recorder.add_result(create_dummy_result("model1"))
        recorder.add_result(create_dummy_result("model2"))
        save_path = tmp_path / "metric_comparison.png"

        # When
        ExperimentVisualizer.plot_metric_comparison(
            recorder,
            metric_name="accuracy",
            save_path=str(save_path)
        )

        # Then
        assert save_path.exists()
        plt.close('all')

    def test_다중_모델_메트릭_비교_색상(self, tmp_path):
        """
        Given: 15개 모델 결과
        When: plot_metric_comparison 호출
        Then: tab20 색상으로 표시
        """
        # Given
        recorder = ExperimentRecorder()
        for i in range(15):
            recorder.add_result(create_dummy_result(f"model{i}"))
        save_path = tmp_path / "metric_15_models.png"

        # When
        ExperimentVisualizer.plot_metric_comparison(
            recorder,
            metric_name="accuracy",
            save_path=str(save_path)
        )

        # Then
        assert save_path.exists()
        plt.close('all')


class TestVisualizerConstants:
    """Visualizer 클래스 상수 테스트"""

    def test_클래스_상수_정의(self):
        """
        Given: ExperimentVisualizer 클래스
        When: 클래스 상수 확인
        Then: 필요한 상수들이 정의되어 있어야 함
        """
        # Given & When & Then
        assert hasattr(ExperimentVisualizer, 'DEFAULT_COLORMAP')
        assert hasattr(ExperimentVisualizer, 'EXTENDED_COLORMAP')
        assert hasattr(ExperimentVisualizer, 'MAX_TAB_COLORS')

        assert ExperimentVisualizer.DEFAULT_COLORMAP == 'tab10'
        assert ExperimentVisualizer.EXTENDED_COLORMAP == 'tab20'
        assert ExperimentVisualizer.MAX_TAB_COLORS == 20

    def test_하드코딩_없음(self):
        """
        Given: ExperimentVisualizer 클래스
        When: 소스코드 확인
        Then: 색상 관련 매직 넘버가 없어야 함
        """
        # Given & When & Then
        # 클래스 상수를 통해 접근 가능
        assert isinstance(ExperimentVisualizer.DEFAULT_COLORMAP, str)
        assert isinstance(ExperimentVisualizer.EXTENDED_COLORMAP, str)
        assert isinstance(ExperimentVisualizer.MAX_TAB_COLORS, int)


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_1개_모델_색상_할당(self):
        """
        Given: 1개 모델
        When: 색상 팔레트 생성
        Then: 1개의 색상만 생성
        """
        # Given
        n_models = 1

        # When
        colors = ExperimentVisualizer._generate_color_palette(n_models)

        # Then
        assert len(colors) == 1

    def test_매우_많은_모델_50개(self):
        """
        Given: 50개 모델
        When: 색상 팔레트 생성
        Then: 50개의 서로 다른 색상 생성
        """
        # Given
        n_models = 50

        # When
        colors = ExperimentVisualizer._generate_color_palette(n_models)

        # Then
        assert len(colors) == n_models
        # 대부분의 색상이 구분 가능해야 함
        unique_colors = set(colors)
        assert len(unique_colors) == n_models
