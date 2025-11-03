"""
ExperimentRecorder 테스트
TDD 방식: Given-When-Then
"""
import pytest
import warnings
import os
from research.experiment.recorder import ExperimentRecorder
from research.experiment.result import ExperimentResult


def create_dummy_result(model_name: str, params: int = 1000, best_metric: float = 0.9) -> ExperimentResult:
    """더미 ExperimentResult 생성"""
    return ExperimentResult(
        model_name=model_name,
        task_type="MultiClass",
        parameters=params,
        train_metrics={"accuracy": [0.8, 0.85, 0.9]},
        val_metrics={"accuracy": [0.75, 0.8, 0.85]},
        test_metrics={"accuracy": [0.9]},
        train_loss=[0.5, 0.4, 0.3],
        val_loss=[0.6, 0.5, 0.4],
        test_loss=[0.35],
        epoch_times=[1.0, 1.1, 1.2],
        inference_time=0.01,
        primary_metric_name="accuracy",
        best_test_metric=best_metric,
        final_overfitting_gap=0.05
    )


class TestExperimentRecorderBasic:
    """기본 기능 테스트"""

    def test_초기화_기본값(self):
        """
        Given: ExperimentRecorder 생성
        When: 인자 없이 초기화
        Then: 기본값이 설정되어야 함
        """
        # Given & When
        recorder = ExperimentRecorder()

        # Then
        assert len(recorder.results) == 0
        assert recorder.max_results == ExperimentRecorder.DEFAULT_MAX_RESULTS
        assert recorder.auto_save_path is None
        assert recorder.allow_duplicate_names is True
        assert recorder.result_count == 0

    def test_초기화_커스텀값(self):
        """
        Given: 커스텀 설정값
        When: ExperimentRecorder 생성
        Then: 설정값이 적용되어야 함
        """
        # Given
        max_results = 50
        auto_save_path = "test_results.txt"

        # When
        recorder = ExperimentRecorder(
            max_results=max_results,
            auto_save_path=auto_save_path,
            allow_duplicate_names=False
        )

        # Then
        assert recorder.max_results == max_results
        assert recorder.auto_save_path == auto_save_path
        assert recorder.allow_duplicate_names is False

    def test_결과_추가(self):
        """
        Given: ExperimentRecorder와 ExperimentResult
        When: add_result 호출
        Then: 결과가 저장되어야 함
        """
        # Given
        recorder = ExperimentRecorder()
        result = create_dummy_result("model1")

        # When
        recorder.add_result(result)

        # Then
        assert len(recorder.results) == 1
        assert "model1" in recorder.results
        assert recorder.result_count == 1

    def test_결과_조회(self):
        """
        Given: 결과가 저장된 ExperimentRecorder
        When: get_result 호출
        Then: 저장된 결과를 반환해야 함
        """
        # Given
        recorder = ExperimentRecorder()
        result = create_dummy_result("model1")
        recorder.add_result(result)

        # When
        retrieved = recorder.get_result("model1")

        # Then
        assert retrieved is not None
        assert retrieved.model_name == "model1"
        assert retrieved.best_test_metric == 0.9

    def test_존재하지_않는_결과_조회(self):
        """
        Given: ExperimentRecorder
        When: 존재하지 않는 모델 조회
        Then: None 반환
        """
        # Given
        recorder = ExperimentRecorder()

        # When
        retrieved = recorder.get_result("nonexistent")

        # Then
        assert retrieved is None

    def test_모든_결과_조회(self):
        """
        Given: 여러 결과가 저장된 ExperimentRecorder
        When: get_all_results 호출
        Then: 모든 결과를 반환해야 함
        """
        # Given
        recorder = ExperimentRecorder()
        recorder.add_result(create_dummy_result("model1"))
        recorder.add_result(create_dummy_result("model2"))
        recorder.add_result(create_dummy_result("model3"))

        # When
        all_results = recorder.get_all_results()

        # Then
        assert len(all_results) == 3
        assert "model1" in all_results
        assert "model2" in all_results
        assert "model3" in all_results

    def test_결과_삭제(self):
        """
        Given: 결과가 저장된 ExperimentRecorder
        When: clear 호출
        Then: 모든 결과가 삭제되어야 함
        """
        # Given
        recorder = ExperimentRecorder()
        recorder.add_result(create_dummy_result("model1"))
        recorder.add_result(create_dummy_result("model2"))
        assert len(recorder.results) == 2

        # When
        recorder.clear()

        # Then
        assert len(recorder.results) == 0


class TestExperimentRecorderMemoryManagement:
    """메모리 관리 테스트"""

    def test_최대_결과_수_제한(self):
        """
        Given: max_results=3인 ExperimentRecorder
        When: 4개의 결과 추가
        Then: 가장 오래된 결과가 삭제되고 3개만 유지
        """
        # Given
        recorder = ExperimentRecorder(max_results=3)

        # When
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            recorder.add_result(create_dummy_result("model1"))
            recorder.add_result(create_dummy_result("model2"))
            recorder.add_result(create_dummy_result("model3"))
            recorder.add_result(create_dummy_result("model4"))  # 이때 model1 삭제

            # Then
            assert len(recorder.results) == 3
            assert "model1" not in recorder.results
            assert "model2" in recorder.results
            assert "model3" in recorder.results
            assert "model4" in recorder.results
            assert len(w) == 1
            assert "Max results" in str(w[0].message)

    def test_무제한_결과_수(self):
        """
        Given: max_results=0인 ExperimentRecorder
        When: 많은 결과 추가
        Then: 제한 없이 모두 저장
        """
        # Given
        recorder = ExperimentRecorder(max_results=0)

        # When
        for i in range(10):
            recorder.add_result(create_dummy_result(f"model{i}"))

        # Then
        assert len(recorder.results) == 10


class TestExperimentRecorderVersioning:
    """버전 관리 테스트"""

    def test_중복_모델명_허용(self):
        """
        Given: allow_duplicate_names=True인 ExperimentRecorder
        When: 동일 모델명으로 결과 추가
        Then: 버전 번호가 추가되어야 함
        """
        # Given
        recorder = ExperimentRecorder(allow_duplicate_names=True)

        # When
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            recorder.add_result(create_dummy_result("model1", best_metric=0.8))
            recorder.add_result(create_dummy_result("model1", best_metric=0.9))
            recorder.add_result(create_dummy_result("model1", best_metric=0.95))

            # Then
            assert len(recorder.results) == 3
            assert "model1" in recorder.results
            assert "model1_v1" in recorder.results
            assert "model1_v2" in recorder.results
            assert len(w) == 2  # 2번의 중복 경고

    def test_중복_모델명_덮어쓰기(self):
        """
        Given: allow_duplicate_names=False인 ExperimentRecorder
        When: 동일 모델명으로 결과 추가
        Then: 이전 결과가 덮어써져야 함
        """
        # Given
        recorder = ExperimentRecorder(allow_duplicate_names=False)

        # When
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            recorder.add_result(create_dummy_result("model1", best_metric=0.8))
            recorder.add_result(create_dummy_result("model1", best_metric=0.9))

            # Then
            assert len(recorder.results) == 1
            assert "model1" in recorder.results
            assert recorder.results["model1"].best_test_metric == 0.9
            assert len(w) == 1
            assert "Overwriting" in str(w[0].message)


class TestExperimentRecorderAutoSave:
    """자동 저장 테스트"""

    def test_자동_저장_비활성화(self):
        """
        Given: auto_save_path=None인 ExperimentRecorder
        When: 결과 추가
        Then: 자동 저장이 실행되지 않아야 함
        """
        # Given
        recorder = ExperimentRecorder(auto_save_path=None)

        # When
        for i in range(12):
            recorder.add_result(create_dummy_result(f"model{i}"))

        # Then
        assert not os.path.exists("any_auto_save_file.txt")

    def test_자동_저장_간격(self, tmp_path):
        """
        Given: auto_save_path가 설정된 ExperimentRecorder
        When: AUTO_SAVE_INTERVAL(10)개의 결과 추가
        Then: 자동 저장이 실행되어야 함
        """
        # Given
        save_path = tmp_path / "auto_save.txt"
        recorder = ExperimentRecorder(auto_save_path=str(save_path))

        # When
        for i in range(ExperimentRecorder.AUTO_SAVE_INTERVAL):
            recorder.add_result(create_dummy_result(f"model{i}"))

        # Then
        assert save_path.exists()


class TestExperimentRecorderBestModel:
    """최고 모델 검색 테스트"""

    def test_최고_모델_검색_높을수록_좋음(self):
        """
        Given: 여러 모델 결과가 저장된 ExperimentRecorder
        When: get_best_model(higher_better=True)
        Then: 가장 높은 메트릭을 가진 모델 반환
        """
        # Given
        recorder = ExperimentRecorder()
        # test_metrics에 실제 값이 있어야 함
        result1 = create_dummy_result("model1", best_metric=0.85)
        result1.test_metrics["accuracy"] = [0.85]  # 명시적으로 설정
        result2 = create_dummy_result("model2", best_metric=0.95)
        result2.test_metrics["accuracy"] = [0.95]
        result3 = create_dummy_result("model3", best_metric=0.90)
        result3.test_metrics["accuracy"] = [0.90]

        recorder.add_result(result1)
        recorder.add_result(result2)
        recorder.add_result(result3)

        # When
        best_model = recorder.get_best_model("accuracy", higher_better=True)

        # Then
        assert best_model == "model2"

    def test_최고_모델_검색_낮을수록_좋음(self):
        """
        Given: 여러 모델 결과가 저장된 ExperimentRecorder
        When: get_best_model(higher_better=False)
        Then: 가장 낮은 메트릭을 가진 모델 반환
        """
        # Given
        recorder = ExperimentRecorder()
        result1 = create_dummy_result("model1", best_metric=0.85)
        result1.test_metrics["accuracy"] = [0.85]
        result2 = create_dummy_result("model2", best_metric=0.75)
        result2.test_metrics["accuracy"] = [0.75]
        result3 = create_dummy_result("model3", best_metric=0.90)
        result3.test_metrics["accuracy"] = [0.90]

        recorder.add_result(result1)
        recorder.add_result(result2)
        recorder.add_result(result3)

        # When
        best_model = recorder.get_best_model("accuracy", higher_better=False)

        # Then
        assert best_model == "model2"

    def test_빈_레코더_최고_모델_검색(self):
        """
        Given: 빈 ExperimentRecorder
        When: get_best_model 호출
        Then: None 반환
        """
        # Given
        recorder = ExperimentRecorder()

        # When
        best_model = recorder.get_best_model("accuracy")

        # Then
        assert best_model is None


class TestExperimentRecorderFileOperations:
    """파일 저장/로드 테스트"""

    def test_파일_저장(self, tmp_path):
        """
        Given: 결과가 저장된 ExperimentRecorder
        When: save_to_file 호출
        Then: 파일이 생성되어야 함
        """
        # Given
        recorder = ExperimentRecorder()
        recorder.add_result(create_dummy_result("model1"))
        recorder.add_result(create_dummy_result("model2"))
        save_path = tmp_path / "results.txt"

        # When
        recorder.save_to_file(str(save_path))

        # Then
        assert save_path.exists()
        content = save_path.read_text(encoding="utf-8")
        assert "model1" in content
        assert "model2" in content
        assert "실험 결과 리포트" in content

    def test_요약_출력(self, capsys):
        """
        Given: 결과가 저장된 ExperimentRecorder
        When: print_summary 호출
        Then: 요약이 출력되어야 함
        """
        # Given
        recorder = ExperimentRecorder()
        recorder.add_result(create_dummy_result("model1", params=1000000))
        recorder.add_result(create_dummy_result("model2", params=2000000))

        # When
        recorder.print_summary()
        captured = capsys.readouterr()

        # Then
        assert "실험 결과 요약" in captured.out
        assert "model1" in captured.out
        assert "model2" in captured.out
        assert "1,000,000" in captured.out
        assert "2,000,000" in captured.out
