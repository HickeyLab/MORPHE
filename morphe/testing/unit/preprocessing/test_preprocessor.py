import pandas as pd
from pandas.testing import assert_frame_equal

from core.preprocessor.config import PreProcessConfig
from core.preprocessor.preprocess import PreProcessor


def test_calculate_dimensions_returns_midpoint_when_distinct() -> None:
	processor = PreProcessor(PreProcessConfig(original_dimensions=(10, 10)))

	result = processor._calculate_dimensions((10, 10), (1, 1))

	assert result == (6, 6)


def test_calculate_dimensions_returns_passing_when_midpoint_equals_failing() -> None:
	processor = PreProcessor(PreProcessConfig(original_dimensions=(10, 10)))
    
	result = processor._calculate_dimensions((3, 3), (2, 2))

	assert result == (3, 3)


def test_compute_new_coord_pair_scales_and_rounds() -> None:
	processor = PreProcessor(PreProcessConfig(original_dimensions=(100, 100)))

	result = processor._compute_new_coord_pair(
		coord_pair=(5.0, 7.0),
		dimensions=(10, 20),
		original_dimensions=(100, 100),
	)

	assert result == (0, 1)


def test_reduce_dimensions_finds_largest_non_colliding_size_and_keeps_input_unchanged() -> None:
	processor = PreProcessor(PreProcessConfig(original_dimensions=(6, 6)))
	df_initial = pd.DataFrame(
		{
			"unique_region": ["r1", "r1", "r1"],
			"x": [1.0, 2.0, 5.0],
			"y": [1.0, 2.0, 5.0],
		}
	)
	df_snapshot = df_initial.copy(deep=True)

	reduced_df, reduced_dimensions = processor._reduce_dimensions(
		df_initial,
		original_dimensions=(6, 6),
	)

	assert reduced_dimensions == (5, 5)
	assert list(reduced_df["x"]) == [1, 2, 4]
	assert list(reduced_df["y"]) == [1, 2, 4]
	assert_frame_equal(df_initial, df_snapshot)


def test_reduce_dimensions_handles_region_uniqueness_independently() -> None:
	processor = PreProcessor(PreProcessConfig(original_dimensions=(10, 10)))
	df_initial = pd.DataFrame(
		{
			"unique_region": ["r1", "r2"],
			"x": [2.0, 2.0],
			"y": [2.0, 2.0],
		}
	)

	_, reduced_dimensions = processor._reduce_dimensions(
		df_initial,
		original_dimensions=(10, 10),
	)

	assert reduced_dimensions == (2, 2)


def test_reduce_dimensions_allows_special_coordinate_collision_exception() -> None:
	processor = PreProcessor(PreProcessConfig(original_dimensions=(10, 10)))

	special_df = pd.DataFrame(
		{
			"unique_region": ["r1", "r1"],
			"x": [4915.0, 4915.0],
			"y": [7055.0, 7055.0],
		}
	)
	normal_df = pd.DataFrame(
		{
			"unique_region": ["r1", "r1"],
			"x": [10.0, 10.0],
			"y": [10.0, 10.0],
		}
	)

	_, reduced_special = processor._reduce_dimensions(
		special_df,
		original_dimensions=(10, 10),
	)
	_, reduced_normal = processor._reduce_dimensions(
		normal_df,
		original_dimensions=(10, 10),
	)

	assert reduced_special == (2, 2)
	assert reduced_normal == (10, 10)


def test_preprocess_uses_config_dimensions_and_returns_reduce_output(monkeypatch) -> None:
	config = PreProcessConfig(original_dimensions=(7, 9))
	processor = PreProcessor(config)
	df = pd.DataFrame({"unique_region": ["r1"], "x": [1.0], "y": [2.0]})

	expected_df = pd.DataFrame({"unique_region": ["r1"], "x": [0], "y": [1]})
	expected_dimensions = (3, 4)
	called: dict[str, object] = {}

	def fake_reduce_dimensions(df_initial, original_dimensions, *, verbose=False):
		called["df_initial"] = df_initial
		called["original_dimensions"] = original_dimensions
		called["verbose"] = verbose
		return expected_df, expected_dimensions

	monkeypatch.setattr(processor, "_reduce_dimensions", fake_reduce_dimensions)

	output_df, output_dimensions = processor.preprocess(df, verbose=True)

	assert called["df_initial"] is df
	assert called["original_dimensions"] == (7, 9)
	assert called["verbose"] is True
	assert output_df is expected_df
	assert output_dimensions == expected_dimensions
