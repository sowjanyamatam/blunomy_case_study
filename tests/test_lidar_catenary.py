"""
Run with: pytest test_lidar_catenary.py -v
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

def make_labeled_df(n_wires=2, points_per_wire=30, noise_rows=0):
    """Build a minimal labeled DataFrame that looks like DBSCAN output."""
    rows = []
    for wire_id in range(n_wires):
        x = np.linspace(-5, 5, points_per_wire) + wire_id * 10
        y = np.zeros(points_per_wire) + wire_id
        z = 0.5 * (np.cosh(x / 5) - 1)          # real catenary shape
        for xi, yi, zi in zip(x, y, z):
            rows.append({"x": xi, "y": yi, "z": zi, "labels": wire_id})
    for _ in range(noise_rows):
        rows.append({"x": 999.0, "y": 999.0, "z": 999.0, "labels": -1})
    return pd.DataFrame(rows)


MOCK_CONFIG = {
    "clustering": {"epsilon_value": 0.5, "min_samples": 5},
    "output": {"save_images": False, "save_model_json": False, "save_clustered_csv": False},
    "logging": {"level": "WARNING", "format": "%(message)s"},
    "min_points_for_clustering": 10,
}


#pca_curve_fitter tests
@patch("lidar_catenary.pca_curve_fitter.CONFIG", MOCK_CONFIG)
class TestCurveEquation:
    """Tests for the static catenary equation."""

    def test_vertex_is_at_x0(self):
        """At x = x0, the curve should return y0 (the minimum)."""
        from lidar_catenary.pca_curve_fitter import PCACurveFitter
        result = PCACurveFitter.curve_equation(x=0.0, x0=0.0, y0=1.0, c=2.0)
        assert result == pytest.approx(1.0)

    def test_curve_is_symmetric(self):
        """Catenary is symmetric: f(x0 + d) == f(x0 - d)."""
        from lidar_catenary.pca_curve_fitter import PCACurveFitter
        x0, y0, c = 3.0, 0.0, 2.0
        assert PCACurveFitter.curve_equation(x0 + 1, x0, y0, c) == pytest.approx(
            PCACurveFitter.curve_equation(x0 - 1, x0, y0, c)
        )

    def test_curve_increases_away_from_vertex(self):
        """Points further from x0 should be higher (larger z)."""
        from lidar_catenary.pca_curve_fitter import PCACurveFitter
        x0, y0, c = 0.0, 0.0, 2.0
        z_near = PCACurveFitter.curve_equation(1.0, x0, y0, c)
        z_far = PCACurveFitter.curve_equation(4.0, x0, y0, c)
        assert z_far > z_near


@patch("lidar_catenary.pca_curve_fitter.CONFIG", MOCK_CONFIG)
class TestPCACurveFitter:
    """Tests for PCACurveFitter.pca_curve_fitting()."""

    def _make_fitter(self, labeled_df):
        from lidar_catenary.pca_curve_fitter import PCACurveFitter
        n_real_clusters = labeled_df[labeled_df["labels"] !=-1]["labels"].nunique()
        return PCACurveFitter(
            labeled_dataset_df=labeled_df,
            dataset_name="test.parquet",
            clusters_count=labeled_df["labels"].nunique(),
            output_dir="/tmp/test_output",
        )

    def test_returns_dict_with_expected_keys(self):
        df = make_labeled_df(n_wires=2)
        result = self._make_fitter(df).pca_curve_fitting()
        for key in ("File_name", "Row_count", "Timestamp", "wires", "summary"):
            assert key in result

    def test_wire_count_matches_clusters(self):
        df = make_labeled_df(n_wires=3)
        result = self._make_fitter(df).pca_curve_fitting()
        assert result["summary"]["number_of_wires"] == 3

    def test_each_wire_has_catenary_params(self):
        df = make_labeled_df(n_wires=2)
        result = self._make_fitter(df).pca_curve_fitting()
        for wire in result["wires"]:
            assert "x0" in wire and "y0" in wire and "c" in wire

    def test_file_name_recorded_correctly(self):
        df = make_labeled_df()
        result = self._make_fitter(df).pca_curve_fitting()
        assert result["File_name"] == "test.parquet"


# loader tests
@patch("lidar_catenary.loader.CONFIG", MOCK_CONFIG)
class TestDataLoaderValidate:
    """Tests for DataLoader.validate()."""

    def _loader(self):
        from lidar_catenary.loader import DataLoader
        return DataLoader(dataset_path="dummy.parquet")

    def _good_df(self, n=20):
        return pd.DataFrame({
            "x": np.random.uniform(-10, 10, n),
            "y": np.random.uniform(-10, 10, n),
            "z": np.random.uniform(0, 5, n),
        })

    def test_valid_dataframe_passes(self):
        self._loader().validate(self._good_df())   # should not raise

    def test_wrong_columns_raises(self):
        bad_df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        with pytest.raises(ValueError, match="Invalid columns"):
            self._loader().validate(bad_df)

    def test_null_values_raises(self):
        df = self._good_df()
        df.loc[0, "x"] = np.nan
        with pytest.raises(ValueError, match="Null values"):
            self._loader().validate(df)

    def test_too_few_points_raises(self):
        tiny_df = self._good_df(n=2)           # below min_points_for_clustering=10
        with pytest.raises(ValueError, match="Too few points"):
            self._loader().validate(tiny_df)

    def test_infinite_values_raises(self):
        df = self._good_df()
        df.loc[0, "z"] = np.inf
        with pytest.raises(ValueError, match="Infinite values"):
            self._loader().validate(df)


# clustering tests
@patch("lidar_catenary.cluster.CONFIG", MOCK_CONFIG)
class TestDataCluster:
    """Tests for DataCluster.clustering()."""

    def _make_cluster_df(self):
        """Two tight blobs well-separated so DBSCAN finds two clusters."""
        rng = np.random.default_rng(42)
        x1 = rng.normal(0, 0.1, 40)
        x2 = rng.normal(10, 0.1, 40)
        df = pd.DataFrame({
            "x": np.concatenate([x1, x2]),
            "y": np.zeros(80),
            "z": np.zeros(80),
        })
        return df

    def test_returns_labeled_df_and_cluster_count(self):
        from lidar_catenary.cluster import DataCluster
        df = self._make_cluster_df()
        dc = DataCluster(df, "test.parquet", "/tmp/test_output")
        labeled_df, n_clusters = dc.clustering()
        assert "labels" in labeled_df.columns
        assert n_clusters >= 1

    def test_labeled_df_same_length_as_input(self):
        from lidar_catenary.cluster import DataCluster
        df = self._make_cluster_df()
        dc = DataCluster(df, "test.parquet", "/tmp/test_output")
        labeled_df, _ = dc.clustering()
        assert len(labeled_df) == len(df)