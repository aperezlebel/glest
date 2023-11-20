import numpy as np
import pytest

from glest.helpers import (
    _validate_clustering,
    bins_from_strategy,
    calibration_curve,
    check_2D_array,
    grouping_loss_bias,
    grouping_loss_lower_bound,
)


def test_bins_from_strategy():
    bins_from_strategy(10, strategy="uniform")
    with pytest.raises(TypeError):
        bins_from_strategy(10, strategy="quantile")

    with pytest.raises(ValueError):
        bins_from_strategy(10, strategy="")


def test_validate_clustering():
    frac_pos = np.zeros((2, 2))
    counts = np.zeros((2, 2))
    mean_scores = np.zeros((2, 2))
    _validate_clustering(frac_pos, counts, mean_scores)
    with pytest.raises(ValueError):
        _validate_clustering(frac_pos, counts, mean_scores, counts)

    with pytest.raises(ValueError):
        _validate_clustering(frac_pos, counts.reshape(-1), mean_scores)

    with pytest.raises(ValueError):
        _validate_clustering(frac_pos, counts, mean_scores.reshape(-1))

    with pytest.raises(ValueError):
        _validate_clustering(
            frac_pos.reshape(-1), counts.reshape(-1), mean_scores.reshape(-1)
        )


def test_check_2d_array():
    x = np.zeros(2)
    check_2D_array(x)
    check_2D_array(x.reshape(-1, 1))
    x = np.zeros((2, 2))
    with pytest.raises(ValueError):
        check_2D_array(x)
    x = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        check_2D_array(x)


def test_grouping_loss_bias():
    frac_pos = np.ones((2, 2))
    counts = np.ones((2, 2))
    grouping_loss_bias(frac_pos, counts, reduce_bin=False)


def test_grouping_loss_lower_bound():
    frac_pos = np.ones((2, 2))
    counts = np.ones((2, 2))
    grouping_loss_lower_bound(frac_pos, counts, reduce_bin=False, debiased=False)


def test_calibration_curve():
    frac_pos = np.ones((2, 2))
    counts = np.ones((2, 2))
    mean_scores = np.zeros((2, 2))
    calibration_curve(frac_pos, counts, mean_scores)
    with pytest.raises(ValueError):
        calibration_curve(frac_pos, counts)
