import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes


from glest.helpers import (
    _validate_clustering,
    bins_from_strategy,
    calibration_curve,
    check_2D_array,
    grouping_loss_bias,
    grouping_loss_lower_bound,
    compute_GL_induced,
    compute_GL_uncorrected,
    # compute_GL_uncorrected2,
    compute_GL_bias,
    psr_name_to_entropy,
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


def _compute_GL_uncorrected_brier(frac_pos, counts):
    prob_bins = calibration_curve(
        frac_pos, counts, remove_empty=False, return_mean_bins=False
    )
    diff = np.multiply(counts, 2 * np.square(frac_pos - prob_bins[:, None]))
    return np.nansum(diff) / np.sum(counts)


def test_compute_GL_uncorrected_one():
    frac_pos = np.array(
        [
            [0, 1],
            [0.5, 0.5],
        ]
    )

    counts = np.array(
        [
            [1, 1],
            [1, 1],
        ]
    )

    GL_uncorrected = compute_GL_uncorrected(frac_pos, counts, psr="brier")
    GL_uncorrected2 = _compute_GL_uncorrected_brier(frac_pos, counts)

    assert np.allclose(GL_uncorrected, 1 / 4)
    assert np.allclose(GL_uncorrected2, 1 / 4)


def partitioning_factory():
    shape_strategy = st.shared(
        array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=10)
    )

    # Define strategies for integer and float arrays
    counts_strategy = shape_strategy.flatmap(lambda s: arrays(np.int32, s))
    frac_pos_strategy = shape_strategy.flatmap(
        lambda s: arrays(np.float64, s, elements=st.floats(0, 1))
    )

    # Use the st.tuples strategy to create a tuple of two arrays with the same shape
    return st.tuples(frac_pos_strategy, counts_strategy)


@given(partitioning_factory())
def test_compute_GL_uncorrected(partitioning):
    frac_pos, counts = partitioning
    GL1 = _compute_GL_uncorrected_brier(frac_pos, counts)
    GL2 = compute_GL_uncorrected(frac_pos, counts, psr="brier")

    assert np.allclose(GL1, GL2, equal_nan=True)


def _compute_GL_induced_brier(c_hat, y_bins):
    """Estimate GL induced for the Brier score."""
    uniques, counts = np.unique(y_bins, return_counts=True)
    var = []

    for i in uniques:
        var.append(2 * np.var(c_hat[y_bins == i]))

    GL_ind = np.vdot(var, counts) / np.sum(counts)

    return GL_ind


def test_compute_GL_induced_one():
    """Example of a calibrated classifier with uniform scores."""
    # Calibrated classifier with uniform scores
    n_bins, n_samples_per_bin = 10, 1000
    c_hat = np.linspace(0, 1, n_bins * n_samples_per_bin)
    y_bins = np.repeat(np.arange(n_bins), n_samples_per_bin)

    GL_induced = compute_GL_induced(c_hat, y_bins, psr="brier")
    GL_induced2 = _compute_GL_induced_brier(c_hat, y_bins)

    assert np.allclose(GL_induced, GL_induced2)

    # From Perez-Lebel et al. 2023 Lemma C.3
    GL_induced_theoretical = 2 * 1 / (12 * n_bins**2)
    assert np.allclose(GL_induced, GL_induced_theoretical, atol=1e-6)
    assert np.allclose(GL_induced2, GL_induced_theoretical, atol=1e-6)


@pytest.mark.parametrize(
    "param",
    [
        ("brier", lambda x: 2 * x * (1 - x)),
        ("log", lambda x: -x * np.log(x) - (1 - x) * np.log(1 - x)),
    ],
)
@given(x=st.floats(min_value=0, max_value=1))
def test_psr_name_to_entropy(param, x):
    "Check the entropies associated with the names."
    psr, entropy = param

    entropy2 = psr_name_to_entropy(psr)

    assert np.allclose(entropy(x), entropy2(x), equal_nan=True)
