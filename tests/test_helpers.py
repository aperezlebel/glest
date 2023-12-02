import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays, array_shapes


from glest.helpers import (
    _validate_clustering,
    bins_from_strategy,
    calibration_curve,
    check_2D_array,
    compute_GL_induced,
    compute_GL_uncorrected,
    psr_name_to_entropy,
    compute_GL_bias,
    filter_valid_counts,
)


@pytest.fixture(scope="module")
def c_hat():
    n_bins, n_samples_per_bin = 10, 1000
    return np.linspace(0, 1, n_bins * n_samples_per_bin)


@pytest.fixture(scope="module")
def y_bins():
    n_bins, n_samples_per_bin = 10, 1000
    return np.repeat(np.arange(n_bins), n_samples_per_bin)


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


def test_calibration_curve():
    frac_pos = np.ones((2, 2))
    counts = np.ones((2, 2))
    mean_scores = np.zeros((2, 2))
    calibration_curve(frac_pos, counts, mean_scores)
    with pytest.raises(ValueError):
        calibration_curve(frac_pos, counts)


def _compute_GL_uncorrected_brier(frac_pos, counts):
    counts = filter_valid_counts(counts)

    prob_bins = calibration_curve(
        frac_pos, counts, remove_empty=False, return_mean_bins=False
    )
    diff = np.multiply(counts, 2 * np.square(frac_pos - prob_bins[:, None]))
    n_samples = np.sum(counts)
    if n_samples > 0:
        return np.nansum(diff) / n_samples
    else:
        return 0


def test_compute_GL_uncorrected_one():
    frac_pos = np.array(
        [
            [0, 1],
            [0.5, 0.5],
        ]
    )

    counts = np.array(
        [
            [2, 2],
            [2, 2],
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
    counts_strategy = shape_strategy.flatmap(
        lambda s: arrays(np.int32, s, elements=st.integers(0, 10))
    )
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

    assert np.isfinite(GL2)
    assert np.allclose(GL1, GL2)

    # GL_uncorrected should be positive
    assert GL2 >= -1e-15  # allow for numerical errors


@given(partitioning_factory())
def test_compute_GL_bias(partitioning):
    frac_pos, counts = partitioning
    GL_bias = compute_GL_bias(frac_pos, counts)
    assert np.isfinite(GL_bias)


def c_hat_factory():
    shape_strategy = st.shared(
        array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100)
    )

    # Define strategies for integer and float arrays
    y_bins_strategy = shape_strategy.flatmap(
        lambda s: arrays(np.int32, s, elements=st.integers(0, 10))
    )
    c_hat_strategy = shape_strategy.flatmap(
        lambda s: arrays(np.float64, s, elements=st.floats(0, 1))
    )

    # Use the st.tuples strategy to create a tuple of two arrays with the same shape
    return st.tuples(c_hat_strategy, y_bins_strategy)


@given(params=c_hat_factory())
def test_compute_GL_induced(params):
    c_hat, y_bins = params
    GL_induced = compute_GL_induced(c_hat, y_bins)
    print(GL_induced)
    assert np.isfinite(GL_induced)

    # GL_induced should be positive
    assert GL_induced >= -1e-15  # allow for numerical errors


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
    assert np.allclose(entropy(x), psr_name_to_entropy(entropy)(x), equal_nan=True)


def test_psr_name_to_entropy_exceptions():
    with pytest.raises(ValueError):
        psr_name_to_entropy("unexisting")


def size_one_factory():
    shape_strategy = st.shared(
        array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=10)
    )

    # Define strategies for integer and float arrays
    # counts_strategy = shape_strategy.flatmap(lambda s: arrays(np.int32, s, elements=1))
    counts_strategy = shape_strategy.flatmap(
        lambda shape: st.just(np.ones(shape, dtype=np.int32))
    )
    frac_pos_strategy = shape_strategy.flatmap(
        lambda s: arrays(np.float64, s, elements=st.floats(0, 1))
    )

    # Use the st.tuples strategy to create a tuple of two arrays with the same shape
    return st.tuples(frac_pos_strategy, counts_strategy)


@given(size_one_factory())
def test_size_one_regions(partitioning):
    """Regions with only one sample must be discarded because there cannot be
    a valid debiasing correction for these regions. The debiasing starts
    with two samples per region (factor 1/(n-1)).

    This test checks that size one regions are discarded for the metrics
    computation.
    """
    frac_pos, counts = partitioning

    # print(counts)

    GL_bias = compute_GL_bias(frac_pos, counts)

    print(GL_bias)

    GL_uncorrected = compute_GL_uncorrected(frac_pos, counts)
    GL_induced = compute_GL_induced(frac_pos, counts)

    # print(GL_uncorrected)
    print(GL_induced)

    assert GL_uncorrected == 0
    assert GL_bias == 0
