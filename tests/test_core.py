import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier

from glest.core import GLEstimator, GLEstimatorCV, Partitioner
from glest.helpers import scores_to_bin_ids
from sklearn.base import BaseEstimator


@pytest.fixture(scope="module")
def data():
    X, y = make_classification(n_samples=500, random_state=0)
    return X, y


@pytest.fixture(scope="module")
def classifier(data):
    X, y = data
    est = LogisticRegression()
    est.fit(X, y)

    return est


@pytest.fixture(scope="module")
def y_scores(data, classifier):
    X, y = data
    y_scores = classifier.predict_proba(X)

    return y_scores


@pytest.fixture(scope="module")
def glest(classifier, data):
    est = classifier
    X, y = data

    glest = GLEstimator(est, random_state=0, train_size=0.5)
    glest.fit(X, y)

    return glest


def test_glest_estimator(data):
    X, y = data

    glest = GLEstimator(LogisticRegression())
    with pytest.raises(NotFittedError):
        glest.fit(X, y)

    glest = GLEstimator(RidgeClassifier().fit(X, y))
    with pytest.raises(ValueError):
        glest.fit(X, y)


@pytest.mark.parametrize(
    "partitioner",
    [
        "decision_stump",
        "decision_tree",
        Partitioner(DecisionTreeClassifier(), predict_method="apply"),
    ],
)
def test_glest_partitioner(data, classifier, partitioner):
    X, y = data

    glest = GLEstimator(classifier, partitioner)
    glest.fit(X, y)


@pytest.mark.parametrize(
    "partitioner",
    [
        Partitioner(DecisionTreeClassifier(random_state=0), predict_method="apply"),
        Partitioner(
            DecisionTreeClassifier(max_depth=1, random_state=0), predict_method="apply"
        ),
        Partitioner(
            KMeans(n_clusters=2, random_state=0, n_init="auto"),
            predict_method="predict",
            verbose=False,
        ),
        Partitioner.from_name("kmeans", random_state=0),
        Partitioner.from_name("decision_tree", random_state=0),
        Partitioner.from_name("decision_stump", random_state=0),
        "decision_tree",
    ],
)
def test_glest(data, classifier, partitioner):
    X, y = data

    if not isinstance(partitioner, str):
        partitioner.n_bins = 15
        partitioner.random_state = 0

    glest = GLEstimator(classifier, partitioner)
    glest.fit(X, y)


def test_glest_user_partition(data, classifier):
    X, y = data
    rng = np.random.default_rng(0)
    P = rng.binomial(1, 0.5, size=X.shape[0])

    glest = GLEstimator(classifier, partitioner=None)
    glest.fit(X, y, partition=P)


@pytest.mark.parametrize("plot_calibration", [False, True])
@pytest.mark.parametrize("plot_bins", [False, True])
@pytest.mark.parametrize("plot_cbar", [False, True])
@pytest.mark.parametrize("plot_hist", [False, True])
@pytest.mark.parametrize("plot_legend", [False, True])
def test_plot(glest, plot_calibration, plot_bins, plot_cbar, plot_hist, plot_legend):
    fig = glest.plot(
        plot_bins=plot_bins,
        plot_calibration=plot_calibration,
        plot_hist=plot_hist,
        plot_legend=plot_legend,
        plot_cbar=plot_cbar,
    )
    plt.close(fig)


@pytest.mark.parametrize("plot", [False, True])
@pytest.mark.parametrize("fig_kw", [None, dict(figsize=(4, 4))])
@pytest.mark.parametrize("scatter_kw", [None, dict(s=10)])
@pytest.mark.parametrize("calibration_kw", [None, dict(color="red")])
@pytest.mark.parametrize("hist_kw", [None, dict(color="red")])
@pytest.mark.parametrize("bin_kw", [None, dict(color="red")])
@pytest.mark.parametrize("legend_kw", [None, dict(ncols=1)])
def test_plot_kw(
    glest, plot, fig_kw, scatter_kw, calibration_kw, hist_kw, bin_kw, legend_kw
):
    fig, ax = plt.subplots(1, 1)
    fig = glest.plot(
        ax=ax,
        plot_bins=plot,
        plot_calibration=plot,
        plot_hist=plot,
        plot_legend=plot,
        plot_cbar=plot,
        fig_kw=fig_kw,
        scatter_kw=scatter_kw,
        calibration_kw=calibration_kw,
        hist_kw=hist_kw,
        bin_kw=bin_kw,
        legend_kw=legend_kw,
    )
    plt.close(fig)


@pytest.mark.parametrize("n_bins", [1, 2, 15, 100, [0, 0.5, 1]])
@pytest.mark.parametrize("strategy", ["uniform", "quantile"])
def test_scores_to_bin_ids(n_bins, strategy):
    rng = np.random.default_rng(0)
    y_scores = rng.random(size=10)
    y_bin, bins = scores_to_bin_ids(y_scores, n_bins, strategy)

    if hasattr(n_bins, "__len__"):
        n_bins = len(n_bins) - 1

    assert len(bins) == n_bins + 1
    assert (0 <= y_bin).all()
    assert (y_bin <= n_bins - 1).all()
    assert y_bin.shape == y_scores.shape
    assert bins.shape == (n_bins + 1,)


@pytest.mark.parametrize(
    "estimator",
    [
        ("decision_stump", "apply"),
        ("decision_tree", "apply"),
        (DecisionTreeClassifier(random_state=0), "apply"),
        (KMeans(n_clusters=1, n_init="auto", random_state=0), "predict"),
    ],
)
@pytest.mark.parametrize("n_bins", [1, 2, 15, 100])
@pytest.mark.parametrize(
    "strategy",
    [
        "uniform",
        "quantile",
    ],
)
def test_partitioner(estimator, n_bins, strategy):
    rng = np.random.default_rng(0)
    n = 10

    X = rng.normal(size=(n, 3))
    y_scores = rng.random(size=n)
    y_true = rng.binomial(1, 0.5, size=n)

    estimator, attribute = estimator

    if isinstance(estimator, str):
        partitioner = Partitioner.from_name(
            estimator, n_bins=n_bins, strategy=strategy, verbose=10
        )
    else:
        partitioner = Partitioner(
            estimator,
            n_bins=n_bins,
            strategy=strategy,
            predict_method=attribute,
            verbose=10,
        )
    partitioner.fit(X, y_scores, y_true=y_true)
    labels = partitioner.predict(X, y_scores)

    assert labels.shape == (n, 2)
    bin_assigned = np.unique(labels[:, 0])
    assert ((0 <= bin_assigned) & (bin_assigned <= n_bins - 1)).all()
    assert not np.isnan(labels).any()

    y_bins, _ = scores_to_bin_ids(y_scores, partitioner.bins_, strategy)

    for i in range(n_bins):
        bin_assigned = np.unique(labels[y_bins == i, 0])
        if (y_bins == i).any():
            assert len(bin_assigned) == 1

            region_assigned = np.unique(labels[y_bins == i, 1])
            if estimator in ["decision_stump", "balanced_stump"]:
                assert 1 <= len(region_assigned) <= 2


def test_glest_cv(classifier, data):
    X, y = data
    glest_cv = GLEstimatorCV(classifier, "decision_tree", random_state=0, verbose=10)
    glest_cv.fit(X, y)
    print(glest_cv)


def test_repr(glest):
    print(glest)


def test_manual_partition(data, classifier):
    X, y = data

    def quantile_partition(x, n):
        quantiles = np.quantile(x, np.linspace(0, 1, n + 1))
        quantile_ids = np.digitize(x, quantiles, right=False)
        quantile_ids[quantile_ids == n] = n - 1
        return quantile_ids

    partition = quantile_partition(X[:, 0], n=5)

    glest = GLEstimator(classifier, None)
    glest.fit(X, y, partition=partition)


def test_partitioner_exceptions(y_scores, data):
    X, y = data

    with pytest.raises(ValueError):
        Partitioner.from_name("")

    with pytest.raises(ValueError):
        Partitioner.from_name("decision_tree").transform_bins(y_scores)

    with pytest.raises(ValueError):
        Partitioner.from_name(None).fit(X, y_scores)

    class Dummy(BaseEstimator):
        def get_labels():
            pass

    with pytest.raises(AttributeError):
        Partitioner(estimator=Dummy()).fit(X, y_scores)

    class Dummy(BaseEstimator):
        def fit(self, X):
            raise AttributeError

        def get_labels():
            pass

    with pytest.raises(AttributeError):
        Partitioner(estimator=Dummy(), predict_method="labels").fit(X, y_scores)

    with pytest.raises(AttributeError):
        Partitioner(
            estimator=Dummy(), raise_on_fit_error=True, predict_method="get_labels"
        ).fit(X, y_scores)
    Partitioner(
        estimator=Dummy(),
        raise_on_fit_error=False,
        predict_method="get_labels",
        verbose=10,
    ).fit(X, y_scores)

    class Dummy:
        def fit(self, X):
            pass

        def labels():
            pass

    with pytest.raises(AttributeError):
        Partitioner(estimator=Dummy(), predict_method="labels").fit(X, y_scores)

    class Dummy:
        def fit(self, X):
            pass

        def labels():
            pass

        def clone(self):
            return self

    Partitioner(estimator=Dummy(), predict_method="labels").fit(X, y_scores)


def test_glest_exceptions(classifier, data, y_scores):
    X, y = data

    GLEstimator(y_scores.reshape(-1, 2)).fit(X, y)
    with pytest.raises(ValueError):
        GLEstimator(y_scores.reshape(-1, 2, 1)).fit(X, y)

    with pytest.raises(ValueError):
        GLEstimator(y_scores[: y_scores.shape[0] // 2]).fit(X, y)

    with pytest.raises(ValueError):
        GLEstimator(classifier).fit(X, y, test_data=(X, y), partition=y)

    with pytest.raises(ValueError):
        GLEstimator(classifier, partitioner=Partitioner.from_name("kmeans")).fit(
            X, y, partition=y
        )
    with pytest.raises(ValueError):
        GLEstimator(classifier, "kmeans").fit(X, y, partition=y)

    with pytest.raises(ValueError):
        GLEstimator(y_scores).fit(X, y, test_data=(X, y))
    GLEstimator(y_scores).fit(X, y, test_data=(X, y, y_scores))

    y2 = np.array([0, 1, 2])
    X2 = np.ones((y2.shape[0], 2))
    y_scores2 = np.ones_like(y2)
    with pytest.raises(ValueError):
        GLEstimator(y_scores2).fit(X2, y2)

    with pytest.raises(ValueError):
        GLEstimator(y_scores, None).fit(X, y, partition=y[: y.shape[0] // 2])


def test_glest_cv_exceptions(classifier):
    glest_cv = GLEstimatorCV(classifier)
    with pytest.raises(ValueError):
        glest_cv.GL_
    with pytest.raises(ValueError):
        glest_cv.GL_bias_
    with pytest.raises(ValueError):
        glest_cv.GL_ind_
    with pytest.raises(ValueError):
        glest_cv.GL_uncorrected_
