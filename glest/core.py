import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, check_X_y, column_or_1d
from sklearn.model_selection._split import check_cv
from .helpers import (
    CEstimator,
    scores_to_bin_ids,
    compute_GL_induced,
    compute_GL_bias,
    list_list_to_array,
    bins_from_strategy,
    compute_GL_uncorrected,
    filter_valid_counts,
)
from .plot import grouping_diagram
from sklearn.utils.validation import indexable
from sklearn.base import clone
from typing import List


class Partitioner:
    """A class for partitionning the feature space, stratified by level sets
    of predicted probabilities.

    Parameters
    ----------
    estimator : object
        An estimator to create the partition within level sets of predicted
        probabilities. It must implement a fit method. In each bin, the
        estimator is fitted using the fit method. Then, region assignments
        are retrieved through the method defined with the `predict_method`
        argument. The estimator must either support `sklearn.base.clone`
        method (e.g. deriving from `sklearn.base.BaseEstimator`),
        or implementing a `clone` method.
    predict_method : str, default=None
        The name of the method to call on `estimator` to get the class
        assignments. If estimator is not None, `predict_method` should be set.
    n_bins : int, default=15
        The number of bins to split the probability space [0, 1] into.
    strategy : {"uniform", "quantile"}, default="uniform"
        The binning strategy used to create the bins. With uniform, same-width
        bins are created. With quantile, same-mass bins are created.
    binwise_fit : bool, default=True
        When True, fits one partitioner per bin. Otherwise, fits one
        partitioner on the whole feature space at once.
    raise_on_fit_error : bool, default=False
        Whether to raise an error when the estimator fails to fit on a bin.
        If False, no partition is created on the failing bin and all samples
        within this bin are assigned the same label. If True, raises an error.
    verbose : int, default=0
        Whether to print progress.

    Raises
    ------
    Exception
        When `raise_on_fit_error` is True and the estimator fails to fit on a
        bin.

    """

    def __init__(
        self,
        estimator,
        predict_method: str = None,
        n_bins: int = 15,
        strategy: str = "uniform",
        binwise_fit: bool = True,
        raise_on_fit_error: bool = False,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.n_bins = n_bins
        self.strategy = strategy
        self.binwise_fit = binwise_fit
        self.predict_method = predict_method
        self.raise_on_fit_error = raise_on_fit_error
        self.verbose = verbose

    @classmethod
    def from_name(
        cls,
        name: str,
        n_bins: int = 15,
        strategy: str = "uniform",
        binwise_fit: bool = True,
        raise_on_fit_error: bool = False,
        verbose: int = 0,
        random_state: int = None,
    ):
        """Load a predefined Partitioner instance from a name.

        Parameters
        ----------
        name : {"decision_tree", "decision_stump", "kmean", None}
            The predefined estimator to use to partition the bins.
        n_bins : int, default=15
            The number of bins to split the probability space [0, 1] into.
        strategy : {"uniform", "quantile"}, default="uniform"
            The binning strategy used to create the bins. With uniform, same-width
            bins are created. With quantile, same-mass bins are created.
        binwise_fit : bool, default=True
            When True, fits one partitioner per bin. Otherwise, fits one
            partitioner on the whole feature space at once.
        raise_on_fit_error : bool, default=False
            Whether to raise an error when the estimator fails to fit on a bin.
            If False, no partition is created on the failing bin and all samples
            within this bin are assigned the same label. If True, raises an error.
        verbose : int, default=0
            Whether to print progress.
        random_state : int, default=none
            Controls the randomness of the estimator used in the partitioner.

        """
        available_names = [
            "decision_tree",
            "decision_stump",
            "kmeans",
            None,
        ]

        if name not in available_names:
            raise ValueError(
                f'Unknown name "{name}". Available names are: {available_names}.'
            )

        if name == "decision_tree":
            estimator = DecisionTreeRegressor(
                max_depth=10,
                random_state=random_state,
            )
            predict_method = "apply"

        elif name == "decision_stump":
            estimator = DecisionTreeRegressor(max_depth=1, random_state=random_state)
            predict_method = "apply"

        elif name == "kmeans":
            estimator = KMeans(
                n_clusters=2,
                random_state=random_state,
                n_init="auto",
            )
            predict_method = "predict"

        elif name is None:
            estimator = None
            predict_method = None

        return cls(
            estimator=estimator,
            n_bins=n_bins,
            strategy=strategy,
            binwise_fit=binwise_fit,
            raise_on_fit_error=raise_on_fit_error,
            verbose=verbose,
            predict_method=predict_method,
        )

    def fit_bins(self, y_scores=None):
        """Create bins from strategy, number of bins and proba distribution
        if necessary.

        Parameters
        ----------
        y_scores : array-like
            The probabilities from which to derive the bins if
            strategy="quantile".

        """
        self.bins_ = bins_from_strategy(self.n_bins, self.strategy, y_scores)

    def transform_bins(self, y_scores):
        """Convert probabilities to their bin assignment.

        Parameters
        ----------
        y_scores : array-like of shape (n,)
            The probabilities from which to derive the assignments.

        Returns
        -------
        array-like of shape (n,)
            The array of bin indices each probability falls into.

        """
        if not hasattr(self, "bins_"):
            raise ValueError("fit_bins must have been called before transform_bins.")
        y_bins, _ = scores_to_bin_ids(y_scores, self.bins_, None)
        return y_bins

    def fit(self, X, y_scores, y_true=None):
        """Fit the partitioner.

        Parameters
        ----------
        X : array-like of shape (n, d)
            The features.
        y_scores : array-like of shape (n,)
            The probabilities of each sample.
        y_true : array-like of shape (n,), optional
            The true labels. Used by some partitioner to find the best
            partitions, by default None

        Returns
        -------
        Partitioner
            Returns the current instance.

        """
        y_scores = GLEstimator._validate_scores(y_scores)

        if self.estimator is None:
            raise ValueError(
                "A Partitioner with estimator=None cannot be fitted. To use "
                "predefined partitions, use the partition argument of "
                "GLEstimator.fit instead."
            )

        if not hasattr(self.estimator, "fit"):
            raise AttributeError(
                f'partitioner {self.estimator} must implement a "fit" method.'
            )

        if not hasattr(self.estimator, self.predict_method):
            raise AttributeError(
                f'"{self.estimator.__class__.__name__}" object has no '
                f'attribute "{self.predict_method}". Make sure `predict_method` '
                f'is set accordingly to the estimator "{self.estimator}".'
            )

        self.fit_bins(y_scores)  # bins are stored in self.bins_

        if self.binwise_fit:
            y_bins = self.transform_bins(y_scores)
            n_bins = len(self.bins_) - 1
        else:
            y_bins = np.zeros_like(y_scores)
            n_bins = 1

        fitted_partitioners_ = []
        for i in range(n_bins):
            if self.verbose > 0:
                print(f"Bin {i+1}/{n_bins}: partitioning.")
            bin_idx = y_bins == i
            X_bin = X[bin_idx, :]
            n_samples_bin = X_bin.shape[0]

            if n_samples_bin > 0:
                try:
                    partitioner_bin = clone(self.estimator)
                except TypeError:
                    if not hasattr(self.estimator, "clone"):
                        raise AttributeError(
                            f'Estimator "{self.estimator}" must either support '
                            f"sklearn.base.clone, or implement a `clone` method "
                            f"itself."
                        )
                    partitioner_bin = self.estimator.clone()
                try:
                    if y_true is None:
                        partitioner_bin.fit(X_bin)
                    else:
                        partitioner_bin.fit(X_bin, y_true[bin_idx])
                except Exception as e:
                    if self.raise_on_fit_error:
                        raise e
                    else:
                        if self.verbose:
                            print(
                                f"WARNING: No partition created in bin #{i}: "
                                f"estimator {self.estimator} failed to fit. "
                                f'"{e}"'
                            )
                        partitioner_bin = None
            else:
                partitioner_bin = None
            fitted_partitioners_.append(partitioner_bin)

        self.fitted_partitioners_ = fitted_partitioners_
        return self

    def predict(self, X, y_scores):
        """Get the region assignments.

        Parameters
        ----------
        X : array-like of shape (n, d)
            The features.
        y_scores : array-like of shape (n,)
            The probabilities of each sample.

        Returns
        -------
        array-like of shape (n,)
            The assignments of each sample to the partition.
        """
        check_is_fitted(self)

        labels = np.full((X.shape[0], 2), np.nan)
        y_bins, _ = scores_to_bin_ids(y_scores, self.bins_, None)

        for i in range(len(self.bins_) - 1):
            if self.verbose > 0:
                print(f"Bin {i+1}/{len(self.bins_)-1}: assigning.")
            bin_idx = y_bins == i  # restrict to samples belonging to bin i
            X_bin = X[bin_idx, :]
            n_samples_bin = X_bin.shape[0]
            partitioner = self.fitted_partitioners_[i if self.binwise_fit else 0]

            # Store partition id
            if partitioner is not None and n_samples_bin > 0:
                predict_method = getattr(partitioner, self.predict_method)
                labels[bin_idx, 1] = predict_method(X_bin)

            else:
                # no partitioner was fit in this bin because not enough training samples
                # hence gather all test samples in same group
                labels[bin_idx, 1] = np.zeros(n_samples_bin)

            # Store bin id
            labels[bin_idx, 0] = i

        return labels


class GLEstimator:
    """Estimate the grouping loss of a fitted probabilistic classifier.

    Parameters
    ----------
    classifier : object
        The classifier for which to estimate the grouping loss. The
        classifier must implement a `predict_proba` method. The classifier
        must already be fit since GLEstimator only evaluates the classifier.
    partitioner : {"decision_tree", "decision_stump", "kmean", None}
                  | Partitioner, optional
        The partitioning strategy to use for estimating the grouping loss.
        If string given, use corresponding predefined strategy. If
        `Partitioner` instance given, use this as partitioner.
        By default "decision_tree".
    train_size : float, optional
        The size of the training set size. To avoid overfitting, the
        estimation of the grouping loss is evaluated on a test set and the
        partition is created on the training set. By default 0.5.
    random_state : int, optional
        Controls the randomness of the estimator used in the partitioner.
        By default None.
    verbose : int, optional
        Whether to print progress. By default 0.
    """

    default_n_bins: int = 15
    default_strategy: str = "uniform"
    default_binwise_fit: bool = True

    def __init__(
        self,
        classifier,
        partitioner: str | Partitioner = "decision_tree",
        train_size: float = 0.5,
        random_state: int = None,
        verbose: int = 0,
    ) -> None:
        self.classifier = classifier
        self.partitioner = partitioner
        self.train_size = train_size
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _validate_scores(y_scores):
        """Uniformize probability array shape to (n,) from either (n,), (n, 1)
        or (n, 2)."""
        if y_scores.ndim == 2 and y_scores.shape[1] == 2:
            y_scores = y_scores[
                :, 1
            ]  # since y_type is binary take only the positive class
        elif y_scores.ndim != 1:
            raise ValueError(
                f"Invalid proba array shape: {y_scores.shape}. Expecting (n,)"
            )

        y_scores = np.array(y_scores).squeeze()
        return y_scores

    @staticmethod
    def _is_valid_classifier(est):
        """Check what is considered a valid classifier."""
        return hasattr(est, "predict_proba")

    @staticmethod
    def _probas_from_estimator(est, X):
        """Get the probability array by checking if estimator is either a
        classifier or an array."""
        if GLEstimator._is_valid_classifier(est):
            y_scores = est.predict_proba(X)
        else:
            try:
                y_scores = np.array(est)
                y_scores.shape[0]
            except Exception:
                raise ValueError(
                    "classifier must either implement a predict_proba method, "
                    "or be an array of probability."
                )
            if X.shape[0] != y_scores.shape[0]:
                raise ValueError(
                    f"Shape mismatch between proba array given as classifier "
                    f"and the data given in fit: X.shape[0]={X.shape[0]} "
                    f"y_scores.shape[0]={y_scores.shape[0]}"
                )
        y_scores = np.array(y_scores)
        y_scores = GLEstimator._validate_scores(y_scores)
        return y_scores

    def fit(self, X, y, test_data=None, partition=None):
        """Create the partitions and evaluate the grouping loss. After fit,
        the metrics are accessible at GL_, GL_ind_, GL_bias_.

        Parameters
        ----------
        X : array-like of shape (n, d)
            The features.
        y : array-like of shape (n,)
            The binary labels.
        test_data : tuple of array-likes, optional
            The test data on which to evaluate the grouping loss.
            If None, the data (X, y) is split into a training and test data
            based on the `train_size` argument. The partitions are created
            on the training data and the grouping loss is evaluated on the test
            data. If `(X2, y2)` given, (X, y) is taken as training set and
            (X2, y2) as test set. If `classifier` is not an estimator but an
            array of probabilities, then `test_data` must either be None or
            a tuple (X2, y2, y_scores2). By default None.
        partition : array-like of shape (n,), optional
            The predefined partition along which to evaluate the grouping loss.
            If set, (X, y) is taken as the test data on which is evaluated the
            grouping loss. `partition` and `test_data` are thus incompatible
            and only one of them can be set at the same time. If None,
            the partition is created using the `partitioner`. By default None.

        Returns
        -------
        GLEstimator
            The fitted instance.

        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y")
        if y_type != "binary":
            raise ValueError(f"y must be binary. Got {y_type}.")
        y = column_or_1d(y)

        if partition is not None and test_data is not None:
            raise ValueError(
                f"partition and test_data cannot be both not None. "
                f"Got partition={type(partition)} and test_data={type(test_data)}."
            )

        # Get the scores
        y_scores = GLEstimator._probas_from_estimator(self.classifier, X)

        if self.partitioner is None or isinstance(self.partitioner, str):
            self.partitioner_ = Partitioner.from_name(
                name=self.partitioner,
                n_bins=GLEstimator.default_n_bins,
                strategy=GLEstimator.default_strategy,
                binwise_fit=GLEstimator.default_binwise_fit,
                random_state=self.random_state,
                verbose=self.verbose - 1,
            )
        else:
            self.partitioner_: Partitioner = self.partitioner

        if partition is not None:
            if (
                hasattr(self.partitioner_, "estimator")
                and self.partitioner_.estimator is not None
            ):
                raise ValueError(
                    "Specifying a custom partition is only available when "
                    "partitioner=None or "
                    "partitioner=Partitioner.from_name(None, ...)"
                )
            self.partitioner_.fit_bins(y_scores)
            return self._evaluate(X, y, y_scores, partition=partition)

        if test_data is None:
            self.partitioner_.fit_bins(y_scores)
            y_bins, _ = scores_to_bin_ids(
                y_scores, self.partitioner_.bins_, self.partitioner_.strategy
            )
            # We use a stratified shuffle split to keep the split balance in each bin
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=self.train_size, random_state=self.random_state
            )
            train_index, test_index = next(sss.split(X, y_bins))
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index]
            y_test = y[test_index]
            y_scores_train = y_scores[train_index]
            y_scores_test = y_scores[test_index]
        else:
            X_train, y_train = X, y
            y_scores_train = y_scores
            if GLEstimator._is_valid_classifier(self.classifier):
                X_test, y_test = test_data
                y_scores_test = GLEstimator._probas_from_estimator(
                    self.classifier, X_test
                )
            else:
                try:
                    X_test, y_test, y_scores_test = test_data
                except Exception as e:
                    raise ValueError(
                        f"When manually passing the probabilities as classifier,"
                        f"the test_data must also pass the probabilities "
                        f"(X, y, y_probas). {e}"
                    )

                y_scores_test = GLEstimator._probas_from_estimator(
                    y_scores_test, X_test
                )

        if self.verbose > 0:
            print("Fitting.")
        self._fit(X_train, y_train, y_scores_train)
        self._evaluate(X_test, y_test, y_scores_test)
        return self

    def _fit(self, X, y, y_scores):
        self.partitioner_.fit(X, y_scores, y)
        self.n_features_in_ = X.shape[1]
        return self

    def _evaluate(self, X, y, y_scores, partition=None):
        if partition is None:
            check_is_fitted(self)

        y_bins = self.partitioner_.transform_bins(y_scores)

        if partition is not None:
            if partition.shape != y_bins.shape:
                raise ValueError(
                    f"Given partition must have the same shape as y_probas. "
                    f"Got partition.shape={partition.shape} and "
                    f"y_probas.shape={y_scores.shape}"
                )
            labels = np.stack([y_bins, partition], axis=1)
        else:
            labels = self.partitioner_.predict(X, y_scores)

        frac_pos = []
        counts = []
        mean_scores = []

        for i in range(len(self.partitioner_.bins_) - 1):
            if self.verbose:
                print(f"Bin {i+1}/{len(self.partitioner_.bins_) - 1}: evaluating.")
            bin_idx = y_bins == i
            y_bin = y[bin_idx]
            y_scores_bin = y_scores[bin_idx]

            unique_labels, unique_counts = np.unique(
                labels[bin_idx, 1], return_counts=True
            )

            frac_pos.append([])
            counts.append([])
            mean_scores.append([])

            for label in unique_labels:
                if len((labels == label)[bin_idx, 1]) > 0:
                    frac_pos[i].append(np.mean(y_bin[(labels == label)[bin_idx, 1]]))
                    mean_scores[i].append(
                        np.mean(y_scores_bin[(labels == label)[bin_idx, 1]])
                    )

            counts[i].extend(unique_counts)

        frac_pos = list_list_to_array(frac_pos, fill_value=0)
        counts = list_list_to_array(counts, fill_value=0, dtype=int)
        mean_scores = list_list_to_array(mean_scores, fill_value=0)

        self.frac_pos_ = frac_pos
        self.counts_ = counts
        self.mean_scores_ = mean_scores

        self.c_hat_ = CEstimator(y_scores, y).c_hat()
        self.y_bins_, _ = scores_to_bin_ids(y_scores, self.partitioner_.bins_, None)

        return self

    def GL(self, psr: str = "brier"):
        return self.GL_uncorrected(psr) - self.GL_bias(psr) - self.GL_induced(psr)

    def GL_uncorrected(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimator is not fitted.")

        return compute_GL_uncorrected(self.frac_pos_, self.counts_, psr)

    def GL_bias(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimator is not fitted.")

        return compute_GL_bias(self.frac_pos_, self.counts_, psr)

    def GL_induced(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimator is not fitted.")

        return compute_GL_induced(self.c_hat_, self.y_bins_, psr)

    def metrics(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError('GLEstimator must be fitted to call "metrics".')

        GL_ind = self.GL_induced(psr)
        GL_uncorrected = self.GL_uncorrected(psr)
        GL_bias = self.GL_bias(psr)
        GL = GL_uncorrected - GL_bias - GL_ind

        return {
            "psr": psr,
            "GL": GL,
            "GL_induced": GL_ind,
            "GL_uncorrected": GL_uncorrected,
            "GL_bias": GL_bias,
        }

    def plot(
        self,
        ax: plt.Axes = None,
        plot_bins: bool = True,
        plot_calibration: bool = True,
        plot_hist: bool = True,
        plot_legend: bool = True,
        plot_cbar: bool = True,
        fig_kw: dict = None,
        scatter_kw: dict = None,
        calibration_kw: dict = None,
        hist_kw: dict = None,
        bin_kw: dict = None,
        legend_kw: dict = None,
    ) -> plt.Figure:
        """Plot the grouping diagram.

        Parameters
        ----------
        ax : plt.Axes, optional
            The axis on which to plot. If None, a new figure is created.
            By default None.
        plot_bins : bool, optional
            Whether to plot the vertical lines for bins.
            By default True.
        plot_calibration : bool, optional
            Whether to plot the calibration curve.
            By default True.
        plot_hist : bool, optional
            Whether to plot the x-axis histogram.
            By default True.
        plot_legend : bool, optional
            Whether to plot the legend.
            By default True.
        plot_cbar : bool, optional
            Whether to plot the colorbar.
            By default True.
        fig_kw : dict, optional
            Keyword arguments to pass to plt.subplots.
            By default None.
        scatter_kw : dict, optional
            Keyword arguments to pass to ax.scatter.
            By default None.
        calibration_kw : dict, optional
            Keyword arguments to pass to ax.plot for the calibration curve.
            By default None.
        hist_kw : dict, optional
            Keyword arguments to pass to ax.hist for the x-axis histogram.
            By default None.
        bin_kw : dict, optional
            Keyword arguments to pass to ax.axvline for the bin edges.
            By default None.
        legend_kw : dict, optional
            Keyword arguments to pass to ax.legend.
            By default None.

        Returns
        -------
        plt.Figure
            The grouping diagram figure.
        """
        check_is_fitted(self)

        counts = filter_valid_counts(self.counts_)

        fig = grouping_diagram(
            frac_pos=self.frac_pos_,
            counts=counts,
            mean_scores=self.mean_scores_,
            bins=self.partitioner_.bins_,
            ax=ax,
            plot_bins=plot_bins,
            plot_calibration=plot_calibration,
            plot_hist=plot_hist,
            plot_legend=plot_legend,
            plot_cbar=plot_cbar,
            fig_kw=fig_kw,
            scatter_kw=scatter_kw,
            calibration_kw=calibration_kw,
            hist_kw=hist_kw,
            bin_kw=bin_kw,
            legend_kw=legend_kw,
        )

        return fig

    def is_fitted(self):
        return (
            hasattr(self, "frac_pos_")
            and hasattr(self, "counts_")
            and hasattr(self, "mean_scores_")
            and hasattr(self, "c_hat_")
            and hasattr(self, "y_bins_")
        )

    def __format__(self, psr: str) -> str:
        """Print the computed metrics."""
        s = "GLEstimator()"

        if self.is_fitted():
            if not psr:
                psr = "brier"

            metrics = self.metrics(psr)

            extra = (
                f"  Scoring Rule      : {psr}\n"
                f"  Grouping loss     : {metrics['GL']:.4f}\n"
                f"   ↳ Uncorrected GL : {metrics['GL_uncorrected']:.4f}\n"
                f"   ↳ Bias           : {metrics['GL_bias']:.4f}\n"
                f"   ↳ Binning induced: {metrics['GL_induced']:.4f}\n"
            )
            s = f"{s}\n{extra}"

        return s

    def __str__(self) -> str:
        return f"{self}"

    def __repr__(self) -> str:
        return f"{self}"


class GLEstimatorCV:
    """Estimate the grouping loss of a probabilistic classifier.

    Parameters
    ----------
    classifier : object
        The classifier for which to estimate the grouping loss. The
        classifier must implement a `predict_proba` method.
    partitioner : {"decision_tree", "decision_stump", "kmean", None}
                  | Partitioner, optional
        The partitioning strategy to use for estimating the grouping loss.
        If string given, use corresponding predefined strategy. If
        `Partitioner` instance given, use this as partitioner.
        By default "decision_tree".
    cv : int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy using
        `sklearn.model_selection._split.check_cv`. See scikit-learn doc
        for more details (e.g. `sklearn.model_selection.cross_validate`).
    random_state : int, optional
        Controls the randomness of the estimator used in the partitioner.
        By default None.
    verbose : int, optional
        Whether to print progress. By default 0.

    """

    def __init__(
        self,
        classifier,
        partitioner="decision_tree",
        cv=None,
        random_state: int = None,  # not the rs of the cv
        verbose: int = 0,
    ):
        self.classifier = classifier
        self.partitioner = partitioner
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose

    def GL(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimatorCV is not fitted.")
        return np.array([glest.GL(psr) for glest in self.glests_])

    def GL_uncorrected(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimatorCV is not fitted.")
        return np.array([glest.GL_uncorrected(psr) for glest in self.glests_])

    def GL_bias(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimatorCV is not fitted.")
        return np.array([glest.GL_bias(psr) for glest in self.glests_])

    def GL_induced(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimatorCV is not fitted.")
        return np.array([glest.GL_induced(psr) for glest in self.glests_])

    def fit(self, X, y, groups=None):
        """Fit a GLEstimator instance on each of the train/test split yield
        by `cv`. Each instance is stored in the `glests_` attribute.

        Parameters
        ----------
        X : array-like of shape (n, d)
            The features.
        y : array-like of shape (n,)
            The binary labels.
        groups : array-like of shape (n,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a “Group” cv
            instance. See `sklearn.model_selection.cross_validate` for
            details. By default None.

        Returns
        -------
        GLEstimatorCV
            The fitted instance.
        """
        X, y, groups = indexable(X, y, groups)
        cv = check_cv(self.cv, y=y, classifier=True)
        indices = cv.split(X, y, groups)
        glests: List[GLEstimator] = []
        for i, (train, test) in enumerate(indices):
            if self.verbose > 0:
                print(f"Split {i+1}/{cv.get_n_splits()}")
            glest = GLEstimator(
                classifier=self.classifier,
                partitioner=self.partitioner,
                random_state=self.random_state,
                verbose=self.verbose - 1,
            )
            glest.fit(X[train], y[train], test_data=(X[test], y[test]))
            glests.append(glest)

        self.glests_ = glests
        self.cv_ = cv

        return self

    def is_fitted(self):
        return hasattr(self, "glests_")

    def metrics(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError('GLEstimator must be fitted to call "metrics".')

        GL_ind = self.GL_induced(psr)
        GL_uncorrected = self.GL_uncorrected(psr)
        GL_bias = self.GL_bias(psr)
        GL = GL_uncorrected - GL_bias - GL_ind

        return {
            "psr": psr,
            "GL": GL,
            "GL_induced": GL_ind,
            "GL_uncorrected": GL_uncorrected,
            "GL_bias": GL_bias,
        }

    def __format__(self, psr: str) -> str:
        """Print the computed average metrics with variance."""
        s = "GLEstimatorCV()"

        def format_trials(values):
            mean = np.mean(values)
            std = np.std(values)
            return f"{mean:.4f} ({std:.4f})"

        if self.is_fitted():
            if not psr:
                psr = "brier"

            metrics = self.metrics(psr)

            extra = (
                # f"  Splits            : {self.cv_}\n"
                f"  Scoring rule      : {psr}\n"
                f"  Grouping loss     : {format_trials(metrics['GL'])}\n"
                f"   ↳ Uncorrected GL : {format_trials(metrics['GL_uncorrected'])}\n"
                f"   ↳ Bias           : {format_trials(metrics['GL_bias'])}\n"
                f"   ↳ Binning induced: {format_trials(metrics['GL_induced'])}\n"
            )
            s = f"{s}\n{extra}"

        return s

    def __str__(self) -> str:
        return f"{self}"

    def __repr__(self) -> str:
        return f"{self}"
