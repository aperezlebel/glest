import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection._split import train_test_split
from .plot import grouping_diagram
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import brier_score_loss


class PartitioningEstimate:
    """A class for partitioning-based estimation with honest splitting.

    This class fits a partitioning estimator to predict residuals from calibrated
    scores, enabling grouping loss estimation and risk analysis.

    Parameters
    ----------
    estimator : object or str
        The estimator to use for partitioning (e.g., DecisionTreeRegressor, KMeans).
        Can also be a string name like "decision_tree", "decision_stump", or "kmeans".
    predict_method : str, optional
        The method to call on the estimator to get partition assignments
        (e.g., "apply" for trees, "predict" for KMeans). Default is None.
    verbose : int, default=1
        Controls verbosity of output during fitting and evaluation.

    Attributes
    ----------
    calibrator : LogisticRegression
        The fitted calibrator for probability scores.
    tree : callable
        Function mapping features to residual predictions.
    r_j : ndarray
        Mean residuals for each partition.
    v_j : ndarray
        Variance of residuals for each partition.
    n_j : ndarray
        Number of samples in each partition.
    group_definitions : dict
        Human-readable definitions of each partition/group.

    """

    def __init__(
        self,
        estimator,
        predict_method: str = None,
        verbose: int = 0,
    ) -> None:
        self.estimator = estimator
        self.predict_method = predict_method
        self.verbose = verbose

    @classmethod
    def from_name(
        cls,
        name: str,
        verbose: int = 0,
        random_state: int = None,
    ):
        """Load a predefined partitioning estimator from a name.

        Parameters
        ----------
        name : {"decision_tree", "decision_stump", "kmeans", None}
            The predefined estimator to use for partitioning.
        verbose : int, default=0
            Controls verbosity of output.
        random_state : int, default=None
            Controls the randomness of the estimator.

        Returns
        -------
        estimator : object or None
            The configured estimator instance.

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
                min_samples_leaf=15,
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

        return estimator, predict_method

    def fit(self, X, y_scores, y_true, seed: int = 42):
        """
        Fit the partitioning estimator with honest splitting.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.verbose > 0:
            print("Starting fit process...")

        y_scores = y_scores.reshape(-1, 1)
        X_train, X_test, y_scores_train, y_scores_test, y_true_train, y_true_test = (
            train_test_split(X, y_scores, y_true, test_size=0.5, random_state=seed)
        )

        X_train, X_cal, y_scores_train, y_scores_cal, y_true_train, y_true_cal = (
            train_test_split(
                X_train, y_scores_train, y_true_train, test_size=0.2, random_state=seed
            )
        )

        if self.verbose > 0:
            print(f"Calibration set size: {len(X_cal)}")
            print(f"Train set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")

        self.calibrate(y_scores_cal, y_true_cal)
        self.train(X_train, y_scores_train, y_true_train)
        self.evaluate(X_test, y_scores_test, y_true_test)

        if hasattr(X_test, "columns"):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = None
        self.get_group_definitions(X_test, feature_names=feature_names)

        if self.verbose > 0:
            print("Fit process completed.")

        return self

    def calibrate(self, y_scores, y_true):
        """
        Calibrate the predicted scores using logistic regression.
        Parameters
        ----------
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted calibrator.
        """
        if self.verbose > 1:
            print("Calibrating scores...")

        calibrator = LogisticRegression()
        calibrator.fit(y_scores, y_true)
        self.calibrator = calibrator

        if self.verbose > 1:
            print("Calibration completed.")

        return self

    def train(self, X, y_scores, y_true):
        """
        Train the partitioning estimator on residuals.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted partitioning estimator.
        """
        if self.verbose > 1:
            print("Training partitioning estimator...")

        if isinstance(self.estimator, str):
            self.estimator, self.predict_method = PartitioningEstimate.from_name(
                self.estimator
            )

        residuals_train = y_true - self.calibrator.predict(y_scores)
        self.estimator.fit(X, residuals_train)

        if self.verbose > 1:
            print("Training completed.")

        return self

    def evaluate(self, X, y_scores, y_true):
        """
        Evaluate the partitioning estimator on a test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Evaluated partitioning estimator with computed statistics.
        """
        if self.verbose > 1:
            print("Evaluating on test set...")

        self.y_scores = y_scores
        self.y_true = y_true
        self.X = X
        leaf_indices = self.estimator.apply(X)

        c_hat = self.calibrator.predict_proba(y_scores)[:, 1]

        v_j = np.zeros(max(leaf_indices) + 1)
        r_j = np.zeros(max(leaf_indices) + 1)
        n_j = np.zeros(max(leaf_indices) + 1)
        # Vectorized computation using bincount for better performance
        n_j = np.bincount(leaf_indices, minlength=max(leaf_indices) + 1)
        # Compute residuals once
        residuals = y_true - c_hat

        # Vectorized computation of means and variances
        r_j = np.divide(
            np.bincount(leaf_indices, weights=residuals),
            n_j,
            out=np.zeros_like(n_j, dtype=float),
            where=n_j > 0,
        )
        # Compute variance using E[X^2] - E[X]^2 formula
        residuals_sq = residuals**2
        mean_sq = np.divide(
            np.bincount(leaf_indices, weights=residuals_sq),
            n_j,
            out=np.zeros_like(n_j, dtype=float),
            where=n_j > 0,
        )
        v_j = mean_sq - r_j**2

        # Apply Bessel's correction (ddof=1)
        v_j *= n_j / (n_j - 1)
        v_j = np.where(n_j > 1, v_j, 0)

        def r(X):
            leaf_indices = self.estimator.apply(X)
            return r_j[leaf_indices]

        self.cal_err = np.mean(np.square(y_scores.flatten() - c_hat))
        self.tree = r
        self.r_j = r_j
        self.v_j = v_j
        self.n_j = n_j

        if self.verbose > 0:
            print(f"Evaluation completed. Found {len(np.unique(leaf_indices))} groups.")
            print(f"Calibration error: {self.cal_err:.4f}")

        return self

    def predict(self, X):
        """
        Predict honest residuals for new data points.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        Returns
        -------
        r_hat : array-like of shape (n_samples,)
            The predicted residuals.
        """
        return self.tree(X)

    def apply(self, X):
        return self.estimator.apply(X)

    def plot(self, groups="all"):
        # check_is_fitted(self)
        leaf_ids = self.apply(self.X)
        n_in_leaf = self.n_j[leaf_ids]
        grouping_diagram(
            c_hat=self.calibrator.predict_proba(self.y_scores)[:, 1],
            r_hat=self.predict(self.X),
            n_in_leaf=n_in_leaf,
            f=self.y_scores.flatten(),
            leaf_ids=leaf_ids,
            groups=groups,
        )

    def get_group_definitions(self, X, feature_names=None):
        """
        Extract human-readable decision rules for each partition/group.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features used to traverse the tree.
        feature_names : list of str, optional
            Names of features for readable output. If None, uses X_0, X_1, etc.

        Returns
        -------
        group_definitions : dict
            Dictionary mapping leaf IDs to group information including rules,
            sample counts, and heterogeneity measures.
        """
        tree = self.estimator
        # Convert to numpy array if pandas DataFrame
        X_array = X.values if hasattr(X, "values") else np.asarray(X)

        # Get unique leaf IDs
        leaf_ids = tree.apply(X_array)
        unique_leaves = np.unique(leaf_ids)

        group_definitions = {}
        if feature_names is None:
            feature_names = [f"X_{i}" for i in range(X_array.shape[1])]
        elif all(isinstance(f, int) for f in feature_names):
            feature_names = [f"X_{i}" for i in feature_names]

        for leaf_id in unique_leaves:
            # Get samples in this leaf
            samples_in_leaf = X_array[leaf_ids == leaf_id]

            # Get the path to this leaf
            path = tree.decision_path(samples_in_leaf[:1]).toarray()[0]

            # Extract the rules
            raw_rules = []

            # Get the path from root to leaf
            feature = tree.tree_.feature
            threshold = tree.tree_.threshold

            for node_id in range(len(path)):
                if path[node_id] == 1:  # This node is in the path
                    if feature[node_id] != -2:  # Not a leaf node
                        # Determine if we went left or right
                        sample_feature_value = samples_in_leaf[0, feature[node_id]]
                        feat_name = feature_names[feature[node_id]]
                        if sample_feature_value <= threshold[node_id]:
                            raw_rules.append((feat_name, "<=", threshold[node_id]))
                        else:
                            raw_rules.append((feat_name, ">", threshold[node_id]))

            # Combine rules for the same feature
            feature_bounds = {}
            for feat_name, operator, value in raw_rules:
                if feat_name not in feature_bounds:
                    feature_bounds[feat_name] = {"min": None, "max": None}

                if operator == "<=":
                    if (
                        feature_bounds[feat_name]["max"] is None
                        or value < feature_bounds[feat_name]["max"]
                    ):
                        feature_bounds[feat_name]["max"] = value
                else:  # operator == ">"
                    if (
                        feature_bounds[feat_name]["min"] is None
                        or value > feature_bounds[feat_name]["min"]
                    ):
                        feature_bounds[feat_name]["min"] = value

            # Convert bounds to readable rules
            combined_rules = []
            for feat_name, bounds in feature_bounds.items():
                if bounds["min"] is not None and bounds["max"] is not None:
                    combined_rules.append(
                        f"{bounds['min']:.1f} < {feat_name} <= {bounds['max']:.1f}"
                    )
                elif bounds["min"] is not None:
                    combined_rules.append(f"{feat_name} > {bounds['min']:.1f}")
                elif bounds["max"] is not None:
                    combined_rules.append(f"{feat_name} <= {bounds['max']:.1f}")

            group_definitions[leaf_id] = {
                "rules": combined_rules,
                "n_samples": len(samples_in_leaf),
                "sample_indices": np.where(leaf_ids == leaf_id)[0],
                "heterogeneity": self.r_j[leaf_id],
            }
        self.group_definitions = group_definitions
        return group_definitions

    def groups(self):
        """
        Convert group definitions to a human-readable string format.

        Parameters
        ----------
        group_definitions : dict
            Dictionary with leaf IDs as keys and group information as values

        Returns
        -------
        str
            A formatted string with group definitions
        """
        group_definitions = self.group_definitions
        lines = []
        lines.append("=" * 80)
        lines.append("GROUP DEFINITIONS")
        lines.append("=" * 80)

        for leaf_id in sorted(group_definitions.keys()):
            info = group_definitions[leaf_id]
            lines.append(f"\nGroup {leaf_id}:")
            lines.append(f"  Heterogeneity detected: {info['heterogeneity']:.4f}")
            lines.append(f"  Number of samples: {info['n_samples']}")
            lines.append("  Rules:")
            if info["rules"]:
                for rule in info["rules"]:
                    lines.append(f"    • {rule}")
            else:
                # lines.append(f"    • No splitting rules (root/single leaf)")
                lines.append("-" * 80)

        result = "\n".join(lines)
        print(result)
        return self.group_definitions


class ResidualEstimator:
    """Estimate residuals for a fitted probabilistic classifier.

    This class provides methods to estimate the residuals of a probabilistic
    classifier by partitioning the feature space and analyzing calibration
    residuals within each partition.

    Parameters
    ----------
    partitioning_estimate : str or PartitioningEstimate, default="decision_tree"
        The partitioning strategy to use for estimating the residuals.
        If string, must be one of {"decision_tree", "decision_stump", "kmeans", None}.
        If PartitioningEstimate instance, uses the provided partitioner.
    train_size : float, default=0.5
        The proportion of the dataset to use for training the partitioner.
        The remaining data is used for evaluation to avoid overfitting.
    random_state : int, default=None
        Controls the randomness of the partitioner and data splitting.
    verbose : int, default=0
        Controls the verbosity of output during fitting and estimation.
        Higher values produce more detailed output.

    Attributes
    ----------
    partitioner : PartitioningEstimate
        The fitted partitioning estimator.

    Examples
    --------
    >>> from glestimation import ResidualEstimator
    >>> estimator = ResidualEstimator(partitioning_estimate="decision_tree")
    >>> estimator.fit(X, y_scores, y_true)
    >>> residuals = estimator.predict(X_new)
    """

    def __init__(
        self,
        estimator: str = "hgb",
        random_state: int = None,
        verbose: int = 0,
    ) -> None:
        self.estimator = HistGradientBoostingRegressor(
            random_state=random_state,
        )
        self.random_state = random_state
        self.verbose = verbose

    def from_name(
        cls,
        name: str,
        verbose: int = 0,
        random_state: int = None,
    ):
        """Load a predefined partitioning estimator from a name.

        Parameters
        ----------
        name : {"decision_tree", "decision_stump", "kmeans", None}
            The predefined estimator to use for partitioning.
        verbose : int, default=0
            Controls verbosity of output.
        random_state : int, default=None
            Controls the randomness of the estimator.

        Returns
        -------
        estimator : object or None
            The configured estimator instance.

        """
        available_names = [
            "hgb",
            "rf",
            "mlp",
            None,
        ]

        if name not in available_names:
            raise ValueError(
                f'Unknown name "{name}". Available names are: {available_names}.'
            )

        if name == "hgb":
            estimator = HistGradientBoostingRegressor(
                random_state=random_state,
            )

        elif name == "rf":
            estimator = RandomForestRegressor(
                n_estimators=100,
                random_state=random_state,
            )

        elif name == "mlp":
            estimator = MLPRegressor(
                hidden_layer_sizes=(100,),
                max_iter=500,
                random_state=random_state,
            )

        elif name is None:
            estimator = None

        return estimator

    def fit(self, X, y_scores, y_true, seed: int = 42):
        """
        Fit the ResidualEstimator with data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted ResidualEstimator.
        """
        if self.verbose > 0:
            print("Starting fit process...")

        y_scores = y_scores.reshape(-1, 1)

        X_train, X_test, y_scores_train, y_scores_test, y_true_train, y_true_test = (
            train_test_split(X, y_scores, y_true, test_size=0.5, random_state=seed)
        )

        X_train, X_cal, y_scores_train, y_scores_cal, y_true_train, y_true_cal = (
            train_test_split(
                X_train, y_scores_train, y_true_train, test_size=0.2, random_state=seed
            )
        )

        if self.verbose > 0:
            print(f"Calibration set size: {len(X_cal)}")
            print(f"Train set size: {len(X_train)}")
            print(f"Test set size: {len(X_test)}")

        self.calibrate(y_scores_cal, y_true_cal)
        self.train(X_train, y_scores_train, y_true_train)
        self.evaluate(X_test, y_scores_test, y_true_test)

        if self.verbose > 0:
            print("Fit process completed.")

        return self

    def calibrate(self, y_scores, y_true):
        """
        Calibrate the predicted scores using logistic regression.
        Parameters
        ----------
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted calibrator.
        """
        if self.verbose > 1:
            print("Calibrating scores...")

        calibrator = LogisticRegression()
        calibrator.fit(y_scores, y_true)
        self.calibrator = calibrator

        if self.verbose > 1:
            print("Calibration completed.")

        return self

    def train(self, X, y_scores, y_true):
        """
        Train the partitioning estimator on residuals.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted partitioning estimator.
        """
        if self.verbose > 1:
            print("Training partitioning estimator...")

        if isinstance(self.estimator, str):
            self.estimator = ResidualEstimator.from_name(self.estimator)

        residuals_train = y_true - self.calibrator.predict(y_scores)
        self.estimator.fit(X, residuals_train)

        if self.verbose > 1:
            print("Training completed.")

        return self

    def evaluate(self, X, y_scores, y_true):
        c_hat = self.calibrator.predict_proba(y_scores)[:, 1]
        self.cal_err = np.mean(np.square(y_scores.flatten() - c_hat))

        self.r_j = self.estimator.predict(X)

        return self

    def predict(self, X):
        return self.estimator.predict(X)


class GLEstimator:
    """Estimate the grouping loss of a fitted probabilistic classifier.

    This class provides methods to estimate the grouping loss (GL) of a probabilistic
    classifier by partitioning the feature space and analyzing calibration residuals
    within each partition.

    Parameters
    ----------
    partitioning_estimate : str or PartitioningEstimate, default="decision_tree"
        The partitioning strategy to use for estimating the grouping loss.
        If string, must be one of {"decision_tree", "decision_stump", "kmeans", None}.
        If PartitioningEstimate instance, uses the provided partitioner.
    train_size : float, default=0.5
        The proportion of the dataset to use for training the partitioner.
        The remaining data is used for evaluation to avoid overfitting.
    random_state : int, default=None
        Controls the randomness of the partitioner and data splitting.
    verbose : int, default=0
        Controls the verbosity of output during fitting and estimation.
        Higher values produce more detailed output.

    Attributes
    ----------
    partitioner : PartitioningEstimate
        The fitted partitioning estimator.
    gl_estimate : float
        The bias-corrected grouping loss estimate.
    gl_uncorrected : float
        The uncorrected grouping loss (without bias correction).
    gl_bias : float
        The estimated bias in the grouping loss.
    gl_j : ndarray
        Per-partition grouping loss values.

    Examples
    --------
    >>> from glestimation import GLEstimator
    >>> estimator = GLEstimator(partitioning_estimate="decision_tree")
    >>> estimator.fit(X, y_scores, y_true)
    >>> estimator.estimate()
    >>> print(estimator.GL())
    """

    def __init__(
        self,
        partitioning_estimate: str | PartitioningEstimate = "decision_tree",
        random_state: int = None,
        verbose: int = 0,
        residual_estimator: ResidualEstimator = None,
    ) -> None:
        self.partitioner = PartitioningEstimate(partitioning_estimate)
        self.random_state = random_state
        self.verbose = verbose
        self.residual_estimator = (
            ResidualEstimator(residual_estimator)
            if residual_estimator is not None
            else None
        )

    def fit(self, X, y_scores, y_true, seed: int = 42):
        """
        Fit the GLEstimator with data.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features.
        y_scores : array-like of shape (n_samples,)
            The predicted probability scores from a classifier.
        y_true : array-like of shape (n_samples,)
            The true binary labels.
        Returns
        -------
        self : object
            Fitted GLEstimator.
        """
        if self.residual_estimator is not None:
            self.residual_estimator.fit(X, y_scores, y_true, seed=seed)

        else:
            self.partitioner.fit(X, y_scores, y_true, seed=seed)
        self.brier = brier_score_loss(y_true, y_scores)
        return self

    def estimate(self):
        """
        Estimate the grouping loss (GL) using the fitted partitioner.
        Returns
        -------
        self : object
            GLEstimator with computed GL estimates.
        """

        if self.residual_estimator is not None:
            r_j = self.residual_estimator.r_j
            gl_uncorrected = np.mean(r_j**2)
            gl_bias = "Non existant due to using a residual estimator"
            gl_estimate = gl_uncorrected
            self.gl_estimate = 2 * gl_estimate
            self.gl_bias = gl_bias
            self.gl_uncorrected = 2 * gl_uncorrected
            self.cal_err = 2 * self.residual_estimator.cal_err
        else:
            r_j = self.partitioner.r_j
            n_j = self.partitioner.n_j
            v_j = self.partitioner.v_j
            N = np.sum(n_j)

            gl_j_uncorrected = r_j**2
            gl_uncorrected = np.sum(n_j * gl_j_uncorrected) / N

            gl_j_bias = np.divide(v_j, n_j, out=np.zeros_like(v_j), where=n_j != 0)
            gl_bias = np.sum(n_j * gl_j_bias) / N

            gl_j = gl_j_uncorrected - gl_j_bias
            gl_estimate = np.sum(n_j * gl_j) / N

            self.gl_j = 2 * gl_j
            self.gl_uncorrected = 2 * gl_uncorrected
            self.gl_bias = 2 * gl_bias
            self.gl_estimate = 2 * gl_estimate
            self.cal_err = 2 * self.partitioner.cal_err

        return self

    def GL(self, psr: str = "brier"):
        return self.gl_estimate

    def GL_uncorrected(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimator is not fitted.")

        return self.gl_uncorrected

    def GL_bias(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError("GLEstimator is not fitted.")

        return self.gl_bias

    def metrics(self, psr: str = "brier"):
        if not self.is_fitted():
            raise ValueError('GLEstimator must be fitted to call "metrics".')

        return {
            "psr": psr,
            "GL": self.gl_estimate,
            "GL_uncorrected": self.gl_uncorrected,
            "GL_bias": self.gl_bias,
            "CL": self.cal_err,
            "EL": self.gl_estimate + self.cal_err,
        }

    def plot(self):
        self.partitioner.plot()
        return self

    def is_fitted(self):
        return (
            hasattr(self, "gl_estimate")
            and hasattr(self, "gl_bias")
            and hasattr(self, "gl_uncorrected")
        )

    def __format__(self, psr: str) -> str:
        """Print the computed metrics."""
        s = "GLEstimator()"

        if self.is_fitted():
            if not psr:
                psr = "Brier"

            metrics = self.metrics(psr)

            extra = (
                f"  Scoring Rule      : {psr}: {self.brier:.4f}\n"
                f"  Grouping loss     : {metrics['GL']:.4f}\n"
                f"  Calibration Loss  : {metrics['CL']:.4f}\n"
                f"  Epistemic Loss    : {metrics['EL']:.4f}\n"
            )
            s = f"{s}\n{extra}"

        return s

    def __str__(self) -> str:
        return f"{self}"

    def __repr__(self) -> str:
        return f"{self}"


class RiskEstimator:
    """
    Estimate the 0-1 risk of a fitted probabilistic classifier.
    This class provides methods to estimate the risk of a probabilistic
    classifier by partitioning the feature space and analyzing calibration
    residuals within each partition.
    Parameters
    ----------
    partitioning_estimate : str or PartitioningEstimate, default="decision_tree"
        The partitioning strategy to use for estimating the risk.
        If string, must be one of {"decision_tree", "decision_stump", "kmeans", None}.
        If PartitioningEstimate instance, uses the provided partitioner.
    train_size : float, default=0.5
        The proportion of the dataset to use for training the partitioner.
        The remaining data is used for evaluation to avoid overfitting.
    random_state : int, default=None
        Controls the randomness of the partitioner and data splitting.
    verbose : int, default=0
        Controls the verbosity of output during fitting and estimation.
        Higher values produce more detailed output.
    Attributes
    ----------
    partitioner : PartitioningEstimate
        The fitted partitioning estimator.
    """

    def __init__(
        self,
        partitioning_estimate: str | PartitioningEstimate = "decision_tree",
        train_size: float = 0.5,
        random_state: int = None,
        verbose: int = 0,
        residual_estimator: ResidualEstimator = None,
        # t: float = 0.5,
    ) -> None:
        self.partitioner = PartitioningEstimate(partitioning_estimate)
        self.train_size = train_size
        self.random_state = random_state
        self.verbose = verbose
        self.residual_estimator = (
            ResidualEstimator(residual_estimator)
            if residual_estimator is not None
            else None
        )
        # self.t = t

    def fit(self, X, y_scores, y_true, seed: int = 42):
        if self.residual_estimator is not None:
            self.residual_estimator.fit(X, y_scores, y_true, seed=seed)
        else:
            self.partitioner.fit(X, y_scores, y_true, seed=seed)
        return self

    def compute_regret(self, C: np.ndarray, t: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Compute Regret estimations.

        Parameters
        ----------
        C : np.ndarray of shape (n,)
            The calibrated scores of each samples.
        t : np.ndarray of shape (k,)
            The thresholds t* derived from the utilities.
        a : np.ndarray of shape (n, k)
            The action taken on each sample.

        Returns
        -------
        RCL : np.ndarray of shape (n, k)
            The regret of the estimated probabilities to the calibrated scores.

        """
        a_star = (C >= t).astype(int)  # (n,)
        R = np.zeros(C.shape[0])  # (n,)
        idx_disagreement = a.flatten() != a_star  # (n,)
        R[idx_disagreement] = np.abs(C - t)[idx_disagreement]

        return R  # (n,)

    def predict(self, X, y_scores, t):
        if self.residual_estimator is not None:
            r_hat = self.residual_estimator.predict(X)
            c_hat = self.residual_estimator.calibrator.predict(y_scores)
        else:
            r_hat = self.partitioner.predict(X)
            c_hat = self.partitioner.calibrator.predict(y_scores)
        a = (y_scores >= t).astype(int)
        RCL = self.compute_regret(c_hat, t, a)

        REL = self.compute_regret(c_hat + r_hat, t, a)
        return RCL, REL

    def predict_total(self, X, y_scores, t):
        RCL, REL = self.predict(X, y_scores, t)

        return RCL.mean(), REL.mean()

    def plot(self):
        self.partitioner.plot()
        return self
