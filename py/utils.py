import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import mean_poisson_deviance


def get_coefs(model):
    """Extract coefficients from linear model."""

    return pd.DataFrame(
        model.coef_, columns=["Estimates"], index=model.feature_names_in_
    )


def poisson_scorer(names, models, X, y, w, reference_model):
    """Calculates mean Poisson deviance and corresponding pseudo R-squared."""

    perf = dict()
    deviance_0 = mean_poisson_deviance(y, reference_model.predict(X), sample_weight=w)
    for name, model in zip(names, models):
        deviance = mean_poisson_deviance(y, model.predict(X), sample_weight=w)
        perf[name] = (deviance, (deviance_0 - deviance) / deviance_0)
    perf_df = pd.DataFrame.from_dict(
        perf, orient="index", columns=("mean_deviance", "Pseudo_R2")
    )
    return perf_df


def plot_scores(scores, title=None):
    """Plots performance scores as calculated e.g. from poisson_scorer()."""

    fig, axes = plt.subplots(1, scores.shape[1], figsize=(6, 3))
    for measure, ax in zip(scores.columns, axes):
        scores[[measure]].plot.bar(ax=ax).legend(loc="lower center")
    if title:
        plt.suptitle(title, fontsize=15)
    fig.tight_layout()


class LogRegressor(RegressorMixin):
    """
    A wrapper class for a Scikit-Learn regressor that evaluates predictions on a log scale.

    Parameters
    ----------
    regressor : object
        A Scikit-Learn regressor object that has already been fit to data.

    Methods
    -------
    predict(X)
        Make predictions for the given input data X.

    fit(*args, **kwargs)
        Not used.
    """

    def __init__(self, estimator):
        self._estimator = estimator
        check_is_fitted(self._estimator)
        self.is_fitted_ = True

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return np.log(self._estimator.predict(X))


"""Friedman and Popescu's H-Statistic"""

import itertools

import numpy as np
from sklearn.base import is_classifier, is_regressor
from sklearn.utils import Bunch, _get_column_indices, _safe_assign, _safe_indexing
from sklearn.utils.random import sample_without_replacement
from sklearn.utils.validation import _check_sample_weight, check_is_fitted


def _calculate_pd_brute_fast(estimator, X, feature_indices, grid, sample_weight=None):
    """Fast version of _calculate_partial_dependence_brute()

    Returns np.array of size (n_grid, ) or (n_ngrid, output_dimension).
    """

    if is_regressor(estimator):
        pred_fun = estimator.predict
    elif is_classifier(estimator) and hasattr(estimator, "predict_proba"):
        pred_fun = estimator.predict_proba
    else:
        raise ValueError("The estimator has no predict or predict_proba method.")

    # X is stacked n_grid times, and grid columns are replaced by replicated grid
    n = X.shape[0]
    n_grid = grid.shape[0]
    X_eval = X.copy()

    X_stacked = _safe_indexing(X_eval, np.tile(np.arange(n), n_grid), axis=0)
    grid_stacked = _safe_indexing(grid, np.repeat(np.arange(n_grid), n), axis=0)
    _safe_assign(X_stacked, values=grid_stacked, column_indexer=feature_indices)

    # Predict on stacked data. Pick positive class probs for binary classification
    preds = pred_fun(X_stacked)
    if is_classifier(estimator) and preds.shape[1] == 2:
        preds = preds[:, 1]

    # Partial dependences are averages per grid block
    pd_values = [
        np.average(Z, axis=0, weights=sample_weight) for Z in np.split(preds, n_grid)
    ]

    return np.array(pd_values)


def _calculate_pd_over_data(estimator, X, feature_indices, sample_weight=None):
    """Calculates centered partial dependence over the data distribution.

    It returns a numpy array of size (n, ) or (n, output_dimension).
    """

    # Select grid columns and remove duplicates (will compensate below)
    grid = _safe_indexing(X, feature_indices, axis=1)

    # np.unique() fails for mixed type and sparse objects
    try:
        ax = 0 if grid.shape[1] > 1 else None  # np.unique works better in 1 dim
        _, ix, ix_reconstruct = np.unique(
            grid, return_index=True, return_inverse=True, axis=ax
        )
        grid = _safe_indexing(grid, ix, axis=0)
        compressed_grid = True
    except (TypeError, np.AxisError):
        compressed_grid = False

    pd_values = _calculate_pd_brute_fast(
        estimator,
        X=X,
        feature_indices=feature_indices,
        grid=grid,
        sample_weight=sample_weight,
    )

    if compressed_grid:
        pd_values = pd_values[ix_reconstruct]

    # H-statistics are based on *centered* partial dependences
    column_means = np.average(pd_values, axis=0, weights=sample_weight)

    return pd_values - column_means


def h_statistic(
    estimator,
    X,
    *,
    features=None,
    n_max=500,
    random_state=None,
    sample_weight=None,
    eps=1e-10,
):
    """Friedman and Popescu's H-statistic of pairwise interaction strength.

    Calculates Friedman and Popescu's H-statistic of interaction strength
    for each feature pair j, k, see [FRI]_. The statistic is defined as::

        H_jk^2 = Numerator_jk / Denominator_jk, where

        - Numerator_jk = 1/n * sum(PD_{jk}(x_ij, x_ik) - PD_j(x_ij) - PD_k(x_ik)^2,
        - Denominator_jk = 1/n * sum(PD_{jk}(x_ij, x_ik)^2),
        - PD_j and PD_jk are the one- and two-dimensional partial dependence
          functions centered to mean 0,
        - and the sums run over 1 <= i <= n, where n is the sample size.

    It equals the proportion of effect variability between two features that cannot
    be explained by their main effects. When there is no interaction, the value is
    precisely 0. The numerator (or its square root) provides an absolute measure
    of interaction strength, enabling direct comparison across feature pairs.

    The computational complexity of the function is :math:`O(p^2 n^2)`,
    where :math:`p` denotes the number of features considered. The size of `n` is
    automatically controlled via `n_max=500`, while it is the user's responsibility
    to select only a subset of *important* features. It is crucial to focus on important
    features because for weak predictors, the denominator might be small, and
    even a weak interaction could result in a high Friedman's H, sometimes exceeding 1.

    Parameters
    ----------
    estimator : object
        An estimator that has already been :term:`fitted`.

    X : ndarray or DataFrame, shape (n_observations, n_features)
        Data for which :term:`estimator` is able to calculate predictions.

    features : array-like of {int, str}, default=None
        List of feature names or column indices used to calculate pairwise statistics.
        The default, None, will use all column indices of X.

    n_max : int, default=500
        The number of rows to draw without replacement from X (and `sample_weight`).

    random_state : int, RandomState instance, default=None
        Pseudo-random number generator used for subsampling via `n_max`.
        See :term:`Glossary <random_state>`.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights used in calculating partial dependencies.

    eps : float, default=1e-10
        Threshold below which numerator values are set to 0.

    Returns
    -------
    result : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        feature_pair : list of length n_feature_pairs
            The list contains tuples of feature pairs (indices) in the same order
            as all pairwise statistics.

        h_squared_pairwise : ndarray of shape (n_pairs, ) or (n_pairs, output_dim)
            Pairwise H-squared statistic. Useful to see which feature pair has
            strongest relative interation (relative with respect to joint effect).
            Calculated as numerator_pairwise / denominator_pairwise.

        numerator_pairwise : ndarray of shape (n_pairs, ) or (n_pairs, output_dim)
            Numerator of pairwise H-squared statistic.
            Useful to see which feature pair has strongest absolute interaction.
            Take square-root to get values on the scale of the predictions.

        denominator_pairwise : ndarray of shape (n_pairs, ) or (n_pairs, output_dim)
            Denominator of pairwise H-squared statistic (not of particular interest).

    References
    ----------
    .. [FRI] :doi:`J. H. Friedman and B. E. Popescu,
            "Predictive Learning via Rule Ensembles",
            The Annals of Applied Statistics, 2(3), 916-954,
            2008. <10.1214/07-AOAS148>`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import HistGradientBoostingRegressor
    >>> from sklearn.inspection import permutation_importance, h_statistic
    >>> from sklearn.datasets import load_diabetes
    >>>
    >>> X, y = load_diabetes(return_X_y=True)
    >>> est = HistGradientBoostingRegressor(max_iter=5, max_depth=2).fit(X, y)
    >>>
    >>> # Get Friedman's H-squared for top m=3 predictors
    >>> m = 3
    >>> imp = permutation_importance(est, X, y, random_state=0)
    >>> top_m = np.argsort(imp.importances_mean)[-m:]
    >>> h_statistic(est, X=X, features=top_m, random_state=4)

    >>> # For features (8, 2), 3.4% of the joint effect variability comes from
    >>> # their interaction. These two features also have strongest absolute
    >>> # interaction, see "numerator_pairwise":
    >>> # {'feature_pair': [(3, 8), (3, 2), (8, 2)],
    >>> # 'h_squared_pairwise': array([0.00985985, 0.00927104, 0.03439926]),
    >>> # 'numerator_pairwise': array([ 1.2955532 ,  1.2419687 , 11.13358385]),
    >>> # 'denominator_pairwise': array([131.39690331, 133.96210997, 323.6576595 ])}

    """
    check_is_fitted(estimator)

    if sample_weight is not None:
        sample_weight = _check_sample_weight(sample_weight, X)

    # Usually, the data is too large and we need subsampling
    if X.shape[0] > n_max:
        row_indices = sample_without_replacement(
            n_population=X.shape[0], n_samples=n_max, random_state=random_state
        )
        X = _safe_indexing(X, row_indices, axis=0)
        if sample_weight is not None:
            sample_weight = _safe_indexing(sample_weight, row_indices, axis=0)
    else:
        X = X.copy()

    if features is None:
        features = feature_indices = np.arange(X.shape[1])
    else:
        feature_indices = np.asarray(
            _get_column_indices(X, features), dtype=np.intp, order="C"
        ).ravel()

    # CALCULATIONS
    pd_univariate = []
    for ind in feature_indices:
        pd_univariate.append(
            _calculate_pd_over_data(
                estimator, X=X, feature_indices=[ind], sample_weight=sample_weight
            )
        )

    num = []
    denom = []

    for j, k in itertools.combinations(range(len(feature_indices)), 2):
        pd_bivariate = _calculate_pd_over_data(
            estimator,
            X=X,
            feature_indices=feature_indices[[j, k]],
            sample_weight=sample_weight,
        )
        num.append(
            np.average(
                (pd_bivariate - pd_univariate[j] - pd_univariate[k]) ** 2,
                axis=0,
                weights=sample_weight,
            )
        )
        denom.append(np.average(pd_bivariate**2, axis=0, weights=sample_weight))

    num = np.array(num)
    num[np.abs(num) < eps] = 0  # Round small numerators to 0
    denom = np.array(denom)
    h2_stat = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)

    return Bunch(
        feature_pair=list(itertools.combinations(features, 2)),
        h_squared_pairwise=h2_stat,
        numerator_pairwise=num,
        denominator_pairwise=denom,
    )


class KerasRegressor(RegressorMixin):
    """
    A wrapper class for a keras model.

    Parameters
    ----------
    regressor : object
        A keras regressor object that has already been fit to data.

    Methods
    -------
    predict(X)
        Make predictions for the given input data X.

    fit(*args, **kwargs)
        Not used.
    """

    def __init__(self, estimator):
        self._estimator = estimator
        self.is_fitted_ = True

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X):
        return self._estimator.predict(X, verbose=0, batch_size=20_000).flatten()


class ColumnSplitter(BaseEstimator, TransformerMixin):
    """
    Transformer that splits a pandas.Dataframe into a dict of numpy arrays.

    Parameters
    ----------
    feature_dict : dictionary
        The keys define the keys of the dict holding the dataframe pieces, and
        the values the corresponding feature column names.

    Methods
    -------
    transform(X)
        Splits input dataframe X into dict of numpy arrays defined by `feature_dict`.

    fit(*args, **kwargs)
        Not used.
    """

    def __init__(self, feature_dict):
        self._feature_dict = feature_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        out = dict()
        for key, value in self._feature_dict.items():
            out[key] = X[value].to_numpy()
        return out
