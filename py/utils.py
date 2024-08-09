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
