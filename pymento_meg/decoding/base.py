import logging
import numpy as np
# we need the imbalanced learn pipeline to apply preprocessing steps
# to both X *and* y.
from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.utils.validation import check_is_fitted

from sklearn.model_selection import StratifiedKFold
# noinspection PyUnresolvedReferences
from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore,
    Vectorizer,
)

from pymento_meg.srm.srm import shared_response


class MyOwnSlidingEstimator(SlidingEstimator):
    """
    Partial custom override of mne's SlidingEstimator. Reimplemented in order to
    forgo dimensionality checks.
    reshape is Reshaper.thicken
    """
    def __init__(self,
                 reshape,
                 base_estimator,
                 scoring=None,
                 n_jobs=None,
                 *,
                 verbose=None):
        super().__init__(
            base_estimator,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self._reshape = reshape

    def fit_transform(self, X, y, **fit_params):
        return super().fit_transform(
            self._reshape(X),
            y,
            **fit_params,
        )

    def fit(self, X, y, **fit_params):
        return super().fit(
            self._reshape(X),
            y,
            **fit_params,
        )

    def _transform(self, X, method):
        return super()._transform(
            self._reshape(X),
            method,
        )


    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['reshape'] = self._reshape
        return params


def confusion_magnitude(est, X, y_true, **kwargs):
    """Custom scorer to be able to compute confusion matrix on predictions
    during cross_val_multiscore. Uses the Magnitude value labels"""
    from sklearn.metrics import confusion_matrix
    y_pred = est.predict(X)
    # compute and return confusion matrices for each time point.
    return np.apply_along_axis(lambda p, t: confusion_matrix(t, p, labels=['M0.5', 'M1.0', 'M2.0', 'M4.0']), 0, y_pred, y_true)


def confusion_probability(est, X, y_true, **kwargs):
    """Custom scorer to be able to compute confusion matrix on predictions
    during cross_val_multiscore. Uses the Probability value labels"""
    from sklearn.metrics import confusion_matrix
    y_pred = est.predict(X)
    # compute and return confusion matrices for each time point.
    return np.apply_along_axis(lambda p, t: confusion_matrix(t, p, labels=['P0.1', 'P0.2', 'P0.4', 'P0.8']), 0, y_pred, y_true)


def confusion_expectedvalue(est, X, y_true, **kwargs):
    """Custom scorer to be able to compute confusion matrix on predictions
    during cross_val_multiscore. Uses the Expected Value value labels"""
    from sklearn.metrics import confusion_matrix
    y_pred = est.predict(X)
    # compute and return confusion matrices for each time point.
    return np.apply_along_axis(lambda p, t: confusion_matrix(t, p, labels=['EV0.2', 'EV0.4', 'EV0.8']), 0, y_pred, y_true)


def decode(X,
           y,
           n_splits=10,
           estimator=LogisticRegression(solver='liblinear'),
           metric='accuracy',
           n_jobs=None
           ):
    """
    Fit an estimator of a given type to every time point in a
    epochs x channel x time matrix using MNE's SlidingEstimator in a stratified
    cross validation. The metric parameter allows to specify a scorer during the
    cross_val_multiscore step. Three custom metrics,
    confusion_{magnitude,probability,expectedvalue} allow to return confusion
    matrices per time point. The custom scoring metrics require that the labels
    y correspond to the form <letter-value> e.g., 'M0.5' for a trial with
    stimulus Magnitude of 0.5, or 'P0.8' for a trial with stimulus probability
    0.8, or 'EV0.2' for a trial with expected value of 0.2.

    Example invocation: decode(X, mags, metric=confusion_magnitude)
    :param X: array; dimensionality epochs X channels X time
    :param y: array; targets
    :param n_splits: int, number of cross-validation folds
    :param estimator: sklearn estimator
    :param metric: callable, either from sklearn or one of three custom
     implementations confusion_{magnitude,probability,expectedvalue}
    :param n_jobs: int or None; determines parallelization
    :return:
    """
    logging.info(f'Fitting {estimator} in a stratified cross-validation with'
                 f' {n_splits} splits using {str(metric)} as the final scoring.'
                 )
    # TODO: pass random states?
    clf = make_pipeline(
        StandardScaler(),
        estimator
    )
    time_decod = SlidingEstimator(
        clf, n_jobs=n_jobs, scoring='accuracy', verbose=True)
    cv = StratifiedKFold(n_splits=n_splits)
    scores = cross_val_multiscore(time_decod,
                                  X,
                                  y,
                                  cv=cv,
                                  scoring=metric,
                                  n_jobs=n_jobs)
    return scores

class Reshaper:
    """Helper class to reshape to and from 3D/2D data"""
    def __init__(self, k=None):
        self._sampleshape = None

    def flatten(self, X):
        self._sampleshape = X.shape[1:]
        return np.reshape(X, (X.shape[0], -1))

    def thicken(self, X):
        print(f"thickening X from {X.shape} to {(X.shape[0],)+self._sampleshape}")
        return np.reshape(X, (X.shape[0],)+self._sampleshape)

    def thickenSRM(self, X):
        print(f"thickening X from {X.shape} to {(X.shape[0], self._k, -1)}")
        return np.reshape(X, (X.shape[0], self._k, -1))


def trialaveraging(X, y, ntrials=4, nsamples=100):
    """Average N=ntrials trials together, and repeat this until we
    have generated N=nsamples average trials. This function will
    become an imblearn FunctionSampler."""
    X_ = np.empty((nsamples * len(np.unique(y)),)+X.shape[1:])
    y_ = []
    sample = 0
    for unique_target in np.unique(y):
        # average trials in batches of the specified foldsize per unique target
        trial_ids = np.where(y == unique_target)[0]
        # draw n = ntrials random trials and average the trials
        for b in range(nsamples):
            X_[sample] = np.mean(
                X[np.random.choice(trial_ids, ntrials)],
                axis=0,
            )
            sample += 1
            # create the appropriate amount of targets
            y_.append(unique_target)
    y_ = np.array(y_)
    assert len(X_) == len(y_)
    return X_, y_



class SRMTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, k, nsamples=10):
        self.k = k
        self.nsamples = nsamples

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        X_ = np.reshape(X, (X.shape[0], 306, -1))
        # TODO: Where to subselect time points?
        targets = np.unique(y)
        # generate nsamples samples for the shared response model
        samples = []
        for i in range(self.nsamples):
            # generate virtual subjects from concatenating data from each target
            sample_ids = [np.random.choice(np.where(y==target)[0], 1)
                          for target in targets]
            samples.append(np.concatenate([np.squeeze(X_[i,:,0:700]) for i in sample_ids], axis=1))
        # fit an SRM model on the samples
        srm = shared_response(samples,
                              features=self.k)
        # average "subject" basis
        avg_basis = np.mean(srm.w_, axis=0)
        self.basis = avg_basis
        print(self.basis.shape)
        return self

    def transform(self, X, y=None):
        X_ = np.reshape(X, (X.shape[0], 306, -1))
        print('X_ shape is: ', X_.shape)
        check_is_fitted(self, 'basis')
        transformed = np.stack([np.dot(self.basis.T, x) for x in X_])
        print('within SRM transform: from ', transformed.shape)
        return np.reshape(transformed, (transformed.shape[0], -1))
