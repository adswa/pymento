import logging
import numpy as np
# we need the imbalanced learn pipeline to apply preprocessing steps
# to both X *and* y.
from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
# noinspection PyUnresolvedReferences
from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore,
    Vectorizer,
)


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
