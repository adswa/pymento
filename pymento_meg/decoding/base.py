import logging
import numpy as np

from functools import partial
# we need the imbalanced learn pipeline to apply preprocessing steps
# to both X *and* y.
from imblearn import FunctionSampler
from imblearn.pipeline import make_pipeline

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
        # X is trials x sensors * time. Reshape turns it into trials x sensors x time
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


def confusion_choice(est, X, y_true, **kwargs):
    """Custom scorer to be able to compute confusion matrix on predictions
    during cross_val_multiscore. Uses the Choice value labels"""
    from sklearn.metrics import confusion_matrix
    y_pred = est.predict(X)
    # compute and return confusion matrices for each time point.
    return np.apply_along_axis(lambda p, t: confusion_matrix(t, p, labels=['choice1.0', 'choice2.0']), 0, y_pred, y_true)


def sliding_averager(X, size):
    """Custom sliding window function that averages a given amount of samples"""
    logging.info('Starting averager sliding')
    ntrials, nsensors, nts = X.shape
    # the output has the same number of trials and sensors, but is shorter by
    # the length of one sliding window
    out = np.empty((ntrials, nsensors, nts-size))
    for t in range(out.shape[-1]):
        out[:, :, t] = np.mean(X[:, :, t:t + size], axis=2)
    logging.info(f"Dimensionality after average sliding is {out.shape}")
    return out


def spatiotemporal_slider(X, size):
    """Custom sliding window function that does spatio-temporal integration over
    a given amount of samples"""
    logging.info('Starting spatiotemporal sliding')
    ntrials, nsensors, nts = X.shape
    # the output is one sliding window shorter in the sample dimension, and has
    # a multitude more sensors as we append other samples sensors into a single
    # sample
    out = np.empty((ntrials, nsensors * size, nts - size))
    for t in range(out.shape[-1]):
        out[:, :, t] = X[:, :, t:t + size].reshape(ntrials, -1)
    logging.info(f"Dimensionality after spatiotemporal sliding is {out.shape}")
    return out


def decode(X,
           y,
           n_splits=10,
           estimator=LogisticRegression(solver='liblinear'),
           metric='accuracy',
           n_jobs=None,
           dimreduction=None,
           trainrange=None,
           srmsamples=None,
           k=None,
           ntrials=4,
           nsamples=100,
           slidingwindow=10,
           slidingwindowtype=spatiotemporal_slider,
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
    :param dimreduction: None or string identifying a transformer;
     currently supports 'srm', 'pca', 'spectralsrm'. Will be added
     to pipeline for dimensionality reduction. Requires parameter k
    :param k: None or int; dimensions to reduce to/features to select
    :param trainrange: None or list of int; if not None its a range of
     samples to subset srm training data to (e.g., [0, 70])
    :param srmsamples: None or int; how many virtual subjects shall be created
    within the shared response modelling. If None, as many subjects as
    bootstrapped trials will be used.
    :param ntrials: int; how many trials of the same type to average together
    :param nsamples: int; how many bootstrapping draws during trial averaging
    :param slidingwindow: int or None; size of a sliding window in samples
    :return:
    """
    logging.info(f'Initializing TrialAverager with {ntrials} trials per '
                 f'average and N={nsamples} bootstrapping draws.')
    trialaverager = FunctionSampler(
        func=trialaveraging,
        kw_args=dict(ntrials=ntrials, nsamples=nsamples),
    )
    logging.info(f'Setting up a Stratified KFold Crossvalidation with '
                 f'{n_splits} splits')
    cv = StratifiedKFold(n_splits=n_splits)
    if dimreduction is not None:
        assert k is not None
        logging.info(
            f'Fitting a {estimator} using {str(metric)} as the final scoring '
            f'and {dimreduction} for dimensionality reduction.')
        reshaper = Reshaper(k=k)
        # if we use a sliding window, supply the necessary parameters
        reshaperfx = reshaper.thickentok if slidingwindowtype is None \
            else partial(reshaper.slide,
                         thickenfx=reshaper.thickentok,
                         size=slidingwindow,
                         slidefx=slidingwindowtype,
                         )
        # both PCA and SRM get the same sliding estimator
        slidingestimator = MyOwnSlidingEstimator(
            reshaperfx,
            estimator,
            n_jobs=n_jobs,
            scoring='accuracy',
            verbose=True)
        if dimreduction in ['srm', 'spectralsrm']:
            # determine how many virtual subjects are generated internally
            assert srmsamples is not None
            srmtransformer = SRMTransformer if dimreduction == 'srm' else SpectralSRMTransformer
            # no scaler prior SRM, we need the time signature, and SRM does
            # demeaning itself. We call the StandardScaler() afterwards to
            # harmonize sensors prior to Logistic Regression
            outer_pipeline = make_pipeline(
                trialaverager,
                srmtransformer(k=k,
                               subjects=srmsamples,
                               trainrange=trainrange,
                               ),
                StandardScaler(),
                slidingestimator,
            )
        elif dimreduction == 'pca':
            outer_pipeline = make_pipeline(
                trialaverager,
                StandardScaler(),
                SpatialPCATransformer(k=k, reshaper=reshaper,
                                      trainrange=trainrange),
                slidingestimator,
            )
    else:
        logging.info(
            f'Fitting {estimator} in a stratified cross-validation with'
            f' {n_splits} splits using {str(metric)} as the final scoring.'
            )
        reshaper = Reshaper()
        reshaperfx = reshaper.thicken if slidingwindowtype is None else partial(
            reshaper.slide,
            thickenfx=reshaper.thicken,
            size=slidingwindow,
            slidefx=slidingwindowtype,
        )

        slidingestimator = MyOwnSlidingEstimator(
            reshaperfx,
            estimator,
            n_jobs=n_jobs,
            scoring='accuracy',
            verbose=True)

        outer_pipeline = make_pipeline(
            trialaverager,
            StandardScaler(),
            slidingestimator,
        )
    scores = cross_val_multiscore(outer_pipeline,
                                  reshaper.flatten(X),
                                  y,
                                  cv=cv,
                                  scoring=metric,
                                  n_jobs=n_jobs
                                  )
    return scores


class Reshaper:
    """Helper class to reshape to and from 3D/2D data"""
    def __init__(self, k=None):
        self._sampleshape = None
        self._k = k

    def flatten(self, X):
        print(f"flattening X from {X.shape} to {(X.shape[0], -1)}")
        self._sampleshape = X.shape[1:]
        return np.reshape(X, (X.shape[0], -1))

    def thicken(self, X):
        print(f"thickening X from {X.shape} to {(X.shape[0],)+self._sampleshape}")
        return np.reshape(X, (X.shape[0],)+self._sampleshape)

    def thickentok(self, X):
        """Thicken X back according to the dimensions the dim reduction reduced
         to"""
        print(f"thickening X from {X.shape} to {(X.shape[0], self._k, -1)}")
        return np.reshape(X, (X.shape[0], self._k, -1))

    def thickentotime(self, X, newtime):
        """Alternative to thicken that allows to reshape back to three,
         dimensions, but with a specified time dimension instead of a prior
         extracted one. This allows time subsetting to train ranges."""
        print(f"thickening X with new time from {X.shape} to "
              f"{(X.shape[0], self._sampleshape[0], newtime)}")
        return np.reshape(X, (X.shape[0], self._sampleshape[0], newtime))

    def slide(self, X, thickenfx=None, size=10, slidefx=spatiotemporal_slider):
        """This reshaper implements a sliding window across a range of samples
        specified by the size parameter. A custom sliding function can be passed
        to determine the sliding behavior. By default, the sliding results in a
        spatio-temporal integration: All samples in the sliding window are
        concatenated such that the one resulting sample per sliding window
        includes the spatia-temporal configuration of all sensors in the window.
        """
        logging.info(f'Setting up a sliding window of {size} samples')
        # First, reshape to trials x sensors x time
        X_ = thickenfx(X)
        out = slidefx(X_, size)
        return out


def trialaveraging(X, y, ntrials=4, nsamples='max'):
    """Average N=ntrials trials together, and repeat this until we
    have generated N=nsamples average trials. This function will
    become an imblearn FunctionSampler.
    :param nsamples: int or str; switch to determine bootstrapping behavior.
    Can be 'min' (will bootstrap the minimum samples per target), 'max' (will
    bootstrap the maximum samples per target) or any integer.
    TODO: maybe make also 'stratified' option
    """
    unique, counts = np.unique(y, return_counts=True)
    logging.info(f'nsamples is set to {nsamples}, ')
    nsamples = np.max(counts) if nsamples == 'max' \
        else np.min(counts) if nsamples == 'min' \
        else nsamples
    logging.info(f'using {nsamples} for bootstrapping in this split')
    X_ = np.empty((nsamples * len(np.unique(y)),)+X.shape[1:])
    y_ = []
    sample = 0
    for unique_target in unique:
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


class SpatialPCATransformer(BaseEstimator, TransformerMixin):

    def __init__(self, k, reshaper, trainrange=None):
        from sklearn.decomposition import PCA
        self.k = k
        self.pca = PCA(n_components=k)
        self.reshaper = reshaper
        if trainrange is not None:
            # trainrange needs to be a set or list with a start and end value
            assert len(trainrange) == 2, 'the trainrange needs to have 2 values'
            assert trainrange[0] < trainrange[1], \
                'start value must be smaller than end value'
            # check that the train range is not negative
            assert all(self.trainrange) > 0, 'train range cannot be negative!'
        self.trainrange = trainrange
        self.newtime = None

    def _dimensionalityvodoo(self, X):
        # restore to trials x sensors x time
        if self.newtime is not None:
            X_ = self.reshaper.thickentotime(X, self.newtime)
            # reset newtime internally
            self.newtime = None
        else:
            X_ = self.reshaper.thicken(X)
        # now bend dimensions: aim is spatial PCA, thus the time dimension is at
        # the wrong place. We need to unwind the time dimension along the trial
        # dimension. **vodoooo**
        X_ = np.rollaxis(X_, 2, 1)
        # trial*time x sensors
        X_ = X_.reshape(np.prod(X_.shape[:2]), -1)
        return X_

    def fit(self, X, y):
        # subset training data to the specified trainrange
        if self.trainrange is not None:
            logging.info(f'subsetting PCA training data into train range '
                         f'{self.trainrange}.')
            # first, turn X back into trials x sensors x time
            X_ = self.reshaper.thicken(X)
            # check that the selected range isn't larger than the available time
            assert self.trainrange[1] <= X_.shape[-1], \
                'train range is larger than available data range!'
            X_ = X_[:, :, self.trainrange[0]:self.trainrange[1]]
            # flatten, but hold on to the new time dimension
            self.newtime = X_.shape[-1]
            X_ = np.reshape(X_, (X_.shape[0], -1))
            self.pca.fit(self._dimensionalityvodoo(X_))
        else:
            self.pca.fit(self._dimensionalityvodoo(X))
        return self

    def transform(self, X):
        # X is trials x sensors*time

        # the input of the PCA X_ is trial*time x components
        X_ = self.pca.transform(self._dimensionalityvodoo(X))
        # restore time axis: trials x times x components
        X_ = X_.reshape(X.shape[0], -1, self.k)
        # restore correct position of time axis: trials x components x times
        X_ = np.rollaxis(X_, 1, 3)
        # let it goooo
        return X_


class SRMTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, k, subjects, trainrange, spectral):
        self.k = k
        self.subjects = subjects
        if trainrange is not None:
            # trainrange needs to be a set or list with a start and end value
            assert len(trainrange) == 2, 'the trainrange needs to have 2 values'
            assert trainrange[0] < trainrange[1], \
                'start value must be smaller than end value'
        self.trainrange = trainrange

    def fit(self, X, y):
        targets, counts = np.unique(y, return_counts=True)
        logging.info(f'Preparing to draw {self.subjects} virtual subjects...')
        # set up the time subselection for the training data
        if self.trainrange is not None:
            # check that the selected range isn't larger than the available time
            assert self.trainrange[1] <= X_.shape[-1], \
                'train range is larger than available data range!'
            # check that the train range is not negative
            assert all(self.trainrange) > 0, 'train range cannot be negative!'
        # generate virtual subjects for the shared response model
        samples = []
        for subject in range(self.subjects):
            # generate virtual subjects from concatenating data from each target
            # first, get trial ids of one trial per unique target value
            sample_ids = [np.random.choice(np.where(y == target)[0], 1)
                          for target in targets]
            # then, select each of those trials, potentially subselecting the
            # number of time points to a specified range
            start = 0
            end = X_.shape[-1]
            if self.trainrange is not None:
                # subset the time series (train only on specific part of trial)
                start = self.trainrange[0]
                end = self.trainrange[1]
            samples.append(
                np.concatenate(
                    # the first dimension is 1, squeeze it away
                    [self._preprocess(
                        np.squeeze(X_[i, :, start:end]))
                        for i in sample_ids],
                    axis=1
                )
            )

        # fit an SRM model on the samples
        srm = shared_response(samples,
                              features=self.k)
        # average "subject" basis
        avg_basis = np.mean(srm.w_, axis=0)
        self.basis = avg_basis
        print(self.basis.shape)
        return self

    def _preprocess(self, data):
        return data

    def transform(self, X, y=None):
        X_ = np.reshape(X, (X.shape[0], 306, -1))
        logging.info(f'X_ shape is: {X_.shape}')
        check_is_fitted(self, 'basis')
        transformed = np.stack([np.dot(self.basis.T, x) for x in X_])
        logging.info(f'within SRM transform: from {transformed.shape}')
        return np.reshape(transformed, (transformed.shape[0], -1))


class SpectralSRMTransformer(SRMTransformer):

    def _preprocess(self, data):
        """Apply a spectral transformation to a virtual SRM subject."""
        from pymento_meg.srm.simulate import transform_to_power
        return transform_to_power(data)
