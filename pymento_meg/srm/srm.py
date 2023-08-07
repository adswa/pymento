"""
Module for fitting shared response models.

"""

import mne
import logging
import random

import numpy as np
import pandas as pd

from brainiak.funcalign import srm
from pathlib import Path
from scipy import stats

from pymento_meg.config import trial_characteristics
from pymento_meg.orig.behavior import read_bids_logfile
from pymento_meg.proc.epoch import get_stimulus_characteristics
from pymento_meg.srm.simulate import (
    transform_to_power,
)

from pymento_meg.srm.utils import (
    _create_splits_from_left_and_right_stimulation,
    _find_data_of_choice
)
from pymento_meg.srm.viz import (
    _plot_transformed_components,
    plot_many_distance_matrices,
    plot_distance_matrix,
    plot_srm_model,
)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
# set a seed to make train and test splits deterministic
random.seed(423)


def srm_with_spectral_transformation(subject=None,
                                     datadir=None,
                                     bidsdir=None,
                                     timespan=None,
                                     k=10,
                                     ntrain=240,
                                     ntest=200,
                                     modelfit='epochwise',
                                     figdir='/data/project/brainpeach/memento-bids-figs/',
                                     custom_name_component=''):
    """
    Fit a shared response model on data transformed into frequency space.
    This is the practical implementation of the work simulated in
    pymento_meg.srm.simulate.

    :param subject: list, all subjects to include
    :param ntrain: int, number of epochs to use in training
    :param ntest: int, number of epochs to use in testing
    :param datadir: str, path to directory with epoched data
    :param bidsdir: str, path to directory with bids data
    :param timespan: str or list, an identifier of the time span of data to be
    used for model fitting. If the epochs span the entire trial, it can be one
    of 'decision' (time locked around the decision in each trial),
    'firststim' (time locked to the first stimulus, for the
    stimulus duration), or 'fulltrial' (entire 7second epoch), 'secondstim', or
    'delay'. Alternatively, specify a list corresponding to a time frame within
    the experiment (in seconds)
    :param k: int, number of components used in SRM fitting
    :param modelfit: str, either 'epochwise', 'subjectwise', or 'trialorder'.
    :param custom_name_component: str, added to plot file name if given
     Determines
     whether SRM is fit on all epochs as if they were subjects, subjectwise
     on averaged epochs, or subject-wise on artificial time series

    :return:
    """

    # set a few defaults if unset
    if subject is None:
        subject = ['001', '002', '003', '004', '005', '006', '007', '008',
                   '009', '010', '011', '012', '013', '014', '015', '016',
                   '017', '018', '019', '020', '021', '022']
    if datadir is None:
        datadir = '/data/project/brainpeach/memento-sss'
    if bidsdir is None:
        bidsdir = '/data/project/brainpeach/memento-bids'
    if modelfit is None:
        modelfit = 'trailtype'
    if timespan is None:
        timespan = [0, 2.7]
    # TODO: distinguish trials with positive and negative feedback
    fullsample, data = get_general_data_structure(subject=subject,
                                                  datadir=datadir,
                                                  bidsdir=bidsdir,
                                                  condition='nobrain-brain',
                                                  timespan=timespan)

    # TODO: turn this into some sort of cross-validation?
    trainset, testset = train_test_set(fullsample,
                                       data,
                                       ntrain=ntrain,
                                       ntest=ntest,
                                       modelfit=modelfit)
    if modelfit == 'epochs':
        # transform  epochs to a time resolved and a spectral space. In order to
        # get components reflecting processes within 3 second epochs, loosen the
        # subject definition, and regard each individual epoch as a subject
        train_spectral, train_series = epochs_to_spectral_space(trainset)
        test_spectral, test_series = epochs_to_spectral_space(testset)
    elif modelfit == 'subjectwise':
        # Alternative: average epochs within subjects (in spectral space)
        train_spectral, train_series = \
            epochs_to_spectral_space(trainset, subjectwise=True)
        test_spectral, test_series = \
            epochs_to_spectral_space(testset, subjectwise=True)
    elif modelfit == 'trialtype':
        # the test and train data should be in a different format in this
        # condition, and require some downstream processing (concatenation)
        train_spectral, train_series = \
            concat_epochs_to_spectral_space(trainset)
        test_spectral, test_series = \
            concat_epochs_to_spectral_space(testset)

    else:
        logging.info(f"Unknown parameter {modelfit} for modelfit.")
        return

    # fit a shared response model on training data in spectral space
    if modelfit == 'epochs' and len(subject) > 1:
        data = [ts for sub in train_spectral for ts in train_spectral[sub]]
    else:
        data = [ts for k, ts in train_spectral.items()]
    model = shared_response(data,
                            features=k)
    # transform the test data with it
    transformed = get_transformations(model, test_series, k)
    # plot the transformed test data by trial features
    _plot_transformed_components(transformed, k, testset,
                                 figdir=figdir,
                                 adderror=False,
                                 stderror=True, modelfit=modelfit,
                                 custom_name_component=custom_name_component)
    return \
        transformed, model, train_spectral, train_series, test_spectral,\
        test_series


def get_transformations(model, test_series, comp):
    """
    Transform raw data into model space.
    :param model:
    :param test_series:
    :param comp: number of components in the model
    :return:
    """
    transformations = {}
    for mid, sub in enumerate(test_series.keys()):
        transformations[sub] = {}
        for k in range(comp):
            transformations[sub][k] = \
                [np.dot(model.w_[mid].T, ts)[k] for ts in test_series[sub]]
    return transformations


def concat_epochs_to_spectral_space(data, shorten=False, separate=False):
    """
    Transform epoch data organized by trial features into spectral space.
    Average within subject and feature, and concatenate within subject over
    features. Takes a nested dictionary with n subjects as keys and features as
    subkeys.
    :param data: nested dict, top level keys are subjects, lower level dicts are
     trial data of one epoch
    :param shorten: None or tuple, if given, the data will be subset in th
     range of the tuple. The tuple needs to have a start and end index in Hz.
    :param separate: bool, if true, returns one spectral and time-resolved data
     slice per trial type. E.g., a subject with 9 trial types will get a dict
     with 9 sets of spectral and time series data
    :return:
    series: list of lists, or dict with list of lists when separate=True
    """
    from collections import defaultdict
    subjects = data.keys()
    series_spectral = {}
    series_time = {}
    conditionwise_spectraldata = defaultdict(dict)
    conditionwise_seriesdata = defaultdict(dict)
    for sub in subjects:
        subject_spectral = []
        subject_time = []
        # conditions are trial types (A, B, C, ...)
        for condition in data[sub]:
            epochs = [info['normalized_data'] for info in data[sub][condition]]
            if shorten:
                # subset the data to the range given by 'shorten'
                epochs = [epo[:, shorten[0]:shorten[1]] for epo in epochs]
            spectral_series = [transform_to_power(e) for e in epochs]
            assert spectral_series[0].shape[0] == epochs[0].shape[0] == 306
            # average over epochs.
            # noinspection PyUnresolvedReferences
            spectral_series = np.mean(np.asarray(spectral_series), axis=0)
            assert spectral_series.shape[0] == epochs[0].shape[0] == 306
            if separate:
                # keep data separate per condition
                conditionwise_spectraldata[sub][condition] = spectral_series
                conditionwise_seriesdata[sub][condition] = epochs
            else:
                # append the condition to concatenate all trialtypes per subject
                subject_spectral.extend(spectral_series.T)
                subject_time.extend(epochs)
        series_spectral[sub] = np.asarray(subject_spectral).T
        series_time[sub] = subject_time
    if separate:
        assert len(conditionwise_spectraldata) == len(conditionwise_seriesdata) == len(subjects)
        return conditionwise_spectraldata, conditionwise_seriesdata
    else:
        assert len(series_spectral) == len(series_time) == len(subjects)
        return series_spectral, series_time


def epochs_to_spectral_space(data, subjectwise=False):
    """
    Transform epoch data into spectral space. Takes a dictionary with n subjects
    as keys, transforms each subjects epochs into spectral space, returns each
    epoch as if it was its own subject
    :param data: nested dict, top level keys are subjects, lower level dicts are
     trial data of one epoch
    :param subjectwise: bool, if given, epochs are averaged resulting in one
     train epoch only
    :return:
    series: list of lists
    """
    subjects = data.keys()
    series_spectral = {}
    series_time = {}
    for sub in subjects:
        epochs = [info['normalized_data'] for info in data[sub]]
        spectral_series = [transform_to_power(e) for e in epochs]
        assert spectral_series[0].shape[0] == epochs[0].shape[0] == 306
        if subjectwise:
            # average over epochs. This results in one train epoch only.
            # noinspection PyUnresolvedReferences
            spectral_series = np.mean(np.asarray(spectral_series), axis=0)
            assert spectral_series.shape[0] == epochs[0].shape[0] == 306
        series_spectral[sub] = spectral_series
        series_time[sub] = epochs
    assert len(series_spectral) == len(series_time) == len(subjects)
    return series_spectral, series_time


def train_test_set(fullsample, data, ntrain=140, ntest=100, modelfit=None):
    """
    Split data into ntrain trainings data and ntest test data, and return them
    as a dictionary. If a particular modelfit is specified, ntrain and ntest are
    ignored, and instead, data are returned as a nested dictionary according to
    the specified experiment featured, with half of all available data in the
    training set and the other half in the testset.
    :param fullsample:
    :param data:
    :param ntrain:
    :param ntest:
    :param modelfit:
    :return:
    """
    testset = {}
    trainset = {}
    subjects = fullsample.keys()
    if modelfit == 'trialtype':
        logging.info(f"Splitting the data into training and testing epochs "
                     f"based on trial type.")
        # because we want to concatenate artificial time series, we let go of
        # ntrain and ntest specification and instead take all data we have
        trialorder = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        for sub in subjects:
            testset[sub] = {}
            trainset[sub] = {}
            for trialtype in trialorder:
                testset[sub][trialtype] = []
                trainset[sub][trialtype] = []
                trials = [i for i in data if (i['subject'] == sub and
                                              i['Lchar'] == trialtype)]
                # take half of the trials for training, half of them for testing
                ntest = ntrain = int(len(trials) / 2)
                shuffled = random.sample(trials, len(trials))
                trainset[sub][trialtype].extend(shuffled[:ntrain])
                testset[sub][trialtype].extend(shuffled[-ntest:])
        # return a nested dictionary. Later functions will need to disect it
        return trainset, testset

    else:
        logging.info(f"Splitting the data into {ntrain} training epochs and "
                     f"{ntest} testing epochs.")
        # just take the trials "as is"
        for subject in subjects:
            train = []
            test = []
            trials = [i for i in data if i['subject'] == subject]
            # we need to have at least as many trials per condition as the
            # sum of training data and testing data
            assert len(trials) >= ntrain + ntest
            # for each trialtype, pick n random trials for training,
            # shuffle the list with a random seed to randomize trial order
            shuffled = random.sample(trials, len(trials))
            train.extend(shuffled[:ntrain])
            # pick test trials from the other side of the shuffled list
            test.extend(shuffled[-ntest:])
            assert all([i not in test for i in train])
            testset[subject] = test
            trainset[subject] = train
            assert len(testset[subject]) == ntest
            assert len(trainset[subject]) == ntrain
    return trainset, testset


def get_general_data_structure(subject,
                               datadir,
                               bidsdir,
                               condition,
                               timespan):
    """
    Retrieve trial-wise data and its characteristics into a nested dictionary
    and the linearization of this structure, a list of dictionaries.

    :param subject: str, subject identifier
    :param datadir: str, path to data with epochs
    :param bidsdir: str, path to bids directory
    :param condition: str, an identifier based on which trials can be split.
    Possible values: 'left-right' (left versus right option choice),
    'nobrain-brain' (no-brainer versus brainer trials)
    :param timespan: str, an identifier of the time span of data to be used for
    model fitting. Must be one of 'decision' (time locked around the decision
    in each trial), 'firststim' (time locked to the first stimulus, for the
    stimulus duration), or 'fulltrial' (entire 7second epoch), 'secondstim',
    'delay'.
    :return:
    fullsample: dict; holds data of all subjects (keys are subjects)
    data: list of dicts, each entry is a dict with one epoch of one subject.
    """
    # initialize a dict to hold data over all subjects in the model
    fullsample = {}
    if not isinstance(subject, list):
        # we may want to combine multiple subject's data.
        subject = [subject]

    for sub in subject:
        fname = Path(datadir) / f'sub-{sub}/meg' / \
                f'sub-{sub}_task-memento_cleaned_epo.fif'
        logging.info(f'Reading in cleaned epochs from subject {sub} '
                     f'from path {fname}.')

        epochs = mne.read_epochs(fname)
        logging.info(
            f'The data has a frequency of {int(epochs.info["sfreq"])}Hz'
        )

        # use experiment logdata to build a data structure with experiment
        # information on every trial
        trials_to_trialtypes = \
            _find_data_of_choice(subject=sub,
                                 bidsdir=bidsdir,
                                 condition=condition)

        all_trial_info = combine_data(epochs=epochs,
                                      sub=sub,
                                      trials_to_trialtypes=trials_to_trialtypes,
                                      bidsdir=bidsdir,
                                      timespan=timespan)
        # add trial order information to the data structure
        all_trial_info = add_more_stuff(subject=sub,
                                        bidsdir=bidsdir,
                                        all_trial_info=all_trial_info)

        # append single subject data to the data dict of the sample
        fullsample[sub] = all_trial_info
    # aggregate the data into a list of dictionaries
    data = [
        dict(info, subject=subject, trial_no=trial_no)
        for subject, trialinfo in fullsample.items()
        for trial_no, info in trialinfo.items()
    ]

    return fullsample, data


def make_distance_matrices(datadir,
                          bidsdir,
                          figdir,
                          subjects=None,
                          ntrain=13,
                          ntest=13,
                          timespan=None,
                          triallength=None,
                          subset=None):
    """
    Create artificially synchronized time series data. In these artificially
    synchronized timeseries, N=ntrain trials per trial type (unique probability-
    magnitude combinations of the first stimulus) are first concatenated in
    blocks. This is done for a train set and a test set.
    Then, trials within each trial type block are averaged for noise reduction.
    :param datadir:
    :param bidsdir:
    :param figdir:
    :param subjects: set of str of subject identifiers. Default are subjects
    with the largest number of suitable trial data (30+ per condition)
    :param ntrain: int, number of trials to put in training set
    :param ntest: int, number of trials to put in test set
    :param timespan: dict, a specification of the time span to use for
     subsetting the data. When it isn't a string identifier from combine_data,
     it can be a list with start and end sample in 100Hz, e.g. {'left': [0, 70],
     'right': [270, 340]}
    :param triallength: int, the number of samples that make up the part of the
    trial that the model is fit on, e.g., 700samples of visual stimulation.
    :return:
    train_series: list of N=subjects lists artificial time series data,
    ready for SRM
    mean_train_series: list of N=subjects lists averaged artificial time series
     data, ready for SRM
    training_set_left: dict; overview of trials used as training and test data
     for each subject
    training_set_right:  dict; overview of trials used as training and test data
     for each subject
    """
    if subjects is None:
        subjects = ['001', '002', '003', '004', '005', '006', '007', '008',
                    '009', '010', '011', '012', '013', '014', '015', '016',
                    '017', '018', '019', '020', '021', '022']
    if timespan is None:
        timespan = {'left': 'firststim',
                    'right': 'secondstim'}
    timespan_left = timespan['left']
    timespan_right = timespan['right']
    if triallength is None:
        triallength = 70 if isinstance(timespan_left, str) \
            else timespan_left[1] - timespan_left[0]
        logging.info(f"Setting trial length to {triallength}")
    # first stimulus data
    leftsample, leftdata = get_general_data_structure(subject=subjects,
                                                      datadir=datadir,
                                                      bidsdir=bidsdir,
                                                      condition='left-right',
                                                      timespan=timespan_left)
    # second stimulus data
    rightsample, rightdata = get_general_data_structure(subject=subjects,
                                                        datadir=datadir,
                                                        bidsdir=bidsdir,
                                                        condition='left-right',
                                                        timespan=timespan_right)
    # define trialorder into which the stimuli are sorted. The trial order below
    # is grouped by increasing probability
    trialorder = ['E', 'H', 'I', 'C', 'F', 'A', 'G', 'B', 'D']
    desc = 'probability'
    # do different data mangling: reorder and/or average data from the left or
    # from the left and right sample.
    results = _create_splits_from_left_and_right_stimulation(
        subjects=subjects,
        ntrain=ntrain,
        ntest=ntest,
        trialorder=trialorder,
        leftdata=leftdata,
        rightdata=rightdata)
    # create plots based on the data
    models = plot_many_distance_matrices(results=results,
                                         triallength=triallength,
                                         figdir=figdir,
                                         subjects=subjects,
                                         trialorder=trialorder,
                                         description=desc,
                                         subset=subset)


def plot_trial_components_from_srm(subject,
                                   datadir,
                                   bidsdir,
                                   figdir,
                                   condition='left-right',
                                   timespan='fulltrial'):
    """
    Previously, we fit a probabilistic SRM to data, transformed the data
    with each trial's  weights. Now, we plot the data feature-wise for
    different conditions.
    :param subject
    :param datadir
    :param bidsdir
    :param figdir
    :param condition: str, an identifier based on which trials can be split.
    Possible values: 'left-right' (left versus right option choice),
    'nobrain-brain' (no-brainer versus brainer trials)
    :param timespan: str, an identifier of the time span of data to be used for
    model fitting. Must be one of 'decision' (time locked around the decision
    in each trial), 'firststim' (time locked to the first stimulus, for the
    stimulus duration), 'fulltrial' (entire epoch), 'secondstim',
    'delay'.
    :return:
    models, dict with feature number as keys and corresponding models as values
    data: list of dicts, with all trial information
    """
    fullsample, data = get_general_data_structure(subject=subject,
                                                  datadir=datadir,
                                                  bidsdir=bidsdir,
                                                  condition=condition,
                                                  timespan=timespan)
    del fullsample
    logging.info(f'Fitting shared response models based on data from subjects '
                 f'{subject}')
    # use a wide range of features to see if anything makes a visible difference
    features = [5, 10, 20]
    models = {}
    for f in features:
        # fit the model. List comprehension to retrieve the data as a list of
        # lists, as required by brainiak
        model = shared_response(data=[d['normalized_data'] for d in data],
                                features=f)
        final_df, data = create_full_dataframe(model, data)

        # plot individual features
        plot_srm_model(df=final_df,
                       nfeatures=f,
                       figdir=figdir,
                       subject=subject,
                       mdl='srm',
                       cond=condition,
                       timespan=timespan,
                       freq=1000)
        models[f] = model
    for idx, model in models.items():
        plot_distance_matrix(model, idx, figdir,
                             freq=1000, subject=subject,
                             condition=condition, timespan=timespan)
    return models, data


def add_more_stuff(subject,
                   bidsdir,
                   all_trial_info):
    """
    Bring trials in a standardized sequence across participants according to
    their characteristics

    Left characteristics
    Event name ->   description ->      typename:   -> count
    lOpt1 -> LoptMag 0.5, LoptProb 0.4 -> A ->  70
    lOpt2 -> LoptMag 0.5, LoptProb 0.8 -> B ->  65
    lOpt3 -> LoptMag 1, LoptProb 0.2 -> C ->    50
    lOpt4 -> LoptMag 1, LoptProb 0.8 -> D ->    70
    lOpt5 -> LoptMag 2, LoptProb 0.1 -> E ->    50
    lOpt6 -> LoptMag 2, LoptProb 0.2 -> F ->    35
    lOpt7 -> LoptMag 2, LoptProb 0.4 -> G ->    50
    lOpt8 -> LoptMag 4, LoptProb 0.1 -> H   ->  70
    lOpt9 -> LoptMag 4, LoptProb 0.2 -> I   ->  50

    Right_characteristics:

    :return:
    """
    stim_char = get_stimulus_characteristics(subject,
                                             bidsdir,
                                             columns=['trial_no',
                                                      'LoptMag',
                                                      'LoptProb',
                                                      'RoptMag',
                                                      'RoptProb',
                                                      'RT',
                                                      'choice',
                                                      'pointdiff']
                                             )

    # add the probability and magnitude information
    for info in all_trial_info.keys():
        LMag = stim_char[stim_char['trial_no'] == info]['LoptMag'].item()
        LPrb = stim_char[stim_char['trial_no'] == info]['LoptProb'].item()
        RMag = stim_char[stim_char['trial_no'] == info]['RoptMag'].item()
        RPrb = stim_char[stim_char['trial_no'] == info]['RoptProb'].item()
        RT = stim_char[stim_char['trial_no'] == info]['RT'].item()
        choice = stim_char[stim_char['trial_no'] == info]['choice'].item()
        pointdiff = stim_char[stim_char['trial_no'] == info]['pointdiff'].item()
        # also add information what the previous trial was, if we have info
        if info > 1:
            prev_choice = stim_char[stim_char['trial_no'] == info - 1][
                'choice'].item()
            prev_RT = stim_char[stim_char['trial_no'] == info - 1]['RT'].item()
            prev_LMag = stim_char[stim_char['trial_no'] == info - 1][
                'LoptMag'].item()
            prev_LPrb = stim_char[stim_char['trial_no'] == info - 1][
                'LoptProb'].item()
            prev_RMag = stim_char[stim_char['trial_no'] == info - 1][
                'RoptMag'].item()
            prev_RPrb = stim_char[stim_char['trial_no'] == info - 1][
                'RoptProb'].item()
        else:
            # this should be ok because None is a singleton
            prev_choice = prev_RT = prev_RPrb = prev_RMag = prev_LPrb = \
                prev_LMag = None
        all_trial_info[info]['prevLoptMag'] = prev_LMag
        all_trial_info[info]['prevLoptProb'] = prev_LPrb
        all_trial_info[info]['prevRoptMag'] = prev_RMag
        all_trial_info[info]['prevRoptProb'] = prev_RPrb
        all_trial_info[info]['prevLchar'] = trial_characteristics[
            (prev_LMag, prev_LPrb)] if prev_LMag else None
        all_trial_info[info]['prevRchar'] = trial_characteristics[
            (prev_RMag, prev_RPrb)] if prev_RMag else None
        all_trial_info[info]['prevRT'] = prev_RT
        all_trial_info[info]['prevchoice'] = prev_choice
        all_trial_info[info]['LoptMag'] = LMag
        all_trial_info[info]['LoptProb'] = LPrb
        all_trial_info[info]['RoptMag'] = RMag
        all_trial_info[info]['RoptProb'] = RPrb
        all_trial_info[info]['Lchar'] = trial_characteristics[(LMag, LPrb)]
        all_trial_info[info]['Rchar'] = trial_characteristics[(RMag, RPrb)]
        all_trial_info[info]['RT'] = RT
        all_trial_info[info]['choice'] = choice
        all_trial_info[info]['pointdiff'] = pointdiff

    # get a count of trials per characteristic
    Lchars = [info['Lchar'] for info in all_trial_info.values()]
    Rchars = [info['Rchar'] for info in all_trial_info.values()]
    # from collections import Counter
    # Lcounts = Counter(Lchars)
    # Rcounts = Counter(Rchars)
    return all_trial_info


def create_full_dataframe(model,
                          data):
    """
    Create a monstrous pandas dataframe.

    :param model: Brainiak SRM model, fitted
    :param data: List of Pandas data frames with MEG data and trial info
    :return: data: completed List of pd DataFrames with transformed data
    :return: finaldf: Large pd Dataframe, used for plotting
    """
    # transform the data with the fitted model. list comprehension as data is
    # a list of dicts
    transformed = model.transform([d['normalized_data'] for d in data])
    # add the transformed data into the dicts. They hold all information
    for d in data:
        d['transformed'] = transformed.pop(0)
    assert transformed == []
    # build a dataframe for plotting
    dfs = []
    for d in data:
        df = pd.DataFrame.from_records(d['transformed']).T
        trial_type = d['trial_type']
        trial = d['trial_no']
        df['trial_type'] = trial_type
        df['trial_no'] = trial
        dfs.append(df)

    finaldf = pd.concat(dfs)
    return finaldf, data


def combine_data(epochs,
                 sub,
                 trials_to_trialtypes,
                 bidsdir,
                 timespan):
    """
    Generate a dictionary that contains all relevant information of a given
    trial, including the data, correctly slices, to train the model on.
    :param epochs: contains the MEG data
    :param sub: str; subject identifier
    :param trials_to_trialtypes: Dict; a mapping of trial numbers to trial type
    :param bidsdir; str, Path to BIDS dir with log files
    :param timespan: str, an identifier of the time span of data to be used for
    model fitting. Must be one of 'decision' (time locked around the decision
    in each trial), 'firststim' (time locked to the first stimulus, for the
    stimulus duration), or 'fulltrial' (entire 7second epoch), 'secondstim',
    'delay', or a time frame within the experiment in samples (beware of the
    sampling rate!)
    :return: all_trial_infos; dict; with trial-wise information
    """
    all_trial_infos = {}

    if timespan == 'decision':
        # extract the information on decision time for all trials at once.
        trials_to_rts = get_decision_timespan_on_and_offsets(subject=sub,
                                                             bidsdir=bidsdir)
    if isinstance(timespan, list):
        logging.info(f"Received a list as a time span. Attempting to "
                     f"subset the available trial data in range "
                     f"{timespan[0]}, {timespan[1]} seconds")
    else:
        logging.info(f"Selecting data for the event description {timespan}.")

    # ugly, because epochs.__iter__ does give an ndarray, not an Epoch object
    for i in range(len(epochs)):
        epoch = epochs[i]
        # get the trial number as a key
        trial_no = epoch.metadata['trial_no'].item()
        # get the trial type, if it hasn't been excluded already
        trial_type = trials_to_trialtypes.get(trial_no, None)
        if trial_type is None:
            continue
        if timespan == 'decision' and trial_no not in trials_to_rts.keys():
            logging.info(f'Could not use epoch {trial_no} because the reaction '
                         f'was not fast enough.')
            continue

        tmin, tmax = _get_timewindows_from_spec(
            trials_to_rts[trial_no] if timespan == 'decision' else timespan
        )

        # get the data of the epoch. sensors X time array
        data = epoch.get_data(
            picks='meg',
            tmin=tmin,
            tmax=tmax,
        # we only have a single epoch, throw away len-1 dimension
        )[0]
        # normalize (z-score) the data within sensors
        normalized_data = stats.zscore(data, axis=1, ddof=0)
        all_trial_infos[trial_no] = {
            # the original MNE epoch ID
            'epoch': epoch.metadata.index.item(),
            'trial_type': trial_type,
            'data': data,
            'normalized_data': normalized_data,
        }
    return all_trial_infos


def _get_timewindows_from_spec(timespan):
    # will be the desired time windows (min, max) in seconds
    if isinstance(timespan, (list, tuple)):
        return timespan[:2]

    # otherwise decode the label
    t = None
    if timespan == 'fulltrial':
        # the data does not need to be shortened.
        t = (None, None)
    elif timespan == 'firststim':
        # we only need the first 700 milliseconds from the trial,
        # corresponding to the first 70 entries since we timelocked to the
        # onset of the first stimulation
        t = (0.0, 0.7)
    elif timespan == 'delay':
        # take the 2 seconds after the first stimulus
        t = (0.7, 2.7)
    elif timespan == 'secondstim':
        # take 700 ms after the first stimulus + delay phase
        t = (2.7, 3.4)
    else:
        raise NotImplementedError(
            f"The timespan {timespan} is not implemented.")
    return t


def get_decision_timespan_on_and_offsets(subject,
                                         bidsdir):
    """
    For each trial, get a time frame around the point in time that a decision
    was made.
    :param subject; str, subject identifier
    :param bidsdir: str, path to BIDS dir with log files
    :return: trials_to_rts: dict, and association of trial numbers to 800ms time
    slices around the time of decision in the given trial
    """
    logs = read_bids_logfile(subject=subject,
                             bidsdir=bidsdir)
    trials_and_rts = logs[['trial_no', 'RT']].values
    logging.info(f'The average reaction time was '
                 f'{np.nanmean(trials_and_rts[:, 1])}')
    # mark nans with larger RTs
    logging.info('Setting nan reaction times to implausibly large values.')
    np.nan_to_num(trials_and_rts, nan=100, copy=False)
    # collect the trial numbers where reaction times are too large to fit into
    # the trial. For now, this is at 1.9 seconds for 5 second epochs.
    trials_to_remove = trials_and_rts[np.where(
        trials_and_rts[:, 1] > 1.9)][:, 0]
    # initialize a dict to hold all information
    trials_to_rts = {}
    for trial, rt in trials_and_rts:
        # Now, add RT to the start of the second visual stimulus to get the
        # approximate decision time from trial onset
        # (0.7s + 2.0s = 2.7s)
        decision_time = rt + 2.7
        # plausibility check, no decision is made before a decision is possible
        assert decision_time > 2.7
        # calculate the slice needed for indexing the data for the specific
        # trial. We round down so that the specific upper or lower time point
        # can be used as an index to subset the data frame
        window = [decision_time - 0.4, decision_time + 0.4]
        if trial not in trials_to_remove:
            trials_to_rts[trial] = window

    return trials_to_rts


def shared_response(data,
                    features,
                    model='SRM'):

    """
    Compute a shared response model from a list of trials
    :param data: list of lists, with MEG data
    :param features: int, specification of feature number for the model
    :return:
    """
    if model == 'SRM':
        logging.info(f'Fitting a probabilistic SRM with {features} features...')
        # fit a probabilistic shared response model
        model = srm.SRM(features=features)
        model.fit(data)
    elif model == 'RSRM':
        logging.info(f'Fitting a robust SRM with {features} features...')
        from brainiak.funcalign import rsrm
        model = rsrm.RSRM(features=features)
        model.fit(data)
    return model
