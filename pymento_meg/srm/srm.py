"""
Module for fitting shared response models.
TODO:
Reduce epoch length to 6 seconds
OR reduce epoch length further, to potentially loose fewer trials

write all shared responses into a matrix, check for consistent correlation
pattern across subjects

"""


import mne
import logging
import numpy as np
import pandas as pd

from brainiak.funcalign import srm
from pathlib import Path
from scipy import stats
from pymento_meg.orig.behavior import read_bids_logfile
from pymento_meg.proc.epoch import get_stimulus_characteristics

import scipy.spatial.distance as sp_distance
import matplotlib.pyplot as plt

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
# set font specifications for plots
font = {'family': 'normal',
        'weight': 'bold',
        'size': 50}
plt.rc('font', **font)

# map a set of reward magnitude and probability (in the order) to trial types.
# the characteristic combinations with letters are frequent (50-75 over the
# course of the experiment) and occur in the left and right stimulus. The
# non-letter combinations occur only on the right side, and some of them are
# infrequent (between 1 and 5? occurrences over the course of the experiment)
trial_characteristics = {
    (0.5, 0.4): 'A',
    (0.5, 0.8): 'B',
    (1, 0.2): 'C',
    (1, 0.8): 'D',
    (2, 0.1): 'E',
    (2, 0.2): 'F',
    (2, 0.4): 'G',
    (4, 0.1): 'H',
    (4, 0.2): 'I',
    (0.5, 0.1): '0.5-0.1',
    (0.5, 0.2): '0.5-0.2',
    (1, 0.4): '1.0-0.4',
    (1, 0.1): '1.0-0.1',
    (2, 0.8): '2.0-0.8',
    (4, 0.4): '4.0-0.4',
    (4, 0.8): '4.0-0.8'
}


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
        logging.info('Ensuring data has a frequency of 100Hz')
        if epochs.info['sfreq'] > 100:
            # after initial preprocessing, they are downsampled to 200Hz.
            # Downsample further to 100Hz
            epochs.resample(sfreq=100, verbose=True)
        assert epochs.info['sfreq'] == 100
        # read the epoch data into a dataframe
        df = epochs.to_data_frame()

        # use experiment logdata to build a data structure with experiment
        # information on every trial
        trials_to_trialtypes, epochs_to_trials = \
            _find_data_of_choice(epochs=epochs,
                                 subject=sub,
                                 bidsdir=bidsdir,
                                 condition=condition,
                                 df=df)

        all_trial_info = combine_data(df=df,
                                      sub=sub,
                                      trials_to_trialtypes=trials_to_trialtypes,
                                      epochs_to_trials=epochs_to_trials,
                                      bidsdir=bidsdir,
                                      timespan=timespan)
        # add trial order information to the data structure
        all_trial_info = add_trial_types(subject=sub,
                                         bidsdir=bidsdir,
                                         all_trial_info=all_trial_info)

        # append single subject data to the data dict of the sample
        fullsample[sub] = all_trial_info
    # aggregate the data into a list of dictionaries
    data = []
    for subject, trialinfo in fullsample.items():
        for trial_no, info in trialinfo.items():
            info['subject'] = subject
            info['trial_no'] = trial_no
            data.append(info)

    return fullsample, data


def test_and_train_split(datadir,
                         bidsdir,
                         figdir,
                         subjects=['011', '012', '014', '016', '017',
                                   '018', '019', '020', '022'],
                         ntrain=15,
                         ntest=15,
                         timespan={'left': 'firststim',
                                   'right': 'secondstim'}):
    """
    Create artificially synchronized time series data. In these artificially
    synchronized timeseries, N=ntrain trials per trial type (unique probability-
    magnitude combinations of the first stimulus) are first concatenated in
    blocks. This is done for a train set and a test set.
    Then, trials within each trial type block are averaged for noise reduction.

    # TODO: Maybe use training data of shorter length (e.g., visual stim), and
    # test data of longer length/different trial segments
    # TODO: use the first 500 ms

    # TODO: threshold für die mean trial distance matrices
    # TODO: sanity check mit test data
    # TODO: Transformation von test data mit srm vom training
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
    import random
    random.seed(423)

    timespan_left = timespan['left']
    timespan_right = timespan['right']
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
    # define trialorder. according to reward magnitude below
    trialorder = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    # order according to probability:
    # trialorder = ['E', 'H', 'I', 'C', 'F', 'A', 'G', 'B', 'D']
    # order according to expected value (prob*reward):
    # trialorder = ['A', 'C', 'E', 'B', 'F', 'H', 'D', 'G', 'I']

    train_data = {}
    mean_train_data = {}
    test_data = {}
    mean_test_data = {}
    for data, position, index in [(leftdata, 'left', 'Lchar'),
                                  (rightdata, 'right', 'Rchar')]:
        train_data[position] = {}
        mean_train_data[position] = {}
        test_data[position] = {}
        mean_test_data[position] = {}
        for sub in subjects:
            train = []
            mean_train = []
            test = []
            mean_test = []
            for trialtype in trialorder:
                trials = [i for i in data if (i['subject'] == sub and
                                              i[index] == trialtype)]
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

                # average data across trials of trialtype for noise reduction
                mean = np.mean(
                    [t['normalized_data'] for t in shuffled[:ntrain]],
                    axis=0)
                assert mean.shape[0] == 306
                mean_train.append(mean)
                # do the same for test trials
                mean = np.mean(
                    [t['normalized_data'] for t in shuffled[-ntest:]],
                    axis=0)
                assert mean.shape[0] == 306
                mean_test.append(mean)
            # after looping through 9 trial types make sure we have the expected
            # amount of data
            assert len(train) == 9 * ntrain
            assert len(test) == 9 * ntest
            assert len(mean_train) == 9
            assert len(mean_test) == 9
            train_data[position][sub] = train
            mean_train_data[position][sub] = mean_train
            test_data[position][sub] = test
            mean_test_data[position][sub] = mean_test
    # join the data from left and right into nested lists. Also make a only-left
    # series
    mean_train_data_fullseries = []
    mean_train_data_leftseries = []
    mean_test_data_fullseries = []
    mean_test_data_leftseries = []
    for sub in subjects:
        left = concatenate_means(mean_train_data['left'][sub])
        right = concatenate_means(mean_train_data['right'][sub])
        assert left.shape == right.shape
        mean_train_data_fullseries.append(concatenate_means([left, right]))
        mean_train_data_leftseries.append(left)
        # also for test data
        left = concatenate_means(mean_test_data['left'][sub])
        right = concatenate_means(mean_test_data['right'][sub])
        assert left.shape == right.shape
        mean_test_data_fullseries.append(concatenate_means([left, right]))
        mean_test_data_leftseries.append(left)
    assert len(mean_train_data_fullseries) == \
           len(mean_train_data_leftseries) == \
           len(subjects) == \
           len(mean_test_data_leftseries) == \
           len(mean_test_data_fullseries)

    train_data_fullseries = []
    train_data_leftseries = []
    for sub in subjects:
        left = concatenate_data(train_data['left'][sub])
        right = concatenate_data(train_data['right'][sub])
        assert left.shape == right.shape
        train_data_fullseries.append(concatenate_means([left, right]))
        train_data_leftseries.append(left)
    assert len(train_data_fullseries) == \
           len(train_data_leftseries) == \
           len(subjects)

    test_data_fullseries = []
    test_data_leftseries = []
    for sub in subjects:
        left = concatenate_data(test_data['left'][sub])
        right = concatenate_data(test_data['right'][sub])
        assert left.shape == right.shape
        test_data_fullseries.append(concatenate_means([left, right]))
        test_data_leftseries.append(left)
    assert len(test_data_fullseries) == \
           len(test_data_leftseries) == \
           len(subjects)

    # create & plot distance matrices of SRMs with different no of components,
    # fit on the averaged and unaveraged artificial train timeseries. Return SRM
    # loop through different number of features
    models = {}
    for n in [5, 10, 20, 40, 80, 160]:
        models[n] = {}
        models[n]['full'] = plot_trialtype_distance_matrix(
            mean_train_data_fullseries,
            n,
            figdir=figdir,
            triallength=triallength
            )
        models[n]['left'] = plot_trialtype_distance_matrix(
            mean_train_data_leftseries,
            n,
            figdir=figdir,
            trialtypes=9,
            clim=[0, 0.5],
            triallength=triallength)
        plot_trialtype_distance_matrix(train_data_fullseries,
                                       n,
                                       figdir=figdir,
                                       trialtypes=270,
                                       triallength=triallength)

    # create subject specific and averaged distance matrices from raw data
    compute_raw_distances(mean_train_data_fullseries,
                          subjects,
                          figdir=figdir,
                          trialtypes=18,
                          triallength=triallength,
                          feature='train'
                          )

    compute_raw_distances(mean_test_data_fullseries,
                          subjects,
                          figdir=figdir,
                          trialtypes=18,
                          triallength=triallength,
                          feature='test'
                          )

    # transform subject raw data into shared model room
    for n in [5, 10, 20, 40, 80, 160]:
        shared_test = models[n]['full'].transform(mean_test_data_fullseries)
        for idx, sub in enumerate(subjects):
            plot_trialtype_distance_matrix(shared_test[idx],
                                           n=sub,
                                           figdir=figdir,
                                           trialtypes=18,
                                           on_model_data=False,
                                           feature='test' + str(n),
                                           triallength=triallength)

    for n in [5, 10, 20, 40, 80, 160]:
        shared_test = models[n]['full'].transform(mean_train_data_fullseries)
        for idx, sub in enumerate(subjects):
            plot_trialtype_distance_matrix(shared_test[idx],
                                           n=sub,
                                           figdir=figdir,
                                           trialtypes=18,
                                           on_model_data=False,
                                           feature='train' + str(n),
                                           triallength=triallength)

    return train_data_fullseries, train_data_leftseries, \
           mean_train_data_fullseries, mean_train_data_leftseries,


def compute_raw_distances(data,
                          subjects,
                          figdir,
                          trialtypes,
                          triallength,
                          feature=None):
    """
    Take a list of lists with time series from N subjects.
    For each subject/list in this data, build a trialtype-by-trialtype
    correlation distance matrix.

    Then, take the N distance matrices, transform them from correlation distance
    to correlation, and Fisher-z-transform them. Afterwards, average them across
    subjects
    :param data: Data to use for distance matrix creation. Should be a list of
    arrays
    :param subjects: list, of subject identifiers
    :param feature: str, some additional identifier, used in figure name
    :param figdir: str, path to where plots are saved
    :param triallength: int, length of a single trial in samples
    :param trialtypes: Number of consecutive trial types in the time series to
    plot. 9 or 18 for single/averaged trials left or left+right, or n*trialtypes
    :return:
    """
    # create & plot distance matrices of raw data per subject. Return dist_mat
    distmat = {}
    for idx, sub in enumerate(subjects):
        distmat[sub] = \
            plot_trialtype_distance_matrix(data[idx],
                                           n=sub,
                                           figdir=figdir,
                                           trialtypes=trialtypes,
                                           on_model_data=False,
                                           triallength=triallength,
                                           feature=feature)
    # Fisher-z transform the matrices
    zdistmat = {}
    for sub in subjects:
        # transform data from correlation distance back to correlation
        corrdist = 1 - distmat[sub]
        assert (corrdist >= -1).all() & (corrdist <= 1).all(),\
            "We have impossible correlations"
        # fisher z-transform
        print(sub)
        zdistmat[sub] = np.arctanh(corrdist)
    # average the matrices
    avg = np.mean(np.array([v for k, v in zdistmat.items()]), axis=0)
    # plot it
    plt.figure(figsize=[50, 50])
    plt.imshow(avg, cmap='viridis')
    plt.colorbar()
    # set a figure title according to the number of trialtypes plotted
    type = 'left and right' if trialtypes in [18, 270] else 'left'
    plt.title(f"Average of subject-wise raw data trial distances for "
              f"{type} stimulation")
    fname = Path(figdir) / f'group/meg' / \
                f'group_task-memento_raw_avg_trialdist_{trialtypes}.png'
    plt.savefig(fname)


def plot_trialtype_distance_matrix(data,
                                   n,
                                   figdir,
                                   trialtypes=18,
                                   clim=[0, 1],
                                   on_model_data=True,
                                   feature=None,
                                   triallength=70):
    """
    A generic function to fit SRMs and plot distance matrices on trial data
    :param data: list of lists, needs to be one list per subject with time
    series data. SRM is fit on this data
    :param n: Number of features to use in SRM
    :param figdir: Path to where figures and plots are saved
    :param trialtypes: Number of trialtypes to plot. 9 if only left trials are
    used, 18 when its both left and right, 135 when its left but not averaged
    :param clim: list, upper and lower limit of the colorbar
    :param feature: str, some additional identifier, used in figure name
    :param on_model_data: bool, if True, fit an SRM model and base the distance
    matrix on the shared components. If False, base the distance matrix on raw
    data. In the latter case, data needs to be a single time series, no list of
    lists, and n needs to be a subject identifier
    :param triallength: int, length of a single trial in samples. Defaults to 70
    :return:
    """
    if on_model_data:
        assert len(data) > 1
        assert isinstance(data, list)
        # fit a probabilistic SRM
        model = shared_response(data, features=n)
        # get the componentXtime series of each trial in SRM, and put it into
        # a nested array. 70 -> length of one trial / averaged trial type
        trialmodels_ = np.array(
            [model.s_[:, triallength * i:triallength * (i + 1)].ravel()
             for i in range(trialtypes)])
    else:
        assert isinstance(data, np.ndarray)
        trialmodels_ = np.array(
            [data[:, triallength * i:triallength * (i + 1)].ravel()
             for i in range(trialtypes)])

    dist_mat = sp_distance.squareform(
        sp_distance.pdist(trialmodels_, metric='correlation'))
    plt.figure(figsize=[50, 50])
    plt.imshow(dist_mat, cmap='viridis', clim=clim)

    # set a figure title according to the number of trialtypes plotted
    type = 'left and right' if trialtypes in [18, 270] else 'left'
    if on_model_data:
        plt.title(f"Correlation distance between trialtypes for "
                  f"{type} stimulation")
        fname = Path(figdir) / f'group/meg' / \
                    f'group_task-memento_srm-{n}_trialdist_{trialtypes}.png'
    else:
        plt.title(f"Correlation distance between trialtypes for "
                  f"{type} stimulation in raw data")
        fname = Path(figdir) / f'group/meg' / \
                    f'group_task-memento_raw_sub-{n}_trialdist_{trialtypes}_feature-{feature}.png'
    plt.colorbar()

    logging.info(f"Saving a distance matrix with {n} features to {fname}")
    plt.savefig(fname)
    plt.close('all')
    if on_model_data:
        return model
    else:
        return dist_mat


def concatenate_data(data, field='normalized_data'):
    """
    Concatenate trial data in a list of dictionaries
    :param data: nested dict, contains all trial infos
    :param field: str, dict key in info dict in general data structure
    :return:
    """
    time_series = np.concatenate([info[field] for info in data],
                                 axis=1)

    assert time_series.shape[0] == 306
    return time_series


def concatenate_means(data):
    time_series = np.concatenate(data, axis=1)
    assert time_series.shape[0] == 306
    return time_series


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
    stimulus duration), 'fulltrial' (entire 7second epoch), 'secondstim',
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

    logging.info(f'Fitting shared response models based on data from subjects '
                 f'{subject}')
    # use a wide range of features to see if anything makes a visible difference
    features = [5, 10, 20, 100]
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
                       subject='group',
                       mdl='srm',
                       cond=condition,
                       timespan=timespan)
        models[f] = model
    for idx, model in models.items():
        plot_distance_matrix(model, idx, figdir)
    return models, data


def add_trial_types(subject,
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
                                                      'RoptProb']
                                             )

    # add the probability and magnitude information
    for info in all_trial_info.keys():
        LMag = stim_char[stim_char['trial_no'] == info]['LoptMag'].item()
        LPrb = stim_char[stim_char['trial_no'] == info]['LoptProb'].item()
        RMag = stim_char[stim_char['trial_no'] == info]['RoptMag'].item()
        RPrb = stim_char[stim_char['trial_no'] == info]['RoptProb'].item()
        all_trial_info[info]['LoptMag'] = LMag
        all_trial_info[info]['LoptProb'] = LPrb
        all_trial_info[info]['RoptMag'] = RMag
        all_trial_info[info]['RoptProb'] = RPrb
        all_trial_info[info]['Lchar'] = trial_characteristics[(LMag, LPrb)]
        all_trial_info[info]['Rchar'] = trial_characteristics[(RMag, RPrb)]

    # get a count of trials per characteristic
    Lchars = [info['Lchar'] for info in all_trial_info.values()]
    Rchars = [info['Rchar'] for info in all_trial_info.values()]
    from collections import Counter
    Lcounts = Counter(Lchars)
    Rcounts = Counter(Rchars)
    print(Lcounts)
    print(Rcounts)
    # make sure we have a minimal amount of trials to fit the model
    assert all([Lcounts[i] > 5 for i in Lcounts])
    #assert all([Rcounts[i] > 5 for i in Rcounts])

    # 10+:1,2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
    # 20+: 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22
    # 30+ trials per condition: 11, 12, 14, 16, 17, 18, 19, 20, 22

    return all_trial_info


def plot_distance_matrix(model, idx, figdir):
    """
    plot a distance matrix between time points from the shared response.
    :param model:
    :param idx
    :param figdir:
    :return:
    """
    dist_mat = sp_distance.squareform(sp_distance.pdist(model.s_.T,
                                                        metric='correlation'))
    plt.xlabel('t (100 = 1sec)')
    plt.ylabel('t (100 = 1sec)')
    plt.imshow(dist_mat, cmap='viridis')
    # TODO: maybe add vertical lines in experiment landmarks
    plt.colorbar()
    fname = Path(figdir) / f'group/meg' / \
                            f'group_task-memento_srm-{idx}_distances.png'
    plt.savefig(fname)
    plt.close()


def create_full_dataframe(model,
                          data):
    """
    Create a monstrous pandas dataframe.

    :param model: Brainiak SRM model, fitted
    :param data: List of Pandas dataframes with MEG data and trial info
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


def combine_data(df,
                 sub,
                 trials_to_trialtypes,
                 epochs_to_trials,
                 bidsdir,
                 timespan):
    """
    Generate a dictionary that contains all relevant information of a given
    trial, including the data, correctly slices, to train the model on.
    :param df: pandas dataframe, contains the MEG data
    :param sub: str; subject identifier
    :param trials_to_trialtypes: Dict; a mapping of trial numbers to trial type
    :param epochs_to_trials: Dict; a mapping of epochs to trial numbers
    :param bidsdir; str, Path to BIDS dir with log files
    :param timespan: str, an identifier of the time span of data to be used for
    model fitting. Must be one of 'decision' (time locked around the decision
    in each trial), 'firststim' (time locked to the first stimulus, for the
    stimulus duration), or 'fulltrial' (entire 7second epoch), 'secondstim',
    'delay', or a time frame within the experiment in samples (100Hz sampling
    rate)
    :return: all_trial_infos; dict; with trial-wise information
    """
    all_trial_infos = {}
    unique_epochs = df['epoch'].unique()

    if timespan == 'decision':
        # extract the information on decision time for all trials at once.
        trials_to_rts = get_decision_timespan_on_and_offsets(subject=sub,
                                                             bidsdir=bidsdir)
    if isinstance(timespan, list):
        logging.info(f"Received a list as a time span. Attempting to "
                     f"subset the available trial data in range "
                     f"{timespan[0]}, {timespan[1]}")
    else:
        logging.info(f"Selecting data for the event description {timespan}.")

    for epoch in unique_epochs:
        # get the trial number as a key
        trial_no = epochs_to_trials[epoch]
        # get the trial type, if it hasn't been excluded already
        trial_type = trials_to_trialtypes.get(trial_no, None)
        if trial_type is None:
            continue
        # get the data of the epoch. Transpose it, to get a sensors X time array
        data = df.loc[df['epoch'] == epoch, 'MEG0111':'MEG2643'].values.T
        if timespan == 'fulltrial':
            # the data does not need to be shortened. just make sure it has the
            # expected dimensions
            assert data.shape == (306, 700)
        elif timespan == 'firststim':
            # we only need the first 700 milliseconds from the trial,
            # corresponding to the first 70 entries since we timelocked to the
            # onset of the first stimulation
            data = data[:, :70]
            assert data.shape == (306, 70)
        elif timespan == 'delay':
            # take the 2 seconds after the first stimulus
            data = data[:, 70:270]
            assert data.shape == (306, 200)
        elif timespan == 'secondstim':
            # take 700 ms after the first stimulus + delay phase
            data = data[:, 270:340]
            assert data.shape == (306, 70)
        elif timespan == 'decision':
            # we need an adaptive slice of data (centered around the exact time
            # point at which a decision was made in a given trial.
            if trial_no not in trials_to_rts.keys():
                # if the trial number has been sorted out before, don't append
                # the data
                continue
            onset, offset = trials_to_rts[trial_no]
            data = data[:, onset:offset]
            assert data.shape == (306, 80)
        else:
            if isinstance(timespan, list):
                data = data[:, timespan[0]:timespan[1]]
            else:
                raise NotImplementedError(f"The timespan {timespan} is not "
                                          f"implemented.")
        # normalize (z-score) the data within sensors
        normalized_data = stats.zscore(data, axis=1, ddof=0)
        all_trial_infos[trial_no] = {'epoch': epoch,
                                     'trial_type': trial_type,
                                     'data': data,
                                     'normalized_data': normalized_data}
    return all_trial_infos


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
                 f'{np.nanmean(trials_and_rts[:,1])}')
    # mark nans with larger RTs
    logging.info('Setting nan reaction times to implausibly large values.')
    np.nan_to_num(trials_and_rts, nan=100, copy=False)
    # collect the trial numbers where reaction times are too large to fit into
    # the trial. For now, this is at 3 seconds.
    trials_to_remove = trials_and_rts[np.where(
        trials_and_rts[:, 1] > 3)][:, 0]
    # initialize a dict to hold all information
    trials_to_rts = {}
    for trial, rt in trials_and_rts:
        # get the right time frame. First, transform decision time into the
        # timing within the trial. this REQUIRES a sampling rate of 100Hz!
        rt = rt * 100
        # Now, add RT to the end of the second visual stimulus to get the
        # decision time from trial onset
        # (70 + 200 + 70 = 340)
        decision_time = rt + 340
        assert decision_time > 340
        # calculate the slice needed for indexing the data for the specific
        # trial. We round down so that the specific upper or lower time point
        # can be used as an index to subset the data frame
        slices = [int(np.floor(decision_time - 40)),
                  int(np.floor(decision_time + 40))]
        assert slices[1] - slices[0] == 80
        if trial not in trials_to_remove:
            trials_to_rts[trial] = slices

    return trials_to_rts


def shared_response(data,
                    features):
    """
    Compute a shared response model from a list of trials
    :param data: list of lists, with MEG data
    :param features: int, specification of feature number for the model
    :return:
    """
    logging.info(f'Fitting a probabilistic SRM with {features} features...')
    # fit a probabilistic shared response model
    model = srm.SRM(features=features)
    model.fit(data)
    return model


def _find_data_of_choice(df,
                         epochs,
                         subject,
                         bidsdir,
                         condition):
    """
    Based on a condition that can be queried from the log files (e.g., right or
    left choice of stimulus), return the trial names, trial types, and epoch IDs
    :param epochs: epochs object
    :param df: pandas dataframe of epochs
    :param subject: str, subject identifier '011'
    :param bidsdir: str, path to bids data with logfiles
    :param condition: str, a condition description. Must be one of 'left-right'
    (for trials with right or left choice), 'nobrain-brain' (for trials with
    no-brainer decisions versus actual decisions)
    :return: trials_to_trialtypes; dictionary with trial - condition
    :return: epochs_to_trials; dict; epoch ID to trial number associations
    """
    # get an association of trial types with trial numbers
    if condition == 'left-right':
        logging.info('Attempting to retrieve trial information for left and '
                     'right stimulus choices')
        choices = get_leftright_trials(subject=subject,
                                       bidsdir=bidsdir)
    elif condition == 'nobrain-brain':
        logging.info('Attempting to retrieve trial information for no-brainer '
                     'and brainer trials')
        choices = get_nobrainer_trials(subject=subject,
                                       bidsdir=bidsdir)

    # Create a mapping between epoch labels and trial numbers based on metadata.
    assert len(df['epoch'].unique()) == len(epochs.metadata.trial_no.values)
    # generate a mapping between trial numbers and epoch names in the dataframe
    # TODO: take this straight from metadata, epochs are indices
    epochs_to_trials = {key: value for (key, value) in
                        zip(df['epoch'].unique(),
                            epochs.metadata.trial_no.values)}
    # make sure we caught all epochs
    assert all([i in df['epoch'].unique() for i in epochs_to_trials.keys()])
    assert len(df['epoch'].unique()) == len(epochs_to_trials.keys())

    # transform the "trial_no"-"trial_type" association in choices into an
    # association of "trial_no that exist in the data" (e.g., survived
    # cleaning) - "trial_types". The epochs_to_trials mapping is used as an
    # indication with trial numbers are actually still existing in the epochs
    trials_to_trialtypes = {}
    for cond, trials in choices.items():
        # trials is a list of all trial numbers in a given condition
        counter = 0
        for trial in trials:
            # we need extra conditions because during the generation of each of
            # the dicts below, some trials may have been excluded
            if trial in epochs_to_trials.values():
                trials_to_trialtypes[trial] = cond
                counter += 1
            else:
                logging.info(f'Trial number {trial} is not '
                             f'included in the data.')

        logging.info(f"Here's my count of matching events in the SRM data for"
                     f" condition {cond}: {counter}")

    return trials_to_trialtypes, epochs_to_trials


def get_nobrainer_trials(subject, bidsdir):
    """
    Return the trials where a decision is a "nobrainer", a trial where both the
    reward probability and magnitude of one option is higher than that of the
    other option.
    :param subject: str, subject identifier
    :param bidsdir: str, path to BIDS dir with logfiles
    :return:
    """
    df = read_bids_logfile(subject=subject,
                           bidsdir=bidsdir)
    # where is the both Magnitude and Probability of reward greater for one
    # option over the other? -> nobrainer trials
    right = df['trial_no'][(df.RoptMag > df.LoptMag) &
                           (df.RoptProb > df.LoptProb)].values
    left = df['trial_no'][(df.LoptMag > df.RoptMag) &
                          (df.LoptProb > df.RoptProb)].values
    # the remaining trials require information integration ('brainer' trials)
    brainer = [i for i in df['trial_no'].values
               if i not in right and i not in left]

    # brainer and no-brainer trials should add up
    assert len(brainer) + len(right) + len(left) == len(df['trial_no'].values)
    # make sure that right and left no brainers do not intersect - if they have
    # common values, something went wrong
    assert not bool(set(right) & set(left))
    # make sure to only take those no-brainer trials where participants actually
    # behaved as expected. Those trials where it would be a nobrainer to pick X,
    # but the subject chose Y, are excluded with this.
    consistent_nobrainer_left = \
        [trial for trial in left if
         df['choice'][df['trial_no'] == trial].values == 1]
    consistent_nobrainer_right = \
        [trial for trial in right if
         df['choice'][df['trial_no'] == trial].values == 2]
    logging.info(f"Subject {subject} underwent a total of {len(right)} "
                 f"no-brainer trials for right choices, and a total of "
                 f"{len(left)} no-brainer trials for left choices. The subject "
                 f"chose consistently according to the no-brainer nature of "
                 f"the trial in N={len(consistent_nobrainer_right)} cases for "
                 f"right no-brainers, and in "
                 f"N={len(consistent_nobrainer_left)} cases for left "
                 f"no-brainers.")
    # create a dictionary with brainer and no-brainer trials. We leave out all
    # no-brainer trials where the participant hasn't responded in accordance to
    # the no-brain nature of the trial
    choices = {'brainer': brainer,
               'nobrainer_left': consistent_nobrainer_left,
               'nobrainer_right': consistent_nobrainer_right}

    return choices


def get_leftright_trials(subject,
                         bidsdir):
    """
    Return the trials where a left choice and where a right choice was made.
    Logdata coding for left and right is probably 1 = left, 2 = right (based on
    experiment file)
    :param subject: str, subject identifier, e.g., '001'
    :param bidsdir: str, Path to the root of a BIDS raw dataset
    :return: choices; dict of trial numbers belonging to left or right choices
    """
    df = read_bids_logfile(subject=subject,
                           bidsdir=bidsdir)
    # get a list of epochs in which the participants pressed left and right
    left_choice = df['trial_no'][df['choice'] == 1].values
    right_choice = df['trial_no'][df['choice'] == 2].values
    choices = {'left (1)': left_choice,
               'right (2)': right_choice}

    return choices


def plot_srm_model(df,
                   nfeatures,
                   figdir,
                   subject,
                   mdl='srm',
                   cond='left-right',
                   timespan='fulltrial'):
    """
    Plot the features of a shared response model
    :param df: concatenated dataframe of trial data, transformed with the
    trial-specific mapping of the shared response model (returned by
    concatenate_transformations().
    :param figdir: str, path to directory to save figures in
    :param nfeatures: int, number of features in the model
    :param subject: str, subject identifier such as '011'
    :param mdl: Name of the SRM model to place in the figure name
    :param cond: Str, name of the condition plotted. Useful values are
     'left-right', 'nobrain-brain'
    :param timespan: Str, name of the timeframe used to fit the model
    :return:
    """
    # TODO: This needs adjustment for trial names!
    import pylab
    import seaborn as sns
    logging.info('Plotting the data transformed with the SRM model.')
    title = 'Full trial duration' if timespan == 'fulltrial' else \
            'Duration of the first stimulus' if timespan == 'firststim' else \
            '400ms +/- decision time' if timespan == 'decision' else \
            None
    # TODO: this needs some indication of which subjects the plot is made from
    for i in range(nfeatures):
        fname = Path(figdir) / f'{subject}/meg' /\
                     f'{subject}_{mdl}_{nfeatures}-feat_task-{cond}_model-{timespan}_comp_{i}.png'
        if cond == 'left-right':
            fig = sns.lineplot(data=df[df['trial_type'] == 'right (2)'][i])
            sns.lineplot(data=df[df['trial_type'] == 'left (1)'][i]).set_title(title)
            fig.legend(title='Condition', loc='upper left',
                       labels=['left choice', 'right choice'])
        elif cond == 'nobrain-brain':
            fig = sns.lineplot(data=df[df['trial_type'] == 'brainer'][i])
            sns.lineplot(data=df[(df['trial_type'] == 'nobrainer_left') |
                                 (df['trial_type'] == 'nobrainer_right')][i]).set_title(title)
            fig.legend(title='Condition', loc='upper left',
                       labels=['brainer', 'nobrainer'])
        if timespan == 'fulltrial':
            # define the timing of significant events in the trial timecourse:
            # onset and offset of visual stimuli
            events = [0, 70, 270, 340]
            # plot horizontal lines to mark the end of visual stimulation
            [pylab.axvline(ev, linewidth=1, color='grey', linestyle='dashed')
             for ev in events]

        plot = fig.get_figure()
        plot.savefig(fname)
        plot.clear()


def _select_channels(epochs):
    """
    Select a subset of channels based on location in helmet
    :param epochs: pandas DataFrame, df of epochs
    :return:
    """

    right_chs = mne.read_vectorview_selection(['Right-occipital'])
    idx_right = [epochs.columns.get_loc(s.replace(' ', '')) for s in right_chs]
    left_chs = mne.read_vectorview_selection(['Left-occipital'])
    idx_left = [epochs.columns.get_loc(s.replace(' ', '')) for s in left_chs]
    idx_left = [186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
                199, 200, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                224, 234, 235, 236, 237, 238, 239, 246, 247, 248]
    idx_right = [231, 232, 233, 240, 241, 242, 243, 244, 245, 261, 262, 263,
                 264, 265, 266, 267, 268, 269, 270, 271, 272, 279, 280, 281,
                 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296]
