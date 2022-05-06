"""
Module providing small utility functions for SRMs.

"""

import mne
import logging
import random
import numpy as np
from pymento_meg.orig.behavior import read_bids_logfile

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


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


def _get_mean_and_std_from_transformed(transformed,
                                       i,
                                       stderror=False
                                       ):
    """Helper to extract mean and standard deviation/standard error vectors
     for a given component i from the dictionary of transformed test data

    :param transformed: dict, contains transformed data
    :param i: component index
    :param stderror: bool, if True, return SEM instead of STD
    """
    mean = np.mean(np.asarray(
        [ts for sub in transformed.keys() for ts in transformed[sub][i]]),
        axis=0)
    # potentially change to standard error by dividing by np.sqrt(nepochs)

    if stderror:
        data = np.asarray(
            [ts for sub in transformed.keys() for ts in transformed[sub][i]]
        )
        std = np.std(data, axis=0, ddof=1) / np.sqrt(data.shape[0])
    else:
        # use standard deviation
        std = np.std(np.asarray(
            [ts for sub in transformed.keys() for ts in transformed[sub][i]]),
            axis=0)

    return mean, std


def _get_trial_indicators(transformed, data, type='choice'):
    """Query the metadata of the transformed data and report indices of trials
     corresponding to different properties"""

    if type == 'choice':
        # Get indices for trials with left and right choice
        i = 0
        left = []
        right = []
        for sub in transformed.keys():
            for epoch in data[sub]:
                if epoch['choice'] == 2:
                    right.append(i)
                elif epoch['choice'] == 1:
                    left.append(i)
                i += 1
        return left, right

    elif type == 'feedback':
        # Get trials with positive or negative feedback
        i = 0
        negative = []
        positive = []
        for sub in transformed.keys():
            for epoch in data[sub]:
                if np.isnan(epoch['pointdiff']):
                    negative.append(i)
                else:
                    positive.append(i)
                i += 1
        return negative, positive

    elif type == 'difficulty':
        # split between brainer and no-brainer trials
        i = 0
        brainer = []
        nobrainer = []
        for sub in transformed.keys():
            for epoch in data[sub]:
                if epoch['trial_type'] == 'brainer':
                    brainer.append(i)
                elif epoch['trial_type'] in (
                        'nobrainer_right', 'nobrainer_left'):
                    nobrainer.append(i)
                i += 1
        return brainer, nobrainer


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


def _create_splits_from_left_and_right_stimulation(subjects,
                                                   leftdata,
                                                   rightdata,
                                                   trialorder,
                                                   ntrain,
                                                   ntest):
    """
    Create a number of data aggregates, split into test and train sets, both in
    raw form and in averaged form.
    :param subjects: list of subjects included in the data
    :param leftdata: dict, contains the data from the left stimulation
    :param rightdata: dict, contains the data from the right stimulation
    :param ntrain: int, number of trials to use in training
    :param ntest: int, number of trials to use in testing
    :param trialorder: list, order in which trials should be concatenated
    :return: nested dict, with keys indicating original or averaged, then train
    or test, and then left or fullseries
    """

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

    # return all this data as a dict
    results = {'original_trials':
                   {'test':
                        {'left': test_data_leftseries,
                         'full': test_data_fullseries
                         },
                    'train':
                        {'left': train_data_leftseries,
                         'full': train_data_fullseries
                         }
                    },
               'averaged_trials':
                   {'test':
                        {'left': mean_test_data_leftseries,
                         'full': mean_test_data_fullseries
                         },
                    'train':
                        {'left': mean_train_data_leftseries,
                         'full': mean_train_data_fullseries
                         }
                    }
               }

    return results


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


def _get_quadrant_corrs(a, b, c, d):
    """
    get the correlations between four quadrants
    :param a, b, c, d: array, 1D vector corresponding to the lower triangle of
     each quadrant of a distance matrix
    :return:
    """
    # correlation between the intertrial distance of the left set of stimuli
    # and the intertrial distance of the right stimuli
    ad = np.corrcoef(a, d)[0][1]
    ab = np.corrcoef(a, b)[0][1]
    ac = np.corrcoef(a, c)[0][1]
    db = np.corrcoef(d, b)[0][1]
    dc = np.corrcoef(d, c)[0][1]
    return ad, ab, ac, db, dc

