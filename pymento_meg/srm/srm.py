"""
Module for fitting shared response models.

TODO: preprocess all files starting from first stimulus, with
downsampling to 100Hz.
Reduce epoch length to 6 seconds

get data from the log files about button presses and sort the trials into
left and right
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


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def plot_trial_components_from_detsrm(subject,
                                      datadir,
                                      bidsdir,
                                      figdir,
                                      condition='left-right',
                                      timespan='fulltrial'):
    """
    Fit a deterministic SRM to one subjects data, transform the data with each
    trial's  weights, and plot the data feature-wise for different conditions.
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
    stimulus duration), or 'fulltrial' (entire 7second epoch).
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
        logging.info('Preparing data for fitting a shared response model')
        if epochs.info['sfreq'] > 100:
            # after initial preprocessing, they are downsampled to 200Hz.
            # Downsample further to 100Hz
            epochs.resample(sfreq=100, verbose=True)
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
        # append single subject data to the data dict of the sample
        fullsample[sub] = all_trial_info

    logging.info(f'Fitting shared response models based on data from subjects '
                 f'{subject}')
    features = [5, 7, 10, 15, 20]
    # aggregate the data from the dictionary into a list of lists, as required
    # by brainiak
    data = []
    for subject, trialinfo in fullsample.items():
        for trial_no, info in trialinfo.items():
            data.append(info['normalized_data'])

    for f in features:
        # fit the model
        model = shared_response(data=data,
                                features=f)
        final_df = create_full_dataframe(fullsample, model, data)

        # plot individual features
        plot_srm_model(df=final_df,
                       nfeatures=f,
                       figdir=figdir,
                       subject='group',
                       mdl='det-srm',
                       cond=condition,
                       timespan=timespan)


def create_full_dataframe(fullsample,
                          model,
                          data):
    """
    Create a monstrous pandas dataframe.

    :param fullsample: dict, holds all experiment information
    :param model: Brainiak SRM model, fitted
    :param data: Pandas dataframe with MEG data
    :return:
    """
    # there must be a way to transform the nested dictionary into a data frame,
    # but I have failed so far
    transformed = model.transform(data)
    # add the transformed data into the dict
    for subject, infodict in fullsample.items():
        for trial in infodict.keys():
            infodict[trial]['transformed'] = transformed.pop(0)
    assert transformed == []
    dfs = []
    for subject, infodict in fullsample.items():
        for trial, info in infodict.items():
            df = pd.DataFrame.from_records(info['transformed']).T
            trial_type = info['trial_type']
            trial = trial
            df['trial_type'] = trial_type
            df['trial_no'] = trial
            dfs.append(df)

    finaldf = pd.concat(dfs)
    return finaldf


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
    stimulus duration), or 'fulltrial' (entire 7second epoch).
    :return: all_trial_infos; dict; with trial-wise information
    """
    all_trial_infos = {}
    unique_epochs = df['epoch'].unique()

    if timespan == 'decision':
        # extract the information on decision time for all trials at once.
        trials_to_rts = get_decision_timespan_on_and_offsets(subject=sub,
                                                             bidsdir=bidsdir)

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
    logging.info(f'Fitting a deterministic SRM with {features} features...')
    # fit a deterministic shared response model
    model = srm.DetSRM(features=features)
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
    assert len(brainer) + len(right) + len (left) == len(df['trial_no'].values)
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
                   mdl='det-srm',
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
            # define the timing of significant events in the timecourse of a trial:
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
    idx_right = [epochs.columns.get_loc(s.replace(' ','')) for s in right_chs]
    left_chs = mne.read_vectorview_selection(['Left-occipital'])
    idx_left = [epochs.columns.get_loc(s.replace(' ','')) for s in left_chs]
    idx_left = [186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 234, 235, 236, 237, 238, 239, 246, 247, 248]
    idx_right = [231, 232, 233, 240, 241, 242, 243, 244, 245, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 279, 280, 281, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296]
