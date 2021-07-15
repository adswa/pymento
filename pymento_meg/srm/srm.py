import mne
import logging
import numpy as np
import pandas as pd

from brainiak.funcalign import srm
from pathlib import Path
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
    :return:
    """
    # initialize a list to hold data over all subjects in the model
    alldata = []
    allassoc = {}
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
        # find out which arrays belong to a desired condition. The indices in assoc
        # should correspond to indices of the data list.
        assoc, mapping = _find_data_of_choice(epochs=epochs,
                                              subject=sub,
                                              bidsdir=bidsdir,
                                              condition=condition,
                                              df=df)
        if timespan == 'decision':
            # we need to get behavioral data
            behav = read_bids_logfile(subject=sub,
                                      bidsdir=bidsdir)
            df, data, assoc = _prep_for_srm_decision(df=df,
                                                     behav=behav,
                                                     epochs=epochs,
                                                     mapping=mapping,
                                                     assoc=assoc)

        else:
            df, data = _prep_for_srm(df,
                                     timespan=timespan)
        alldata.extend(data)
        allassoc[sub] = assoc


    features = [5, 7, 10, 15, 20]
    for f in features:
        model, data = shared_response(data=data,
                                      features=f)
        final_df = concatenate_transformations(model,
                                               data,
                                               assoc,
                                               condition=condition)
        # plot individual features
        plot_srm_model(df=final_df,
                       nfeatures=f,
                       figdir=figdir,
                       subject=sub,
                       mdl='det-srm',
                       cond=condition,
                       timespan=timespan)


def shared_response(data,
                    features):
    """
    Compute a shared response model from a list of trials
    :param epochs: Epoch object, cleaned epochs in FIF format
    :param features
    :return:
    """
    logging.info(f'Fitting a deterministic SRM with {features} features...')
    # fit a deterministic shared response model
    model = srm.DetSRM(features=features)
    model.fit(data)
    return model, data


def _prep_for_srm_decision(df, behav, epochs, mapping, assoc):
    """
    This function does the same as _prep_for_srm for time spans that should be
    centered at the time of decision making.
    Its an extra function because it does the heavy lifting of extracting
    decision times and slicing the data based on the decision times dynamically.
    It then also strips any removed data from the trial-condition assocaciations.
    This function is not elegant, it needs a good chunk of testing and future
    refactoring to ensure it does not return gibberish that looks sane
    :param df:
    :param timespan:
    :param behav:
    :param epochs:
    :param mapping:
    :param assoc
    :return:
    """

    # get all the reaction times for the trial numbers we have data for
    assert behav is not None
    trials_and_rts = [behav[behav['trial_no'] == e][['trial_no', 'RT']].values
                      for e in epochs.metadata.trial_no.values]
    trials_and_rts = np.concatenate(trials_and_rts)
    logging.info(f'The average reaction time was '
                 f'{np.nanmean(trials_and_rts[:,1])}')
    # mark nans with larger RTs
    logging.info('Setting nan reaction times to implausibly large values.')
    np.nan_to_num(trials_and_rts, nan=100, copy=False)
    # kick out trial numbers where reaction times are too large to fit into
    # the trial. For now, this is at 3 seconds.
    # collect the trial numbers
    trials_to_remove = trials_and_rts[np.where(
        trials_and_rts[:, 1] > 3)][:, 0]
    # and also their indices in the data
    idx = np.where(np.in1d(list(epochs.metadata.trial_no), trials_to_remove))[0]
    assert len(trials_to_remove) == len(idx)
    removal = {'trial_no': list(trials_to_remove),
               'idx': list(idx)}
    remaining_epochs = [e for e in epochs.metadata.trial_no.values
                        if e not in trials_to_remove]
    RTs = [behav.RT[behav.trial_no == e].values
           for e in remaining_epochs]
    # unravel
    RTs = np.concatenate(RTs).ravel()
    assert len(RTs) == len(df['epoch'].unique()) - len(trials_to_remove)
    # the reaction times are in seconds, we can convert them into
    # milliseconds to fit the sampling rate of the data. We should
    # make sure that we're really at 100Hz
    assert epochs.info['sfreq'] == 100
    RTs = RTs * 100
    # Now, add RT to the end of the second visual stimulus to get the
    # decision time from trial onset
    # (70 + 200 + 70 = 340)
    decision_time = RTs + 340
    if trials_to_remove is not None:
        logging.info(f"The following trials showed decision times that "
                     f"were later than 7 seconds after trial start, or no "
                     f"decision at all: "
                     f"{trials_to_remove}")
    # readiness potential is reported to start 400 to 500ms before movement
    # in motor areas, let's select 400ms before and after decision
    # this means we need a slice for each trial. To ensure consistent length
    # we round down
    slices = [[int(np.floor(t - 40)), int(np.floor(t + 40))]
              for t in decision_time]
    assert all([s[1] - s[0] == 80 for s in slices])
    # now we can slice the data
    # use the association of epochs to trial numbers
    existing_epochs = [k for k in mapping.keys()
                       if mapping[k] not in trials_to_remove]
    # associate epochs with slices:
    assert len(existing_epochs) == len(slices)
    epochs_to_slices = list(zip(existing_epochs, slices))
    data = [df.loc[df['epoch'] == e, 'MEG0111':'MEG2643'][s[0]:s[1]].values.T
            for e, s in epochs_to_slices]
    assert len(data) == len(trials_and_rts) - len(trials_to_remove)
    assert all([d.shape[1] == 80 for d in data])
    # remove all removed trials from assoc
    for k, v in assoc.items():
        logging.info("Removing trial data for invalid trials from sample.")
        matches = set(np.intersect1d(list(v['trial_no']), trials_to_remove))
        if len(matches) != 0:
            # we have found trials in that have gotten removed. Get their index
            # to remove them from the trial_no and idx list
            remove_idx = np.where(np.in1d(v['trial_no'], list(matches)))[0]
            new_trial_no = [t for idx, t in enumerate(v['trial_no'])
                            if idx not in remove_idx]
            assert len(new_trial_no) == len(v['trial_no']) - len(remove_idx)
            new_idx = [i for idx, i in enumerate(v['idx'])
                       if idx not in remove_idx]
            assert len(new_idx) == len(new_trial_no)
            # overwrite the previous lists
            assoc[k]['trial_no'] = new_trial_no
            assoc[k]['idx'] = new_idx

    return df, data, assoc


def _prep_for_srm(df, timespan):
    """
    Prepare the data for computing a shared response model.
    This function reads in epochs in a dataframe, and returns
    the data as a list of sensor x time points arrays.
    :param timespan: Str, an indication of the time span that should become a
    part of the data. Possible choices: 'fulltrial' (all data), 'firststim'
    (the duration of the first stimulus), 'decision' (time around the motor
    response)
    :return: data; list of arrays
    """
    # create a list of arrays from the dataframe: each array consists of the
    # data of one trial (a unique epoch in the sample), for each sensor.
    if timespan == 'firststim':
        # let's cut the data to the required time span
        # first 70 milliseconds
        data = [df.loc[df['epoch'] == e, 'MEG0111':'MEG2643'][:70].values.T
                for e in df['epoch'].unique()]
        assert all([d.shape[1] == 70 for d in data])
    elif timespan == 'fulltrial':
        # get all data
        data = [df.loc[df['epoch'] == e, 'MEG0111':'MEG2643'].values.T
                for e in df['epoch'].unique()]
    assert len(data) > 0
    # make sure that the first dimension is the number of sensors
    assert all([d.shape[0] == 306 for d in data])
    # return the dataframe, and the data representation as a list
    return df, data


def _find_data_of_choice(df, epochs, subject, bidsdir, condition):
    """
    Based on a condition that can be queried from the log files (e.g., right or
    left choice of stimulus), return the indices that the respective epochs have
    in the list of sensor x time points arrays.
    :param epochs: epochs object
    :param df: pandas dataframe of epochs
    :param subject:
    :param bidsdir:
    :param condition: str, a condition description. Must be one of 'left-right'
    (for trials with right or left choice), 'nobrain-brain' (for trials with
    no-brainer decisions versus actual decisions)
    :return: assoc; dictionary with condition - index associations
    """
    if condition == 'left-right':
        logging.info('Attempting to retrieve trial information for left and right '
              'stimulus choices')
        choices = get_leftright_trials(subject=subject,
                                       bidsdir=bidsdir)
    elif condition == 'nobrain-brain':
        logging.info('Attempting to retrieve trial information for no-brainer and '
              'brainer trials')
        choices = get_nobrainer_trials(subject=subject,
                                       bidsdir=bidsdir)
    # Create a map between epoch labels and trial numbers based on metadata.
    # Horray to dict comprehensions!
    mapping = {key: value for (key, value) in
               zip(df['epoch'].unique(), epochs.metadata.trial_no.values)}
    assert all([i in df['epoch'].unique() for i in mapping.keys()])
    assert len(df['epoch'].unique()) == len(mapping.keys())

    # initialize a dictionary that holds the condition-epochindex associations
    assoc = {}
    total_trials = 0
    for cond, trials in choices.items():
        # find the association of trial numbers to epochs that have been kept
        # by generating the intersection of trials that match the condition and
        # trials in the epoch data frame.
        # CAVE: The trials are not the epoch numbers - we first need to map the
        # trial info from the metadata to the epoch names!
        fit = np.intersect1d(list(mapping.values()), trials)
        # ... and getting their index
        # CAVE: The input should be ordered (monotonically increasing) so that
        # returned indices match the order of the trials existent in epochs
        assert all(x < y for x, y in zip(fit, fit[1:]))
        idx = np.where(np.in1d(list(mapping.values()), fit))[0]
        assert len(idx) == len(fit)
        assoc[cond] = {'idx': idx,
                       'trial_no': fit}
        # the trial number can't be larger than the index
        assert all(x <= y for x, y in zip(
             assoc[cond]['idx'],
             assoc[cond]['trial_no']))
        total_trials += len(idx)
        logging.info(f"Here's my count of matching events in the SRM data for"
              f" condition {cond}: {len(idx)}")
    # does the number of conditions match the number of trials?
    # may not work all of the time
    # assert total_trials == len(epochs)

    return assoc, mapping


def concatenate_transformations(model,
                                data,
                                assoc,
                                condition='left-right'):
    """Transform the trial data with the SRM model transformation matrix.
    Attach trial labels to each trial data
    :returns df: pandas dataframe, a DataFrame with all transformed trial data
    and a trial association
    """
    transformed = model.transform(data)
    dfs = []

    ### TODO:
    # This function is wrong if trials have been removed. It relies on the index
    # of a list or trial, and this is 100%error prone. The whole thing has to be
    # refactored.
    if condition == 'left-right':
        left = assoc['left (1)']['idx']
        right = assoc['right (2)']['idx']

        for idx, l in enumerate(transformed):
            df = pd.DataFrame.from_records(l).T
            trial_type = 'left' if idx in left else 'right' if idx in right else None
            # trial type can be None! I assume this happens in trials where no
            # button press was made
            if trial_type == None:
                logging.info(f'Could not find a matching condition for trial '
                             f'ID {idx}')
            df['trialtype'] = trial_type
            df['trial'] = idx
            dfs.append(df)

    elif condition == 'nobrain-brain':
        brainer = assoc['brainer']['idx']
        nobrainerleft = assoc['nobrainer_left']['idx']
        nobrainerright = assoc['nobrainer_right']['idx']

        for idx, l in enumerate(transformed):
            df = pd.DataFrame.from_records(l).T
            if idx in brainer:
                trial_type = 'brainer'
            elif idx in nobrainerleft:
                trial_type = 'nobrainer'
            elif idx in nobrainerright:
                trial_type = 'nobrainer'
            else:
                trial_type = None

            # trial type can be None! e.g., if the participant did not press the
            # right button in nobrainer trials
            if trial_type is None:
                logging.info(f'Could not find a matching condition for trial {idx}')
            df['trialtype'] = trial_type
            df['trial'] = idx
            dfs.append(df)
    else:
        raise NotImplementedError(f'The condition {condition} is not '
                                  f'implemented.')
    df = pd.concat(dfs)
    return df


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

    import pylab
    import seaborn as sns
    for i in range(nfeatures):
        fname = Path(figdir) / f'sub-{subject}/meg' /\
                     f'sub-{subject}_{mdl}_{nfeatures}-feat_task-{cond}_model-{timespan}_comp_{i}.png'
        if cond == 'left-right':
            fig = sns.lineplot(data=df[df['trialtype'] == 'right'][i])
            sns.lineplot(data=df[df['trialtype'] == 'left'][i])
        elif cond == 'nobrain-brain':
            fig = sns.lineplot(data=df[df['trialtype'] == 'brainer'][i])
            sns.lineplot(data=df[df['trialtype'] == 'nobrainer'][i])
        if timespan == 'fulltrial':
            # define the timing of significant events in the timecourse of a trial:
            # onset and offset of visual stimuli
            events = [0, 70, 270, 340]
            # plot horizontal lines to mark the end of visual stimulation
            [pylab.axvline(ev, linewidth=1, color='grey', linestyle='dashed')
             for ev in events]
        if timespan == 'decision':


        # add a legend
        if cond == 'left-right':
            fig.legend(title='Condition', loc='upper left',
                       labels=['left choice', 'right choice'])
        elif cond == 'nobrain-brain':
            fig.legend(title='Condition', loc='upper left',
                       labels=['brainer', 'nobrainer'])
        plot = fig.get_figure()
        plot.savefig(fname)
        plot.clear()


def get_leftright_trials(subject, bidsdir):
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


def get_nobrainer_trials(subject, bidsdir):
    """
    Return the trials where a decision is a "nobrainer", a trial where both the
    reward probability and magnitude of one option is higher than that of the
    other option.
    :return:
    """
    df = read_bids_logfile(subject=subject,
                           bidsdir=bidsdir)
    # where is the both Magnitude and Probability of reward greater for one
    # option over the other
    right = df['trial_no'][(df.RoptMag > df.LoptMag) &
                           (df.RoptProb > df.LoptProb)].values
    left = df['trial_no'][(df.LoptMag > df.RoptMag) &
                          (df.LoptProb > df.RoptProb)].values
    brainer = [i for i in df['trial_no'].values if i not in right and i not in left]
    # brainer and no-brainer trials should add up
    assert len(brainer) + len(right) + len (left) == len(df['trial_no'].values)
    # make sure that right and left no brainers do not intersect - if they have
    # common values, something went wrong
    assert not bool(set(right) & set(left))
    # make sure to only take those no-brainer trials where participants actually
    # behaved as expected
    consistent_nobrainer_left = [trial for trial in left if
                                 df['choice'][df['trial_no'] == trial].values == 1]
    consistent_nobrainer_right = [trial for trial in right if
                                  df['choice'][df['trial_no'] == trial].values == 2]
    logging.info(f"Subject {subject} underwent a total of {len(right)} no-brainer "
          f"trials for right choices, and a total of {len(left)} no-brainer "
          f"trials for left choices. The subject chose consistently according "
          f"to the no-brainer nature of the trial in "
          f"N={len(consistent_nobrainer_right)} cases for right no-brainers, "
          f"and in N={len(consistent_nobrainer_left)} cases for left "
          f"no-brainers.")
    # create a dictionary with brainer and no-brainer trials. We leave out all
    # no-brainer trials where the participant hasn't responded in accordance to
    # the no-brain nature of the trial
    choices = {'brainer': brainer,
               'nobrainer_left': consistent_nobrainer_left,
               'nobrainer_right': consistent_nobrainer_right}

    return choices


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



def get_nobrain_brain_trials():
    return


# TODO: preprocess all files starting from first stimulus, with downsampling to 100Hz.
# Reduce epoch length to 6 seconds

# get data from the log files about button presses and sort the trials into left and right
# write all shared responses into a matrix, check for consistent correlation pattern across subjects