"""
In the memento task, the behavioral responses of participants were written to
log files.
However, different participants played different versions of the task, and
different versions of the task saved a different amount of variables as a
Matlab struct into the log file.
This file contains information on the variables and their indexes per subject.
Indexing is done according to Python, i.e., zero-based.
"""

from scipy.io import loadmat
from pathlib import Path


smaller_onsets = ['fix_onset', 'LoptOnset', 'or_onset', 'RoptOnset',
                  'response_onset', 'feedback_onset']
larger_onsets = ['fix_onset', 'pause_start', 'LoptOnset', 'or_onset',
                 'RoptOnset', 'response_onset', 'feedback_onset',
                 'Empty_screen', 'timeoutflag']
largest_onsets = ['fix_onset', 'pause_start', 'LoptOnset', 'or_onset',
                  'second_delay_screen', 'RoptOnset', 'response_onset',
                  'feedback_onset', 'Empty_screen', 'timeoutflag']
larger_probmagrew = ['trial_no', 'LoptProb', 'LoptMag', 'RoptProb', 'RoptMag',
                     'LoptRew', 'RoptRew', 'choice', 'RT', 'points',
                     'pointdiff', 'timeoutflag', 'breaktrial']
smaller_probmagrew = ['trial_no', 'LoptProb', 'LoptMag', 'RoptProb', 'RoptMag',
                      'LoptRew', 'RoptRew', 'choice', 'RT', 'points',
                      'pointdiff', 'breaktrial']
disptimes = ['trial_no', 'FixReqT', 'FixTime', 'orReqTime', 'orTime', 'LoptT',
             'RoptT', 'FeedbackT']
single_onsets = ['empty_start_screen', 'start', 'instruction_onset', 'end_onset']

subjectmapping = {
    'memento_001': {
        'probmagrew': larger_probmagrew,
        'onsets': smaller_onsets,
        'single_onsets': False,
        'disptimes': disptimes,
        'logfilename': 'memento_001/mementoLOG_1.mat'
        },
    'memento_002': {
        'probmagrew': larger_probmagrew,
        'onsets': smaller_onsets,
        'single_onsets': False,
        'disptimes': disptimes,
        'logfilename': 'memento_002/mementoLOG_2.mat'
    },
    'memento_003': {
        'probmagrew': larger_probmagrew,
        'onsets': smaller_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_003/mementoLOG_3.mat'
    },
    'memento_004': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_004/mementoLOG_4.mat'
    },
    'memento_005': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_005/mementoLOG_5.mat'
    },
    'memento_006': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_006/mementoLOG_6.mat'
    },
    'memento_007': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_007/mementoLOG_7.mat'
    },
    'memento_008': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_008/mementoLOG_8.mat'
    },
    'memento_009': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_009/mementoLOG_9.mat'
    },
    'memento_010': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_010/mementoLOG_10.mat'
    },
    'memento_011': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_011/mementoLOG_11.mat'
    },
    'memento_012': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_012/mementoLOG_12.mat'
    },
    'memento_013': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_013/mementoLOG_13.mat'
    },
    'memento_014': {
        'probmagrew': smaller_probmagrew,
        'onsets': larger_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_014/mementoLOG_14.mat'
    },
    'memento_015': {
        'probmagrew': smaller_probmagrew,
        'onsets': largest_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_0015/mementoLOG_15.mat'
    },
    'memento_016': {
        'probmagrew': smaller_probmagrew,
        'onsets': largest_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_0016/mementoLOG_16.mat'
    },
    'memento_017': {
        'probmagrew': smaller_probmagrew,
        'onsets': largest_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_0017/mementoLOG_17.mat'
    },
    'memento_018': {
        'probmagrew': smaller_probmagrew,
        'onsets': largest_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_0018/mementoLOG_18.mat'
    },
    'memento_019': {
        'probmagrew': smaller_probmagrew,
        'onsets': largest_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_0019/mementoLOG_19.mat'
    },
    'memento_020': {
        'probmagrew': smaller_probmagrew,
        'onsets': largest_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_0020/mementoLOG_20.mat'
    },
    'memento_021': {
        'probmagrew': smaller_probmagrew,
        'onsets': largest_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_0021/mementoLOG_21.mat'
    },
    'memento_022': {
        'probmagrew': smaller_probmagrew,
        'onsets': largest_onsets,
        'single_onsets': True,
        'disptimes': disptimes,
        'logfilename': 'memento_0022/mementoLOG_22.mat'
    }
}


def get_behavioral_data(subject,
                        behav_dir,
                        fieldname,
                        variable=None):
    """
    Read in behavioral data and return the values of one variable.
    :param subject:
    :param subjectmapping:
    :param behav_dir: Path to the directory that contains subject-specific
    log directories (e.g., data/DMS_MEMENTO/Data_Behav/Data_Behav_Memento)
    :param fieldname: Fieldname where the variable is in. Can be "probmagrew",
    "single_onset", "disptimes", or "onset"
    :param variable: str, variable name that should be retrieved. If None is
    specified, it will get all variables of this fieldname
    :return:
    """

    key = f'memento_{subject}'
    print(f"Reading in experiment log files of {key} for {fieldname}...")
    # get the information about the subject's behavioral data out of the subject
    # mapping, but make sure it is actually there first
    assert key in subjectmapping.keys()
    subinfo = subjectmapping[key]
    # based on the information in subinfo, load the file
    fname = subinfo['logfilename']
    path = Path(behav_dir) / fname
    res = loadmat(path)
    # get the subject ID out of the behavioral data struct. It is buried quite
    # deep, and typically doesn't have a leading zero
    subID = str(res['mementores']['subID'][0][0][0][0])
    assert subID in subject
    # first, retrieve all possible variables given the fieldname
    var = subinfo[fieldname]
    if variable:
        # make sure the required variable is inside this list
        assert variable in var
        # get the index from the required variable. This is necessary to index
        # the struct in the right place. Only fieldnames seem to be indexable
        # by name, not their variables
        idx = var.index(variable)
        # for all relevant fieldnames, it takes two [0] indices to get to an unnested
        # matrix of all variables
        wanted_var = res['mementores'][fieldname][0][0][idx]
        return wanted_var
    else:
        return res['mementores'][fieldname][0][0]


def write_to_df(participant,
                behav_dir,
                bids_dir
                ):
    """
    Write logfile data to a dataframe to get rid of the convoluted matlab
    structure.
    All variables should exist 510 times.
    :param: str, subject identifier in the form of "001"
    """
    import pandas as pd
    # read the data in as separate dataframes
    # Onset times are timestamps! View with datetime
    # first, get matlab data
    onsets = get_behavioral_data(subject=participant,
                                 behav_dir=behav_dir,
                                 fieldname='onsets')
    disps = get_behavioral_data(subject=participant,
                                behav_dir=behav_dir,
                                fieldname='disptimes')
    probs = get_behavioral_data(subject=participant,
                                behav_dir=behav_dir,
                                fieldname='probmagrew')
    # we need to transpose the dataframe to get variables as columns and
    # trials as rows
    df_onsets = pd.DataFrame(onsets).transpose()
    df_disps = pd.DataFrame(disps).transpose()
    df_probs = pd.DataFrame(probs).transpose()
    # set header:
    df_onsets.columns = subjectmapping[f'memento_{participant}']['onsets']
    df_disps.columns = subjectmapping[f'memento_{participant}']['disptimes']
    df_probs.columns = subjectmapping[f'memento_{participant}']['probmagrew']
    # assert that all series are monotonically increasing in onsets, but skip
    # Series with NaNs:
    assert all([df_onsets[i].is_monotonic for i in df_onsets.columns if not
                df_onsets[i].isna().values.any()])
    assert all([d.shape[0] == 510 for d in [df_disps, df_onsets, df_probs]])

    # concatenate the dataframes to one
    df = pd.concat([df_disps, df_onsets, df_probs], axis=1)

    # write dataframe to file. bids_dir should be a Path object
    fname = bids_dir / f'sub-{participant}_task-memento_events-log.tsv'
    df.to_csv(fname, sep='\t', index=False)
