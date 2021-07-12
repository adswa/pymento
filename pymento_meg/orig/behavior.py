"""
In the memento task, the behavioral responses of participants were written to
log files.
However, different participants played different versions of the task, and
different versions of the task saved a different amount of variables as a
Matlab struct into the log file.
This file contains information on the variables and their indexes per subject.
Indexing is done according to Python, i.e., zero-based.
"""
import logging
from pymento_meg.config import subjectmapping
from scipy.io import loadmat
from pathlib import Path
import pandas as pd


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def get_behavioral_data(subject, behav_dir, fieldname, variable=None):
    """
    Read in behavioral data and return the values of one variable.
    :param subject:
    :param behav_dir: Path to the directory that contains subject-specific
    log directories (e.g., data/DMS_MEMENTO/Data_Behav/Data_Behav_Memento)
    :param fieldname: Fieldname where the variable is in. Can be "probmagrew",
    "single_onset", "disptimes", or "onset"
    :param variable: str, variable name that should be retrieved. If None is
    specified, it will get all variables of this fieldname
    :return:
    """

    key = f"memento_{subject}"
    logging.info(f"Reading in experiment log files of {key} for {fieldname}...")
    # get the information about the subject's behavioral data out of the subject
    # mapping, but make sure it is actually there first
    assert key in subjectmapping.keys()
    subinfo = subjectmapping[key]
    # based on the information in subinfo, load the file
    fname = subinfo["logfilename"]
    path = Path(behav_dir) / fname
    res = loadmat(path)
    # get the subject ID out of the behavioral data struct. It is buried quite
    # deep, and typically doesn't have a leading zero
    subID = str(res["mementores"]["subID"][0][0][0][0])
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
        # for all relevant fieldnames, it takes two [0] indices to get to an
        # unnested matrix of all variables
        wanted_var = res["mementores"][fieldname][0][0][idx]
        return wanted_var
    else:
        return res["mementores"][fieldname][0][0]


def write_to_df(participant, behav_dir, bids_dir, write_out=False):
    """
    Write logfile data to a dataframe to get rid of the convoluted matlab
    structure.
    All variables should exist 510 times.
    :param: str, subject identifier in the form of "001"
    """

    # read the data in as separate dataframes
    # Onset times are timestamps! View with datetime
    # first, get matlab data
    onsets = get_behavioral_data(
        subject=participant, behav_dir=behav_dir, fieldname="onsets"
    )
    disps = get_behavioral_data(
        subject=participant, behav_dir=behav_dir, fieldname="disptimes"
    )
    probs = get_behavioral_data(
        subject=participant, behav_dir=behav_dir, fieldname="probmagrew"
    )
    # we need to transpose the dataframe to get variables as columns and
    # trials as rows
    df_onsets = pd.DataFrame(onsets).transpose()
    df_disps = pd.DataFrame(disps).transpose()
    df_probs = pd.DataFrame(probs).transpose()
    # set header:
    df_onsets.columns = subjectmapping[f"memento_{participant}"]["onsets"]
    df_disps.columns = subjectmapping[f"memento_{participant}"]["disptimes"]
    df_probs.columns = subjectmapping[f"memento_{participant}"]["probmagrew"]
    # assert that all series are monotonically increasing in onsets, but skip
    # Series with NaNs:
    assert all(
        [
            df_onsets[i].is_monotonic
            for i in df_onsets.columns
            if not df_onsets[i].isna().values.any()
        ]
    )
    assert all([d.shape[0] == 510 for d in [df_disps, df_onsets, df_probs]])

    # concatenate the dataframes to one
    df = pd.concat([df_disps, df_onsets, df_probs], axis=1)

    if write_out:
        # write dataframe to file. bids_dir should be a Path object
        fname = bids_dir / f"sub-{participant}_task-memento_log.tsv"
        df.to_csv(fname, sep="\t", index=False)
    return df


def read_bids_logfile(subject, bidsdir):
    """
    Read the log file data from one subject from their BIDS directory
    :param subject: str, a subject identifier such as '001'
    :param bidsdir: str, a path to the root of the BIDS directory
    :return: df, Pandas Dataframe, containing log file data
    """
    # construct name of the first split
    fname = Path(bidsdir) / f'sub-{subject}/meg' / \
            f'sub-{subject}_task-memento_log.tsv'
    logging.info(f'Reading in log file from subject sub-{subject} from the path'
          f' {fname}...')
    # read in as a dataframe and return
    df = pd.read_csv(fname, sep='\t')
    return df
