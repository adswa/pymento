"""
This is a collection of functions for working with BIDS data read in with
mne-python.
"""
import logging
from mne_bids import (
    read_raw_bids,
    BIDSPath,
)
from mne import events_from_annotations
from pymento_meg.config import event_dict
from pymento_meg.utils import _check_if_bids_directory_exists


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def read_bids_data(bids_root, subject, datatype="meg", task="memento", suffix="meg"):
    """
    Read in a BIDS directory.
    :param subject: str, subject identifier, takes the form '001'
    :param bids_root: str, path to the root of a BIDS directory to read from
    :param datatype: str, descriptor of the data type, e.g., 'meg'
    :param task: str, descriptor of the task name ('memento')
    :param suffix: str, BIDS suffix for the data, e.g., 'meg'
    """
    bids_path = BIDSPath(
        root=bids_root, datatype=datatype, subject=subject, task=task, suffix=suffix
    )
    try:
        # Only now (Apr. 2021) MNE python gained the ability to read in split
        # annexed data. Until this is released and established, we're making
        # sure files are read fully, and if not, we attempt to unlock them first
        raw = read_raw_bids(
            bids_path=bids_path,
            extra_params=dict(on_split_missing="raise"),
        )
    except ValueError:
        logging.warn(
            "Ooops! I can't load all splits of the data. This may be because "
            "you run a version of MNE-python that does not read in annexed "
            "data automatically. I will try to datalad-unlock them for you."
        )
        import datalad.api as dl

        dl.unlock(bids_path.directory)
        raw = read_raw_bids(
            bids_path=bids_path,
            extra_params=dict(on_split_missing="raise"),
        )

    # return the raw data, and also the bids path as it has
    return raw, bids_path


def get_events(raw, event_dict=event_dict):
    """
    Convert the annotations of the raw data into events
    """
    # supply the original event dict, else the events extracted from annotations
    # are ordered and numbered alphabetically
    events, event_dict = events_from_annotations(raw, event_id=event_dict)

    return events, event_dict


def save_derivatives_to_bids_dir(raw_sss, subject, bidsdir, figdir):
    """
    Save data in BIDS format into a new directory.
    This can't be done with mne-bids!
    """

    bids_path = _get_BIDSPath_processed(subject, bidsdir)
    logging.info(
        f"Saving BIDS-compliant signal-space-separated data from subject "
        f"{subject} into "
        f"{bids_path}"
    )
    # save raw fif data and events
    events_data, event_dict = get_events(raw_sss)
    _check_if_bids_directory_exists(bids_path)
    raw_sss.save(bids_path)
    # TODO: use figdir variable


def _get_BIDSPath_processed(subject, bidsdir):
    from pymento_meg.utils import _construct_path

    _construct_path([bidsdir, f"sub-{subject}/"])
    bids_path = BIDSPath(
        subject=subject,
        task="memento",
        root=bidsdir,
        suffix="meg",
        extension=".fif",
        processing="sss",
    )
    return bids_path


def add_durations():
    """
    TODO: add event durations!
    Delay: always 2 seconds
    stimduration = 0.7;
    FeedbackTime= 1;
    """
    return
