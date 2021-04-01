"""
This is a collection of functions for working with BIDS data read in with
mne-python.
"""

from mne_bids import (
    read_raw_bids,
    write_raw_bids,
    BIDSPath,
)
from mne import events_from_annotations
from pymento_meg.config import event_dict

def read_bids_data(bids_root,
                   subject,
                   datatype='meg',
                   task='memento',
                   suffix='meg'):
    """
    Read in a BIDS directory.
    :param subject: str, subject identifier, takes the form '001'
    """
    bids_path = BIDSPath(root=bids_root,
                         datatype=datatype,
                         subject=subject,
                         task=task,
                         suffix=suffix)
    try:
        # Only now (Apr. 2021) MNE python gained the ability to read in split
        # annexed data. Until this is released and established, we're making
        # sure files are read fully, and if not, we attempt to unlock them first
        raw = read_raw_bids(bids_path=bids_path,
                            extra_params=dict(on_split_missing='raise'),
                            )
    except ValueError as e:
        print("Ooops! I can't load all splits of the data. This may be because "
              "you run a version of MNE-python that does not read in annexed "
              "data automatically. I will try to datalad-unlock them for you.")
        import datalad.api as dl
        dl.unlock(bids_path.directory)
        raw = read_raw_bids(bids_path=bids_path,
                            extra_params=dict(on_split_missing='raise'),
                            )

    # return the raw data, and also the bids path as it has
    return raw, bids_path


def get_events(raw):
    """
    Convert the annotations of the raw data into events
    """
    events = events_from_annotations(raw, event_id=event_dict)

    return events


def add_dur_to_annot(raw):
    """
    Add event durations to the annotation
    """
    return


def save_to_bids_dir(raw_sss,
                     subject,
                     bidsdir,
                     figdir):

    bids_path = _get_BIDSPath_processed(subject, bidsdir)
    print(
        f"Saving BIDS-compliant signal-space-separated data from subject "
        f"{subject} into " f"{bids_path}"
    )
    # save raw fif data and events
    events_data, event_dict = _events(raw_sss, subject, figdir)
    write_raw_bids(raw, bids_path, events_data=events_data,
                   event_id=event_dict, overwrite=True)


def _get_BIDSPath_processed(subject, bidsdir):
    from pymento_meg.utils import _construct_path
    _construct_path([bidsdir, f'sub-{subject}/'])
    bids_path = BIDSPath(subject=subject,
                         task='memento',
                         root=bidsdir,
                         suffix='meg',
                         extension='.fif',
                         processing='sss')
    return bids_path


# TODO: add delay durations? It is always 2 seconds
#stimduration = 0.7;
#FeedbackTime= 1;
#delay = 2;

# CAVE: The events are numbered in alphabetical order in the BIDS annotation!