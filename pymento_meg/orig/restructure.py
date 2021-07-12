import mne
import logging

import os.path as op

from mne_bids import (
    write_raw_bids,
    write_meg_calibration,
    write_meg_crosstalk,
    BIDSPath,
)
from pathlib import Path
from glob import glob
from pymento_meg.config import channel_types
from pymento_meg.orig.behavior import write_to_df


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def read_data_original(
    directory,
    subject,
    savetonewdir=False,
    bidsdir=None,
    preprocessing="Raw",
    figdir="/tmp",
    crosstalk_file=None,
    fine_cal_file=None,
    behav_dir=None,
):
    """
    The preprocessed MEG data is split into three files that are quite
    unstructured.
    We read in all files, and concatenate them by hand.
    If needed, we save the raw data in a BIDS compliant way.
    :param directory: path to a subject directory.
    :param subject: str, subject identifier ('001'), used for file names
     and logging
    :param savetonewdir: Boolean, if True, save the data as BIDS conform
    files into a new directory using mnebids
    :param bidsdir: str, Path to where BIDS conform data shall be saved
    :param preprocessing: Data flavour to load. Existing directories are
     'Move_correc_SSS_realigneddefault_nonfittoiso' and 'Raw' (default)
     :param fine_cal_file: str, path to fine_cal file
     :param figdir: str, path to directory for diagnostic plots
     :param crosstalk_file; str, path to crosstalk file
    :return:
    """

    # We're starting with the original data from Luca. The files were
    # transferred from Hilbert as is, and have a non-BIDS and partially
    # inconsistent naming and directory structure
    # First, construct a Path to a preprocessed or Raw directory
    directory = Path(directory) / f"memento_{subject}"
    path = (
        Path(directory) / preprocessing / "*.fif"
        if preprocessing
        else Path(directory) / "*.fif"
    )
    if not op.exists(op.split(path)[0]):
        # some subjects have an extra level of directories
        path = (
            Path(directory) / "*" / preprocessing / "*.fif"
            if preprocessing
            else Path(directory) / "*" / "*.fif"
        )
    logging.info(f"Reading files for subject sub-{subject} from {path}.")
    # file naming is a mess. We need to make sure to sort the three files
    # correctly
    unsorted_files = glob(str(path))
    first, second, third = _get_first_file(unsorted_files)
    if subject not in ["001", "005"]:
        # subjects one and five don't have consistent subject identifiers or
        # file names.
        # automatic reading in would only load the first.
        logging.info(f"reading in subject {subject}")
        try:
            raw = mne.io.read_raw_fif(first)
        except ValueError:
            logging.warning(
                f"WARNING Irregular file naming. "
                f"Will read files in sequentially "
                f"in the following order: {first}{second}{third}"
            )
            # read the splits
            split1 = mne.io.read_raw_fif(first, on_split_missing="warn")
            split2 = mne.io.read_raw_fif(second, on_split_missing="warn")
            split3 = mne.io.read_raw_fif(third, on_split_missing="warn")
            # concatenate all three split files
            raw = mne.concatenate_raws([split1, split2, split3])
    else:
        # read the splits
        split1 = mne.io.read_raw_fif(first, on_split_missing="warn")
        split2 = mne.io.read_raw_fif(second, on_split_missing="warn")
        split3 = mne.io.read_raw_fif(third, on_split_missing="warn")
        # concatenate all three split files
        raw = mne.concatenate_raws([split1, split2, split3])
    # explicitly set channel types to EOG and ECG sensors
    raw.set_channel_types(channel_types)

    if savetonewdir:
        if not bidsdir:
            logging.info(
                "I was instructed to save BIDS conform raw data into a"
                "different directory, but did not get a path."
            )
            return raw

        save_bids_data(
            raw=raw,
            subject=subject,
            bidsdir=bidsdir,
            figdir=figdir,
            ctfile=crosstalk_file if crosstalk_file else None,
            fcfile=fine_cal_file if fine_cal_file else None,
            behav_dir=behav_dir,
        )
    return raw


def save_bids_data(raw, subject, bidsdir, figdir, ctfile, fcfile, behav_dir):
    """
    Saves BIDS-structured data subject-wise
    :param raw: raw fif data
    :param subject: str, subject identifier
    :param bidsdir: str, path to where a bids-conform output dir shall exist
    :param figdir: str, path to diagnostic plot directory
    :param ctfile: str, path to crosstalk file
    :param fcfile: str, path to fine cal file
    :return:
    """

    bids_path = _get_BIDSPath(subject, bidsdir)
    logging.info(
        f"Saving BIDS-compliant raw data from subject "
        f"{subject} into "
        f"{bids_path}"
    )
    # a workaround of https://github.com/mne-tools/mne-bids/issues/718
    if subject in ["001", "005"]:
        # these subjects can't be written directly - mne-bids would perform an
        # internal check that fails on the basis that the hand-concatenated
        # files are longer than the single split file that would be read-in
        # automatically.
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(suffix="_raw.fif", delete=False) as f:
            fname = f.name
            raw.save(fname, overwrite=True)
            raw = mne.io.read_raw_fif(fname, preload=False)

    # save raw fif data and events
    # get log files, but don't yet write them out as the path doesn't exist yet
    df = write_to_df(
        participant=subject,
        behav_dir=behav_dir,
        bids_dir=bids_path.directory,
        write_out=False,
    )
    events_data, event_dict = _events(raw, subject, figdir, df)
    write_raw_bids(
        raw, bids_path, events_data=events_data, event_id=event_dict, overwrite=True
    )

    # save crosstalk and calibration file
    _elektas_extras(crosstalk_file=ctfile, fine_cal_file=fcfile, bids_path=bids_path)
    # write out log files
    df = write_to_df(
        participant=subject,
        behav_dir=behav_dir,
        bids_dir=bids_path.directory,
        write_out=True,
    )


def _get_BIDSPath(subject, bidsdir):
    from pymento_meg.utils import _construct_path

    _construct_path([bidsdir, f"sub-{subject}/"])
    bids_path = BIDSPath(
        subject=subject, task="memento", root=bidsdir, suffix="meg", extension=".fif"
    )
    return bids_path


def _events(raw, subject, figdir, df=None):
    from pymento_meg.utils import eventreader, event_dict

    events = eventreader(
        raw=raw, subject=subject, event_dict=event_dict, outputdir=figdir, df=df
    )
    return events, event_dict


def _elektas_extras(crosstalk_file, fine_cal_file, bids_path):
    """
    Save the crosstalk and fine-calibration files
    :param crosstalk_file: str, path to crosstalk file
    :param fine_cal_file: str, path to fine_cal file
    :param bids_path: mne-bids BIDSPath instance
    :return:
    """

    write_meg_calibration(fine_cal_file, bids_path)
    write_meg_crosstalk(crosstalk_file, bids_path)


def _get_first_file(files):
    """
    Helper function to return the first split of a range of files.
    This is necessary because the file names are inconsistent across subjects.
    This function should return file names of any preprocessing flavor or Raw
    directory in the correct order for reading in.

    :param files: list of str, with file names
    :return:

    """
    first, second, third = None, None, None

    # basic sanity check:
    # there should be three fif files
    assert len(files) == 3
    # check if there are any files starting with a number
    starts_with_digit = [op.basename(f)[0].isdigit() for f in files]
    if not any(starts_with_digit):
        # phew, we're not super bad
        for f in files:
            # only check the filenames
            base = op.basename(f)
            if len(base.split("-")) == 2 and base.split("-")[-1].startswith("1"):
                second = f
            elif len(base.split("-")) == 2 and base.split("-")[-1].startswith("2"):
                third = f
            elif len(base.split("-")) == 1:
                first = f
            else:
                # we shouldn't get here
                raise ValueError(f"Cannot handle file list {files}")
    else:
        # at least some file names start with a digit
        if all(starts_with_digit):
            # this is simple, files start with 1_, 2_, 3_
            first, second, third = sorted(files)
        else:
            # only some file names start with a digit. This is really funky.
            for f in files:
                base = op.basename(f)
                if base[0].isdigit() and base[0] == "1" and len(base.split("-")) == 1:
                    first = f
                elif base[0].isdigit() and base[0] == "2" and len(base.split("-")) == 1:
                    second = f
                elif base[0].isdigit() and base[0] == "2" and len(base.split("-")) == 2:
                    if base.split("-")[-1].startswith("1"):
                        second = f
                    elif base.split("-")[-1].startswith("2"):
                        third = f
                elif len(base.split("-")) == 2 and base[0].isalpha():
                    if base.split("-")[-1].startswith("1"):
                        second = f
                    elif base.split("-")[-1].startswith("2"):
                        third = f
                else:
                    # this shouldn't happen
                    raise ValueError(f"Cannot handle file list {files}")
    # check that all files are defined
    assert all([v is not None for v in [first, second, third]])
    logging.info(f"Order the files as follows: {first}, {second}, {third}")
    return first, second, third
