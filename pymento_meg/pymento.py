"""Main module."""

import mne
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from glob import glob
from matplotlib import interactive
from pathlib import Path
from pymento_meg.orig.restructure import (
    read_data_original,
    )
from pymento_meg.proc.preprocess import (
    maxwellfilter
    )
from mne_bids import (
    read_raw_bids,
    BIDSPath
    )


def restructure_to_bids(rawdir,
                        subject,
                        bidsdir,
                        figdir,
                        crosstalk_file,
                        fine_cal_file):
    """
    Transform the original memento MEG data into something structured.
    :return:
    """

    print(f"Starting to restructure original memento data into BIDS for"
          f"subject sub-{subject}.")

    raw = read_data_original(directory=rawdir,
                             subject=subject,
                             savetonewdir=True,
                             bidsdir=bidsdir,
                             figdir=figdir,
                             crosstalk_file=crosstalk_file,
                             fine_cal_file=fine_cal_file,
                             preprocessing="Raw")


def signal_space_separation(bidspath,
                            subject,
                            figdir,
                            derived_path):
    """
    Reads in the raw data from a bids structured directory, applies a basic
    signal space separation with motion correction, and saves the result in a
    derivatives BIDS directory
    :param bidspath:
    :param subject: str, subject identifier, e.g., '001'
    :param figdir: str, path to a diagnostics directory to save figures into
    :param derived_path: str, path to where a derivatives dataset with sss data
    shall be saved
    :return:
    """
    print(f"Starting to read in raw memento data from BIDS directory for"
          f"subject sub-{subject}.")

    bids_path = BIDSPath(subject=subject, task='memento', suffix='meg',
                         datatype='meg', root=bidspath)

    raw = read_raw_bids(bids_path)
    # Events are now Annotations!

    fine_cal_file = bids_path.meg_calibration_fpath
    crosstalk_file = bids_path.meg_crosstalk_fpath

    print(f"Starting signal space separation with motion correction "
          f"for subject sub{subject}.")

    raw_sss = maxwellfilter(raw=raw,
                            crosstalk_file=crosstalk_file,
                            fine_cal_file=fine_cal_file,
                            subject=subject,
                            headpos_file=None,
                            compute_motion_params=True,
                            head_pos_outdir=None,  # isn't saved atm
                            figdir=figdir,
                            outdir=derived_path,
                            filtering=False,
                            filter_args=None,)
