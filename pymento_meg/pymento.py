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

    raw = read_data_original(directory=rawdir,
                             subject=subject,
                             savetonewdir=True,
                             bidsdir=bidsdir,
                             figdir=figdir,
                             crosstalk_file=crosstalk_file,
                             fine_cal_file=fine_cal_file,
                             preprocessing="Raw")


def signal_space_separation(bidspath, subject):
    """
    Reads in the raw data from a bids structured directory.
    :param bidspath:
    :return:
    """


    bids_path = BIDSPath(subject=subject, task='memento', suffix='meg',
                         datatype='meg', root=directory)

    raw = read_raw_bids(bids_path)
    # Events are now Annotations!

    fine_cal_file = bids_path.meg_calibration_fpath
    crosstalk_file = bids_path.meg_crosstalk_fpath


    raw_sss = maxwellfilter(raw=raw,
                            crosstalk_file=crosstalk_file,
                            fine_cal_file=fine_cal_file,
                            subject=subject,
                            headpos_file=None,
                            compute_motion_params=True,
                            head_pos_outdir=bidsdir,
                            figdir=figdir,
                            outdir=bidsdir,
                            filtering=False,
                            filter_args=None,)
