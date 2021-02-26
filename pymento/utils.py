#!/usr/bin/env python

import mne
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from glob import glob
from matplotlib import interactive
from pathlib import Path

# plotting settings
interactive(True)
import matplotlib
matplotlib.use('Qt5Agg')
mne.set_log_level('info')

# Set a few global variables
# The data is not preconditioned unless this variable is reset
preconditioned = False

from .config import (channel_types,
                     reject_criteria,
                     flat_criteria,
                     crosstalk_file,
                     fine_cal_file,
                     subject_list
                     )

# Define data processing functions and helper functions

def _filter_data(raw,
                 l_freq=None,
                 h_freq=40,
                 picks=None,
                 fir_window='hamming',
                 filter_length='auto',
                 iir_params=None,
                 method='fir',
                 phase='zero',
                 l_trans_bandwidth='auto',
                 h_trans_bandwidth='auto',
                 pad='reflect_limited',
                 skip_by_annotation=('edge', 'bad_acq_skip'),
                 fir_design='firwin'):
    """
    Filter raw data. This is an exact invocation of the filter function of mne 0.23 dev.
    It uses all defaults of this version to ensure future updates to the defaults will not
    break the analysis result reproducibility.
    :param raw:
    :param l_freq:
    :param h_freq:
    :param fir_window:
    :param filter_length:
    :param phase:
    :param l_trans_bandwidth:
    :param h_trans_bandwidth:
    :param fir_design:
    :return:
    """
    # make sure that the data is loaded
    raw.load_data()
    raw.filter(h_freq=h_freq,
               l_freq=l_freq,
               picks=picks,
               filter_length=filter_length,
               l_trans_bandwidth=l_trans_bandwidth,
               h_trans_bandwidth=h_trans_bandwidth,
               iir_params=iir_params,
               method=method,
               phase=phase,
               skip_by_annotation=skip_by_annotation,
               pad=pad,
               fir_window=fir_window,
               fir_design=fir_design,
               verbose=True
               )
    return raw


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
    import os.path as op
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
            if len(base.split('-')) == 2 and base.split('-')[-1].startswith('1'):
                second = f
            elif len(base.split('-')) == 2 and base.split('-')[-1].startswith('2'):
                third = f
            elif len(base.split('-')) == 1:
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
                if base[0].isdigit() and base[0]== '1' and len(base.split('-')) == 1:
                    first = f
                elif base[0].isdigit() and base[0]== '2' and len(base.split('-')) == 1:
                    second = f
                elif base[0].isdigit() and base[0] == '2' and len(base.split('-')) == 2:
                    if base.split('-')[-1].startswith('1'):
                        second = f
                    elif base.split('-')[-1].startswith('2'):
                        third = f
                elif len(base.split('-')) == 2 and base[0].isalpha():
                    if base.split('-')[-1].startswith('1'):
                        second = f
                    elif base.split('-')[-1].startswith('2'):
                        third = f
                else:
                    # this shouldn't happen
                    raise ValueError(f"Cannot handle file list {files}")
    # check that all files are defined
    assert all([v is not None for v in [first, second, third]])
    print(f'Order the files as follows: {first}, {second}, {third}')
    return first, second, third


def read_data_original(directory,
                       subject,
                       savetonewdir=False,
                       bidsdir=None,
                       preprocessing='Raw'):
    """
    The preprocessed MEG data is split into three files that MNE Python
    can't automatically co-load.
    We read in all files, and concatenate them by hand.
    :param directory: path to a subject directory.
    :param subject: str, subject identifier ('001'), used for file names
     and logging
    :param savetonewdir: Boolean, if True, save the data as BIDS conform
    files into a new directory
    :param newdir: str, Path to where BIDS conform data shall be saved
    :param preprocessing: Data flavour to load. Existing directories are
     'Move_correc_SSS_realigneddefault_nonfittoiso' and 'Raw' (default)
    :return:
    """

    # We're starting with the original data from Luca. The files were
    # transferred from Hilbert as is, and have a non-BIDS and partially
    # inconsistent naming and directory structure
    # First, construct a Path to a preprocessed or Raw directory
    path = Path(directory) / preprocessing / '*.fif' if preprocessing \
            else Path(directory) / '*.fif'
    if not os.path.exists(os.path.split(path)[0]): # TODO: test this
        # some subjects have an extra level of directories
        path = Path(directory) / '*' / preprocessing / '*.fif' if preprocessing \
            else Path(directory) / '*' / '*.fif'
    print(f"Reading files for subject sub-{subject} from {path}.")
    # file naming is a mess. We need to make sure to sort the three files
    # correctly
    unsorted_files = glob(str(path))
    first, second, third = _get_first_file(unsorted_files)
    try:
        raw = mne.io.read_raw_fif(first)
    except ValueError:
        print(f'WARNING Irregular file naming. Will read files in sequentially '
              f'in the following order: {first}{second}{third}')
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
            print("I was instructed to save BIDS conform raw data into a"
                  "different directory, but did not get a path.")
            return raw

        outpath = Path(bidsdir) / f'sub-{subject}' / 'meg' / \
                  f'sub-{subject}_task-memento_meg.fif'
        # make sure there is a directory to save into
        _check_if_bids_directory_exists(outpath, subject)
        print(f"Saving BIDS-compliant raw data from subject {subject} into "
              f"{outpath}")
        raw.save(outpath, split_naming="bids", overwrite=True)
    return raw


def motion_estimation(subject,
                      raw,
                      head_pos_outdir='/tmp/'):
    """
    Calculate head positions from HPI coils as a prerequisite for movement
    correction.
    :param subject: str, subject identifier; used for writing file names &
    logging
    :param raw: Raw data object
    :param head_pos_outdir: directory to save the head position file to. Should
    be the root of a bids directory
    :return: head_pos: head positions estimates from HPI coils
    """
    # Calculate head motion parameters to remove them during maxwell filtering
    # First, extract HPI coil amplitudes to
    print(f'Extracting HPI coil amplitudes for subject sub-{subject}')
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    # compute time-varying HPI coil locations from amplitudes
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    print(f'Computing head positions for subject sub-{subject}')
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)
    # save head positions
    outpath = Path(head_pos_outdir) / f'sub-{subject}' / 'meg' / \
              f'sub-{subject}_task-memento_headshape.pos'
    # make sure there is a directory to save into
    _check_if_bids_directory_exists(outpath, subject)
    print(f'Saving head positions as {outpath}')
    mne.chpi.write_head_pos(outpath, head_pos)
    figpath = Path(head_pos_outdir) / f'sub-{subject}_ses-01_headmovement.png'
    fig = mne.viz.plot_head_positions(head_pos, mode='traces')
    fig.savefig(figpath)
    return head_pos


def _check_if_bids_directory_exists(outpath, subject):
    """
    Helper function that checks if a directory exists, and if not, creates it.
    """
    check_dir = os.path.dirname(outpath)
    if not os.path.isdir(Path(check_dir) / f'sub-{subject}' / 'meg'):
        print(f"The BIDS directory {check_dir} does not seem to exist. "
              f"Attempting creation...")
        os.makedirs(Path(check_dir) / f'sub-{subject}' / 'meg')


# TODO: We could do maxwell filtering without applying a filter when we remove
# chpi and line noise beforehand.
# mne.chpi.filter_chpi is able to do this
def filter_chpi_and_line(raw):
    """
    Remove Chpi and line noise from the data. This can be useful in order to
    use no filtering during bad channel detection for maxwell filtering.
    :param raw: Raw data, preloaded
    :return:
    """
    from mne.chpi import filter_chpi
    # make sure the data is loaded first
    print('Loading data for CHPI and line noise filtering')
    raw.load_data()
    print('Applying CHPI and line noise filter')
    # all parameters are set to the defaults of 0.23dev
    filter_chpi(raw, include_line=True, t_step=0.01, t_window='auto',
                ext_order=1, allow_line_only=False, verbose=None)
    # the data is now preconditioned, hence we change the state of the global
    # variable
    global preconditioned
    preconditioned = True
    return raw


def maxwellfilter(raw,
                  crosstalk_file,
                  fine_cal_file,
                  subject,
                  headpos_file=None,
                  compute_motion_params=True,
                  head_pos_outfile='/tmp/',
                  figdir='/tmp/',
                  outdir='/tmp/'):
    """

    :param raw:
    :param crosstalk_file: crosstalk compensation file from the Elekta system to
     reduce interference between gradiometers and magnetometers
    :param calibration_file: site-specific sensor orientation and calibration
    :param figdir; str, path to directory to save figures in
    :return:
    """
    from mne.preprocessing import find_bad_channels_maxwell


    if not compute_motion_params:
        if not headpos_file or not os.path.exists(headpos_file):
            print(f'Could not find or read head position files under the supplied'
                  f'path: {headpos_file}. Recalculating from scratch.')
            head_pos = motion_estimation(subject, raw, head_pos_outfile)
        print(f'Reading in head positions for subject sub-{subject} '
              f'from {headpos_file}.')
        head_pos = mne.chpi.read_head_pos(headpos_file)

    else:
        print(f'Starting motion estimation for subject sub-{subject}.')
        head_pos = motion_estimation(subject, raw, head_pos_outfile)

    raw.info['bads'] = []
    raw_check = raw.copy()
    if preconditioned:
        # preconditioned is a global variable that is set to True if some form
        # of filtering (CHPI and line noise removal or general filtering) has
        # been applied.
        # the data has been filtered, and we can pass h_freq=None
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
            return_scores=True, verbose=True, h_freq=None)
    else:
        # the data still contains line noise (50Hz) and CHPI coils. It will
        # filter the data before extracting bad channels
        auto_noisy_chs, auto_flat_chs, auto_scores = find_bad_channels_maxwell(
            raw_check, cross_talk=crosstalk_file, calibration=fine_cal_file,
            return_scores=True, verbose=True)
    print(f'Found the following noisy channels: {auto_noisy_chs} \n '
          f'and the following flat channels: {auto_flat_chs} \n'
          f'for subject sub-{subject}')
    bads = raw.info['bads'] + auto_noisy_chs + auto_flat_chs
    raw.info['bads'] = bads
    # free up space
    del raw_check
    # plot as a sanity check
    for ch_type in ['grad', 'mag']:
        plot_noisy_channel_detection(auto_scores,
                                     ch_type=ch_type,
                                     subject=subject,
                                     outpath=figdir
                                     )
    print(f'Signal Space Separation with movement compensation '
          f'starting for subject sub-{subject}')
    ## TODO: movement compensation can be done during maxwell filtering but also during
    raw_sss = mne.preprocessing.maxwell_filter(raw,
                                               cross_talk=crosstalk_file,
                                               calibration=fine_cal_file,
                                               head_pos=head_pos,
                                               verbose=True)
    # save sss files
    fname = Path(outdir) / f'sub-{subject}_task-memento_proc-sss.fif'
    raw_sss.save(fname, split_naming='bids')
    return raw_sss


def plot_noisy_channel_detection(auto_scores,
                                 subject='test',
                                 ch_type='grad',
                                 outpath='/tmp/'
                                 ):

    # Select the data for specified channel type
    ch_subset = auto_scores['ch_types'] == ch_type
    ch_names = auto_scores['ch_names'][ch_subset]
    scores = auto_scores['scores_noisy'][ch_subset]
    limits = auto_scores['limits_noisy'][ch_subset]
    bins = auto_scores['bins']  # The the windows that were evaluated.
    # We will label each segment by its start and stop time, with up to 3
    # digits before and 3 digits after the decimal place (1 ms precision).
    bin_labels = [f'{start:3.3f} – {stop:3.3f}'
                  for start, stop in bins]

    # We store the data in a Pandas DataFrame. The seaborn heatmap function
    # we will call below will then be able to automatically assign the correct
    # labels to all axes.
    data_to_plot = pd.DataFrame(data=scores,
                                columns=pd.Index(bin_labels, name='Time (s)'),
                                index=pd.Index(ch_names, name='Channel'))

    # First, plot the "raw" scores.
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(f'Automated noisy channel detection: {ch_type}, subject sub-{subject}',
                 fontsize=16, fontweight='bold')
    sns.heatmap(data=data_to_plot, cmap='Reds', cbar_kws=dict(label='Score'),
                ax=ax[0])
    [ax[0].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[0].set_title('All Scores', fontweight='bold')

    # Now, adjust the color range to highlight segments that exceeded the limit.
    sns.heatmap(data=data_to_plot,
                vmin=np.nanmin(limits),  # bads in input data have NaN limits
                cmap='Reds', cbar_kws=dict(label='Score'), ax=ax[1])
    [ax[1].axvline(x, ls='dashed', lw=0.25, dashes=(25, 15), color='gray')
     for x in range(1, len(bins))]
    ax[1].set_title('Scores > Limit', fontweight='bold')

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = outpath + '/' + f'noise_detection_sub-{subject}_{ch_type}.png'
    fig.savefig(fname)


def eventreader(raw, outputdir = '/tmp'):
    """
    the Triggers 32790 32792 seem spurious. TODO.
    :param raw:
    :return:
    """
    # for some reason, events only start at sample 628416 (sub 4)
    events = mne.find_events(raw,
                             min_duration=0.002,    # ignores spurious events
                             uint_cast=True,    # workaround an Elekta acquisition bug that causes negative values
                             #initial_event=True # unsure - the first on is 32772
                             )
    # TriggerName description based on experiment matlab files
    # lOpt10 and rOpt1 don't seem to exist (for sub 4 at least?)
    event_dict = {'end': 2, 'fixCross': 10, 'lOpt1': 12, 'lOpt2': 13,
                  'lOpt3': 14, 'lOpt4': 15, 'lOpt5': 16, 'lOpt6': 17,
                  'lOpt7': 18, 'lOpt8': 19, 'lOpt9': 20, 'lOpt10': 21,
                  'rOpt': 24, 'delay': 22, 'empty_screen': 26,
                  'pauseStart': 25, 'feedback': 27,
                  'weirdone': 32790, 'weirdtwo': 32792}
    # plot events. This works without raw data
    fig = mne.viz.plot_events(events,
                              sfreq=raw.info['sfreq'],
                              first_samp=raw.first_samp,
                              event_id=event_dict,
                              on_missing='warn')
    fig.suptitle('Full event protocol for {} ({})'.format(
        raw.info['subject_info']['first_name'],
        raw.info['subject_info']['last_name'])
    )
    # TODO: How do I increase/set the size of the plot?
    fpath = Path(outputdir) / 'eventplot_{}.png'.format(raw.info['subject_info']['first_name'])
    fig.savefig(str(fpath))

    # TODO: MNE crashes when it tries to plot events on top of raw data, doesn't plot any events

    # epochs
    epochs = mne.Epochs(raw,
                        events,
                        tmin=-0.3, # TODO: change
                        tmax=0.7, # TODO: change
                        event_id=event_dict,
                        on_missing='warn')
    # plotting epochs crashes (too many?)
    # which epoch is actually relevant? delay?
    return events, epochs, event_dict


def artifacts(raw):
    # see https://mne.tools/stable/auto_tutorials/preprocessing/plot_10_preprocessing_overview.html#sphx-glr-auto-tutorials-preprocessing-plot-10-preprocessing-overview-py
    # low frequency drifts in magnetometers
    mag_channels = mne.pick_types(raw.info, meg='mag')
    raw.plot(block=True, duration=60, order=mag_channels, n_channels=len(mag_channels),
             remove_dc=False)

    # power line noise
    fig = raw.plot_psd(block=True, tmax=np.inf, fmax=250, average=True)

    # heartbeat artifacts
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
    avg_ecg_epochs = ecg_epochs.average().apply_baseline((-0.5, -0.2))


# run everything
def main(subject,
         preprocessing='Raw',
         ):
    """
    Do a computation
    :param subject:
    :param preprocessing:
    :return:
    """

    directory = 'memento_' + subject
    raw = read_data(directory, preprocessing)
    raw_sss = maxwellfilter(raw,
                            crosstalk_file,
                            fine_cal_file,
                            subject,
                            compute_motion_params=True,
                            head_pos_outfile=directory)
    fname = directory + '/' + f'sub-{subject}_ses-01_raw_sss.fif'
    raw_sss.save(fname)

main('008')
