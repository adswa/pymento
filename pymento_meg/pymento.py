import mne
import logging
import numpy as np
from pathlib import Path
from pymento_meg.utils import (
    _construct_path,
)
from pymento_meg.orig.restructure import (
    read_data_original,
)

from pymento_meg.proc.preprocess import (
    maxwellfilter,
    ZAPline,
    _filter_data,
)
from pymento_meg.proc.bids import (
    read_bids_data,
    get_events,
    save_derivatives_to_bids_dir,
)
from pymento_meg.proc.artifacts import (
    remove_eyeblinks_and_heartbeat,
)
from autoreject import (
    AutoReject,
)
# prevent qt errors in mne browsers
mne.viz.set_browser_backend('matplotlib')

def restructure_to_bids(
        rawdir, subject, bidsdir, figdir, crosstalk_file, fine_cal_file,
        behav_dir
):
    """
    Transform the original memento MEG data into something structured.
    :return:
    """

    logging.info(
        f"Starting to restructure original memento data into BIDS for "
        f"subject sub-{subject}."
    )

    raw = read_data_original(
        directory=rawdir,
        subject=subject,
        savetonewdir=True,
        bidsdir=bidsdir,
        figdir=figdir,
        crosstalk_file=crosstalk_file,
        fine_cal_file=fine_cal_file,
        preprocessing="Raw",
        behav_dir=behav_dir,
    )


def signal_space_separation(bidspath, subject, figdir, derived_path):
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
    logging.info(
        f"Starting to read in raw memento data from BIDS directory for"
        f"subject sub-{subject}."
    )

    raw, bids_path = read_bids_data(
        bids_root=bidspath,
        subject=subject,
        datatype="meg",
        task="memento",
        suffix="meg",
    )

    # Events are now Annotations, also get them as events
    events = get_events(raw)

    fine_cal_file = bids_path.meg_calibration_fpath
    crosstalk_file = bids_path.meg_crosstalk_fpath

    logging.info(
        f"Starting signal space separation with motion correction "
        f"for subject sub{subject}."
    )

    raw_sss = maxwellfilter(
        raw=raw,
        crosstalk_file=crosstalk_file,
        fine_cal_file=fine_cal_file,
        subject=subject,
        headpos_file=None,
        compute_motion_params=True,
        figdir=figdir,
        outdir=derived_path,
        filtering=False,
        filter_args=None,
    )
    # don't ZAPline the data of subject 9, it introduces counter artifacts and
    # causes problems with the later ICA
    if subject == '009':
        # save processed files into their own BIDS directory
        save_derivatives_to_bids_dir(raw_sss=raw_sss, subject=subject,
                                     bidsdir=derived_path, figdir=figdir)
        return

    # ZAPline power-line and presentation screen noise
    raw_sss_zaplined = ZAPline(raw=raw_sss,
                               subject=subject,
                               figdir=figdir)
    # save processed files into their own BIDS directory
    save_derivatives_to_bids_dir(raw_sss=raw_sss_zaplined, subject=subject,
                                 bidsdir=derived_path, figdir=figdir)



def epoch_and_clean_trials(subject,
                           diagdir,
                           bidsdir,
                           datadir,
                           derivdir,
                           epochlength=3,
                           eventid={'visualfix/fixCross': 10}):
    """
    Chunk the data into epochs starting at the eventid specified per trial,
    lasting 7 seconds (which should include all trial elements).
    Do automatic artifact detection, rejection and fixing for eyeblinks,
    heartbeat, and high- and low-amplitude artifacts.
    :param subject: str, subject identifier. takes the form '001'
    :param diagdir: str, path to a directory where diagnostic plots can be saved
    :param bidsdir: str, path to a directory with BIDS data. Needed to load
    event logs from the experiment
    :param datadir: str, path to a directory with SSS-processed data
    :param derivdir: str, path to a directory where cleaned epochs can be saved
    :param epochlength: int, length of epoch
    :param eventid: dict, the event to start an Epoch from
    """
    # construct name of the first split
    raw_fname = Path(datadir) / f'sub-{subject}/meg' / \
                f'sub-{subject}_task-memento_proc-sss_meg.fif'
    logging.info(f"Reading in SSS-processed data from subject sub-{subject}. "
          f"Attempting the following path: {raw_fname}")
    raw = mne.io.read_raw_fif(raw_fname)
    events, event_dict = get_events(raw)
    # filter the data to remove high-frequency noise. Minimal high-pass filter
    # based on
    # https://www.sciencedirect.com/science/article/pii/S0165027021000157
    # ensure the data is loaded prior to filtering
    raw.load_data()
    if subject == '017':
        logging.info('Setting additional bad channels for subject 17')
        raw.info['bads'] = ['MEG0313', 'MEG0513', 'MEG0523']
        raw.interpolate_bads()
    # high-pass doesn't make sense, raw data has 0.1Hz high-pass filter already!
    _filter_data(raw, h_freq=100)
    # ICA to detect and repair artifacts
    logging.info('Removing eyeblink and heartbeat artifacts')
    rng = np.random.RandomState(28)
    remove_eyeblinks_and_heartbeat(raw=raw,
                                   subject=subject,
                                   figdir=diagdir,
                                   events=events,
                                   eventid=eventid,
                                   rng=rng,
                                   )
    # retrieve metadata to later add SUBJECT SPECIFIC TRIAL NUMBER TO THE EPOCH
    # THIS WAY WE CAN LATER RECOVER WHICH TRIAL PARAMETERS WE'RE LOOKING AT
    # BASED ON THE LOGS, AS THE EPOCH REJECTION WILL REMOVE TRIALS
    logging.info("Retrieving trial metadata.")
    from pymento_meg.proc.epoch import get_trial_features
    metadata = get_trial_features(bids_path=bidsdir,
                                  subject=subject,
                                  column=['trial_no'])
    # get the actual epochs: chunk the trial into epochs starting from the
    # event ID. Do not baseline correct the data.
    logging.info(f'Creating epochs of length {epochlength}')
    if eventid == {'press/left': 1,
                   'press/right': 4
                   }:
        # we need to clean up the events, as not all button presses are valid.
        # Some happen too late, others at the wrong time, and yet others to
        # restart a block of trials after a break. We only want those that
        # actually select an option
        events, trials_without_press = clean_response_events(events)
        # when centered on the response, move back in time
        epochs = mne.Epochs(raw, events, event_id=eventid,
                            tmin=-epochlength/2, tmax=epochlength/2,
                            picks='meg', baseline=None)
        # drop trials without button presses from the metadata
        if trials_without_press:
            logging.info(f"Removing the following trials without button presses:"
                         f"{trials_without_press}")
            metadata = metadata.drop(index=trials_without_press)
    else:
        epochs = mne.Epochs(raw, events, event_id=eventid,
                            tmin=0, tmax=epochlength,
                            picks='meg', baseline=None)

    # transform trial numbers to integers
    metadata = metadata.astype(int)
    # this does not work if we start at fixation cross for subject 5, because 1
    # fixation cross trigger is missing from the data, and it becomes impossible
    # to associate the trial metadata to the correct trials in the data
    epochs.metadata = metadata
    epochs.load_data()
    ## downsample the data to 200Hz
    #logging.info('Resampling epoched data down to 200 Hz')
    #epochs.resample(sfreq=200, verbose=True)
    # use autoreject to repair bad epochs
    ar = AutoReject(random_state=rng,
                    n_interpolate=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    # save the cleaned, epoched data to disk.
    outpath = _construct_path(
        [
            Path(derivdir),
            f"sub-{subject}",
            "meg",
            f"sub-{subject}_task-memento_cleaned_epo.fif",
        ]
    )
    logging.info(f"Saving cleaned, epoched data to {outpath}")
    epochs_clean.save(outpath, overwrite=True)
    # visualize the bad sensors for each trial
    fig = ar.get_reject_log(epochs).plot()
    fname = _construct_path(
        [
            Path(diagdir),
            f"sub-{subject}",
            "meg",
            f"epoch-rejectlog_sub-{subject}_{epochlength}_.png",
        ]
    )
    fig.savefig(fname)
    # plot the average of all cleaned epochs
    fig = epochs_clean.average().plot()
    fname = _construct_path(
        [
            Path(diagdir),
            f"sub-{subject}",
            "meg",
            f"clean-epoch_average_sub-{subject}_{epochlength}.png",
        ]
    )
    fig.savefig(fname)
    # plot psd of cleaned epochs
    psd = epochs_clean.plot_psd()
    fname = _construct_path(
        [
            Path(diagdir),
            f"sub-{subject}",
            "meg",
            f"psd_cleaned-epochs-{subject}_{epochlength}.png",
        ]
    )
    psd.savefig(fname)


def clean_response_events(events):
    """Returns only those button press event IDs that are "valid", i.e.,
    occuring between the second visual stimulus in a trial and the first visual
    stimulus in the next trial."""
    select_evs = []
    ignored_trials = []
    want_button_press = False
    for ev in events:
        # ev is 3-item sequence
        start, duration, evtype = ev
        # fish out every 2 visual stimulus - we know there are 510
        if evtype == 24:
            want_button_press = True
            continue
        if not want_button_press:
            continue
        assert want_button_press
        if evtype in (1, 4):
            select_evs.append(ev)
            want_button_press = False
        # once we reach the first visual stimulation without a button press,
        # register the trial to be removed
        elif evtype in (12, 13, 14, 15, 16, 17, 18, 19, 20):
            want_button_press = False
            ignored_trials.append(len(select_evs) + len(ignored_trials))
        # there already was a button press before, we can safely ignore
        # everything until the next trial
    return np.array(select_evs), ignored_trials


def _test_clean_events():
    first = np.arange(0, 18)
    second = np.repeat(0, 18)
    # three stray presses at the start, one stray press before option 2, one
    # second button press after option 2, one trial
    third = np.array([1, 4, 1, 10, 12, 22, 1, 24, 4, 4, 27, 10, 12, 22, 24, 28, 10, 12])
    events, remove = clean_response_events(np.stack([first, second, third],
                                                    axis=1))
    # trial 2 didn't have a button press
    assert remove == [1]
    # there is one valid event left
    assert len(events[events[:, 2] ==  4]) == 1
    assert len(events[events[:, 2] == 1]) == 0


def SRM(subject,
        datadir,
        bidsdir,
        figdir,
        condition='left-right',
        model='srm',
        timespan='fulltrial'):
    """
    Fit an SRM of a certain type with varying amount of features to a condition
    of choice
    :param subject:
    :param datadir:
    :param bidsdir:
    :param figdir:
    :param condition:
    :param model:
    :return:
    """

    if model == 'srm':
        from pymento_meg.srm.srm import plot_trial_components_from_srm
        plot_trial_components_from_srm(subject=subject,
                                          datadir=datadir,
                                          bidsdir=bidsdir,
                                          figdir=figdir,
                                          condition=condition,
                                          timespan=timespan)
