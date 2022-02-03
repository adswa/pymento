import mne
from mne.preprocessing import (
    create_ecg_epochs,
    create_eog_epochs,
    ICA,
)
from pymento_meg.config import ica_comps
from autoreject import (
    AutoReject
)
import logging
from pymento_meg.utils import _construct_path
from pathlib import Path
import matplotlib.pyplot as plt
plt.tight_layout()



def remove_eyeblinks_and_heartbeat(raw,
                                   subject,
                                   figdir,
                                   events,
                                   eventid
                                   ):
    """
    Find and repair eyeblink and heartbeat artifacts in the data.
    Data should be filtered.
    Importantly, ICA is fitted on artificially epoched data with reject
    criteria estimates via the autoreject package - this is done to reject high-
    amplitude artifacts to influence the ICA solution.
    The ICA fit is then applied to the raw data.
    :param raw: Raw data
    :param subject: str, subject identifier, e.g., '001'
    :param figdir:
    """
    # get ICA components for the given subject
    eog_indices = ica_comps[subject]['eog']
    ecg_indices = ica_comps[subject]['ecg']

    # prior to an ICA, it is recommended to high-pass filter the data
    # as low frequency artifacts can alter the ICA solution. We fit the ICA
    # to high-pass filtered (1Hz) data, and apply it to non-highpass-filtered
    # data
    logging.info("Applying a temporary high-pass filtering prior to ICA")
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=1., h_freq=None)
    # evoked eyeblinks and heartbeats for diagnostic plots
    logging.info("Searching for eyeblink and heartbeat artifacts in the data")
    eog_evoked = create_eog_epochs(filt_raw).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    if subject == '008':
        # subject 008's ECG channel is flat. It will not find any heartbeats by
        # default. We let it estimate heartbeat from magnetometers. For this,
        # we'll drop the ECG channel
        filt_raw.drop_channels('ECG003')
    ecg_evoked = create_ecg_epochs(filt_raw).average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))
    # make sure that we actually found sensible artifacts here
    eog_fig = eog_evoked.plot_joint()
    for i, fig in enumerate(eog_fig):
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"evoked-artifact_eog_sub-{subject}_{i}.png",
            ]
        )
        fig.savefig(fname)
    ecg_fig = ecg_evoked.plot_joint()
    for i, fig in enumerate(ecg_fig):
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"evoked-artifact_ecg_sub-{subject}_{i}.png",
            ]
        )
        fig.savefig(fname)
    # define the actual events (7 seconds from onset of event_id)
    # No baseline correction as it would interfere with ICA.
    logging.info("Epoching filtered data")
    if eventid == {'press/left': 1,
                   'press/right': 4
                   }:
        # when centered on the response, move back in time
        epochs = mne.Epochs(filt_raw, events, event_id=eventid,
                            tmin=-3, tmax=0,
                            picks='meg', baseline=None)
    else:
        epochs = mne.Epochs(filt_raw, events, event_id=eventid,
                            tmin=0, tmax=3,
                            picks='meg', baseline=None)

    # First, estimate rejection criteria for high-amplitude artifacts. This is
    # done via autoreject
    logging.info('Estimating bad epochs quick-and-dirty, to improve ICA')
    ar = AutoReject(random_state=11)
    # fit on first 200 epochs to save (a bit of) time
    epochs.load_data()
    ar.fit(epochs[:200])
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    # run an ICA to capture heartbeat and eyeblink artifacts.
    # set a seed for reproducibility.
    # When left to figure out the component number by itself, it ends up with
    # about 80. I'm setting n_components to 45 to have a chance at checking them
    # by hand.
    # We fit it on a set of epochs excluding the initial bad epochs following
    # https://github.com/autoreject/autoreject/blob/dfbc64f49eddeda53c5868290a6792b5233843c6/examples/plot_autoreject_workflow.py
    logging.info('Fitting the ICA')
    ica = ICA(max_iter='auto', n_components=45, random_state=42)
    ica.fit(epochs[~reject_log.bad_epochs])

    # visualize the components
    components = ica.plot_components()
    for i, fig in enumerate(components):
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"ica-components_sub-{subject}_{i}.png",
            ]
        )
        fig.savefig(fname)
    # visualize the time series of components and save it
    plt.rcParams['figure.figsize'] = 30, 20
    comp_sources = ica.plot_sources(epochs)
    fname = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"ica-components_sources_sub-{subject}.png",
        ]
    )
    comp_sources.savefig(fname)
    # reset plotting params
    plt.rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

    # plot EOG components
    overlay_eog = ica.plot_overlay(eog_evoked, exclude=ica_comps[subject]['eog'])
    fname = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"ica-eog-components_over-avg-epochs_sub-{subject}.png",
        ]
    )
    overlay_eog.savefig(fname)
    # plot ECG components
    overlay_ecg = ica.plot_overlay(ecg_evoked, exclude=ica_comps[subject]['ecg'])
    fname = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"ica-ecg-components_over-avg-epochs_sub-{subject}.png",
        ]
    )
    overlay_ecg.savefig(fname)
    # plot EOG component properties
    figs = ica.plot_properties(filt_raw, picks=eog_indices)
    for i, fig in enumerate(figs):
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"ica-property{i}_artifact-eog_sub-{subject}.png",
            ]
        )
        fig.savefig(fname)
    # plot ECG component properties
    figs = ica.plot_properties(filt_raw, picks=ecg_indices)
    for i, fig in enumerate(figs):
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"ica-property{i}_artifact-ecg_sub-{subject}.png",
            ]
        )
        fig.savefig(fname)

    # Set the indices to be excluded
    ica.exclude = eog_indices
    ica.exclude.extend(ecg_indices)

    # plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
    sources = ica.plot_sources(eog_evoked)
    fname = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"ica-sources_artifact-eog_sub-{subject}.png",
        ]
    )
    sources.savefig(fname)

    # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
    sources = ica.plot_sources(ecg_evoked)
    fname = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"ica-sources_artifact-ecg_sub-{subject}.png",
        ]
    )
    sources.savefig(fname)
    # apply the ICA to the raw data
    logging.info('Applying ICA to the raw data.')
    raw.load_data()
    ica.apply(raw)