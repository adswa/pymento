import mne
from mne.preprocessing import (
    create_ecg_epochs,
    create_eog_epochs,
    ICA,
)
from autoreject import (
    AutoReject
)
import logging
from pymento_meg.utils import _construct_path
from pathlib import Path


def remove_eyeblinks_and_heartbeat(raw,
                                   subject,
                                   figdir,
                                   events,
                                   eventid):
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
    eog_evoked.apply_baseline(baseline=(-0.5, -0.2))
    if subject == '008':
        # subject 008's ECG channel is flat. It will not find any heartbeats by
        # default. We let it estimate heartbeat from magnetometers. For this,
        # we'll drop the ECG channel
        filt_raw.drop_channels('ECG003')
    ecg_evoked = create_ecg_epochs(filt_raw).average()
    ecg_evoked.apply_baseline(baseline=(-0.5, -0.2))
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
    epochs = mne.Epochs(filt_raw, events, event_id=eventid,
                        tmin=0, tmax=7,
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
    # use picard algorithm because it promised
    # set a seed for reproducibility.
    # ICA should figure its component number out itself.
    # We fit it on a set of epochs excluding the initial bad epochs following
    # https://github.com/autoreject/autoreject/blob/dfbc64f49eddeda53c5868290a6792b5233843c6/examples/plot_autoreject_workflow.py
    logging.info('Fitting the ICA')
    ica = ICA(method='picard',
              max_iter='auto', random_state=42)
    ica.fit(epochs[~reject_log.bad_epochs])

    # use the EOG channel to select ICA components:
    ica.exclude = []
    # find which ICs match the EOG pattern
    logging.info("Search for ICA components that capture eye blink artifacts")
    eog_indices, eog_scores = ica.find_bads_eog(filt_raw)
    ica.exclude = eog_indices

    # barplot of ICA component "EOG match" scores
    scores = ica.plot_scores(eog_scores)
    fname = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"ica-scores_artifact-eog_sub-{subject}.png",
        ]
    )
    scores.savefig(fname)
    # plot diagnostics
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
    # find ECG components
    logging.info("Search for ICA components that capture heartbeat artifacts")
    ecg_indices, ecg_scores = ica.find_bads_ecg(filt_raw, method='ctps',
                                                threshold='auto')
    if subject == '008':
        # because this subject misses the ECG channel, automatic detection of
        # ECG components fails. However, visual inspection shows a clear heart
        # beat in component 17. This should be stable across reruns as long as
        # the seed isn't changed.
        logging.info("For subject 8, setting a predefined component.")
        ecg_indices = [17]

    ica.exclude.extend(ecg_indices)

    scores = ica.plot_scores(ecg_scores)
    fname = _construct_path(
        [
            Path(figdir),
            f"sub-{subject}",
            "meg",
            f"ica-scores_artifact-ecg_sub-{subject}.png",
        ]
    )
    scores.savefig(fname)

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
    return raw
