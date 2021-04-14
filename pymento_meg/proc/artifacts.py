from mne.preprocessing import (
    create_ecg_epochs,
    create_eog_epochs,
    ICA
)

from pymento_meg.utils import _construct_path

from pathlib import Path


def remove_eyeblinks_and_heartbeat(raw,
                                   subject,
                                   figdir,
                                   reject,
                                   epochs,
                                   tstep):
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
    :param reject: list, reject criteria for high-amplitude artifacts from
    autoreject
    :param epochs: artificial epochs built from raw data,
    used to determine rejection criteria
    :param tstep:
    """
    # don't overwrite raw data in place
    #filt_raw = raw.copy()
    # evoked eyeblinks and heartbeats
    eog_evoked = create_eog_epochs(raw).average()
    eog_evoked.apply_baseline(baseline=(None, -0.2))
    ecg_evoked = create_ecg_epochs(raw).average()
    ecg_evoked.apply_baseline(baseline=(None, -0.2))

    # temporarily filter the raw data with a highpass filter of 1Hz to
    # remove low-frequency drifts
    #filt_raw.load_data().filter(l_freq=1., h_freq=None)

    # run an ICA to capture heartbeat and eyeblink artifacts.
    # 15 components are hopefully enough to capture them.
    # set a seed for reproducibility
    ica = ICA(n_components=30, max_iter='auto', random_state=13)
    ica.fit(epochs, reject=reject, tstep=tstep)

    # use the EOG channel to select ICA components:
    ica.exclude = []
    # find which ICs match the EOG pattern
    eog_indices, eog_scores = ica.find_bads_eog(raw)
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
    figs = ica.plot_properties(raw, picks=eog_indices)
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
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='correlation',
                                                threshold='auto')
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

    figs = ica.plot_properties(raw, picks=ecg_indices)
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
    ica.apply(raw)
    return raw
