import mne
import matplotlib

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pymento_meg.utils import _construct_path
from pathlib import Path
from matplotlib import interactive

interactive(True)


def plot_psd(raw, subject, figdir, filtering):
    """
    Helper to plot spectral densities
    """
    print(
        f"Plotting spectral density plots for subject sub-{subject}"
        f"after Maxwell filtering."
    )
    if filtering:
        # append a 'filtered' suffix to the file name
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"sub-{subject}_task-memento_spectral-density_filtered.png",
            ]
        )
    else:
        fname = _construct_path(
            [
                Path(figdir),
                f"sub-{subject}",
                "meg",
                f"sub-{subject}_task-memento_spectral-density.png",
            ]
        )
    fig = raw.plot_psd()
    fig.savefig(fname)


def plot_noisy_channel_detection(
    auto_scores, subject="test", ch_type="grad", outpath="/tmp/"
):

    # Select the data for specified channel type
    ch_subset = auto_scores["ch_types"] == ch_type
    ch_names = auto_scores["ch_names"][ch_subset]
    scores = auto_scores["scores_noisy"][ch_subset]
    limits = auto_scores["limits_noisy"][ch_subset]
    bins = auto_scores["bins"]  # The the windows that were evaluated.
    # We will label each segment by its start and stop time, with up to 3
    # digits before and 3 digits after the decimal place (1 ms precision).
    bin_labels = [f"{start:3.3f} – {stop:3.3f}" for start, stop in bins]

    # We store the data in a Pandas DataFrame. The seaborn heatmap function
    # we will call below will then be able to automatically assign the correct
    # labels to all axes.
    data_to_plot = pd.DataFrame(
        data=scores,
        columns=pd.Index(bin_labels, name="Time (s)"),
        index=pd.Index(ch_names, name="Channel"),
    )

    # First, plot the "raw" scores.
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle(
        f"Automated noisy channel detection: {ch_type}, subject sub-{subject}",
        fontsize=16,
        fontweight="bold",
    )
    sns.heatmap(data=data_to_plot, cmap="Reds", cbar_kws=dict(label="Score"), ax=ax[0])
    [
        ax[0].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
        for x in range(1, len(bins))
    ]
    ax[0].set_title("All Scores", fontweight="bold")

    # Now, adjust the color range to highlight segments that exceeded the limit.
    sns.heatmap(
        data=data_to_plot,
        vmin=np.nanmin(limits),  # bads in input data have NaN limits
        cmap="Reds",
        cbar_kws=dict(label="Score"),
        ax=ax[1],
    )
    [
        ax[1].axvline(x, ls="dashed", lw=0.25, dashes=(25, 15), color="gray")
        for x in range(1, len(bins))
    ]
    ax[1].set_title("Scores > Limit", fontweight="bold")

    # The figure title should not overlap with the subplots.
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = _construct_path(
        [
            Path(outpath),
            f"sub-{subject}",
            "meg",
            f"noise_detection_sub-{subject}_{ch_type}.png",
        ]
    )
    fig.savefig(fname)


# TODO: Do a psd plot for each trial
