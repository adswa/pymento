import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

import pandas as pd
from scipy.stats import zscore
from scipy.signal import decimate

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from mne.decoding import GeneralizingEstimator
from pymento_meg.decoding.logreg import known_targets
from pymento_meg.srm.srm import get_general_data_structure
from pymento_meg.utils import _construct_path
from pymento_meg.proc.behavior import logreg

# order trials according to values of stimulus parameters
extreme_targets = {
    'probability': {'low': [0.1],
                    'medium': [0.2, 0.4],
                    'high': [0.8]
                    },
    'magnitude': {'low': [0.5],
                  'medium': [1, 2],
                  'high': [4]
                  }
}


# custom colormap to overlay p-values:
c_white = mc.colorConverter.to_rgba('white', alpha=0)
c_black = mc.colorConverter.to_rgba('gray', alpha=1)
cmap_rb = mc.LinearSegmentedColormap.from_list('rb_cmap',
                                               [c_white, c_black],
                                               512)


def generalize(subject,
               trainingdir,
               testingdir,
               bidsdir,
               figdir,
               n_permutations=100,
               ):
    """
    Perform several temporal generalization analyses: For each subject, take
    1000ms centered around the motor response part of a trial and train a
    Logistic Regression to classify left or right choice on every time point.
    Afterwards, evaluate the models on all timepoints from the start of a trial
    until the end of the delay period. The generalization analysis is done
    separately for trials corresponding to different values (high, medium, and
    low values) on the two stimulus dimensions 'magnitude' and 'probability'
    via test-trial selection: We test how well the models predicting eventual
    choice later in the trial generalize for test periods with low, medium, or
    high magnitude or probability. As the y data is eventual choice later in the
    trial, the results from these analyses depict how well the true response is
    already decodable earlier in the trial.
    In addition to this, we also run a generalization analysis that doesn't use
    true choice behavior later in the trail, but hypothetical choice behavior
    given the stimulus value on a given stimulus dimension. High magnitude and
    probability values of the first stimulus are assumed to predict left
    choices, whereas low magnitude and probability values are assumed to predict
    right choices. These analysis depict how well a potentially prepared
    response is already decodable earlier in the trial.
    Both generalization analyses are complemented with a permutation test. This
    permutation test shuffles labels to break X-y associations, and generates a
    null distribution from it. With a one-sided 5% alpha level, a binary mask
    hides all time points except those with accuracies exceeding the accuracies
    in the null distribution at least in 95% of shuffles.

    This function saves scoring from real and hyptothetical y-values, and
    plain temporal generalization plots as well as plots overlayed with p-value
    masks from the permutation tests.
    Parameters
    ----------
    :param subject: str, subject identifier, e.g. '001'
    :param trainingdir: Directory with epochs centered around response
    :param testingdir: Directory with epochs centered around visual stimulus 1
    :param bidsdir: Directory with BIDS structured raw data
    :param figdir: Directory to save plots in
    :return:
    """
    dec_factor = 5
    fpath = Path(_construct_path([figdir, f'sub-{subject}/']))
    # read in the training data (1s, centered around response)
    train_fullsample, train_data = get_general_data_structure(
        subject=subject,
        datadir=trainingdir,
        bidsdir=bidsdir,
        condition='nobrain-brain',
        timespan=[-0.5, 0.5])
    # read in the testing data (2.7s, first visual stimulus + delay
    test_fullsample, test_data = get_general_data_structure(
        subject=subject,
        datadir=testingdir,
        bidsdir=bidsdir,
        condition='nobrain-brain',
        timespan=[0, 3.4])

    # do the analysis for both stimulus features
    for target in extreme_targets:
        tname = known_targets[target]['tname']
        for condition, value in extreme_targets[target].items():
            # train on all trials, except for trials where no reaction was made
            X_train = np.array([decimate(epoch['normalized_data'], dec_factor)
                               for i, epoch in train_fullsample[subject].items()
                               ])
            y_train = np.array(['choice' + str(epoch['choice'])
                               for i, epoch in train_fullsample[subject].items()
                               ])
            if any(y_train == 'choice0.0'):
                # remove trials where the participant did not make a choice
                idx = np.where(y_train == 'choice0.0')
                logging.info(f"Subject sub-{subject} did not make a choice in "
                             f"{len(idx)} training trials")
                y_train = np.delete(y_train, idx)
                X_train = np.delete(X_train, idx, axis=0)

            # first, test on data corresponding to the target value (e.g., high
            # probability) with choice labels from the later actual choice.
            # As before, exclude trials without a reaction
            X_test = np.array([decimate(epoch['normalized_data'], dec_factor)
                               for id, epoch in test_fullsample[subject].items()
                               if epoch[tname] in value])

            y_test = np.array(['choice' + str(epoch['choice'])
                               for i, epoch in test_fullsample[subject].items()
                               if epoch[tname] in value])
            if any(y_test == 'choice0.0'):
                # remove trials where the participant did not make a choice
                idx = np.where(y_test == 'choice0.0')
                logging.info(f"Subject sub-{subject} did not make a choice in "
                             f"{len(idx)} testing trials")
                y_test = np.delete(y_test, idx)
                X_test = np.delete(X_test, idx, axis=0)

            # set up a generalizing estimator
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(solver='liblinear')
            )

            time_gen = GeneralizingEstimator(clf, scoring='accuracy',
                                             n_jobs=-1, verbose=True)
            # train on the motor response
            time_gen.fit(X=X_train, y=y_train)
            # test on the stimulus presentation, with true labels
            scores = time_gen.score(X=X_test, y=y_test)
            # save the scores
            fname = fpath / \
                    f'sub-{subject}_gen-scores_{target}-{condition}_true-y.npy'
            logging.info(f"Saving generalization scores into {fname}")
            np.save(fname, scores)

            # next, repeat the classification but test with labels implied from
            # value of the target (e.g., high probability -> left choice)
            if condition is not 'medium':
                choice = 'choice1.0' if condition == 'high' \
                    else 'choice2.0'
                hypothetical_y = np.repeat(choice, len(X_test))
                scores_hypothetical = time_gen.score(X=X_test, y=hypothetical_y)
                # save the scores
                fname = fpath / \
                        f'sub-{subject}_gen-scores_{target}-{condition}_hypo-y.npy'
                logging.info(f"Saving generalization scores into {fname}")
                np.save(fname, scores_hypothetical)
            else:
                scores_hypothetical = None
                binary_mask_hypo = None

            y_train_copy = y_train.copy()
            # do a permutation test comparison
            null_distribution = []
            for i in range(n_permutations):
                # shuffle works in place
                np.random.shuffle(y_train_copy)
                # build a new classifier based on scrambled data
                null_time_gen = GeneralizingEstimator(clf, scoring='accuracy',
                                                 n_jobs=-1, verbose=True)
                # train on the motor response
                null_time_gen.fit(X=X_train, y=y_train_copy)
                # test on the stimulus presentation, with true labels
                scrambled_scores = time_gen.score(X=X_test, y=y_test)
                null_distribution.append(scrambled_scores)
            null_distribution = np.asarray(null_distribution)
            # save the null distribution
            fname = fpath / \
                    f'sub-{subject}_gen-scores_{target}-{condition}_scrambled.npy'
            logging.info(f"Saving scrambled scores into {fname}")
            np.save(fname, null_distribution)

            # create distributions for each time point and model
            p_vals = np.zeros(scores.shape)
            for n in null_distribution:
                p_vals += n >= scores
            p_vals += 1
            p_vals /= (len(null_distribution + 1))
            # create a binary mask from p_vals
            binary_mask = p_vals > 0.05

            if scores_hypothetical is not None:
                # create p_values and mask also for hypothetical target data
                p_vals = np.zeros(scores_hypothetical.shape)
                for n in null_distribution:
                    p_vals += n >= scores_hypothetical
                p_vals += 1
                p_vals /= (len(null_distribution + 1))
                binary_mask_hypo = p_vals > 0.05

            for scoring, description, mask in \
                    [(scores, 'actual', binary_mask),
                     (scores_hypothetical, 'hypothetical', binary_mask_hypo)]:
                if scores_hypothetical is None:
                    continue
                plot_generalization(scoring=scoring,
                                    description=description,
                                    condition=condition,
                                    target=target,
                                    fpath=fpath,
                                    subject=subject,
                                    mask=mask)


def plot_generalization(scoring, description, condition, target,
                        fpath, subject, mask=None, fixed_cbar=True):
    """Plot a temporal generalization matrix for the trial"""
    fig, ax = plt.subplots(1, figsize=[9, 4], frameon=False)
    pltkwargs = {'origin': 'lower',
                 'extent': np.array([0, 3400, -500, 500])}
    if fixed_cbar:
        # set cbar limits to 0, 1
        pltkwargs['vmin'] = 0.
        pltkwargs['vmax'] = 1.
    im = ax.matshow(scoring, cmap='RdBu_r', **pltkwargs)
    ax.axhline(0, color='k', linestyle='dotted', label='motor response')
    ax.axvline(700, color='k', label='stimulus offset')
    ax.axvline(2700, color='green', label='stimulus onset')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Test Time (ms), stimulus 1 and delay period')
    ax.set_ylabel('Train Time (ms), \n response-centered')
    plt.suptitle(f'Generalization based on {condition} {target} '
                 f'({description} targets)')
    ax.set_title("Decoding choice (accuracy)")
    axins = inset_axes(
        ax,
        width="2%",
        height="100%",
        loc="lower left",
        bbox_to_anchor=(1.01, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    fig.colorbar(im, cax=axins)
    ax.legend(bbox_to_anchor=(0, 0, 1, 0.2))
    ax.set_aspect('auto')
    fname = fpath / \
            f'sub-{subject}_generalization_{target}-{condition}_{description}.png'
    logging.info(f"Saving generalization plot into {fname}")
    fig.savefig(fname)
    if mask is None:
        return
    # overlay the p-values. non-significant areas will become black
    im2 = ax.matshow(mask, cmap=cmap_rb, **pltkwargs)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_aspect('auto')
    fname = fpath / \
            f'sub-{subject}_generalization_{target}-{condition}_{description}_pval-mask.png'
    logging.info(f"Saving generalization plot into {fname}")
    fig.savefig(fname)


def aggregate_generalization(
        figdir='/data/project/brainpeach/generalization',
):
    """
    Create aggregate plots across generalization results.
    :param figdir: str, directory where to save figures in and where to find
    scores
    :return:
    """
    figdir = Path(figdir)
    for target in extreme_targets:
        for condition, value in extreme_targets[target].items():
            hypo_scores = []
            true_scores = []
            for sub in np.arange(1, 23):
                subject = f'00{sub}' if sub < 10 else f'0{sub}'

                fname = figdir / f'sub-{subject}' / \
                        f'sub-{subject}_gen-scores_{target}-{condition}_true-y.npy'
                truescore = np.load(fname)
                true_scores.append(truescore)
            avg_trues = np.mean(true_scores, axis=0)
            description = f'averaged_actual'
            plot_generalization(avg_trues, description=description,
                                condition=condition, target=target,
                                fpath=figdir, subject='group', mask=None,
                                fixed_cbar=False)
            if condition == 'medium':
                continue
            for sub in np.arange(1, 23):
                subject = f'00{sub}' if sub < 10 else f'0{sub}'
                fname = figdir / f'sub-{subject}' / \
                        f'sub-{subject}_gen-scores_{target}-{condition}_hypo-y.npy'
                hyposcore = np.load(fname)
                hypo_scores.append(hyposcore)
            avg_hypos = np.mean(hypo_scores, axis=0)
            description = f'averaged_hypothetical'
            plot_generalization(avg_hypos, description=description,
                                condition=condition, target=target,
                                fpath=figdir, subject='group', mask=None,
                                fixed_cbar=False)



def generalization_intergrating_behavior(subject,
                                         trainingdir,
                                         testingdir,
                                         bidsdir,
                                         figdir,
                                         n_permutations=100,
                                         ):

    dec_factor = 5
    fpath = Path(_construct_path([figdir, f'sub-{subject}/']))
    # read in the training data (1s, centered around response)
    train_fullsample, train_data = get_general_data_structure(
        subject=subject,
        datadir=trainingdir,
        bidsdir=bidsdir,
        condition='nobrain-brain',
        timespan=[-0.5, 0.5])
    # read in the testing data (2.7s, first visual stimulus + delay
    test_fullsample, test_data = get_general_data_structure(
        subject=subject,
        datadir=testingdir,
        bidsdir=bidsdir,
        condition='nobrain-brain',
        timespan=[0, 3.4])
    # train on all trials, except for trials where no reaction was made
    X_train = np.array([decimate(epoch['normalized_data'], dec_factor)
                        for i, epoch in train_fullsample[subject].items()
                        ])
    y_train = np.array(['choice' + str(epoch['choice'])
                        for i, epoch in train_fullsample[subject].items()
                        ])
    if any(y_train == 'choice0.0'):
        # remove trials where the participant did not make a choice
        idx = np.where(y_train == 'choice0.0')
        logging.info(f"Subject sub-{subject} did not make a choice in "
                     f"{len(idx)} training trials")
        y_train = np.delete(y_train, idx)
        X_train = np.delete(X_train, idx, axis=0)

    # get the test data
    X_test = np.array([decimate(epoch['normalized_data'], dec_factor)
                       for id, epoch in test_fullsample[subject].items()])
    # calculate hypothetical labels. First, get regression coefficients
    prob, mag, EV = logreg(bidsdir=bidsdir,
                           figdir='/tmp',
                           n_splits=10,
                           subject=subject)[subject]['pure_coefs'][:3]
    # make a dataframe for easier data manipulation
    df = pd.DataFrame(test_data)
    # drop everything we don't need
    df = df.drop(columns=['epoch', 'trial_type', 'data', 'normalized_data',
                          'prevLchar', 'prevRchar', 'prevRT', 'prevchoice',
                          'Lchar', 'Rchar', 'RT',  'prevLoptMag',
                          'prevLoptProb', 'prevRoptMag', 'prevRoptProb',
                          'choice', 'pointdiff', 'subject', 'trial_no'])
    # calculate EV from demeaned prob & magnitude
    df['LEV'] = (df.LoptProb - df.LoptProb.mean()) * \
                     (df.LoptMag - df.LoptMag.mean())
    # z-score everything
    df.apply(zscore)
    df['integrate'] = (df.LoptMag * mag) + (df.LoptProb * prob) + (df.LEV * EV)
    # split in the highest and lowest 25%
    col = 'integrate'
    upper, lower = df[col].quantile([0.25, 0.75])
    # everything is negative, more negative = left choice
    conditions = [df[col] <= upper,
                  df[col] >= lower]
    choices = ["choice1.0", 'choice2.0']
    df["choice"] = np.select(conditions, choices, default=None)
    # get indices of all trials that did not make the cut
    medium_trials = np.where(df['choice'].values == None)[0]

    # remove the trials that didn't make the cut from the data
    X_test = np.delete(X_test, medium_trials, axis=0)
    y_test = df['choice'].fillna(np.nan).dropna().values

    # set up a generalizing estimator
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver='liblinear')
    )

    time_gen = GeneralizingEstimator(clf, scoring='accuracy',
                                     n_jobs=-1, verbose=True)
    # train on the motor response
    time_gen.fit(X=X_train, y=y_train)
    # test on the stimulus presentation, with true labels
    scores = time_gen.score(X=X_test, y=y_test)
    # save the scores
    fname = fpath / \
            f'sub-{subject}_gen-scores_estimated-y.npy'
    logging.info(f"Saving generalization scores into {fname}")
    np.save(fname, scores)

    y_train_copy = y_train.copy()
    # do a permutation test comparison
    null_distribution = []
    for i in range(n_permutations):
        # shuffle works in place
        np.random.shuffle(y_train_copy)
        # build a new classifier based on scrambled data
        null_time_gen = GeneralizingEstimator(clf, scoring='accuracy',
                                              n_jobs=-1, verbose=True)
        # train on the motor response
        null_time_gen.fit(X=X_train, y=y_train_copy)
        # test on the stimulus presentation, with true labels
        scrambled_scores = time_gen.score(X=X_test, y=y_test)
        null_distribution.append(scrambled_scores)
    null_distribution = np.asarray(null_distribution)
    # save the null distribution
    fname = fpath / \
            f'sub-{subject}_gen-scores_estimated-y_scrambled.npy'
    logging.info(f"Saving scrambled scores into {fname}")
    np.save(fname, null_distribution)

    # create distributions for each time point and model
    p_vals = np.zeros(scores.shape)
    for n in null_distribution:
        p_vals += n >= scores
    p_vals += 1
    p_vals /= (len(null_distribution + 1))
    # create a binary mask from p_vals
    binary_mask = p_vals > 0.05
    plot_generalization(scoring=scores,
                        description='estimated',
                        condition='trials',
                        target='all',
                        fpath=fpath,
                        subject=subject,
                        mask=binary_mask)
