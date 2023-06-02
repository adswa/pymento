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
                    'high': [0.8],
                    'extreme': [0.1, 0.8]
                    },
    'magnitude': {'low': [0.5],
                  'medium': [1, 2],
                  'high': [4],
                  'extreme': [0.5, 4]
                  }
}


# custom colormap to overlay p-values:
c_white = mc.colorConverter.to_rgba('white', alpha=0)
c_black = mc.colorConverter.to_rgba('gray', alpha=1)
cmap_rb = mc.LinearSegmentedColormap.from_list('rb_cmap',
                                               [c_white, c_black],
                                               512)

# get parietal sensors only:
# raw = mne.io.read_raw_fif(
# '/data/project/brainpeach/memento-sss/sub-001/meg/sub-001_task-memento_proc-sss_meg.fif')
# parietal_channels = [raw.info.ch_names.index(ch)
# for ch in mne.read_vectorview_selection(name=["parietal"], info=raw.info)]
parietal_channels = [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 180, 181, 182, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 222, 223, 224, 225, 226, 227, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 279, 280, 281]


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
    fpath = Path(_construct_path([figdir, f'sub-{subject}/']))
    dec_factor = 5
    train_fullsample, train_data, test_fullsample, test_data = \
        _read_test_n_train(subject=subject, trainingdir=trainingdir,
                           testingdir=testingdir, bidsdir=bidsdir)
    # do the analysis for both stimulus features
    for target in extreme_targets:
        tname = known_targets[target]['tname']
        for condition, value in extreme_targets[target].items():
            # train on all trials, except for trials where no reaction was made
            X_train, y_train = _make_X_n_y(train_fullsample, subject,
                                           dec_factor, drop_non_responses=True,
                                           ch_subset=parietal_channels)

            # first, test on data corresponding to the target value (e.g., high
            # probability) with choice labels from the later actual choice.
            # As before, exclude trials without a reaction
            X_test, y_test = _make_X_n_y(test_fullsample, subject, dec_factor,
                                         drop_non_responses=True, tname=tname,
                                         value=value, ch_subset=parietal_channels)
            fname = fpath / \
                    f'sub-{subject}_gen-scores_{target}-{condition}_true-y.npy'
            scores, clf, time_gen = \
                _train_and_score_generalizer(X_train=X_train, X_test=X_test,
                                             y_train=y_train, y_test=y_test,
                                             fname=fname)

            # next, repeat the classification but test with labels implied from
            # value of the target (e.g., high probability -> left choice)
            if condition != 'medium':
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

            fname = fpath / \
                    f'sub-{subject}_gen-scores_{target}-{condition}_scrambled.npy'
            null_distribution, binary_mask =\
                _permute(y_train=y_train, X_train=X_train, y_test=y_test,
                         X_test=X_test, clf=clf, fname=fname, scores=scores,
                         n_permutations=n_permutations, alpha=0.05)

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


def _read_test_n_train(subject, trainingdir, testingdir, bidsdir):
    """Helper to read in training and testing data"""
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
    return train_fullsample, train_data, test_fullsample, test_data


def _make_X_n_y(fullsample, subject, dec_factor, drop_non_responses=True,
                tname=None, value=None, ch_subset=None):
    if tname is None and value is None:
        X = np.array([decimate(epoch['normalized_data'], dec_factor)
                          for i, epoch in fullsample[subject].items()
                          ])
        y = np.array(['choice' + str(epoch['choice'])
                      for i, epoch in fullsample[subject].items()
                      ])
    else:
        # subset X and y to trials that match a stimulus condition
        X = np.array([decimate(epoch['normalized_data'], dec_factor)
                      for id, epoch in fullsample[subject].items()
                      if epoch[tname] in value])
        y = np.array(['choice' + str(epoch['choice'])
                      for i, epoch in fullsample[subject].items()
                      if epoch[tname] in value])
    if drop_non_responses:
        if any(y == 'choice0.0'):
            # remove trials where the participant did not make a choice
            idx = np.where(y == 'choice0.0')
            logging.info(f"Subject sub-{subject} did not make a choice in "
            f"{len(idx)} training trials")
            y = np.delete(y, idx)
            X = np.delete(X, idx, axis=0)
    if ch_subset is not None:
        # keep only selected channels, identified by index
        logging.info(f'Selecting a subset of channels: {ch_subset}')
        X = X[:, ch_subset, :]
    return X, y


def _permute(y_train, X_train, y_test, X_test, clf, fname, scores,
             n_permutations=100, alpha=0.05):
    """Perform a permutation test for the generalization analysis by permuting
    y_train labels, retraining a new generalizing estimator, and testing it on
    true test data n_permutations times. It will save the scrambled scores
    (null distribution), and return a binary mask corresponding to the specified
    alpha level of significance as well as the null distribution
    Parameters
    ----------
    :param y_train: 1D array, true y training labels
    :param X_train: 3D array, true training epochs
    :param y_test: 1D array, true y testing labels
    :param X_test: 3D array, true testing labels
    :param clf: sklearn pipeline, contains the estimator for generalization
    :param fname: str, Path to save scrambled scores at
    :param n_permutations: int, how often to draw null distributions
    :param alpha: significance level for the p-value map
    """
    y_train_scrambled = y_train.copy()
    # do a permutation test comparison
    null_distribution = []
    for i in range(n_permutations):
        # shuffle works in place
        np.random.shuffle(y_train_scrambled)
        # build a new classifier based on scrambled data
        null_time_gen = GeneralizingEstimator(clf, scoring='accuracy',
                                              n_jobs=-1, verbose=True)
        # train on the motor response
        null_time_gen.fit(X=X_train, y=y_train_scrambled)
        # test on the stimulus presentation, with true labels
        scrambled_scores = null_time_gen.score(X=X_test, y=y_test)
        null_distribution.append(scrambled_scores)
    null_distribution = np.asarray(null_distribution)
    # save the null distribution
    logging.info(f"Saving scrambled scores into {fname}")
    np.save(fname, null_distribution)

    # create distributions for each time point and model
    p_vals = np.zeros(scores.shape)
    for n in null_distribution:
        p_vals += n >= scores
    p_vals += 1
    p_vals /= (len(null_distribution + 1))
    # create a binary mask from p_vals
    binary_mask = p_vals > alpha
    return null_distribution, binary_mask


def _train_and_score_generalizer(X_train, X_test, y_train, y_test, fname):
    """Helper to set up and initially train and score a pipeline and
     Generalizing Estimator. Saves the scores, and returns the pipelines and
     generated scores"""
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
    logging.info(f"Saving generalization scores into {fname}")
    np.save(fname, scores)
    return scores, clf, time_gen


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
        n_permutations=10000
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
            scrambled_scores = {}
            for sub in np.arange(1, 23):
                subject = f'00{sub}' if sub < 10 else f'0{sub}'

                fname = figdir / f'sub-{subject}' / \
                        f'sub-{subject}_gen-scores_{target}-{condition}_true-y.npy'
                truescore = np.load(fname)
                true_scores.append(truescore)

                fname_scrambled = figdir / f'sub-{subject}' / \
                    f'sub-{subject}_gen-scores_{target}-{condition}_scrambled.npy'
                scrambled_score = np.load(fname_scrambled)
                scrambled_scores[sub] = scrambled_score
            avg_trues = np.mean(true_scores, axis=0)
            # permutate
            null_distribution = []
            permutation_count = scrambled_scores[sub].shape[0]
            for i in range(n_permutations):
                null_distribution.append(
                    np.mean(
                        [scrambled_scores[sub][np.random.randint(
                            0, high=permutation_count, size=1)[0]]
                         for sub in np.arange(1, 23)],
                        axis=0
                    ),
                )
            null_distribution = np.asarray(null_distribution)
            # create distributions for each time point and model
            p_vals = np.zeros(avg_trues.shape)
            for n in null_distribution:
                p_vals += n >= avg_trues
            p_vals += 1
            p_vals /= (len(null_distribution + 1))
            # create a binary mask from p_vals
            binary_mask = p_vals > 0.05
            description = f'averaged_actual'
            plot_generalization(avg_trues, description=description,
                                condition=condition, target=target,
                                fpath=figdir, subject='group', mask=binary_mask,
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
            # create distributions for each time point and model
            p_vals = np.zeros(avg_hypos.shape)
            for n in null_distribution:
                p_vals += n >= avg_hypos
            p_vals += 1
            p_vals /= (len(null_distribution + 1))
            # create a binary mask from p_vals
            binary_mask = p_vals > 0.05
            description = f'averaged_hypothetical'
            plot_generalization(avg_hypos, description=description,
                                condition=condition, target=target,
                                fpath=figdir, subject='group', mask=binary_mask,
                                fixed_cbar=False)
    # finally, compute group aggregates for the estimated plots
    estimated_scores = []
    estimated_scrambled = {}
    for sub in np.arange(1, 23):
        subject = f'00{sub}' if sub < 10 else f'0{sub}'

        fname = figdir / f'sub-{subject}' / \
                f'sub-{subject}_gen-scores_estimated-y.npy'
        estimatedscore = np.load(fname)
        estimated_scores.append(estimatedscore)

        fname_scrambled = figdir / f'sub-{subject}' / \
                          f'sub-{subject}_gen-scores_estimated-y_scrambled.npy'
        scrambled_score = np.load(fname_scrambled)
        estimated_scrambled[sub] = scrambled_score
    avg_estimates = np.mean(estimated_scores, axis=0)
    # permutate
    null_distribution = []
    permutation_count = estimated_scrambled[sub].shape[0]
    for i in range(n_permutations):
        null_distribution.append(
            np.mean(
                [estimated_scrambled[sub][np.random.randint(
                    0, high=permutation_count, size=1)[0]]
                 for sub in np.arange(1, 23)],
                axis=0
            ),
        )
    null_distribution = np.asarray(null_distribution)
    # create distributions for each time point and model
    p_vals = np.zeros(avg_estimates.shape)
    for n in null_distribution:
        p_vals += n >= avg_estimates
    p_vals += 1
    p_vals /= (len(null_distribution + 1))
    # create a binary mask from p_vals
    binary_mask = p_vals > 0.05
    description = f'averaged_estimated'
    plot_generalization(avg_estimates, description=description,
                        condition='trials', target='all',
                        fpath=figdir, subject='group', mask=binary_mask,
                        fixed_cbar=False)

    # finally, compute group aggregates for the estimated plots
    for condition in ['flip', 'non-flip']:
        estimated_scores = []
        estimated_scrambled = {}
        for sub in np.arange(1, 23):
            subject = f'00{sub}' if sub < 10 else f'0{sub}'

            fname = figdir / f'sub-{subject}' / \
                    f'sub-{subject}_gen-scores_{condition}-estimated-y.npy'
            estimatedscore = np.load(fname)
            estimated_scores.append(estimatedscore)

            fname_scrambled = figdir / f'sub-{subject}' / \
                              f'sub-{subject}_gen-scores_estimated-y_scrambled.npy'
            scrambled_score = np.load(fname_scrambled)
            estimated_scrambled[sub] = scrambled_score
        avg_estimates = np.mean(estimated_scores, axis=0)
        # permutate
        null_distribution = []
        permutation_count = estimated_scrambled[sub].shape[0]
        for i in range(n_permutations):
            null_distribution.append(
                np.mean(
                    [estimated_scrambled[sub][np.random.randint(
                        0, high=permutation_count, size=1)[0]]
                     for sub in np.arange(1, 23)],
                    axis=0
                ),
            )
        null_distribution = np.asarray(null_distribution)
        # create distributions for each time point and model
        p_vals = np.zeros(avg_estimates.shape)
        for n in null_distribution:
            p_vals += n >= avg_estimates
        p_vals += 1
        p_vals /= (len(null_distribution + 1))
        # create a binary mask from p_vals
        binary_mask = p_vals > 0.05
        description = f'{condition}_averaged_estimated'
        plot_generalization(avg_estimates, description=description,
                            condition='trials', target='all',
                            fpath=figdir, subject='group', mask=binary_mask,
                            fixed_cbar=False)


def generalization_integrating_behavior(subject,
                                        trainingdir,
                                        testingdir,
                                        bidsdir,
                                        figdir,
                                        n_permutations=100,
                                        ):

    dec_factor = 5
    fpath = Path(_construct_path([figdir, f'sub-{subject}/']))

    train_fullsample, train_data, test_fullsample, test_data = \
        _read_test_n_train(subject=subject, trainingdir=trainingdir,
                           testingdir=testingdir, bidsdir=bidsdir)
    # train on all trials, except for trials where no reaction was made
    # train on all trials, except for trials where no reaction was made
    X_train, y_train = _make_X_n_y(train_fullsample, subject,
                                   dec_factor, drop_non_responses=True)

    # get the test data
    X_test = np.array([decimate(epoch['normalized_data'], dec_factor)
                       for id, epoch in test_fullsample[subject].items()])
    # calculate hypothetical labels. First, get regression coefficients
    # TODO: logistic regression with only left characteristics?
    Lprob, Lmag, LEV, Rprob, Rmag, REV = logreg(bidsdir=bidsdir,
                                                figdir='/tmp',
                                                n_splits=100,
                                                subject=subject)[subject]['pure_coefs']
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
    df['REV'] = (df.RoptProb - df.RoptProb.mean()) * \
                     (df.RoptMag - df.RoptMag.mean())
    # z-score everything
    #df = df.apply(zscore)
    # min-max scale everything
    df = (df - df.min()) / (df.max() - df.min())
    df['integrate'] = (df.LoptMag * Lmag) + (df.LoptProb * Lprob) + (df.LEV * LEV)
    df['integrate_all_infos'] = (df.LoptMag * Lmag) + (df.LoptProb * Lprob) + \
                                (df.LEV * LEV) + (df.RoptMag * Rmag) + \
                                (df.RoptProb * Rprob) + (df.REV * REV)
    # split in the highest and lowest 25%
    for col, target in [('integrate', 'choice'),
                        ('integrate_all_infos', 'choice_all_infos')]:
        lower, upper = df[col].quantile([0.49, 0.51])
        # more negative = left choice
        conditions = [df[col] >= upper,
                      df[col] <= lower]
        # we believe that choice2.0 is right, choice1.0 is left
        choices = ['choice2.0', "choice1.0"]
        df[target] = np.select(conditions, choices, default=None)
    # get indices of all trials that did not make the cut
    medium_trials = np.where(df['choice'].values == None)[0]
    # get all trials were infos from left set of weights and all weights diverge
    flipping_trials = np.where(df['choice'].values != df['choice_all_infos'].values)[0]
    # combine them
    trials_to_exclude = np.union1d(medium_trials, flipping_trials)

    # remove the trials that didn't make the cut from the data
    X_test_flips = X_test[flipping_trials]
    X_test = np.delete(X_test, trials_to_exclude, axis=0)
    y_test = np.delete(np.array(df.choice), trials_to_exclude, axis=0)
    y_test_flips = np.array(df.choice)[flipping_trials]

    for condition, X, y in [('non-flip', X_test, y_test),
                            ('flip', X_test_flips, y_test_flips)]:
        fname = fpath / \
                f'sub-{subject}_gen-scores_{condition}-estimated-y.npy'
        scores, clf, time_gen = \
            _train_and_score_generalizer(X_train=X_train, X_test=X,
                                         y_train=y_train, y_test=y,
                                         fname=fname)

        fname = fpath / \
                f'sub-{subject}_gen-scores_{condition}-estimated-y_scrambled.npy'
        null_distribution, binary_mask = \
            _permute(y_train=y_train, X_train=X_train, y_test=y,
                     X_test=X, clf=clf, fname=fname, scores=scores,
                     n_permutations=n_permutations, alpha=0.05)

        plot_generalization(scoring=scores,
                            description='estimated',
                            condition=condition,
                            target='all',
                            fpath=fpath,
                            subject=subject,
                            mask=binary_mask,
                            fixed_cbar=False)


