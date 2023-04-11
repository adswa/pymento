import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

from scipy.signal import decimate

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from mne.decoding import GeneralizingEstimator
from pymento_meg.decoding.logreg import known_targets
from pymento_meg.srm.srm import get_general_data_structure
from pymento_meg.utils import _construct_path


def generalize(subject,
               trainingdir,
               testingdir,
               bidsdir,
               figdir,
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
    :param subject:
    :param trainingdir: Directory with epochs centered around response
    :param testingdir: Directory with epochs centered around visual stimulus 1
    :param bidsdir:
    :param figdir:
    :return:
    """
    dec_factor = 5
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
        timespan=[0, 2.7])

    # custom colormap to overlay p-values:
    import matplotlib.colors as mc
    c_white = mc.colorConverter.to_rgba('white', alpha=0)
    c_black = mc.colorConverter.to_rgba('gray', alpha=1)
    cmap_rb = mc.LinearSegmentedColormap.from_list('rb_cmap',
                                                   [c_white, c_black],
                                                   512)

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

            y_test_copy = y_test.copy()
            # do a permutation test comparison
            null_distribution = []
            n_permutations = 50
            for i in range(n_permutations):
                # shuffle works in place
                np.random.shuffle(y_test_copy)
                scrambled_scores = time_gen.score(X=X_test, y=y_test_copy)
                null_distribution.append(scrambled_scores)
            null_distribution = np.asarray(null_distribution)
            # save the null distribution
            fname = fpath / \
                    f'sub-{subject}_gen-scores_{target}-{condition}_scrambled.npy'
            logging.info(f"Saving scrambled scores into {fname}")
            np.save(fname, null_distribution)

            # create distributions for each time point and model
            p_vals = np.zeros(null_distribution.shape[-2:])
            for model in range(null_distribution.shape[1]):
                for timepoint in range(null_distribution.shape[2]):
                    # the proportion of null_values greater than real values
                    p_vals[model, timepoint] = \
                        ((null_distribution[:, model, timepoint] >= scores[model, timepoint]).sum() + 1) / (
                                n_permutations + 1)
            binary_mask = p_vals > 0.05

            if scores_hypothetical is not None:
                p_vals = np.zeros(null_distribution.shape[-2:])
                for model in range(null_distribution.shape[1]):
                    for timepoint in range(null_distribution.shape[2]):
                        # the proportion of null_values greater than real values
                        p_vals[model, timepoint] = \
                            ((null_distribution[:, model, timepoint] >= scores_hypothetical[
                                model, timepoint]).sum() + 1) / (
                                    n_permutations + 1)
                binary_mask_hypo = p_vals > 0.05

            for scoring, description, mask in \
                    [(scores, 'actual', binary_mask),
                     (scores_hypothetical, 'hypothetical', binary_mask_hypo)]:
                if scores_hypothetical is None:
                    continue
                # plot
                fig, ax = plt.subplots(1, figsize=[9, 4], frameon=False)
                im = ax.matshow(scoring, vmin=0., vmax=1.,
                                cmap='RdBu_r', origin='lower',
                                extent=np.array([0, 2700, -500, 500]))
                ax.axhline(0, color='k', linestyle='dotted', label='motor response')
                ax.axvline(700, color='k', label='stimulus offset')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xlabel('Test Time (ms), stimulus 1 and delay period')
                ax.set_ylabel('Train Time (ms), \n response-centered')
                plt.suptitle(f'Generalization based on {condition} {target} ({description} targets)')
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
                # overlay the p-values. non-significant areas will become black
                im2 = ax.matshow(mask, cmap=cmap_rb, vmin=0.,
                                 vmax=1., origin='lower',
                                extent=np.array([0, 2700, -500, 500]))
                ax.xaxis.set_ticks_position('bottom')
                ax.set_aspect('auto')
                fname = fpath / \
                        f'sub-{subject}_generalization_{target}-{condition}_{description}_pval-mask.png'
                logging.info(f"Saving generalization plot into {fname}")
                fig.savefig(fname)
