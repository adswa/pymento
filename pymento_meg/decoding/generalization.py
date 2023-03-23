import logging
import numpy as np
import matplotlib.pyplot as plt
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

            time_gen = GeneralizingEstimator(clf, scoring='roc_auc',
                                             n_jobs=-1, verbose=True)
            # train on the motor response
            time_gen.fit(X=X_train, y=y_train)
            # test on the stimulus presentation
            scores = time_gen.score(X=X_test, y=y_test)
            # save the scores
            fname = fpath / f'sub-{subject}_gen-scores_{target}-{condition}.npy'
            logging.info(f"Saving generalization scores into {fname}")
            np.save(fname, scores)

            y_test_copy = y_test.copy()
            # do a permutation test comparison
            null_distribution = []
            for i in range(25):
                # shuffle works in place
                np.random.shuffle(y_test_copy)
                scrambled_scores = time_gen.score(X=X_test, y=y_test_copy)
                null_distribution.append(scrambled_scores)
            scrambled_scores = np.mean(null_distribution, axis=0)
            for scoring, description in [(scores, 'actual'),
                                         (scrambled_scores, 'scrambled')]:
                # plot
                fig, ax = plt.subplots(1)
                im = ax.matshow(scoring, vmin=0., vmax=1.,
                                cmap='RdBu_r', origin='lower')
                ax.axhline(500/dec_factor, color='k')
                ax.axvline(700/dec_factor, color='k')
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xlabel('Test Time (5ms), stim 1')
                ax.set_ylabel('Train Time (5ms), response')
                ax.set_title(f'Generalization based on {condition} {target} ({description} data)')
                plt.suptitle("Decoding choice (ROC AUC)")
                plt.colorbar(im, ax=ax)
                plt.tight_layout()
                fname = fpath / f'sub-{subject}_generalization_{target}-{condition}_{description}.png'
                logging.info(f"Saving generalization plot into {fname}")
                fig.savefig(fname)
