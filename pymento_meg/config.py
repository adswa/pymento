"""
General configurations and mappings

"""


# In the memento task, the behavioral responses of participants were written to
# log files.
# However, different participants played different versions of the task, and
# different versions of the task saved a different amount of variables as a
# Matlab struct into the log file.
# This file contains information on the variables and their indexes per subject.
# Quick overview on the meaning of the variables:
# pointdiff = points gained in the trial (0.5, 1, 2, 4 if success, nan if not)
# points = total points (only non-nan when the trial was successful)
# breaktrial = takes value of 1 if trial ends with a break (5 times)

smaller_onsets = [
    "fix_onset",
    "LoptOnset",
    "or_onset",
    "RoptOnset",
    "response_onset",
    "feedback_onset",
]
larger_onsets = [
    "fix_onset",
    "pause_start",
    "LoptOnset",
    "or_onset",
    "RoptOnset",
    "response_onset",
    "feedback_onset",
    "Empty_screen",
    "timeoutflag",
]
largest_onsets = [
    "fix_onset",
    "pause_start",
    "LoptOnset",
    "or_onset",
    "second_delay_screen",
    "RoptOnset",
    "response_onset",
    "feedback_onset",
    "Empty_screen",
    "timeoutflag",
]
larger_probmagrew = [
    "trial_no",
    "LoptProb",
    "LoptMag",
    "RoptProb",
    "RoptMag",
    "LoptRew",
    "RoptRew",
    "choice",
    "RT",
    "points",
    "pointdiff",
    "timeoutflag",
    "breaktrial",
]
smaller_probmagrew = [
    "trial_no",
    "LoptProb",
    "LoptMag",
    "RoptProb",
    "RoptMag",
    "LoptRew",
    "RoptRew",
    "choice",
    "RT",
    "points",
    "pointdiff",
    "breaktrial",
]
disptimes = [
    "trial_no",
    "FixReqT",
    "FixTime",
    "orReqTime",
    "orTime",
    "LoptT",
    "RoptT",
    "FeedbackT",
]
single_onsets = ["empty_start_screen", "start", "instruction_onset", "end_onset"]


subjectmapping = {
    "memento_001": {
        "probmagrew": larger_probmagrew,
        "onsets": smaller_onsets,
        "single_onsets": False,
        "disptimes": disptimes,
        "logfilename": "memento_001/mementoLOG_1.mat",
    },
    "memento_002": {
        "probmagrew": larger_probmagrew,
        "onsets": smaller_onsets,
        "single_onsets": False,
        "disptimes": disptimes,
        "logfilename": "memento_002/mementoLOG_2.mat",
    },
    "memento_003": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_003/mementoLOG_3.mat",
    },
    "memento_004": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_004/mementoLOG_4.mat",
    },
    "memento_005": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_005/mementoLOG_5.mat",
    },
    "memento_006": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_006/mementoLOG_6.mat",
    },
    "memento_007": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_007/mementoLOG_7.mat",
    },
    "memento_008": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_008/mementoLOG_8.mat",
    },
    "memento_009": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_009/mementoLOG_9.mat",
    },
    "memento_010": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_010/mementoLOG_10.mat",
    },
    "memento_011": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_011/mementoLOG_11.mat",
    },
    "memento_012": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_012/mementoLOG_12.mat",
    },
    "memento_013": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_013/mementoLOG_13.mat",
    },
    "memento_014": {
        "probmagrew": smaller_probmagrew,
        "onsets": larger_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_014/mementoLOG_14.mat",
    },
    "memento_015": {
        "probmagrew": smaller_probmagrew,
        "onsets": largest_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_0015/mementoLOG_15.mat",
    },
    "memento_016": {
        "probmagrew": smaller_probmagrew,
        "onsets": largest_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_0016/mementoLOG_16.mat",
    },
    "memento_017": {
        "probmagrew": smaller_probmagrew,
        "onsets": largest_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_0017/mementoLOG_17.mat",
    },
    "memento_018": {
        "probmagrew": smaller_probmagrew,
        "onsets": largest_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_0018/mementoLOG_18.mat",
    },
    "memento_019": {
        "probmagrew": smaller_probmagrew,
        "onsets": largest_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_0019/mementoLOG_19.mat",
    },
    "memento_020": {
        "probmagrew": smaller_probmagrew,
        "onsets": largest_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_0020/mementoLOG_20.mat",
    },
    "memento_021": {
        "probmagrew": smaller_probmagrew,
        "onsets": largest_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_0021/mementoLOG_21.mat",
    },
    "memento_022": {
        "probmagrew": smaller_probmagrew,
        "onsets": largest_onsets,
        "single_onsets": True,
        "disptimes": disptimes,
        "logfilename": "memento_0022/mementoLOG_22.mat",
    },
}

# all relevant subjects (for now, not those from ping)
subject_list = [
    "001",
    "002",
    "003",
    "004",
    "005",
    "006",
    "007",
    "008",
    "009",
    "010",
    "011",
    "012",
    "013",
    "014",
    "015",
    "016",
    "017",
    "018",
    "019",
    "020",
    "021",
    "022",
]

# set channel types explicitly
channel_types = {
    "EOG001": "eog",
    "EOG002": "eog",
    "ECG003": "ecg",
}

# criteria for bad or flat data
# TODO: check if these criteria make sense. Taken from
# https://github.com/hoechenberger/pybrain_mne/blob/main/04-cleaning_data.ipynb
reject_criteria = dict(
    mag=3000e-15, grad=3000e-13, eog=200e-6  # 3000 fT  # 3000 fT/cm
)  # 200 µV

flat_criteria = dict(
    mag=1e-15,  # 1 fT
    grad=1e-13,  # 1 fT/cm
)

# Configuration file names from the Elekta system
crosstalk_file = "ct_sparse.fif"
fine_cal_file = "sss_cal.data"


# human readable names for MEG triggers
# TriggerName description based on experiment matlab files.
# all left options (shown first in each trial) are summarized as "visualfirst".
# lOpt10 and rOpt1 don't seem to exist (for sub 4 at least?)
event_dict = {
    "press/left": 1,
    "end": 2,
    "start": 3,
    "press/right": 4,
    "photodiode/onset": 5,
    "visualfix/fixCross": 10,
    "visualfirst/lOpt1": 12,
    "visualfirst/lOpt2": 13,
    "visualfirst/lOpt3": 14,
    "visualfirst/lOpt4": 15,
    "visualfirst/lOpt5": 16,
    "visualfirst/lOpt6": 17,
    "visualfirst/lOpt7": 18,
    "visualfirst/lOpt8": 19,
    "visualfirst/lOpt9": 20,
    # "visualfirst/lOpt10": 21, # does not exist according to Luca!
    "visualsecond/rOpt": 24,
    "delay": 22,
    "empty_screen": 26,
    "pauseStart": 25,
    "feedback": 27,
    "feedback-fast": 28,
}


# Notes on ICA artifacts for each subject. They were found by hand by
# - visually inspecting ICA components
# - visually inspecting automatically detected EOG and ECG components found by
# eog_indices, eog_scores = ica.find_bads_eog(filt_raw)
# ecg_indices, ecg_scores = ica.find_bads_ecg(filt_raw, method='ctps',
#                                             threshold='auto')
ica_comps = {
    '001': {'ecg': [1, 4],
            'eog': [0, 23]},    # drops 50 epochs
    '002': {'ecg': [24, 34],
            'eog': [33]},       # drops 120 epochs
    '003': {'ecg': [13, 29],
            'eog': [0, 1]},     # drops 218 epochs
    '004': {'ecg': [24, 17],
            'eog': [0]},        # drops 240 epochs
    '005': {'ecg': [2, 37],
            'eog': [1, 13]},    # drops 23 epochs
    '006': {'ecg': [32],
            'eog': [4, 26]},    # drops 18
    '007': {'ecg': [1, 14],
            'eog': [0]},        # drops 269 epochs
    '008': {'ecg': [27],
            'eog': [4]},        # drops 19 epochs
    '009': {'ecg': [6, 41],     # without zapline filtering
            'eog': [0, 1, 2]},  # drops 189 epochs (without ZAPline)
    '010': {'ecg': [5, 15],
            'eog': [0]},        # drops 24 epochs
    '011': {'ecg': [2, 3],
            'eog': [0, 12]},    # drops 3 epochs
    '012': {'ecg': [6, 34],
            'eog': [0, 5]},     # drops 51 epochs
    '013': {'ecg': [7, 14],
            'eog': [2, 24]},    # drops 10  (40?) epochs
    '014': {'ecg': [5, 17, 25],
            'eog': [0, 6]},     # drops 27 epochs
    '015': {'ecg': [1, 6],
            'eog': [0, 16]},    # drops 20 epochs
    '016': {'ecg': [1],
            'eog': [2, 15]},    # drops 4 epochs
    '017': {'ecg': [16, 6],     # set bad channels: 0313 0513 0523
            'eog': [4, 21]},    # drops 62 epochs
    '018': {'ecg': [22, 31],
            'eog': [2]},        # drops 3 epochs
    '019': {'ecg': [4, 31],
            'eog': [13]},       # drops 42 (11?) epochs
    '020': {'ecg': [10, 43],
            'eog': [7, 31]},    # drops 3 epochs
    '021': {'ecg': [5, 32, 42],
            'eog': [11]},       # drops 109 epochs
    '022': {'ecg': [8, 33],
            'eog': [2, 24]},    # drops 18 epochs
}


# map a set of reward magnitude and probability (in the order) to trial types.
# the characteristic combinations with letters are frequent (50-75 over the
# course of the experiment) and occur in the left and right stimulus. The
# non-letter combinations occur only on the right side, and some of them are
# infrequent (between 1 and 5? occurrences over the course of the experiment)
trial_characteristics = {
    (0.5, 0.4): 'A',
    (0.5, 0.8): 'B',
    (1, 0.2): 'C',
    (1, 0.8): 'D',
    (2, 0.1): 'E',
    (2, 0.2): 'F',
    (2, 0.4): 'G',
    (4, 0.1): 'H',
    (4, 0.2): 'I',
    (0.5, 0.1): '0.5-0.1',
    (0.5, 0.2): '0.5-0.2',
    (1, 0.4): '1.0-0.4',
    (1, 0.1): '1.0-0.1',
    (2, 0.8): '2.0-0.8',
    (4, 0.4): '4.0-0.4',
    (4, 0.8): '4.0-0.8'
}
