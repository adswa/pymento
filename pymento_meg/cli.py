import argparse
import logging
from pathlib import Path
import pymento_meg

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
version = pymento_meg.__version__
# TODO: redo this with less duplication in argparsing


def parse_args_main():

    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        prog="pymento",
        description="{}".format(main.__doc__),
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--subject",
        "-s",
        metavar="<subject identifier>",
        help="""
        Subject identifier, e.g., '001'. If several identifiers are given,
        an SRM is trained on data from all subjects""",
    )
    parser.add_argument(
        "--raw_data_dir",
        "-r",
        metavar="<raw data directory>",
        help="""Provide a path to the raw data directory for the
                        complete memento sample.""",
        default="/data/project/brainpeach/memento/data/DMS_MEMENTO/Data_MEG/RawData_MEG_memento/",
    )
    parser.add_argument(
        "--behav_data_dir",
        "-b",
        metavar="<behavioral data directory>",
        help="""Provide a path to the behavioral data directory for the
                        complete memento sample.""",
        default="/data/project/brainpeach/memento/data/DMS_MEMENTO/Data_Behav/Data_Behav_Memento/",
    )
    parser.add_argument(
        "--bids_dir",
        help="""A path to a directory where raw data and
                        processed data will be saved in, complying to BIDS
                        naming conventions""",
        default=None,
    )
    parser.add_argument(
        "--calfiles_dir",
        "-c",
        help="""A path to a directory where the fine-calibration
                        and crosstalk-compensation files of the Elekta system
                        lie. They should be named 'ct_sparse.fif and sss_cal.dat'.
                        They are required during Maxwell Filtering.""",
    )
    parser.add_argument(
        "--diagnostics_dir",
        "-d",
        help="""A path to a directory where diagnostic figures
                        will be saved under""",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="""If true, the data is cropped to 10 minutes to
                        do less computationally intensive test-runs.""",
    )

    args = parser.parse_args()
    if args.version:
        print(version)
    return args


def parse_args_restructure():

    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        prog="pymento",
        description="{}".format(restructure.__doc__),
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--subject",
        "-s",
        metavar="<subject identifier>",
        help="""Subject identifier, e.g., '001'""",
        required=True,
    )
    parser.add_argument(
        "--raw_data_dir",
        "-r",
        metavar="<raw data directory>",
        help="""Provide a path to the raw data directory for the
                        complete memento sample.""",
        default="/data/project/brainpeach/memento/data/DMS_MEMENTO/Data_MEG/RawData_MEG_memento/",
    )
    parser.add_argument(
        "--bids_dir",
        help="""A path to a directory where raw data and
                        processed data will be saved in, complying to BIDS
                        naming conventions""",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--calfiles_dir",
        "-c",
        help="""A path to a directory where the fine-calibration
                        and crosstalk-compensation files of the Elekta system
                        lie. They should be named 'ct_sparse.fif and sss_cal.dat'.
                        They are required during Maxwell Filtering.""",
        required=True,
    )
    parser.add_argument(
        "--diagnostics_dir",
        "-d",
        help="""A path to a directory where diagnostic figures
                        will be saved under""",
        required=True,
    )
    parser.add_argument(
        "--log-file-dir",
        "-l",
        help="""A path to a directory where the matlab log files are stored.""",
        required=True,
    )
    args = parser.parse_args()
    if args.version:
        print(version)
    return args


def parse_args_sss():

    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        prog="pymento",
        description="{}".format(sss.__doc__),
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--subject",
        "-s",
        metavar="<subject identifier>",
        help="""Subject identifier, e.g., '001'""",
        required=True,
    )
    parser.add_argument(
        "--bids_data_dir",
        "-r",
        metavar="<bids data directory>",
        help="""Provide a path to the bids-structured directory for the
                        memento sample.""",
        default="/data/project/brainpeach/memento/memento-bids/",
        required=True,
    )
    parser.add_argument(
        "--bids_deriv_dir",
        help="""A path to a directory where sss processed data will be saved in,
                        complying to BIDS naming conventions""",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--diagnostics_dir",
        "-d",
        help="""A path to a directory where diagnostic figures
                        will be saved under""",
        required=True,
    )

    args = parser.parse_args()
    if args.version:
        print(version)
    return args


def parse_args_epochnclean():

    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        prog="pymento",
        description="{}".format(epoch_and_clean.__doc__),
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--subject",
        "-s",
        metavar="<subject identifier>",
        help="""Subject identifier, e.g., '001'""",
        required=True,
    )
    parser.add_argument(
        "--bidsdir",
        "-b",
        metavar="<bids data directory>",
        help="""Provide a path to the bids-structured directory for the
                memento sample.""",
        default="/data/project/brainpeach/memento/memento-bids/",
        required=True,
    )
    parser.add_argument(
        "--derivdir",
        help="""A path to a directory where cleaned epochs will be saved in""",
        default="/data/project/brainpeach/memento/memento-bids-sss/",
        required=True,
    )
    parser.add_argument(
        "--datadir",
        help="""A path to the directory with sss-processed memento data""",
        default="/data/project/brainpeach/memento/memento-bids-sss/",
        required=True,
    )
    parser.add_argument(
        "--diagdir",
        "-d",
        help="""A path to a directory where diagnostic figures will be saved""",
        required=True,
    )
    parser.add_argument(
        "--event",
        "-e",
        help="An identifier for the timepoint at which an epoch will start. "
             "Currently implemented: 'visualfix' (the start of the "
             "fixation cross in each trial), 'visualfirst' (the start of the "
             "left visual stimulus).",
        default='visualfirst',
        choices=['visualfirst', 'visualfix'],
    )

    args = parser.parse_args()
    if args.version:
        print(version)
    return args


def parse_args_srm():

    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        prog="pymento",
        description="{}".format(srm.__doc__),
    )
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--subject",
        "-s",
        nargs="+",
        metavar="<subject identifier>",
        help="""Subject identifier, e.g., '001'""",
        required=True,
    )
    parser.add_argument(
        "--bids_data_dir",
        "-r",
        metavar="<bids data directory>",
        help="""Provide a path to the bids-structured directory for the
                        memento sample.""",
        default="/data/project/brainpeach/memento/memento-bids/",
        required=True,
    )
    parser.add_argument(
        "--bids_deriv_dir",
        help="""A path to a directory where sss processed data will be saved in,
                        complying to BIDS naming conventions""",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--diagnostics_dir",
        "-d",
        help="""A path to a directory where diagnostic figures
                        will be saved under""",
        required=True,
    )
    parser.add_argument(
        "--condition",
        "-c",
        help="""The condition to split the trials over""",
        choices=['left-right', 'nobrain-brain'],
        default='left-right'
    )
    parser.add_argument(
        "--timespan",
        "-t",
        help="""The timespan to use""",
        choices=['fulltrial', 'decision', 'firststim'],
        default='fulltrial'
    )



    args = parser.parse_args()
    if args.version:
        print(version)
    return args


def restructure():
    """
    Restructure raw, original memento data into a raw BIDS directory.

    """
    from pymento_meg.pymento import restructure_to_bids

    args = parse_args_restructure()
    crosstalk_file = Path(args.calfiles_dir) / "ct_sparse.fif"
    fine_cal_file = Path(args.calfiles_dir) / "sss_cal.dat"

    restructure_to_bids(
        rawdir=args.raw_data_dir,
        subject=args.subject,
        bidsdir=args.bids_dir,
        figdir=args.diagnostics_dir,
        crosstalk_file=crosstalk_file,
        fine_cal_file=fine_cal_file,
        behav_dir=args.log_file_dir,
    )


def sss():
    """
    Based on a raw BIDS directory, create and save SSS processed data BIDS-like

    """
    from pymento_meg.pymento import signal_space_separation

    args = parse_args_sss()
    signal_space_separation(
        bidspath=args.bids_data_dir,
        subject=args.subject,
        figdir=args.diagnostics_dir,
        derived_path=args.bids_deriv_dir,
    )


def epoch_and_clean():
    """
    Epoch the data and remove artifacts and bad trials
    :return: Will save cleaned epochs into <DERIVDIR>
    """
    from pymento_meg.pymento import epoch_and_clean_trials
    args = parse_args_epochnclean()
    if args.event == 'visualfirst':
        # set eventid to be all first stimulus types
        eventid = {"visualfirst/lOpt1": 12,
                   "visualfirst/lOpt2": 13,
                   "visualfirst/lOpt3": 14,
                   "visualfirst/lOpt4": 15,
                   "visualfirst/lOpt5": 16,
                   "visualfirst/lOpt6": 17,
                   "visualfirst/lOpt7": 18,
                   "visualfirst/lOpt8": 19,
                   "visualfirst/lOpt9": 20}
    elif args.event == 'visualfix':
        eventid = {'visualfix/fixCross': 10}
    logging.info(f"Using the following event ids for epoching: {eventid}")
    epoch_and_clean_trials(subject=args.subject,
                           diagdir=args.diagdir,
                           bidsdir=args.bidsdir,
                           datadir=args.datadir,
                           derivdir=args.derivdir,
                           eventid=eventid)


def srm():
    """
    Fit a shared response model
    :return:
    """
    from pymento_meg.pymento import SRM
    # for now, reuse the argparse arguments
    args = parse_args_srm()
    SRM(
        subject=args.subject,
        datadir=args.bids_deriv_dir,
        bidsdir=args.bids_data_dir,
        figdir=args.diagnostics_dir,
        condition=args.condition,
        model='srm',
        timespan=args.timespan)


def main():
    """
    pymento is a library of Python functions to analyze memento project data
    """
    print("no main method yet, use command line subfunctionality.")


if __name__ == "__main__":
    main()
