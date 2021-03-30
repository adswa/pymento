import argparse
from pathlib import Path
import os.path as op
import pymento_meg

version = pymento_meg.__version__
# TODO: redo this with less duplication in argparsing

def parse_args_main():

    formatter_class = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(
        formatter_class=formatter_class,
        prog="pymento",
        description="{}".format(main.__doc__)
    )
    parser.add_argument(
        "--version",
        action="store_true"
    )
    parser.add_argument(
        "--subject",
        "-s",
        metavar="<subject identifier>",
        help="""Subject identifier, e.g., '001'""",
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
        description="{}".format(restructure.__doc__)
    )
    parser.add_argument(
        "--version",
        action="store_true"
    )
    parser.add_argument(
        "--subject",
        "-s",
        metavar="<subject identifier>",
        help="""Subject identifier, e.g., '001'""",
        required=True
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
        required=True
    )
    parser.add_argument(
        "--calfiles_dir",
        "-c",
        help="""A path to a directory where the fine-calibration
                        and crosstalk-compensation files of the Elekta system
                        lie. They should be named 'ct_sparse.fif and sss_cal.dat'.
                        They are required during Maxwell Filtering.""",
        required=True
    )
    parser.add_argument(
        "--diagnostics_dir",
        "-d",
        help="""A path to a directory where diagnostic figures
                        will be saved under""",
        required=True
    )
    parser.add_argument(
        "--log-file-dir",
        "-l",
        help="""A path to a directory where the matlab log files are stored.""",
        required=True
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
        description="{}".format(sss.__doc__)
    )
    parser.add_argument(
        "--version",
        action="store_true"
    )
    parser.add_argument(
        "--subject",
        "-s",
        metavar="<subject identifier>",
        help="""Subject identifier, e.g., '001'""",
        required=True
    )
    parser.add_argument(
        "--bids_data_dir",
        "-r",
        metavar="<bids data directory>",
        help="""Provide a path to the bids-structured directory for the
                        memento sample.""",
        default="/data/project/brainpeach/memento/memento-bids/",
        required=True
    )
    parser.add_argument(
        "--bids_deriv_dir",
        help="""A path to a directory where sss processed data will be saved in,
                        complying to BIDS naming conventions""",
        default=None,
        required=True
    )
    parser.add_argument(
        "--diagnostics_dir",
        "-d",
        help="""A path to a directory where diagnostic figures
                        will be saved under""",
        required=True
    )

    args = parser.parse_args()
    if args.version:
        print(version)
    return args


def restructure():
    """
    Restructure raw, original memento data into a raw BIDS directory.

    """
    from .pymento import restructure_to_bids
    args = parse_args_restructure()
    crosstalk_file = Path(args.calfiles_dir) / "ct_sparse.fif"
    fine_cal_file = Path(args.calfiles_dir) / "sss_cal.dat"

    restructure_to_bids(rawdir=args.raw_data_dir,
                        subject=args.subject,
                        bidsdir=args.bids_dir,
                        figdir=args.diagnostics_dir,
                        crosstalk_file=crosstalk_file,
                        fine_cal_file=fine_cal_file,
                        behav_dir=args.log_file_dir)


def sss():
    """
    Based on a raw BIDS directory, create and save SSS processed data BIDS-like

    """
    from .pymento import signal_space_separation
    args = parse_args_sss()
    signal_space_separation(bidspath=args.bids_data_dir,
                            subject=args.subject,
                            figdir=args.diagnostics_dir,
                            derived_path=args.bids_deriv_dir)



def main():
    """
    pymento is a library of Python functions to analyze memento project data
    """
    from pymento_meg import utils as ut
    args = parse_args()


    assert op.exists(crosstalk_file)
    assert op.exists(fine_cal_file)

    print(
        f"I received the following command line arguments: \n"
        f"Subject: {args.subject}\n"
        f"Raw data directory: {args.raw_data_dir}\n"
        f"Behavioral data directory: {args.behav_data_dir}\n"
    )

    # read in raw data
    raw = ut.read_data_original(directory=args.raw_data_dir, subject=args.subject)

    if args.crop:
        raw = raw.crop(tmax=100)

    raw_sss = ut.maxwellfilter(
        raw=raw,
        subject=args.subject,
        crosstalk_file=crosstalk_file,
        fine_cal_file=fine_cal_file,
        headpos_file=None,
        compute_motion_params=True,
        head_pos_outdir=args.bids_dir,
        figdir=args.diagnostics_dir,
        outdir=args.bids_dir,
        filtering=True,
        filter_args={"h_freq": 45},
    )

    from .config import event_dict
    ut.evoked_visual_potentials(raw=raw_sss,
                                subject=args.subject,
                                event_dict=event_dict,
                                figdir=args.diagnostics_dir)


if __name__ == "__main__":
    main()
