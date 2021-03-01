import argparse
from pathlib import Path
import os.path as op


def parse_args():

    parser = argparse.ArgumentParser()
    prog = "pymento"
    description = "{}".format(main.__doc__)
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
    return args


def main():
    """pymento is a library of Python functions to analyze MEG data
    from the memento project."""
    from pymento import utils as ut

    args = parse_args()

    crosstalk_file = Path(args.calfiles_dir) / "ct_sparse.fif"
    fine_cal_file = Path(args.calfiles_dir) / "sss_cal.dat"

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
