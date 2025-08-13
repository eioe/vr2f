"""Runs the ICA calculation."""

import os
import sys
from pathlib import Path

import mne
from vr2f import helpers, preprocess
from vr2f.staticinfo import TIMINGS

from vr2f.staticinfo import PATHS


def main(sub_nr: int):
    """
    Process EEG data for a specified subject or all subjects if no specific subject is provided.

    This function performs the following steps:
        - loading epoched & filtered data
        - preliminary artifact rejection using AutoReject
        - ICA weight computation (using the Picard algorithm)
        - storing of ICA weights to disc.

    Parameters
    ----------
    sub_nr : int
        Index of the subject to process. If None, processes all subjects.

    """
    paths = PATHS()
    timings = TIMINGS()
    path_data = Path(paths.DATA_01_EPO, "erp")
    # load data
    sub_list_str = [s.split("-epo")[0] for s in os.listdir(path_data) if s.startswith("VR2FEM")]
    sub_list_str.sort()

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]

    for sub_id in sub_list_str:
        fname = Path(paths.DATA_01_EPO, "ica", "clean", f"{sub_id}-epo.fif")
        data_for_ica = mne.read_epochs(fname)

        # clean it with autoreject local to remove bad epochs for better ICA fit:
        data_for_ar = data_for_ica.copy().apply_baseline((-timings.DUR_BL, 0))
        # AR does not perform well on non-baseline corrected data

        _, ar, reject_log = preprocess.clean_with_ar_local(
            sub_id,
            data_for_ar,
            ar_from_disc=False,
            save_to_disc=True,
            ar_path=paths.DATA_02_ICA_AR,
        )

        # Get ICA weights
        _ = preprocess.get_ica_weights(
            sub_id,
            data_for_ica[~reject_log.bad_epochs],
            picks=["eeg", "eog"],
            reject=None,
            method="picard",
            fit_params=None,
            ica_from_disc=False,
            save_to_disc=True,
            ica_path=paths.DATA_02_ICA,
        )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None
    main(JOB_NR)
