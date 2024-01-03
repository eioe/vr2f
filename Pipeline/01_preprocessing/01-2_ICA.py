import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import create_eog_epochs

from vr2fem_analyses.staticinfo import PATHS
from vr2fem_analyses.staticinfo import TIMINGS
from vr2fem_analyses import preprocess, helpers


def main(sub_nr: int):
    paths = PATHS()
    timings = TIMINGS()
    path_data = Path(paths.DATA_01_EPO, "erp")
    # load data
    sub_list_str = [s.split("-epo")[0] for s in os.listdir(path_data) if s.startswith("VR2FEM")]
    sub_list_str.sort()

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]

    for subID in sub_list_str:

        fname = Path(paths.DATA_01_EPO, "ica", "clean", f"{subID}-epo.fif")
        data_forICA = mne.read_epochs(fname)

        # clean it with autoreject local to remove bad epochs for better ICA fit:
        data_forAR = data_forICA.copy().apply_baseline((-timings.DUR_BL, 0))
        # AR does not perform well on non-baseline corrected data

        _, ar, reject_log = preprocess.clean_with_ar_local(
            subID,
            data_forAR,
            ar_from_disc=False,
            save_to_disc=True,
            ar_path=paths.DATA_02_ICA_AR,
        )

        # Get ICA weights
        _ = preprocess.get_ica_weights(
            subID,
            data_forICA[~reject_log.bad_epochs],
            picks=['eeg', 'eog'],
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
