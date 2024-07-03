"""Runs local (per trial & channel) automatized rejection (and interpolation)."""
import os
import sys
from pathlib import Path

import mne
import numpy as np

from vr2f import helpers
from vr2f.preprocess import clean_with_ar_local
from vr2f.staticinfo import CONFIG, PATHS, TIMINGS


def norm_vec(x):
    """Normalize a vector."""
    return x / np.sqrt(np.sum(x**2))


def main(sub_nr: int):
    """Run main."""
    paths = PATHS()
    timings = TIMINGS()
    config = CONFIG()
    path_in = Path(paths.DATA_02_ICA, "cleaneddata")
    # load data
    sub_list_str = [s.split("-postICA-epo")[0] for s in os.listdir(path_in)]

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]

    for sub_id in sub_list_str:
        try:
            fname = Path(path_in, f"{sub_id}-postICA-epo.fif")
            helpers.print_msg(str(fname))
            data_pre = mne.read_epochs(fname)
        except FileNotFoundError:
            print(f"No data for {sub_id}.")
            continue

        # Good time to baseline the data (AR runs better on bl'ed data.)
        data_bl = data_pre.copy().apply_baseline((-timings.DUR_BL, 0))

        ars = []
        reject_logs = []
        # bunch of random seeds:
        rand_ints = [30, 7, 19, 88, 307, 198, 8, 3, 0, 71988]
        for rs in rand_ints:
            _, ar, reject_log = clean_with_ar_local(
                sub_id,
                data_bl,
                n_jobs=config.N_JOBS,
                ar_path=paths.DATA_03_AR,
                ar_from_disc=False,
                save_to_disc=False,
                rand_state=rs,
            )
            ars.append(ar)
            reject_logs.append(reject_log)

        all_badepos = np.stack([rl.bad_epochs for rl in reject_logs])
        avg_badepos = all_badepos.mean(axis=0)

        # Pick the seed which is most representative:
        sims = [np.dot(norm_vec(avg_badepos), norm_vec(be)) for be in all_badepos]
        idx_max = np.argmax(sims)

        _, ar, reject_log = clean_with_ar_local(
            sub_id,
            data_bl,
            n_jobs=config.N_JOBS,
            ar_from_disc=False,
            save_to_disc=True,
            ar_path=paths.DATA_03_AR,
            rand_state=rand_ints[idx_max],
        )

        fpath_out = Path(paths.DATA_03_AR, "robustinfo")
        fpath_out.mkdir(exist_ok=True)

        file_diag = Path(fpath_out, "info.txt")
        n_bad_epos = [sum(rl.bad_epochs) for rl in reject_logs]
        n_epos_min = np.min(n_bad_epos)
        n_epos_max = np.max(n_bad_epos)
        with Path.open(file_diag, "a+") as f:
            f.write(f"{sub_id};{n_epos_min};{n_epos_max};{n_bad_epos};{n_bad_epos[idx_max]}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None
    main(JOB_NR)
