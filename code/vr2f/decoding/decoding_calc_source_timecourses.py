import multiprocessing as mp
import os.path as op
import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from mne import EvokedArray
from mne.datasets import fetch_fsaverage

from vr2f import helpers
from vr2f.staticinfo import COLORS, CONFIG, PATHS


def load_patterns(
    sub_list_str,
    contrast_str,
    viewcond="",
    scoring="accuracy",
    reg="",
    labels_shuffled=False,
):
    """
    Load the patterns from sensor space decoding.

    Parameters
    ----------
    sub_list_str : list, str
        List of subject IDs to load patterns from.
    contrast_str : str
        Decoded contrast.
    viewcond : str
        Viewing condition. 'mono', 'stereo', or ''(default) for data pooled across both viewing conditions.
    scoring: str
        Scoring metric used during decoding. "roc_auc", accuracy" (default), or "balanced_accuracy";
    reg: str, float
        Regularization method used; Ints are interpreted as fixed shrinkage values; defaults to an empty string
    labels_shuffled : bool
        Allows to load the data from the run with shuffled labels.


    Returns
    -------
    patterns: ndarray
        Array with the patterns (subs x csp_components x channels x freqs x times)
    times: array, 1d

    """
    paths = PATHS()

    if isinstance(reg, float):
        reg_str = "shrinkage" + str(reg)
    else:
        reg_str = reg
    shuf_labs = "labels_shuffled" if labels_shuffled else ""

    patterns_list = []
    times = []

    for subID in sub_list_str:
        fpath = Path(paths.DATA_04_DECOD_SENSORSPACE, viewcond, contrast_str, scoring, "patterns")
        fname = op.join(fpath, f"{subID}-patterns_per_sub.npy")
        patterns_ = np.load(fname)
        patterns_list.append(patterns_)
        if len(times) == 0:
            times = np.load(str(fname)[:-4] + "__times" + ".npy")
        else:
            assert np.all(
                times == np.load(str(fname)[:-4] + "__times" + ".npy")
            ), "Times are different between subjects."

    patterns = np.concatenate(patterns_list)
    return patterns, times


def get_info(return_inst=False):
    paths = PATHS()
    fname = Path(
        paths.DATA_03_AR,
        "cleaneddata",
        "VR2FEM_S01-postAR-epo.fif",
    )
    epos = mne.read_epochs(fname, verbose=False)
    epos = epos.pick_types(eeg=True)

    if return_inst:
        return epos.info, epos
    else:
        return epos.info


def get_epos(subID):
    paths = PATHS()
    fname = Path(
        paths.DATA_03_AR,
        "cleaneddata",
        f"{subID}-postAR-epo.fif",
    )
    epos = mne.read_epochs(fname, verbose=False)
    epos = epos.pick_types(eeg=True)
    epos = epos.set_eeg_reference("average", projection=True)
    return epos


def l2norm(vec):
    out = np.sqrt(np.sum(vec**2))
    return out



def get_fsavg_src(from_disk=False):  # noqa: D103
  fs_dir = fetch_fsaverage(verbose=False)
  src_path = Path(fs_dir, "bem", "fsaverage" + "-oct6" + "-src.fif")
  Path.mkdir(src_path.parent, parents=True, exist_ok=True)
  if not from_disk:
    subjects_dir = Path(fs_dir).parent
    src = mne.setup_source_space("fsaverage", spacing="oct6", subjects_dir=subjects_dir, add_dist=False)
    src.save(src_path, overwrite=True)
  else:
    src = mne.read_source_spaces(src_path)
  return src


def create_fake_info(epos, pattern_times):
  info = mne.create_info(ch_names=epos.ch_names, sfreq=1 / np.median(np.diff(pattern_times)), ch_types="eeg")
  easycap_montage = mne.channels.make_standard_montage("easycap-M1")
  info.set_montage(easycap_montage)
  return info


def get_fwd_solution(src, info, from_disk=False):
  config = CONFIG()
  fs_dir = fetch_fsaverage(verbose=True)
  path_fwd = Path(fs_dir, "bem", "fsaverage" + "-oct6" + "-fwd.fif")
  Path.mkdir(path_fwd.parent, parents=True, exist_ok=True)
  if not from_disk:
    fwd = mne.make_forward_solution(
        info, trans="fsaverage", src=src, bem=Path(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif"),
        eeg=True, n_jobs=config.N_JOBS
    )
    mne.write_forward_solution(path_fwd, fwd, overwrite=True, verbose=None)
  else:
    fwd = mne.read_forward_solution(path_fwd)
  return fwd


def get_inv_operator(fwd, epos, pattern_times):
    noise_cov = mne.compute_covariance(epos, tmax=0)
    # Build new pseudo Evoke obj so we can adapt sfreq:
    easycap_montage = mne.channels.make_standard_montage("easycap-M1")
    info = mne.create_info(ch_names=epos.ch_names, sfreq=1 / np.median(np.diff(pattern_times)), ch_types="eeg")
    info.set_montage(easycap_montage)
    inv_op = mne.minimum_norm.make_inverse_operator(
        epos.info,
        fwd,
        noise_cov,
        fixed=True,
        loose=0.0,  # needs to be zero if Fixed is true
        depth=0.8,  # ignored by eLoreta
    )
    return inv_op, info


def get_src_timecourse(sub_id, contrast, pattern, pattern_times, inv_op, info, from_disk=False):
    paths = PATHS()
    fpath = Path(paths.DATA_04_DECOD_SENSORSPACE, contrast, "roc_auc_ovr", "patterns", "src_timecourses", f"{sub_id}")
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    if not from_disk:
        sub_pattern_src = EvokedArray(pattern, info, tmin=pattern_times[0])
        inst = sub_pattern_src.set_eeg_reference("average", projection=True)
        inst.data = sub_pattern_src.data
        stc = abs(
            mne.minimum_norm.apply_inverse(
                inst,
                inv_op,
                lambda2=0.1,
                method="eLORETA",
                pick_ori=None,  # leads to signed values for fixed orientations
                verbose=True)
        )
        stc.save(fpath, overwrite=True)
    else:
        stc = mne.read_source_estimate(fpath)
    return stc


def process_sub(sub_id, contrast, sub_pattern, pat_times):
    # Get the source space and the forward solution (needs to be done only once)
    src = get_fsavg_src(from_disk=True)
    epos = get_epos(sub_id)
    info = create_fake_info(epos, pat_times)
    fwd = get_fwd_solution(src, info, from_disk=True)  # takes a long time if not read from disc
    info = create_fake_info(epos, pat_times)
    inv_op, info = get_inv_operator(fwd, epos, pat_times)
    stc = get_src_timecourse(sub_id, contrast, sub_pattern, pat_times, inv_op, info, from_disk=False)
    return stc



def main(sub_nr):

  contrasts = ["mono_vs_stereo",
               "angry_vs_happy",
               "angry_vs_neutral",
               "angry_vs_surprised",
               "happy_vs_neutral",
               "happy_vs_surprised",
               "surprised_vs_neutral",
               ]
  for contrast in contrasts:
    paths = PATHS()
    path_in = Path(paths.DATA_04_DECOD_SENSORSPACE, contrast, "roc_auc_ovr", "patterns")
    sub_list_str = [s.split("-patterns_per")[0] for s in os.listdir(path_in)]
    sub_list_str = np.unique(sub_list_str)  # remove duplicates because there are two files per subject
    sub_list_str = sorted(sub_list_str, reverse=False)

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]

    sub_patterns, pat_times = load_patterns(
        sub_list_str,
        contrast_str=contrast,  # "mono_vs_stereo",  #'surprised_vs_neutral_vs_angry_vs_happy',  #   #
        viewcond="",  # "stereo",  # "mono", # ,
        scoring="roc_auc_ovr",
        reg="",
        labels_shuffled=False,
    )

    if len(sub_patterns.shape) > 3:
        raise ValueError("source reconstruction for multiclass needs separate implementation.")

    if len(sub_list_str) == 1:
        process_sub(sub_list_str[0], contrast, sub_patterns[0], pat_times)
    else:
        for sub_id, sub_pattern in zip(sub_list_str, sub_patterns, strict=True):
            process_sub(sub_id, contrast, sub_pattern, pat_times)



if __name__ == "__main__":


    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None

    main(JOB_NR)








