import multiprocessing as mp
import os
import os.path as op
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from mne import EvokedArray
from mne.datasets import fetch_fsaverage

from vr2f import helpers
from vr2f.decoding.plotters import load_patterns
from vr2f.staticinfo import COLORS, CONFIG, PATHS


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


def get_src_timecourse(sub_id, contrast, pattern, pattern_times, inv_op, info,
                       viewcond="", from_disk=False, mc_contrast=""):
    paths = PATHS()
    mc_dir = "multiclass" if mc_contrast else ""
    fpath = Path(paths.DATA_04_DECOD_SENSORSPACE,
                 viewcond,
                 contrast,
                 "roc_auc_ovr",
                 "patterns",
                 "src_timecourses",
                 mc_dir,
                 mc_contrast,
                 f"{sub_id}")
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


def get_src_timecourse_multiclass(sub_id, contrast, pattern, pattern_times, inv_op, info,
                                  viewcond="", from_disk=False):
    paths = PATHS()
    stcs = []
    for i in range(pattern.shape[-2]):
        pattern_ = pattern[:, i, :].squeeze()
        emos = contrast.split("_vs_")
        stc_ = get_src_timecourse(sub_id,
                           contrast,
                           pattern_,
                           pattern_times,
                           inv_op,
                           info,
                           viewcond=viewcond,
                           from_disk=from_disk,
                           mc_contrast=f"{emos[i]}_vs_rest")
        stcs.append(stc_)
    data_ = np.stack([stc.data for stc in stcs])
    # normalize per sub-contrast and timepoint:
    sums = np.sum(data_, axis=1, keepdims=True)
    data_ = data_ / sums
    # use last one as template:
    stc = stc_.copy()
    # average across sub-contrasts:
    stc.data = data_.mean(axis=0)
    fpath = Path(paths.DATA_04_DECOD_SENSORSPACE,
                 viewcond,
                 contrast,
                 "roc_auc_ovr",
                 "patterns",
                 "src_timecourses",
                 "multiclass",
                 "ovr_avg",
                 f"{sub_id}",
                 )
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    stc.save(fpath, overwrite=True)
    return stc


def process_sub(sub_id, contrast, sub_pattern, pat_times, viewcond=""):
    # Get the source space and the forward solution (needs to be done only once)
    src = get_fsavg_src(from_disk=True)
    epos = get_epos(sub_id)
    info = create_fake_info(epos, pat_times)
    fwd = get_fwd_solution(src, info, from_disk=True)  # takes a long time if not read from disc
    info = create_fake_info(epos, pat_times)
    inv_op, info = get_inv_operator(fwd, epos, pat_times)
    if sub_pattern.ndim == 3 and len(contrast.split("_vs_")) > 2:
        print(f"Processing {sub_id} for multiclass contrast {contrast}")
        stc = get_src_timecourse_multiclass(sub_id, contrast, sub_pattern, pat_times, inv_op, info,
                                            viewcond=viewcond, from_disk=False)
    else:
        stc = get_src_timecourse(sub_id, contrast, sub_pattern, pat_times, inv_op, info,
                                 viewcond=viewcond, from_disk=False)
    return stc


def main(sub_nr):

    contrasts = ["neutral_vs_happy_vs_angry_vs_surprised",
               "id1_vs_id2_vs_id3",
            #    "mono_vs_stereo",
            #    "angry_vs_happy",
            #    "angry_vs_neutral",
            #    "angry_vs_surprised",
            #    "happy_vs_neutral",
            #    "happy_vs_surprised",
            #    "surprised_vs_neutral",
            ]
    viewconds = ["mono", "stereo", ""]  # ""

    for vc in viewconds:
        for contrast in contrasts:
            if vc in ["mono", "stereo"] and contrast == "mono_vs_stereo":
                continue
            paths = PATHS()
            path_in = Path(paths.DATA_04_DECOD_SENSORSPACE, contrast, "roc_auc_ovr", "patterns")
            sub_list_str = [str(s.name).split("-patterns_per")[0] for s in path_in.iterdir() if s.is_file()]
            sub_list_str = np.unique(sub_list_str)  # remove duplicates because there are two files per subject
            sub_list_str = sorted(sub_list_str, reverse=False)

            if sub_nr is not None:
                sub_list_str = [sub_list_str[sub_nr]]

            sub_patterns, pat_times = load_patterns(
                sub_list_str,
                contrast_str=contrast,
                viewcond=vc,
                scoring="roc_auc_ovr",
                )

            if len(sub_list_str) == 1:
                process_sub(sub_list_str[0], contrast, sub_patterns[0], pat_times, viewcond=vc)
            else:
                for sub_id, sub_pattern in zip(sub_list_str, sub_patterns, strict=True):
                    process_sub(sub_id, contrast, sub_pattern, pat_times, viewcond=vc)



if __name__ == "__main__":


    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None

    main(JOB_NR)








