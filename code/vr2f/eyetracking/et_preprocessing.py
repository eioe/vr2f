import os
from pathlib import Path

import numpy as np
import pandas as pd

from vr2f.eyetracking import lag_calculator_et_vs_eog
from vr2f.staticinfo import COLORS, PATHS


def cart2sph_custom(x, y, z):
    """Convert cartesian coordinates (xyz) to spherical coordinates (theta-phi-r)."""
    hypotxz = np.hypot(x, z)
    r = np.hypot(y, hypotxz)
    phi = np.arctan2(y, hypotxz) * -1  # mind Unity's left-handed coordinate system
    theta = np.arctan2(x, z)

    # translate both to degree
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)

    # concatenate the 3 arrays to a 2d array
    return np.stack((theta, phi, r), axis=1)


def interpolate_blinks(et_data: pd.DataFrame):
    """
    Interpolate blinks in the eye tracking data.

    According to the method suggested by
    Kret, M.E., Sjak-Shie, E.E. Preprocessing pupil size data: Guidelines and code.
    Behav Res 51, 1336-1342 (2019). https://doi.org/10.3758/s13428-018-1075-y

    Parameters
    ----------
    et_data : pd.DataFrame
        Eye tracking data

    Returns
    -------
    et_data : pd.DataFrame
        Eye tracking data with blinks interpolated.

    """
    df_et = et_data.copy()

    p_dil = df_et["diameter_left"].to_numpy()
    t_et = df_et["timestamp_et"].to_numpy()
    d = np.diff(p_dil)
    d_pre = d[:-1]
    d_post = d[1:]
    t = np.diff(t_et)
    t_pre = t[:-1]
    t_post = t[1:]

    o = np.max(np.array((np.abs(d_pre / t_pre), np.abs(d_post / t_post))), axis=0)

    mad = np.median(np.abs(o - np.median(o)))
    thresh = np.median(o) + 50 * mad
    label = o > thresh
    # repeat the first and last label to get the same length as the original array
    label = np.insert(label, 0, label[0])
    label = np.append(label, label[-1])
    df_et["blink"] = label
    # set blink to 1 if dilation is < 0
    df_et.loc[df_et["diameter_left"] < 0, "blink"] = 1

    blink_times = df_et.loc[df_et["blink"] == 1, "timestamp_et"]
    # find all timestamps_et which are closer than 50ms to a blink
    blink_times = blink_times.to_numpy()
    et_times = df_et["timestamp_et"].to_numpy()
    if len(blink_times) == 0:
        blink_times = np.array([-99])
    # repeat blink times len(et_times) along new axis
    blink_times_r = np.repeat(blink_times[:, np.newaxis], len(et_times), axis=1)

    mindist = np.min(np.abs(et_times - blink_times_r), axis=0)
    df_et["blink"] = mindist < 100

    # set theta and phi to NaN for blinks
    df_et.loc[df_et["blink"] == 1, "theta"] = np.nan
    df_et.loc[df_et["blink"] == 1, "phi"] = np.nan
    df_et.loc[df_et["blink"] == 1, "r"] = np.nan
    df_et.loc[df_et["blink"] == 1, "diameter_left"] = np.nan
    df_et.loc[df_et["blink"] == 1, "diameter_right"] = np.nan

    # interpolate the data for the blinks
    df_et["theta"] = df_et["theta"].interpolate(method="linear")
    df_et["phi"] = df_et["phi"].interpolate(method="linear")
    df_et["r"] = df_et["r"].interpolate(method="linear")
    df_et["diameter_right"] = df_et["diameter_right"].interpolate(method="linear")
    df_et["diameter_left"] = df_et["diameter_left"].interpolate(method="linear")

    return df_et


def get_stimonset_df(sub_id: str) -> pd.DataFrame|None:
    """
    Retrieve the corrected stimulus onset times for a given subject.

    Adjusting the time offsets between different time measurements.
    The corrected times are returned as a numpy array. If the required files do
    not exist, the function returns `None`.

    Parameters
    ----------
    sub_id : str
        The subject identifier for which to retrieve the stimulus onset times.

    Returns
    -------
    pd.DataFrame | None
        A pandas DataFrame containing the corrected stimulus onset times for the
        given subject. Returns `None` if the required files are not found.

    Notes
    -----
    - The function assumes a specific directory structure under
    `PATHS.DATA_SUBJECTS` to locate the trial results and marker log files.
    - The times in the marker log file are initially recorded using Unity's
    `Time.realtimeSinceStartup`, which differs from the time used in the trial
    results and tracking files (`Time.time()`). This function corrects for that
    difference.
    - The correction for time offsets is done on a trial-by-trial basis since
    the offset is not constant throughout the experiment.
    - The offset is calculated by subtracting the "ITI start" time (plus a
    fixed duration of 1020ms) from the trial end time. This offset is then
    applied to the stimulus onset times in the marker log file.
    - The function prints a message and returns `None` if the trial results
    or marker log file does not exist for the given subject.

    """
    paths = PATHS()
    # load trial results
    path_in = Path(paths.DATA_SUBJECTS, sub_id, "MainTask", "Unity", "S001")
    fname = Path(path_in, "trial_results.csv")
    # check if exists:
    if Path(fname).is_file():
        trial_results = pd.read_csv(fname, sep=",")
    else:
        print(f"    No trial_results for {sub_id}")
        return None

    # grab marker logfile
    path_in = Path(paths.DATA_SUBJECTS, sub_id, "MainTask", "Unity", "S001", "other")
    fname = Path(path_in, "markerLog.csv")
    # try if file exists
    if Path(fname).is_file():
        markerlog = pd.read_csv(fname, sep=",")
    else:
        print(f"    No markerlog for {sub_id}")
        return None

    # the marker logfile gives us (among others) the times of the stimulus onsets
    # However, the times in the marker logfile have been written using Unity's
    # Time.realtimeSinceStartup, which is not the same as the Time.time() used for
    # the times in the (eye) tracking files (i.e., in /trackers/...) and trial_results.
    # Therefore, we need to translate the times into each other.
    # We can do so by subtracting a constant offset from the times in the marker
    # logfile to get them into the same time frame as the tracking files.
    # We need to do this on a trial-by-trial basis, because the offset is not
    # constant across the entire course of the experiment (triggering the eye tracking
    # calibration changes the offset).
    # We can get the offset for each trial by calculating the difference between the
    # time of the trial END ("end_time" in trial_results) and the time of the
    # "ITI start" marker in the marker logfile + 1020ms (ITI duration + 2 frames that we lost on avg).

    # get the times of the "ITI start" markers
    t_iti_start = markerlog[markerlog["annotation"].str.contains("ITI Start")]["timestamp"].to_numpy()
    t_iti_end = t_iti_start + 1.020

    # get the times of the trial ends
    t_trial_end = trial_results["end_time"]

    # calculate the offset for each trial
    time_offset_per_trial = t_trial_end - t_iti_end
    time_offset_per_trial = time_offset_per_trial.to_numpy().repeat(5)

    # add the offset to the times in the marker logfile
    markerlog["timestamp_corrected"] = markerlog["timestamp"] + time_offset_per_trial
    # get correct stimulus onset times
    markerlog_stimonsets = markerlog[markerlog["annotation"].str.contains("Stimulus Onset")]

    return markerlog_stimonsets  # noqa: RET504


def get_et_rawfiles(sub_id: str) -> tuple[list, os.PathLike] | None:
    """
    Retrieve the eye tracking data files for a given subject.

    Parameters
    ----------
    sub_id : str
        The subject identifier for which to retrieve the eye tracking data files.

    Returns
    -------
    tuple
      list | None
          A sorted list of eye tracking data filenames for the given subject. Returns `None` if
          the required files are not found.
      Path | ""
          The path to the eye tracking data files for the given subject.

    Notes
    -----
    - The function assumes a specific directory structure under
    `PATHS.DATA_SUBJECTS` to locate the eye tracking data files.
    - The function looks for files that start with `[_eye_]` in the subject's
    eye tracking data directory.
    - The function prints a message and returns `None` if no eye tracking files
    are found for the given subject.

    """
    paths = PATHS()
    # load the ET data
    path_in = Path(paths.DATA_SUBJECTS, sub_id, "MainTask", "Unity", "S001", "trackers")

    # find all files that start wit [_eye_]
    et_files = [f for f in os.listdir(path_in) if f.startswith("[_eye_]")]
    # skip the participant if there are no eye tracking files
    if len(et_files) == 0:
        print(f"Skipping {sub_id} because there are no eye tracking files.")
        return None, ""

    return sorted(et_files), path_in


def set_to_fixed_sample_length(df_in, sfreq, dur_pre, dur_post):
    # make fixed length relative to t0 (assuming sfreq of 120Hz)
    df_et = df_in.copy().reset_index()
    n_neg = int(sfreq * dur_pre)
    n_pos = int(sfreq * dur_post)
    # make mask:
    times = df_et["times"]
    idx_tzero = times.abs().idxmin()
    mask = np.zeros(len(times), dtype=bool)
    mask[(idx_tzero - n_neg):(idx_tzero + n_pos + 1)] = True
    df_out = df_et[mask]
    times_new = np.linspace(-1*n_neg*1/sfreq, n_pos*1/sfreq, len(df_out))
    times_old = df_out["times"].dropna().to_numpy()
    if ((np.abs(np.min(times_new) - np.min(times_old)) > 2/sfreq) or 
        (np.abs(np.max(times_new) - np.max(times_old)) > 2/sfreq)):
        print("WARNING: something is off with the timings.")
    df_out["times"] = times_new

    return df_out