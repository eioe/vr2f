"""
Calculate the lag between eye tracking (ET) and electrooculography (EOG) data for a given subject.

This module provides a class to calculate the lag between eye tracking (ET) and electrooculography
(EOG) data for a given subject.
The class includes methods to extract and process ET and EOG data, calculate
cross-correlations between the two data types, and plot the results. The class also includes a
method to write the calculated lag information to a CSV file for later reference.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags

from vr2f.helpers import chkmkdir
from vr2f.staticinfo import CONSTANTS, PATHS


class LagCalculatorEyetrackingVsEog:
  """Calculate the lag between eye tracking (ET) and electrooculography (EOG) data for a given subject."""

  def __init__(self):
    """Initialize constants."""
    self.constants = CONSTANTS()
    self.paths = PATHS()


  def get_data(self, sub_id, picks="eog"):
      """Get EOG data for given subject."""
      path_in = Path(self.paths.DATA_01_EPO, "erp", "clean")
      fname = Path(path_in, f"{sub_id}-epo.fif")
      epos = mne.read_epochs(fname, verbose=False).pick(picks)
      times = epos.times
      info = epos.info

      return epos, times, info


  def zscore_array(self, arr, axis=0):
      """Z-score an array."""
      return (arr - np.nanmean(arr, axis=axis)) / np.nanstd(arr, axis=axis)


  def get_et_eog_df(self, et_data_full, sub_id, trial_range=None):
    """
    Extract and process eye tracking (ET) and electrooculography (EOG) data for a given subject.

    This function processes ET and EOG data for a specified subject. It computes vertical and
    horizontal components for both ET and EOG data, z-scores them, and aligns them in time.
    Optionally, it can plot the aligned data. The function returns a list of pandas DataFrames,
    each containing aligned and processed ET and EOG data for each trial within the specified trial range.

    Parameters
    ----------
    et_data_full : pandas.DataFrame
        A DataFrame containing the full eye tracking data for all subjects and trials.
    sub_id : int or str
        The subject identifier for which to extract and process ET and EOG data.
    trial_range : range, optional
        A range object specifying the trials to process. If None, all trials for the subject are processed.
        Default is None.

    Returns
    -------
    et_eog_dfs : list of pandas.DataFrame
        A list of DataFrames, each containing the processed and aligned ET and EOG data for a trial.
        Each DataFrame has columns for times, ET vertical and horizontal components (phi, theta),
        andEOG vertical and horizontal components (vert, hor).

    Notes
    -----
    - The function assumes a specific structure for the input `et_data_full` DataFrame, including
    columns for trial numbers and eye tracking metrics (phi, theta).
    - EOG data is loaded and processed using predefined picks for EEG and EOG channels.
    - The function z-scores the ET and EOG components within each trial.
    - If `plot_it` is True, the function generates a plot for each trial, showing both ET and EOG components.
    - The function aligns ET and EOG data in time and interpolates EOG data to match ET data timestamps.

    Examples
    --------
    >>> et_data_full = pd.read_csv('eye_tracking_data.csv')
    >>> sub_id = 1
    >>> et_eog_dfs = get_et_eog_df(et_data_full, sub_id)
    >>> print(et_eog_dfs[0].head())

    """
    n_trials_train = self.constants.N_TRIALS_TRAINING

    # load eog data
    eog_data, eog_times, info = self.get_data(sub_id, picks=("eeg","eog"))

    d_fp1 = eog_data.get_data(picks="Fp1")
    d_fp2 = eog_data.get_data(picks="Fp2")
    d_io1 = eog_data.get_data(picks="IO1")
    d_io2 = eog_data.get_data(picks="IO2")
    d_lo1 = eog_data.get_data(picks="LO1")
    d_lo2 = eog_data.get_data(picks="LO2")

    if trial_range is None:
      trial_range = range(len(eog_data))

    et_eog_dfs= []

    for trial_idx in trial_range:
      eog_vert = -1 * ((d_fp1 - d_io1) + (d_fp2 - d_io2)) / 2
      eog_vert = eog_vert.squeeze()[trial_idx, :]
      eog_vert = self.zscore_array(eog_vert, axis=0)

      eog_hor = -1 * (d_lo1 - d_lo2)
      eog_hor = eog_hor.squeeze()[trial_idx, :]
      eog_hor = self.zscore_array(eog_hor, axis=0)

      et_sub = et_data_full.copy()
      et_sub = et_sub[et_sub["trial_num"] > n_trials_train]
      # subtract the number of train trials from the trial number
      et_sub.loc[:,"trial_idx"] = et_sub.loc[:,"trial_num"] - (n_trials_train+1)

      et_sub_trial = et_sub[(et_sub["trial_idx"] == trial_idx)]
      if et_sub_trial.empty:
        print(f"WARNING: No data found for trial {trial_idx} in {sub_id}. Skipping...")
        continue
      et_sub_trial = et_sub_trial.reset_index()
      et_times = et_sub_trial["times"]

      idx_times = ((et_times >= np.min(eog_times)) &
                  (et_times <= np.max(eog_times)))
      # make fixed length relative to t0 (assuming sfreq of 120Hz)
      n_neg = int(self.constants.SFREQ_ET * 0.250)
      n_pos = int(self.constants.SFREQ_ET * 0.950)
      # make mask:
      idx_tzero = et_times.abs().idxmin()
      mask = np.zeros(len(et_times), dtype=bool)
      mask[(idx_tzero - n_neg):(idx_tzero + n_pos + 1)] = True

      et_sub_trial = et_sub_trial[idx_times & mask]
      et_times = et_times[idx_times & mask]

      et_vert = et_sub_trial["phi"].to_numpy()
      et_vert = self.zscore_array(et_vert)
      et_hor = et_sub_trial["theta"].to_numpy()
      et_hor = self.zscore_array(et_hor)

      times_union = np.union1d(eog_times, et_times)

      df_et = pd.DataFrame({"times": et_times, "phi": et_vert, "theta": et_hor})
      df_et = df_et.set_index("times")

      df_eog = pd.DataFrame({"times": eog_times, "vert": eog_vert, "hor": eog_hor})
      df_eog = df_eog.set_index("times")
      df_eog = df_eog.reindex(times_union).interpolate()
      df_all = df_et.merge(df_eog, left_on="times", right_index=True)

      et_eog_dfs.append(df_all)

    return et_eog_dfs


  def calc_xcorrs(self, df_et_eogs):
    """Calculate cross-correlations and lags between ET and EOG data."""
    xcorrs = {}
    xcorrs["vert"] = []
    xcorrs["hor"] = []
    lags = {}
    lags["vert"] = []
    lags["hor"] = []

    for df_et_eog in df_et_eogs:
      xcorr_v = correlate(df_et_eog["phi"], df_et_eog["vert"])
      xcorr_h = correlate(df_et_eog["theta"], df_et_eog["hor"])
      lags_v = correlation_lags(len(df_et_eog["phi"]), len(df_et_eog["vert"]))
      lags_h = correlation_lags(len(df_et_eog["theta"]), len(df_et_eog["hor"]))

      xcorrs["vert"].append(xcorr_v)
      xcorrs["hor"].append(xcorr_h)
      lags["vert"].append(lags_v)
      lags["hor"].append(lags_h)

    return xcorrs, lags


  def plot_xcorrs(self, xcorrs, lags, sub_id=""):
    """Plot cross-correlations and lags."""
    m_v = np.nanmean(np.asarray(np.abs(xcorrs["vert"])), axis=0)
    m_h = np.nanmean(np.asarray(xcorrs["hor"]), axis=0)
    lags_v = np.nanmean(np.asarray(lags["vert"]), axis=0)
    lags_h = np.nanmean(np.asarray(lags["hor"]), axis=0)
    sd_v = np.asarray(xcorrs["vert"]).std(axis=0)
    sd_h = np.asarray(xcorrs["hor"]).std(axis=0)

    max_corr_v = np.argmax(np.abs(m_v))
    max_corr_h = np.argmax(np.abs(m_h))

    print(f"#### {sub_id} ####")
    print(f"Top correlation for vertical: {max_corr_v} at {lags_v[max_corr_v]}")
    print(f"Top correlation for horizontal: {max_corr_h} at {lags_h[max_corr_h]}")

    lag_v_max = lags_v[max_corr_v] * 1/self.constants.SFREQ_ET
    lag_h_max = lags_h[max_corr_h] * 1/self.constants.SFREQ_ET
    lag_mean = np.mean([lag_v_max, lag_h_max])

    plt.figure()

    plt.fill_between(lags_v, m_v - sd_v, m_v + sd_v, alpha=0.5)
    plt.fill_between(lags_h, m_h - sd_h, m_h + sd_h, alpha=0.5)

    plt.plot(lags_v, m_v)
    plt.plot(lags_h, m_h)
    plt.vlines(lag_mean * self.constants.SFREQ_ET, np.min(m_v), np.max(m_v), colors="r", linestyles="--")
    plt.xlabel("Lag (samples)")
    plt.ylabel("Correlation (a.u.)")
    plt.title(f"XCorr EOG and ET: {sub_id}")
    plt.legend(["Vertical", "Horizontal"])
    plt.text(lag_mean * self.constants.SFREQ_ET, 0, f"mean lag: {(1000 * lag_mean):.1f}ms", color="r")

    # save to file
    path_out = Path(self.paths.DATA_ET_PREPROC, "lags", "plots")
    chkmkdir(path_out)
    fpath_out = Path(path_out, f"{sub_id}_xcorr.png")
    plt.savefig(fpath_out)
    plt.close()


  def write_lag_df_csv(self, xcorrs, lags, sub_id):
    """Write lags to a CSV file."""
    m_v = np.nanmean(np.asarray(np.abs(xcorrs["vert"])), axis=0)
    m_h = np.nanmean(np.asarray(xcorrs["hor"]), axis=0)
    lags_v = np.nanmean(np.asarray(lags["vert"]), axis=0)
    lags_h = np.nanmean(np.asarray(lags["hor"]), axis=0)

    max_corr_v = np.argmax(np.abs(m_v))
    max_corr_h = np.argmax(np.abs(m_h))

    lag_v_max = lags_v[max_corr_v] * 1/self.constants.SFREQ_ET
    lag_h_max = lags_h[max_corr_h] * 1/self.constants.SFREQ_ET
    lag_mean = np.mean([lag_v_max, lag_h_max])

    # write to DF
    df_lag = pd.DataFrame({"lag_mean": lag_mean,
                          "lag_vert": lag_v_max, "lag_hor": lag_h_max},
                          index=[sub_id])
    path_df_lags = Path(self.paths.DATA_ET_PREPROC, "lags", "DF_lags.csv")
    if path_df_lags.exists():
      df_old = pd.read_csv(path_df_lags).set_index("sub_id")
      if sub_id in df_old.index:
        df_old = df_old.drop(index=sub_id)
        print(f"WARNING: overwriting existing entry for {sub_id}.")
      df_lag = pd.concat([df_old, df_lag], axis=0)
    df_lag.to_csv(path_df_lags, index=True, index_label="sub_id")


  def get_et_vs_eog_lag(self, sub_id, et_data, plot_it=False, write_csv=True):
    """
    Calculate the lag between eye tracking (ET) and electrooculography (EOG) data for a given subject.

    This function processes ET and EOG data for a specified subject to find the lag between these two types of data.
    It optionally plots the cross-correlations and writes the lag information to a CSV file.

    Parameters
    ----------
    sub_id : int or str
        The subject identifier for which to calculate the ET and EOG lag.
    et_data : pandas.DataFrame
        A DataFrame containing the full eye tracking data for all subjects and trials.
    plot_it : bool, optional
        If True, plots the cross-correlations between ET and EOG data. Default is False.
    write_csv : bool, optional
        If True, writes the calculated lag information to a CSV file. Default is True.

    Returns
    -------
    float
        The mean lag between ET and EOG data, calculated as the average of the maximum
        correlation lags for vertical and horizontal components.

    Notes
    -----
    - This function relies on several helper functions to process the data, calculate
    cross-correlations, and optionally plot and write results.
    - The lag is calculated as the mean of the maximum correlation lags for both
    vertical and horizontal components of the ET and EOG data.
    - The function assumes that the ET data has been preprocessed and is passed as a DataFrame.
    - If `plot_it` is True, the function generates a plot showing the cross-correlations
    for both vertical and horizontal components.
    - If `write_csv` is True, the function writes the calculated lag information to a CSV file for later reference.

    Examples
    --------
    >>> et_data = pd.read_csv('eye_tracking_data.csv')
    >>> sub_id = "VR2FEM_S23"
    >>> mean_lag = get_et_vs_eog_lag(sub_id, et_data, plot_it=True, write_csv=False)
    >>> print(f"Mean lag: {mean_lag}")

    """
    df_et_eog = self.get_et_eog_df(et_data, sub_id)
    xcorrs, lags = self.calc_xcorrs(df_et_eog)
    if plot_it:
      self.plot_xcorrs(xcorrs, lags, sub_id)
    if write_csv:
      self.write_lag_df_csv(xcorrs, lags, sub_id)

    m_v = np.nanmean(np.asarray(np.abs(xcorrs["vert"])), axis=0)
    m_h = np.nanmean(np.asarray(xcorrs["hor"]), axis=0)
    lags_v = np.nanmean(np.asarray(lags["vert"]), axis=0)
    lags_h = np.nanmean(np.asarray(lags["hor"]), axis=0)

    max_corr_v = np.argmax(np.abs(m_v))
    max_corr_h = np.argmax(np.abs(m_h))

    lag_v_max = lags_v[max_corr_v] * 1/self.constants.SFREQ_ET
    lag_h_max = lags_h[max_corr_h] * 1/self.constants.SFREQ_ET
    return np.mean([lag_v_max, lag_h_max])
