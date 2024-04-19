import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from vr2f.eyetracking import ms_toolbox
from vr2f.staticinfo import COLORS, CONSTANTS, PATHS


def get_data_sub_trialnum(df_data, sub_id, trial_num):  # noqa: ARG001
    """Get data for a specific subject and trial number."""
    df_data_sub = (df_data
                    .query("sub_id == @sub_id")
                    .query("trial_num == @trial_num")
    )
    return df_data_sub

def read_preproc_data(sub_list_str_et, file_end_pattern = "-preproc.csv"):
  paths = PATHS()

  data_preproc = []
  for sub_id in sorted(sub_list_str_et):
      fname = Path(paths.DATA_ET_PREPROC, f"{sub_id}-ET-{file_end_pattern}")
      df_clean = pd.read_csv(fname, sep=",")
      df_clean["sub_id"] = sub_id
      data_preproc.append(df_clean)

  df_all = pd.concat(data_preproc, ignore_index=True)
  return df_all


def plot_gaze_sacc_per_trial(data, sub_id, trial_range, min_sac_amp = 1, sfreq = 120,
                             vfac = 5, mindur = 3):
  """
  Plot gaze and microsaccades per trial for a given subject and range of trials.

  This function generates a set of plots for each trial specified in the trial range for a given subject.
  Each plot consists of three subplots: the first shows the gaze plot in "D (theta-phi) space, the second and third
  show the gaze position over time for each cardinal viewing axis (phi and theta, respectively.
  (Micro)saccades are highlighted in red.

  Parameters
  ----------
  data : DataFrame
      The input data containing gaze information, blink information, times, and other relevant details.
      Use the preprocessed data frames.
  sub_id : int or str
      The subject identifier (e.g., "VR2FEM_S01").
  trial_range : iterable
      An iterable (e.g., list or range) specifying the trial numbers to be plotted.
  min_sac_amp : float, optional
      The minimum amplitude in dva of (micro)saccades to be considered for highlighting.
      Default is 1.
  sfreq : int, optional
      The sampling frequency of the gaze data. Default is 120 Hz.
  vfac : int, optional
      Velocity factor used in microsaccade detection algorithm. Default is 5.
  mindur : int, optional
      Minimum duration (in samples) a (micro)saccade must last to be considered in the analysis. Default is 3.

  Returns
  -------
  fig : matplotlib.figure.Figure
      The figure object containing the generated plots for each trial.

  """
  paths = PATHS()
  colors = COLORS()
  constants = CONSTANTS()
  cm = constants.CM

  df_in = data.copy()
  n_trials = len(trial_range)
  cm = constants.CM

  fig, ax = plt.subplots(n_trials, 3, figsize=(32 * cm, n_trials * 11 * cm))

  for i, trial_idx in enumerate(trial_range):
    df_st = get_data_sub_trialnum(df_in, sub_id, trial_idx)

    data = df_st.loc[:,["theta", "phi"]].to_numpy()
    sac, rad = ms_toolbox.microsacc(data, sfreq, vfac=vfac, mindur=mindur)
    sac["amp_tot"] = np.sqrt(sac["amp_x"]**2 + sac["amp_y"]**2)
    times = df_st["times"]

    df_st["phi"][df_st["blink"]] = 0
    df_st["theta"][df_st["blink"]] = 0
    # flip phi for more intuitive plotting:
    df_st["phi"] = df_st["phi"] * -1

    # gaze plot
    sns.lineplot(data=df_st, x="theta", y="phi", ax=ax[i,0], sort=False)
    ax[i,0].set_ylim([-5,5])
    ax[i,0].set_xlim([-5,5])

    # per cardinal viewing axis
    axs = ax[i,1:]
    for _y, _ax in zip(["phi", "theta"], axs, strict=True):
      sns.lineplot(data=df_st, x="times", y=_y, ax=_ax)
      _ax.set_ylim([-5,5])
      _ax.hlines(0, times.min(), times.max(), color="black", linestyle="--")
      _ax.vlines(0, -5, 5, color="black", linestyle="--")
      # remove right and top spine:
      _ax.spines["right"].set_visible(False)
      _ax.spines["top"].set_visible(False)

    for _, s in sac.iterrows():
      if s["amp_tot"] < min_sac_amp:
        continue
      a = int(s["idx_onset"])
      b = int(s["idx_offset"])
      df_plt = df_st.copy().iloc[a:(b+1),:]
      sns.lineplot(data=df_plt, x="theta", y="phi", ax=ax[i,0], color="red", sort=False)
      sns.lineplot(data=df_plt, x="times", y="phi", ax=ax[i,1], color="red")
      sns.lineplot(data=df_plt, x="times", y="theta", ax=ax[i,2], color="red")

    av_name = df_st["avatar_id"].iloc[0]
    emotion = df_st["emotion"].iloc[0]
    trial_num = df_st["trial_num"].iloc[0]
    viewcond = df_st["viewcond"].iloc[0]
    img = plt.imread(Path(paths.STIMULIIMAGES, f"{av_name}_{emotion.capitalize()}.png"))
    cutval = 205
    im2 = img[cutval + 35 : -cutval, cutval + 35 : -cutval, :]
    height, width, _ = im2.shape
    # Overlay the image onto the plot, centered on the axes
    ax[i,0].imshow(im2, extent=[-5, 5, -5, 5], alpha=1)

    ax[i,0].text(0.05, 0.95, f"#{trial_num}", transform=ax[i,0].transAxes, ha="left",
                 va="top", fontsize=12, color="white")
    ax[i,0].text(0.05, 0.9, f"{viewcond}", transform=ax[i,0].transAxes, ha="left",
                 va="top", fontsize=12, color="white")
  return fig


def save_fig(fig, sub_id, trial_range):
  fpath = Path(paths.FIGURES, "ET", "Gaze_saccades", sub_id)
  fpath.mkdir(parents=True, exist_ok=True)
  fname = f"{sub_id}_gaze_{trial_range[0]}-{trial_range[-1]}.pdf"
  fig.savefig(Path(fpath, fname), bbox_inches="tight")
  plt.close()


if __name__ == "__main__":

  paths = PATHS()

  # read in the preprocessed data
  file_end_pattern = "preproc.csv"

  sub_list_str_et = [f for f in os.listdir(paths.DATA_ET_PREPROC) if file_end_pattern in f]
  sub_list_str_et = [f.split("-")[0] for f in sub_list_str_et]
  sub_list_str_et = np.unique(sorted(sub_list_str_et))

  if len(sys.argv) > 1:
    new_list = []
    for arg in sys.argv[1:]:
      if arg.startswith("VR2FEM") and arg in sub_list_str_et:
        new_list.append(arg)
      elif arg.isdigit() and int(arg) < len(sub_list_str_et):
        new_list.append(sub_list_str_et[int(arg)])
      else:
        print(f"Invalid argument: {arg}")
    sub_list_str_et = new_list

  df_all = read_preproc_data(sub_list_str_et, file_end_pattern)

  for sub_id in sub_list_str_et:
    # training trials:
    trial_range = range(1, 25)
    myfig = plot_gaze_sacc_per_trial(df_all, sub_id, trial_range)
    save_fig(myfig, sub_id, trial_range)

    # exp trials (in chunks to manage file size):
    for i in range(25, 745, 30):
      trial_range = range(i, i + 30)
      myfig = plot_gaze_sacc_per_trial(df_all, sub_id, trial_range)
      save_fig(myfig, sub_id, trial_range)
