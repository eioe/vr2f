"""Plotting functions for decoding results."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats

from vr2f import helpers
from vr2f.staticinfo import COLORS, PATHS
from vr2f.utils.stats import run_cbp_test


def load_decod_res_per_viewcond(
    sub_list_str,
    conditions,
    vc_list=None,
    scoring="roc_auc_ovr",
    picks_str=None,
    gen_str=None,
):
    """
    Load decoding results per viewing condition for a set of subjects and conditions.

    This function loads scores, times, and patterns associated with decoding results for
    each specified viewing condition. Data for each condition and subject is stored in
    a nested dictionary.

    Parameters
    ----------
    sub_list_str : list of str
        List of subject identifiers as strings, e.g., ['sub01', 'sub02'].
    conditions : list of str
        List of conditions to contrast (i.e., the decoding target classes), e.g., ['angry', 'neutral'].
    vc_list : list of str, optional
        List of viewing conditions to load, e.g. ["mono", "stereo"]. If empty, defaults to [""].
        "" (empty string) loads data pooled across viewing conditions.
    scoring : str, optional
        Metric used to score decoding results, by default "roc_auc_ovr".
    picks_str : str, optional
        String identifier for channel or sensor selections used during decoding. Defaults to None.
    gen_str : str, optional
        String identifier for generalization parameters (i.e., temporal generalization). Defaults to None.

    Returns
    -------
    dict
        A dictionary where each key is a viewing condition from `vc_list`.
        Each viewing condition maps to a dictionary with the following keys:

        - 'scores' : ndarray
            Array of decoding scores for each subject.
        - 'times' : ndarray
            Array of time points for decoding scores.
        - 'patterns' : ndarray
            Array of decoding patterns for each subject.

    Notes
    -----
    - The function assumes a specific directory structure defined by the `PATHS` object.
    - Decoding scores, times, and patterns are loaded from `.npy` files.
    - Ensures that 'times' arrays are consistent across subjects for each viewing condition.

    """
    if vc_list is None:
        vc_list = [""]
    data_dict = {}
    picks_str_folder = picks_str if picks_str is not None else ""
    gen_folder = gen_str if gen_str is not None else ""

    paths = PATHS()

    contrast_str = "_vs_".join(conditions)

    for vc in vc_list:
        data_dict[vc] = dict(scores=[], times=[], patterns=[])
        for sub_id in sub_list_str:
            fpath = Path(
                paths.DATA_04_DECOD_SENSORSPACE,
                vc,
                contrast_str,
                gen_folder,
                scoring,
                picks_str_folder,
                "scores",
            )
            fname = Path(fpath, f"{sub_id}-scores_per_sub.npy")
            scores_ = np.load(fname)
            data_dict[vc]["scores"].append(scores_)

            if len(data_dict[vc]["times"]) == 0:
                data_dict[vc]["times"] = np.load(str(fname)[:-4] + "__times" + ".npy")
            elif not np.all(data_dict[vc]["times"] == np.load(str(fname)[:-4] + "__times" + ".npy")):
                raise ValueError("Times are different between subjects.")

            fpath = Path(
                paths.DATA_04_DECOD_SENSORSPACE,
                vc,
                contrast_str,
                gen_folder,
                scoring,
                picks_str_folder,
                "patterns",
            )
            fname = Path(fpath, f"{sub_id}-patterns_per_sub.npy")
            patterns_ = np.load(fname)
            data_dict[vc]["patterns"].append(patterns_)

        data_dict[vc]["scores"] = np.array(data_dict[vc]["scores"])
        data_dict[vc]["patterns"] = np.array(data_dict[vc]["patterns"])

    return data_dict


def load_decod_res_crossdecod_viewcond(
    sub_list_str,
    conditions,
    vc_train_list=None,
    vc_test_list=None,
    scoring="roc_auc_ovr",
    picks_str=None,
    gen_str=None,
):
    """
    Load decoding results per viewing condition for a set of subjects and conditions.

    This function loads scores, times, and patterns associated with decoding results for
    each specified viewing condition. Data for each condition and subject is stored in
    a nested dictionary.

    Parameters
    ----------
    sub_list_str : list of str
        List of subject identifiers as strings, e.g., ['sub01', 'sub02'].
    conditions : list of str
        List of conditions to contrast (i.e., the decoding target classes), e.g., ['angry', 'neutral'].
    vc_list : list of str, optional
        List of viewing conditions to load, e.g. ["mono", "stereo"]. If empty, defaults to [""].
        "" (empty string) loads data pooled across viewing conditions.
    scoring : str, optional
        Metric used to score decoding results, by default "roc_auc_ovr".
    picks_str : str, optional
        String identifier for channel or sensor selections used during decoding. Defaults to None.
    gen_str : str, optional
        String identifier for generalization parameters (i.e., temporal generalization). Defaults to None.

    Returns
    -------
    dict
        A dictionary where each key is a viewing condition from `vc_list`.
        Each viewing condition maps to a dictionary with the following keys:

        - 'scores' : ndarray
            Array of decoding scores for each subject.
        - 'times' : ndarray
            Array of time points for decoding scores.
        - 'patterns' : ndarray
            Array of decoding patterns for each subject.

    Notes
    -----
    - The function assumes a specific directory structure defined by the `PATHS` object.
    - Decoding scores, times, and patterns are loaded from `.npy` files.
    - Ensures that 'times' arrays are consistent across subjects for each viewing condition.

    """
    data_dict = {}
    picks_str_folder = picks_str if picks_str is not None else ""
    gen_folder = gen_str if gen_str is not None else ""

    paths = PATHS()

    contrast_str = "_vs_".join(conditions)

    for vc_train in vc_train_list:
        for vc_test in vc_test_list:
            vc = f"train_{vc_train}-test_{vc_test}"
            data_dict[vc] = dict(scores=[], times=[], patterns=[])
            for sub_id in sub_list_str:
                fpath = Path(
                    paths.DATA_04_DECOD_SENSORSPACE,
                    "cross_decod_vc",
                    vc,
                    contrast_str,
                    gen_folder,
                    scoring,
                    picks_str_folder,
                    "scores",
                )
                fname = Path(fpath, f"{sub_id}-scores_per_sub.npy")
                scores_ = np.load(fname)
                data_dict[vc]["scores"].append(scores_)

                if len(data_dict[vc]["times"]) == 0:
                    data_dict[vc]["times"] = np.load(str(fname)[:-4] + "__times" + ".npy")
                elif not np.all(data_dict[vc]["times"] == np.load(str(fname)[:-4] + "__times" + ".npy")):
                    raise ValueError("Times are different between subjects.")

                fpath = Path(
                    paths.DATA_04_DECOD_SENSORSPACE,
                    "cross_decod_vc",
                    vc,
                    contrast_str,
                    gen_folder,
                    scoring,
                    picks_str_folder,
                    "patterns",
                )
                fname = Path(fpath, f"{sub_id}-patterns_per_sub.npy")
                patterns_ = np.load(fname)
                data_dict[vc]["patterns"].append(patterns_)

            data_dict[vc]["scores"] = np.array(data_dict[vc]["scores"])
            data_dict[vc]["patterns"] = np.array(data_dict[vc]["patterns"])

    return data_dict


def load_patterns(
    sub_list_str,
    contrast_str,
    viewcond="",
    scoring="accuracy"
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


    Returns
    -------
    patterns: ndarray
        Array with the patterns (subs x channels x (OvR contrasts) x times)
        OvR contrast: for multiclass classification with the OvR scheme we get n_classes contrasts;
        for binary contrasts, this axis is reduced.
    times: array, 1d

    """
    paths = PATHS()

    patterns_list = []
    times = []

    for sub_id in sub_list_str:
        fpath = Path(paths.DATA_04_DECOD_SENSORSPACE, viewcond, contrast_str, scoring, "patterns")
        fname = Path(fpath, f"{sub_id}-patterns_per_sub.npy")
        patterns_ = np.load(fname)
        patterns_list.append(patterns_)
        if len(times) == 0:
            times = np.load(str(fname)[:-4] + "__times" + ".npy")
        elif not np.all(times == np.load(str(fname)[:-4] + "__times" + ".npy")):
                raise ValueError("Times are different between subjects.")

    patterns = np.concatenate(patterns_list)
    return patterns, times


def get_info(sub_id="VR2FEM_S01", return_inst=False):
    """Load a template `info` object to grab the channel names."""
    paths = PATHS()
    fname = Path(
        paths.DATA_03_AR,
        "cleaneddata",
        f"{sub_id}-postAR-epo.fif",
    )
    epos = mne.read_epochs(fname, verbose=False)
    epos = epos.pick_types(eeg=True)

    if return_inst:
        return epos.info, epos

    return epos.info


def get_epos(sub_id):
    """Load the epochs for a given subject."""
    paths = PATHS()
    fname = Path(
        paths.DATA_03_AR,
        "cleaneddata",
        f"{sub_id}-postAR-epo.fif",
    )
    epos = mne.read_epochs(fname, verbose=False)
    epos = epos.pick_types(eeg=True)
    epos = epos.set_eeg_reference("average", projection=True)
    return epos


def plot_score_per_factor(  # noqa: C901, PLR0913
    factor,
    data,
    scoring="roc_auc",
    sign_clusters=None,
    p_lvl=0.05,
    chancelvl=0.5,
    ylims=None,
    xlims=None,
    ax=None,
    n_boot=10000,
    show_legend=True,
    hide_xaxis_labs=False,
    hide_yaxis_labs=False,
    fontsize=12,
):
    """
    Plot decoding scores over time, grouped by a specified factor.

    Parameters
    ----------
    factor : str
        The column in `data` used to group and color the plot lines.
    data : pandas.DataFrame
        DataFrame containing the decoding scores, with columns including 'time', 'score',
        and `factor`.
    scoring : str, optional
        Metric for decoding performance, by default "roc_auc". Only used for labelling the
        y-axis. If set to "roc_auc", auto-formatted to "ROC AUC".
    sign_clusters : list or dict, optional
        Time intervals where scores are significantly different, formatted as:
        - List of tuples, each defining a start and end time for a single cluster
        (e.g., [(start1, end1), (start2, end2)]), or
        - Dictionary with keys as factor levels and values as lists of tuples
        (e.g., {'factor1': [(start1, end1)], 'factor2': [(start2, end2)]}).
    p_lvl : float, optional
        Significance level, by default 0.05.
    chancelvl : float, optional
        Chance level line for reference on the y-axis, by default 0.5.
    ylims : tuple of float, optional
        Limits for the y-axis as (min, max). If None, the current y-axis limits are used.
    xlims : tuple of float, optional
        Limits for the x-axis as (min, max). If None, the current x-axis limits are used.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses the current axis.
    n_boot : int, optional
        Number of bootstrap samples for computing confidence intervals, by default 10000.
    show_legend : bool, optional
        Whether to display the legend, by default True.
    hide_xaxis_labs : bool, optional
        Whether to hide the x-axis labels, by default False.
    hide_yaxis_labs : bool, optional
        Whether to hide the y-axis labels, by default False.
    fontsize : int, optional
        Font size for axis labels and annotations, by default 12.

    Returns
    -------
    None
        The function modifies the given `ax` object with the plot, labels, and annotations.

    Notes
    -----
    - The function uses seaborn's `lineplot` to plot time-resolved decoding scores, grouped by `factor`.
    - Significance clusters are visualized as horizontal lines, with color corresponding to
      factor levels if `sign_clusters` is a dictionary.
    - If `sign_clusters` is not empty, a significance level label (e.g., "$p$ < .05") is added to
      the plot.

    """
    if sign_clusters is None:
        sign_clusters = []
    colors = COLORS()

    sns.lineplot(
        x="time",
        y="score",
        hue=factor,
        data=data,
        n_boot=n_boot,
        palette=colors.COLDICT,
        ax=ax,
        linewidth=1.5,
        legend=False,
        errorbar="se",
    )
    ytick_range = ax.get_ylim()
    if ylims is None:
        ylims = ytick_range
    ax.set(xlim=xlims, ylim=ylims)
    ax.tick_params(axis="both", labelsize=fontsize)
    scoring_str = "ROC AUC" if scoring == "roc_auc" else scoring
    ax.set_ylabel(scoring_str, fontsize=fontsize)
    ax.set_xlabel("Time (s)", fontsize=fontsize)

    ax.text(x=1.0, y=chancelvl + 0.001, s="chance", ha="right", fontsize=fontsize)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.vlines(
        (0),
        ymin=ylims[0],
        ymax=ylims[1],
        linestyles="dashed",
        linewidth=0.5,
        color="black",
    )
    ax.hlines(
        chancelvl,
        xmin=xlims[0],
        xmax=xlims[1],
        linewidth=0.5,
        color="black",
    )
    p_lvl_str = "$p$ < ." + str(p_lvl).split(".")[-1]
    if isinstance(sign_clusters, dict):
        for i, key in enumerate(sign_clusters):
            col = colors.COLDICT[key.lower()]
            for sc in sign_clusters[key.lower()]:
                xmin = sc[0]
                xmax = sc[-1]
                y_ = ylims[0] + (0.01 - ((i + 1) * 0.015) * np.ptp(ylims))
                ax.hlines(
                    y_,
                    xmin=xmin,
                    xmax=xmax,
                    color=col,
                    label=key,
                )
            if i == 0:
                ax.text(x=1.0, y=y_ + 0.001, s=p_lvl_str, ha="right", fontsize=fontsize)

    else:
        for sc in sign_clusters:
            xmin = sc[0]
            xmax = sc[-1]
            y_ = ytick_range[0] + 0.05 * np.ptp(ytick_range)
            ax.hlines(
                y_,
                xmin=xmin,
                xmax=xmax,
                color="purple",
                label="",
            )
        if len(sign_clusters) > 0:
            ax.text(x=1.0, y=y_ + 0.001, s=p_lvl_str, ha="right", fontsize=fontsize)

    if hide_xaxis_labs:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    if hide_yaxis_labs:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    if show_legend:
      handles_, labels_ = ax.get_legend_handles_labels()
      # remove duplicate labels and accompanying handles
      unique_labels = {label: handles_[i] for i, label in enumerate(labels_)}
      labels = list(unique_labels.keys())
      handles = list(unique_labels.values())

      ax.legend(
          handles=handles,
          labels=labels,
          loc="upper right",
          bbox_to_anchor=(1.0, 1.0),
          frameon=False,
          ncol=len(labels),
          fontsize=fontsize,
      )



def prep_and_plot_from_data(data_dict,
                            subsets,
                            ax,
                            chancelvl=0.25,
                            ylims=(0.2, 0.32),
                            scoring="roc_auc",
                            pval_cbp=0.05,
                            show_legend=True,
                            hide_xaxis_labs=False,
                            hide_yaxis_labs=False,
                            nperm=10000,
                            fontsize=12):
    """
    Prepare and plot decoding performance data with significant clusters highlighted.

    This function prepares decoding data for plotting by transforming it to a long format
    suitable for seaborn and identifying statistically significant clusters where decoding
    performance exceeds the chance level by running a cluster-based permutation test.
    It then visualizes the data, highlighting clusters in which decoding performance is significantly
    above chance or differs between subsets.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing decoding scores, times, and patterns for each viewing condition.
        Expected format: `data_dict[view_condition]["scores"]` (array-like of scores),
        `data_dict[view_condition]["times"]` (array of time points).
    subsets : list of str
        List of viewing conditions (keys in `data_dict`) to plot and analyze.
    ax : matplotlib.axes.Axes
        Matplotlib axis on which to plot the decoding results.
    chancelvl : float, optional
        Chance level threshold for decoding performance, by default 0.25.
    ylims : tuple of float, optional
        Y-axis limits for the plot, by default (0.2, 0.32).
    scoring : str, optional
        Scoring metric name (e.g., "roc_auc" or "accuracy"), by default "roc_auc".
    pval_cbp : float, optional
        Significance level for cluster-based permutation tests (cluster-level), by default 0.05.
    show_legend : bool, optional
        Whether to display the legend on the plot, by default True.
    hide_xaxis_labs : bool, optional
        Whether to hide the x-axis labels, by default False.
    hide_yaxis_labs : bool, optional
        Whether to hide the y-axis labels, by default False.
    nperm : int, optional
        Number of permutations to run for cluster-based permutation tests, by default 10000.
    fontsize : int, optional
        Font size for axis labels and annotations, by default 12.

    Returns
    -------
    None
        The function modifies the provided matplotlib axis `ax` in place.

    Notes
    -----
    - The function computes clusters of significant decoding scores for each viewing
      condition individually, as well as between-condition differences if two conditions are provided.
    - Cluster-based permutation tests (CBP) are performed using `run_cbp_test`.
    - Decoding performance across different conditions and significant clusters are visualized
      using `plot_score_per_factor()`.

    """
    results_df_list = []
    for vc in subsets:
        times = data_dict[vc]["times"]
        acc = np.asarray(data_dict[vc]["scores"])
        acc_df = pd.DataFrame(acc)
        acc_df.columns = times
        df_tmp = acc_df.melt(var_name="time", value_name="score")  # put into long format
        df_tmp["vc"] = vc
        results_df_list.append(df_tmp)
    data_plot = pd.concat(results_df_list)

    sign_cluster_times = {}
    # run CBP to find differences from chance:
    for vc in subsets:
        data = np.asarray(data_dict[vc]["scores"]) - chancelvl
        t_values, clusters, p_values = run_cbp_test(data, tail=1, nperm=nperm)
        idx_sign_clusters = np.argwhere(p_values < pval_cbp)
        sign_cluster_times[vc] = [times[clusters[idx[0]]][[0, -1]] for idx in idx_sign_clusters]
        print(f"Found {len(sign_cluster_times[vc])} significant clusters for viewing condition '{vc}'.")
        for i in range(len(sign_cluster_times[vc])):
            print(f"Significant cluster: {sign_cluster_times[vc][i][0] * 1000:.3f} - "
                                       f"{sign_cluster_times[vc][i][1] * 1000:.3f}")

    if len(subsets) == 2:  # noqa: PLR2004
        # run CBP to find difference between conditions:
        data = np.asarray(data_dict[subsets[0]]["scores"]) - np.asarray(data_dict[subsets[1]]["scores"])
        t_values, clusters, p_values = run_cbp_test(data, tail=0, nperm=nperm)
        idx_sign_clusters = np.argwhere(p_values < pval_cbp)
        sign_cluster_times["diff"] = [times[clusters[idx[0]]][[0, -1]] for idx in idx_sign_clusters]
        if len(sign_cluster_times["diff"]) > 0:
            helpers.print_msg("Found significant difference between conditions! Do you see that?")

    plot_score_per_factor(
        factor="vc",
        data=data_plot.reset_index(),
        scoring=scoring,
        sign_clusters={ecc: sign_cluster_times[ecc] for ecc in subsets},
        p_lvl=pval_cbp,
        chancelvl=chancelvl,
        ylims=ylims,
        xlims=(-0.3, 1.0),
        n_boot=10000,
        ax=ax,
        show_legend=show_legend,
        hide_xaxis_labs=hide_xaxis_labs,
        hide_yaxis_labs=hide_yaxis_labs,
        fontsize=fontsize,
    )


def get_max_decod_score_and_time(data_dict):
    """
    Compute the peak decoding score and its timing with associated statistics.

    This function calculates the peak mean decoding score, its standard deviation, and 95% confidence interval
    (CI) across samples in `data_dict["scores"]`. Additionally, it identifies the time at which the peak decoding
    score occurs, along with the timing's standard deviation and CI, based on the data provided in `data_dict`.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing `scores` (2D array-like, where each row is a sample and each column is a time point)
        and `times` (1D array of time points corresponding to columns in `scores`).

    Returns
    -------
    peak_mean : float
        Mean of the peak decoding scores across samples.
    peak_sd : float
        Standard deviation of the peak decoding scores across samples.
    peak_cil : float
        Lower bound of the 95% confidence interval for the peak decoding score.
    peak_ciu : float
        Upper bound of the 95% confidence interval for the peak decoding score.
    peak_time : float
        Median time of peak decoding across samples (in milliseconds).
    peak_time_sd : float
        Standard deviation of peak times across samples (in milliseconds).
    peak_time_cil : float
        Lower bound of the 95% confidence interval for the peak decoding time.
    peak_time_ciu : float
        Upper bound of the 95% confidence interval for the peak decoding time.

    Notes
    -----
    - The peak score is determined by the maximum score across time points for each sample.
    - The peak time is reported as the median across samples, and timing is converted from seconds to milliseconds.

    """
    peak_mean = data_dict["scores"].max(axis=1).mean()
    peak_sd = data_dict["scores"].max(axis=1).std()
    peak_cil = peak_mean - 1.96 * peak_sd / np.sqrt(len(data_dict["scores"]))
    peak_ciu = peak_mean + 1.96 * peak_sd / np.sqrt(len(data_dict["scores"]))
    times = data_dict["times"]
    tidx = int(np.median(data_dict["scores"].argmax(axis=1)))
    peak_time = times[tidx] * 1000
    peak_time_sd = np.std(times[data_dict["scores"].argmax(axis=1)]) * 1000
    peak_time_cil = peak_time - 1.96 * peak_time_sd / np.sqrt(len(data_dict["scores"]))
    peak_time_ciu = peak_time + 1.96 * peak_time_sd / np.sqrt(len(data_dict["scores"]))
    time_mode = stats.mode(times[data_dict["scores"].argmax(axis=1)])
    peak_time_mode = time_mode.mode * 1000
    peak_time_mode_n = time_mode.count
    print(f"Peak mean: M = {peak_mean:.2f}, SD = {peak_sd:.2f}, 95% CI [{peak_cil:.2f}, {peak_ciu:.2f}]")
    print(f"Peak time: Mdn = {peak_time:.2f}, SD = {peak_time_sd:.2f}, 95% CI [{peak_time_cil:.2f}, "
                                                                            f"{peak_time_ciu:.2f}]")
    return peak_mean, peak_sd, peak_cil, peak_ciu, peak_time, peak_time_sd, peak_time_cil, peak_time_ciu, peak_time_mode, peak_time_mode_n


def get_decod_df(data_dict):
    """Get a DataFrame with peak scores and times."""
    peaks = data_dict["scores"].max(axis=1)
    times = data_dict["times"]
    peak_times = times[data_dict["scores"].argmax(axis=1)] * 1000
    peak_df = pd.DataFrame({"peak": peaks, "peak_time": peak_times})
    return peak_df

def get_binary_results(data_dict_na, contrast_str):
    """Get a dataframe with decoding stats per viewcond."""
    # print stats:
    print("\nMono:")
    res_mono  = get_max_decod_score_and_time(data_dict_na["mono"])
    print("\nStereo:")
    res_stereo = get_max_decod_score_and_time(data_dict_na["stereo"])

    dd = {("Mono", "Peak", "Mean"): res_mono[0],
        ("Mono", "Peak", "SD"): res_mono[1],
        ("Mono", "Peak", "cil"): res_mono[2],
        ("Mono", "Peak", "ciu"): res_mono[3],
        ("Mono", "Peak", "CI"): f"[{res_mono[2]:.2f}, {res_mono[3]:.2f}]",
        ("Mono", "Time", "Median"): res_mono[4],
        ("Mono", "Time", "SD"): res_mono[5],
        ("Mono", "Time", "cil"): res_mono[6],
        ("Mono", "Time", "ciu"): res_mono[7],
        ("Mono", "Time", "CI"): f"[{res_mono[6]:.2f}, {res_mono[7]:.2f}]",
        ("Mono", "Time", "Mode"): res_mono[8],
        ("Mono", "Time", "Mode Count"): res_mono[9],
        ("Stereo", "Peak", "Mean"): res_stereo[0],
        ("Stereo", "Peak", "SD"): res_stereo[1],
        ("Stereo", "Peak", "cil"): res_stereo[2],
        ("Stereo", "Peak", "ciu"): res_stereo[3],
        ("Stereo", "Peak", "CI"): f"[{res_stereo[2]:.2f}, {res_stereo[3]:.2f}]",
        ("Stereo", "Time", "Median"): res_stereo[4],
        ("Stereo", "Time", "SD"): res_stereo[5],
        ("Stereo", "Time", "cil"): res_stereo[6],
        ("Stereo", "Time", "ciu"): res_stereo[7],
        ("Stereo", "Time", "CI"): f"[{res_stereo[6]:.2f}, {res_stereo[7]:.2f}]",
        ("Stereo", "Time", "Mode"): res_stereo[8],
        ("Stereo", "Time", "Mode Count"): res_stereo[9],
        }

    # run paired t-test:
    df_mono = get_decod_df(data_dict_na["mono"])
    df_stereo = get_decod_df(data_dict_na["stereo"])
    ttest_res = stats.ttest_rel(df_mono["peak"], df_stereo["peak"])
    print("\nStats (mono vs stereo):")
    print(f"Peak stats (paired t test): t = {ttest_res.statistic:.2f}, p = {ttest_res.pvalue:.3f}")
    dd[("Stats", "Peak", f"t({ttest_res.df})")] = ttest_res.statistic
    dd[("Stats", "Peak", "p")] = ttest_res.pvalue

    ttest_res_time = stats.ttest_rel(df_mono["peak_time"], df_stereo["peak_time"])
    print(f"Peak time stats: t = {ttest_res_time.statistic:.2f}, p = {ttest_res_time.pvalue:.3f}")
    dd[("Stats", "Time", f"t({ttest_res_time.df})")] = ttest_res_time.statistic
    dd[("Stats", "Time", "p")] = ttest_res_time.pvalue

    del ttest_res, ttest_res_time, res_mono, res_stereo, df_mono, df_stereo

    df_out = pd.DataFrame(dd, index=[contrast_str])
    return df_out


def print_table_contrasts(sub_list_str, variable="Peak", notebook=True, print_latex=True):
    """
    Print a table of binary contrasts for decoding results.

    This function computes decoding results for a predefined set of binary contrasts between
    emotional facial expressions or identities, loads the results, formats them, and displays
    them as a styled table. It also outputs the table in LaTeX format for use in scientific
    reports or publications.

    Parameters
    ----------
    sub_list_str : list of str
        List of subject identifiers to include.
    variable : str, optional
        The decoding metric to display in the table. Can be "Peak" or "Time". Defaults to "Peak".
    notebook : bool, optional
        If run from a Jupyter notebook, set to True to display the table in the notebook. Defaults to True.
    print_latex : bool, optional
        Set to True to print the LaTeX code to produce the table. Defaults to True.

    Returns
    -------
    None
        Displays a styled DataFrame table and prints a LaTeX representation of the table.

    Notes
    -----
    - The function defines a set of binary contrasts for comparison, primarily between emotion types 
      (e.g., "angry vs neutral", "happy vs surprised") and different identities (e.g., "id1", "id2", "id3").
    - It loads decoding results for each contrast using `load_decod_res_per_viewcond` and retrieves binary 
      results using `get_binary_results`.
    - The displayed table includes formatted p-values (to three decimal places) and peak scores, and excludes 
      confidence intervals.
    - The LaTeX output is formatted with centered column alignment for ease of integration into scientific documents.

    """
    binary_contrasts = [("angry", "neutral"),
                        ("angry", "happy"),
                        ("angry", "surprised"),
                        ("happy", "neutral"),
                        ("happy", "surprised"),
                        ("surprised", "neutral"),
                        ("id1", "id2", "id3")]

    mydf = pd.DataFrame()
    for c in binary_contrasts:
        print(f"\n\n\nContrast: {' vs '.join(c)}")
        data_dict_na = load_decod_res_per_viewcond(
                sub_list_str=sub_list_str,
                conditions=list(c),
                vc_list=["mono", "stereo"],
                scoring="roc_auc_ovr",
        )
        ddf = get_binary_results(data_dict_na, "  vs  ".join(c))
        mydf = pd.concat([mydf, ddf]) if mydf.size else ddf

    # Peak scores:
    idx = pd.IndexSlice
    peak_df = mydf.loc[:, idx[:, variable, :]]
    # remove useless level of the multiindex:
    peak_df.columns = peak_df.columns.droplevel(1)

    idx = pd.IndexSlice
    pval_cols = idx[:, idx[:,"p"]]
    sdf = peak_df.drop(columns=["cil", "ciu"], level=1) \
            .style \
            .set_table_styles([dict(selector="th", props=[("text-align", "center")])]) \
            .format(lambda x: f"{x:.2f}" if isinstance(x, float) else x) \
            .format("{:.3f}", subset=pval_cols)
    if notebook:
      display(sdf)
    if print_latex:
      print(sdf.to_latex(multicol_align="c", hrules=True, column_format="lcccccccc"))