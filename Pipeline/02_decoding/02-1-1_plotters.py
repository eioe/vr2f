from collections import defaultdict
from os import path as op
from pathlib import Path
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.lib.npyio import load

import seaborn as sns

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

from scipy import stats

import mne
from mne import EvokedArray

# from mne.epochs import concatenate_epochs
from mne.decoding import (
    SlidingEstimator,  # GeneralizingEstimator,
    cross_val_multiscore,
    LinearModel,
    get_coef,
)
from mne.stats import permutation_cluster_1samp_test, f_mway_rm, f_threshold_mway_rm

from vr2fem_analyses.staticinfo import CONFIG, PATHS, COLORS
from vr2fem_analyses import helpers


def load_decod_res_per_viewcond(
    sub_list_str,
    conditions,
    vc_list=[],
    scoring="accuracy",
    picks_str=None,
    gen_str=None,
):
    data_dict = dict()
    if picks_str is not None:
        picks_str_folder = picks_str
    else:
        picks_str_folder = ""

    if gen_str is not None:
        gen_folder = gen_str
    else:
        gen_folder = ""

    if len(vc_list) == 0:
        vc_list = [""]

    paths = PATHS()

    contrast_str = "_vs_".join(conditions)

    for vc in vc_list:
        data_dict[vc] = dict(scores=[], times=[], patterns=[])
        for subID in sub_list_str:
            fpath = Path(
                paths.DATA_04_DECOD_SENSORSPACE,
                vc,
                contrast_str,
                gen_folder,
                scoring,
                picks_str_folder,
                "scores",
            )
            fname = Path(fpath, f"{subID}-scores_per_sub.npy")
            scores_ = np.load(fname)
            data_dict[vc]["scores"].append(scores_)

            if len(data_dict[vc]["times"]) == 0:
                data_dict[vc]["times"] = np.load(str(fname)[:-4] + "__times" + ".npy")
            else:
                assert np.all(
                    data_dict[vc]["times"]
                    == np.load(str(fname)[:-4] + "__times" + ".npy")
                )

            fpath = Path(
                paths.DATA_04_DECOD_SENSORSPACE,
                contrast_str,
                gen_folder,
                scoring,
                picks_str_folder,
                "patterns",
            )
            fname = op.join(fpath, f"{subID}-patterns_per_sub.npy")
            patterns_ = np.load(fname)
            data_dict[vc]["patterns"].append(patterns_)

        data_dict[vc]["scores"] = np.array(data_dict[vc]["scores"])
        data_dict[vc]["patterns"] = np.array(data_dict[vc]["patterns"])

    return data_dict


def run_cbp_test(data, tail=0):
    # number of permutations to run
    n_permutations = 10000
    # set initial threshold
    p_initial = 0.05
    # set family-wise p-value
    p_thresh = 0.05
    connectivity = None
    tail = tail  # 1 or -1 for one-sided test, 0 for two-sided

    config = CONFIG()

    # set cluster threshold
    n_samples = len(data)
    threshold = -stats.t.ppf(p_initial / (1 + (tail == 0)), n_samples - 1)
    if np.sign(tail) < 0:
        threshold = -threshold

    cluster_stats = mne.stats.permutation_cluster_1samp_test(
        data,
        threshold=threshold,
        n_jobs=config.N_JOBS,
        verbose=False,
        tail=tail,
        step_down_p=0.0005,
        adjacency=connectivity,
        n_permutations=n_permutations,
        seed=42,
        out_type="mask",
    )

    T_obs, clusters, cluster_p_values, _ = cluster_stats
    return T_obs, clusters, cluster_p_values


def plot_score_per_factor(
    factor,
    data,
    scoring="accuracy",
    sign_clusters=[],
    p_lvl=0.05,
    chancelvl=0.5,
    ylims=None,
    xlims=None,
    ax=None,
    n_boot=1000,
):

    colors = COLORS()

    sns.lineplot(
        x="time",
        y="score",
        hue=factor,
        data=data,
        n_boot=n_boot,
        palette=colors.COLDICT,
        ax=ax,
        linewidth=0.5,
        legend=False,
        errorbar="se",
    )
    ytick_range = ax.get_ylim()
    if ylims is None:
        ylims = ytick_range
    ax.set(xlim=xlims, ylim=ylims)
    if scoring == "roc_auc":
        scoring_str = "ROC AUC"
    else:
        scoring_str = scoring
    ax.set_ylabel(scoring_str)
    ax.set_xlabel("Time (s)")

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
    p_lvl_str = "p < ." + str(p_lvl).split(".")[-1]
    if isinstance(sign_clusters, dict):
        for i, key in enumerate(sign_clusters):
            col = colors.COLDICT[key.lower()]
            for sc in sign_clusters[key.lower()]:
                xmin = sc[0]
                xmax = sc[-1]
                ax.hlines(
                    ylims[0] + (0.01 - ((i + 1) * 0.025) * np.ptp(ylims)),
                    xmin=xmin,
                    xmax=xmax,
                    color=col,
                    label=p_lvl_str,
                )

    else:
        for sc in sign_clusters:
            xmin = sc[0]
            xmax = sc[-1]
            ax.hlines(
                ytick_range[0] + 0.05 * np.ptp(ytick_range),
                xmin=xmin,
                xmax=xmax,
                color="purple",
                label=p_lvl_str,
            )
    handles, labels = ax.get_legend_handles_labels()
    n_sgn_clu = None if len(sign_clusters) <= 1 else -(len(sign_clusters))
    # ax.legend(handles=handles[1:n_sgn_clu+1], labels=labels[1:n_sgn_clu+1])


def prep_and_plot_from_data(data_dict, subsets, ax, chancelvl=0.25, ylims=(0.2, 0.32)):
    # Prepare data for plotting with seaborn:
    results_df_list = list()
    for vc in subsets:
        times = data_dict[vc]["times"]
        acc = np.asarray(data_dict[vc]["scores"])
        acc_df = pd.DataFrame(acc)
        acc_df.columns = times
        df = acc_df.melt(var_name="time", value_name="score")  # put into long format
        df["vc"] = vc
        results_df_list.append(df)
    data_plot = pd.concat(results_df_list)

    sign_cluster_times = {}
    # run CBP to find differences from chance:
    for vc in subsets:
        data = np.asarray(data_dict[vc]["scores"]) - chancelvl
        t_values, clusters, p_values = run_cbp_test(data, tail=1)
        idx_sign_clusters = np.argwhere(p_values < p_val_cbp)
        sign_cluster_times[vc] = [
            times[clusters[idx[0]]][[0, -1]] for idx in idx_sign_clusters
        ]

    if len(subsets) == 2:
        # run CBP to find difference between conditions:
        data = np.asarray(data_dict[subsets[0]]["scores"]) - np.asarray(
            data_dict[subsets[1]]["scores"]
        )
        t_values, clusters, p_values = run_cbp_test(data, tail=0)
        idx_sign_clusters = np.argwhere(p_values < p_val_cbp)
        sign_cluster_times["diff"] = [
            times[clusters[idx[0]]][[0, -1]] for idx in idx_sign_clusters
        ]
        if len(sign_cluster_times["diff"]) > 0:
            helpers.print_msg(
                "Found significant difference between conditions! Do you see that?"
            )

    # Plot it:

    plot_score_per_factor(
        factor="vc",
        data=data_plot.reset_index(),
        scoring="Accuracy",
        sign_clusters={ecc: sign_cluster_times[ecc] for ecc in subsets},
        p_lvl=p_val_cbp,
        chancelvl=chancelvl,
        ylims=ylims,
        xlims=(-0.2, 1.0),
        n_boot=10000,
        ax=ax,
    )


# Setup:
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
cm = 1 / 2.54

# plotting:

p_val_cbp = 0.05

paths = PATHS()
path_in = Path(paths.DATA_03_AR, "cleaneddata")

# load data
sub_list_str = [s.split("-postAR-epo")[0] for s in os.listdir(path_in)]

data_dict_allemos = load_decod_res_per_viewcond(
    sub_list_str=sub_list_str,
    conditions=["neutral", "angry", "happy", "surprised"],
    vc_list=["", "mono", "stereo"],
)
data_dict_allemos["all"] = data_dict_allemos.pop("")

fig, ax = plt.subplots(1, figsize=(20 * cm, 18 * cm))
prep_and_plot_from_data(
    data_dict_allemos, subsets=["all"], ax=ax, chancelvl=0.25, ylims=(0.2, 0.32)
)


fig, ax = plt.subplots(1, figsize=(20 * cm, 18 * cm))
prep_and_plot_from_data(
    data_dict=data_dict_allemos,
    subsets=["stereo", "mono"],
    ax=ax,
    chancelvl=0.25,
    ylims=(0.2, 0.32),
)


data_dict_vc = load_decod_res_per_viewcond(
    sub_list_str=sub_list_str,
    conditions=["mono", "stereo"],
    vc_list=[""],
)
data_dict_vc["viewcond"] = data_dict_vc.pop("")

fig, ax = plt.subplots(1, figsize=(20 * cm, 18 * cm))
prep_and_plot_from_data(
    data_dict=data_dict_vc,
    subsets=["viewcond"],
    ax=ax,
    chancelvl=0.5,
    ylims=(0.45, 0.6),
)



## Contrast: angry-neutral

data_dict_na = load_decod_res_per_viewcond(
    sub_list_str=sub_list_str,
    conditions=["neutral", "angry"],
    vc_list=["", "mono", "stereo"],
)
data_dict_na["all"] = data_dict_na.pop("")

fig, ax = plt.subplots(1, figsize=(20 * cm, 18 * cm))
prep_and_plot_from_data(
    data_dict_na, subsets=["all"], ax=ax, chancelvl=0.5, ylims=(0.45, 0.6)
)


fig, ax = plt.subplots(1, figsize=(20 * cm, 18 * cm))
prep_and_plot_from_data(
    data_dict=data_dict_na,
    subsets=["stereo", "mono"],
    ax=ax,
    chancelvl=0.5,
    ylims=(0.45, 0.6),
)


## Contrast: sursprised-angry

data_dict_na = load_decod_res_per_viewcond(
    sub_list_str=sub_list_str,
    conditions=["surprised", "angry"],
    vc_list=["", "mono", "stereo"],
)
data_dict_na["all"] = data_dict_na.pop("")

fig, ax = plt.subplots(1, figsize=(20 * cm, 18 * cm))
prep_and_plot_from_data(
    data_dict_na, subsets=["all"], ax=ax, chancelvl=0.5, ylims=(0.45, 0.6)
)


fig, ax = plt.subplots(1, figsize=(20 * cm, 18 * cm))
prep_and_plot_from_data(
    data_dict=data_dict_na,
    subsets=["stereo", "mono"],
    ax=ax,
    chancelvl=0.5,
    ylims=(0.45, 0.6),
)
