"""
Calculate behavioral confusion matrix.

This script calculates the confusion matrix for the behavioral data of the main task.
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from vr2f.staticinfo import PATHS
from vr2f.utils.access_keeper import copy_to_keeper


def get_confusion_matrix(sub_id, viewcond):
    n_block_training = 2
    paths = PATHS()
    path_subs = paths.DATA_SUBJECTS
    path_in = Path(path_subs, sub_id, "MainTask", "Unity", "S001")
    fname = Path(path_in, "trial_results.csv")
    t_res = pd.read_csv(fname)

    # drop training trials:
    t_res = t_res[t_res.block_type == n_block_training]
    # and trials without response
    t_res = t_res[t_res.selected_emotion != -1]

    match viewcond:
        case "stereo":
            t_res_c = t_res[t_res["stereo"] == 1]
        case "mono":
            t_res_c = t_res[t_res["stereo"] == 0]
        case _:
            t_res_c = t_res
    cf_ing = t_res_c[["emo", "selected_emotion"]]

    cf = confusion_matrix(cf_ing.emo, cf_ing.selected_emotion, labels=(1,2,3,4,5))

    # normalize by the number of trials per condition
    cf = cf / (90, 180)[viewcond == "all"]
    return cf



# Plot confusion matrices

def plot_confusion_matrix(cf_avg, viewcond, save_to_disk=True, figsize_factor=1, show_numbers=True):
    emo_list = ["neutral", "happy", "angry", "surprised", "other"]
    paths = PATHS()
    cpal = sns.blend_palette(["darkblue", "green", "darkred"], n_colors=6000, as_cmap=True, input="rgb")

    cm = 1 / 2.54  # centimeters in inches
    figsize=(19.5, 16.8)
    figsize = (figsize[0] * cm * figsize_factor, figsize[1] * cm * figsize_factor)
    sns.set(rc={"axes.facecolor": "#0000FF", "figure.facecolor": (0, 0, 0, 0)})
    sns.set(font_scale=figsize_factor)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    o = sns.heatmap(
        cf_avg[viewcond][:4],
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=emo_list,
        yticklabels=emo_list[:-1],
        annot=show_numbers,
        cbar=show_numbers,
        ax=ax,
    )  # cmap=cpal)  #
    ax.tick_params(axis="y", labelrotation=45)
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_ylabel("ground truth")
    ax.set_xlabel("choice")
    ax.set_title(f"Viewing condition: {viewcond}")
    fig.tight_layout()
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "Roboto"
    if save_to_disk:
        fig.savefig(
            Path(paths.FIGURES, f"confusion_matrix_{viewcond}.pdf"),
            dpi=300,
            transparent=True,
            bbox_inches="tight",
    )
    return fig


def main():
    """Run main."""
    paths = PATHS()

    path_subs = paths.DATA_SUBJECTS
    sub_list_str = sorted(os.listdir(path_subs))

    subs_incomplete = ["VR2FEM_S09"]
    sub_list_str = [s for s in sub_list_str if s not in subs_incomplete]

    cfs = {"all": [], "stereo": [], "mono": []}
    for sub_id in sub_list_str:
        for cond in cfs:
            cf = get_confusion_matrix(sub_id, cond)
            cfs[cond].append(cf)

    # Average across subjects
    cf_avg = {}
    for cond in cfs:
        cf_avg[cond] = np.array(cfs[cond]).mean(axis=0)
        plot_confusion_matrix(cf_avg, cond)

        source = str(Path(paths.FIGURES, f"confusion_matrix_{cond}.pdf"))
        copy_to_keeper(source_file=source, dest_folder="Figures")


if __name__ == "__main__":
    main()

