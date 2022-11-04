import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from vr2fem_analyses.staticinfo import CONFIG, PATHS, COLORS


paths = PATHS()

path_subs = paths.DATA_SUBJECTS
sub_list_str = sorted(os.listdir(path_subs))

subID = sub_list_str[0]

emo_list = ["neutral", "happy", "angry", "surprised", "other"]

cfs = {"all": [], "stereo": [], "mono": []}
for subID in sub_list_str:
    path_in = Path(path_subs, subID, "MainTask", "Unity", "S001")
    fname = Path(path_in, "trial_results.csv")
    t_res = pd.read_csv(fname)

    # drop training trials:
    t_res = t_res[t_res.block_type == 2]
    # and trials without response
    t_res = t_res[t_res.selected_emotion != -1]

    for cond in cfs:
        match cond:
            case "stereo":
                t_res_c = t_res[t_res["stereo"] == 1]
            case "mono":
                t_res_c = t_res[t_res["stereo"] == 0]
            case _:
                t_res_c = t_res
        cf_ing = t_res_c[["emo", "selected_emotion"]]

        cf = confusion_matrix(cf_ing.emo, cf_ing.selected_emotion)
        # cf = cf / 180
        if len(cf) < 5:
            cf = np.hstack((cf, np.zeros((4, 1))))
            cf = np.vstack((cf, np.zeros((1, 5))))
        cf = cf / (90, 180)[cond == 'all']
        cfs[cond].append(cf)

cf_avg = {}
for cond in cfs:
    cf_avg[cond] = np.array(cfs[cond]).mean(axis=0)

cpal = sns.blend_palette(
    ["darkblue", "green", "darkred"], n_colors=6000, as_cmap=True, input="rgb"
)

plt.style.use('dark_background')
for cond, annot in zip(cfs, [True, False, False]):
    fig, ax = plt.subplots(1,1)
    sns.heatmap(cf_avg[cond][:4], cmap="viridis", vmin=0, vmax=1, xticklabels=emo_list, yticklabels=emo_list[:-1], annot=annot, cbar=annot, ax=ax)  # cmap=cpal)  #
    ax.tick_params(axis='y', labelrotation=45)
    ax.tick_params(axis='x', labelrotation=45)
    ax.set_ylabel("ground truth")
    ax.set_xlabel("choice")
    ax.set_title(cond)
