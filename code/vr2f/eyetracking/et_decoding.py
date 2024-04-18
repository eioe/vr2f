import os
import sys

# %% load libs:
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.decoding import GeneralizingEstimator, LinearModel, SlidingEstimator, cross_val_multiscore, get_coef
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from vr2f import helpers
from vr2f.staticinfo import PATHS


def decode_core(X, y, groups, scoring="roc_auc", temp_gen=False, n_cv_folds=5, cv_random_state=None):
    clf = make_pipeline(
        mne.decoding.Vectorizer(), LinearModel(LogisticRegression(solver="liblinear", random_state=42, verbose=False))
    )

    if temp_gen:
        gen_str = "gen_temp"
        se = GeneralizingEstimator(clf, scoring=scoring, n_jobs=15, verbose=0)
    else:
        gen_str = ""
        se = SlidingEstimator(clf, scoring=scoring, n_jobs=15, verbose=0)
    if groups is None:
        cv = StratifiedKFold(n_splits=n_cv_folds)
    else:
        # raise not yet implemented error:
        raise ValueError("Grouped cross validation not yet implemented")

    scores = cross_val_multiscore(se, X, y, cv=cv, groups=groups, n_jobs=n_cv_folds)

    se.fit(X, y)
    patterns = get_coef(se, "patterns_", inverse_transform=True)

    return scores, patterns


def decode_et(df, sub_id, factor, contrast, scoring, reps=50):
    df_ = df.query("sub_id == @sub_id").query(f"{factor} in @contrast")

    # add an index which counts from 0 up, for each trial_num
    df_["times_idx"] = df_.groupby("trial_num").cumcount()
    iidx = pd.MultiIndex.from_product(
        [df_["trial_num"].unique(), df_["times_idx"].unique()],
        names=["trial_num", "times_idx"],
    )
    df_p = pd.pivot_table(df_, values=["phi", "theta"], index=["trial_num", "times_idx"]).reindex(iidx)
    df_p["phi"] = df_p["phi"].interpolate(method="linear")
    df_p["theta"] = df_p["theta"].interpolate(method="linear")

    # throw away trials which still contain nan values
    times = df_.groupby("times_idx").mean()["times"].to_numpy()
    idx_na = df_p.groupby("trial_num")["phi"].apply(lambda x: x.isna().sum() > 0).to_numpy()
    df_p = df_p[~np.repeat(idx_na, len(times))]

    X = (
        df_p.to_numpy()
        .reshape(df_p.reset_index()["trial_num"].nunique(), df_p.reset_index()["times_idx"].nunique(), 2)
        .swapaxes(1, 2)
    )

    y = df_.query("times_idx == 0")[factor].to_numpy()
    y = y[~idx_na]
    sc_tmp = []
    for i in range(reps):
        sc, _ = decode_core(X, y, groups=None, temp_gen=False, n_cv_folds=5, scoring=scoring)
        sc_tmp.append(sc)
    sc = np.array(sc_tmp).mean(axis=0)
    return sc, times


def main(sub_nr: int):
    paths = PATHS()
    pattern = "preproc.csv"
    sub_list_str_et = [f for f in os.listdir(paths.DATA_ET_PREPROC) if pattern in f]
    sub_list_str_et = [f.split("-")[0] for f in sub_list_str_et]
    sub_list_str_et = np.unique(sorted(sub_list_str_et))

    cond_dict = {  # noqa: F841
        "viewcond": {1: "mono", 2: "stereo"},
        "emotion": {1: "neutral", 2: "happy", 3: "angry", 4: "surprised"},
        "avatar_id": {1: "Woman_01", 2: "Woman_04", 3: "Woman_08"},
    }

    data_preproc = []
    for sub_id in sorted(sub_list_str_et):
        fname = Path(paths.DATA_ET_PREPROC, f"{sub_id}-ET-preproc.csv")
        df = pd.read_csv(fname, sep=",")
        df["sub_id"] = sub_id
        data_preproc.append(df)

    df_all = pd.concat(data_preproc, ignore_index=True)

    # select only columns phi and theta
    df_s = df_all[["times", "theta", "phi", "viewcond", "avatar_id", "emotion", "sub_id", "trial_num"]]
    factor = "avatar_id"  # "viewcond"  #  "emotion"  #
    scoring = "roc_auc_ovr"  # _ovr'
    contrasts = [
        tuple(cond_dict["avatar_id"].values())
    ]
        
    # ("mono", "stereo")

    # tuple(cond_dict["emotion"].values())

    # ("angry", "neutral"),  ("angry", "surprised"), ("angry", "happy"),
    #     ("happy", "neutral"), ("happy", "surprised"),
    #     ("surprised", "neutral")

    scores_all = defaultdict(list)
    times_all = defaultdict(list)

    save_scores = True

    if sub_nr is not None:
        sub_list_str_et = [sub_list_str_et[sub_nr]]

    for contrast in contrasts:
        for sub_id in sorted(sub_list_str_et):
            print("Running subject: ", sub_id)
            sc, times = decode_et(df_s, sub_id, factor=factor, contrast=contrast, scoring=scoring, reps=50)
            scores_all[sub_id].append(sc)
            times_all[sub_id].append(times)

            # save shizzle:
            contrast_str = "_vs_".join([c.lower() for c in contrast])
            conditions_vc = ""

            path_save = Path(
                paths.DATA_ET_DECOD,
                conditions_vc,
                contrast_str,
                scoring,
            )

            # save accuracies:
            if save_scores:
                fpath = Path(path_save, "scores")
                fpath.mkdir(exist_ok=True, parents=True)
                fname = Path(fpath, f"{sub_id}-scores_per_sub.npy")
                np.save(fname, sc)
                np.save(str(fname)[:-4] + "__times" + ".npy", times)
                del (fpath, fname)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None
    main(JOB_NR)


# %%
