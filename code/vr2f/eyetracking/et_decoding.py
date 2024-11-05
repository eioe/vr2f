"""Run eye tracking decoding."""
import multiprocessing as mp
import os
import sys
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


def decode_core(X, y, groups=None, scoring="roc_auc", temp_gen=False, n_cv_folds=5, cv_random_state=42):
    """
    Perform cross-validated decoding analysis using logistic regression and MNE's decoding utilities.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to be used for decoding.

    y : array-like of shape (n_samples,)
        The target labels corresponding to the input data.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into train/test set.
        Currently, grouped cross-validation is not implemented and will raise a ValueError if provided.
        When `None`, StratifiedKFold is used for cross-validation.

    scoring : str, default='roc_auc'
        The scoring method to evaluate the performance of the estimator.

    temp_gen : bool, default=False
        If True, use a GeneralizingEstimator for temporal generalization.
        If False, use a SlidingEstimator for standard time-resolved decoding.

    n_cv_folds : int, default=5
        Number of folds to use in StratifiedKFold cross-validation.

    cv_random_state : int or None, default=None
        Random state for cross-validation splitting.

    Returns
    -------
    scores : array of shape (n_cv_folds,)
        The cross-validated scores for each fold.

    patterns : array of shape (n_features,)
        The patterns (Haufe, 2014) derived from the fitted estimator.

    """
    clf = make_pipeline(
        mne.decoding.Vectorizer(), LinearModel(LogisticRegression(solver="liblinear", random_state=42, verbose=False))
    )

    if temp_gen:
        se = GeneralizingEstimator(clf, scoring=scoring, n_jobs=-1, verbose=0)
    else:
        se = SlidingEstimator(clf, scoring=scoring, n_jobs=-1, verbose=0)

    if groups is None:
        cv = StratifiedKFold(n_splits=n_cv_folds, random_state=cv_random_state, shuffle=True)
    else:
        raise ValueError("Grouped cross validation not yet implemented")

    scores = cross_val_multiscore(se, X, y, cv=cv, groups=groups, n_jobs=n_cv_folds)

    se.fit(X, y)
    patterns = get_coef(se, "patterns_", inverse_transform=True)

    return scores, patterns


def decode_et(df, sub_id, factor, contrast, scoring, reps=50):  # noqa: ARG001
    """
    Perform eye-tracking data decoding analysis for a specific subject and condition.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the eye-tracking data.

    sub_id : int or str
        The subject ID to filter the data.

    factor : str
        The column name in the DataFrame representing the domain to be decoded (possible values: "emotion", "viewcond",
        "avatar_id").

    contrast : list
        The list of target labels in the factor domain to be classified.

    scoring : str
        The scoring method to evaluate the performance of the estimator (e.g., "roc_auc_ovr").

    reps : int, default=50
        The number of repetitions for decoding to obtain stable results.

    Returns
    -------
    scores : numpy.ndarray
        The average cross-validated scores for each fold across repetitions.

    times : numpy.ndarray
        The array of time points corresponding to the data.

    """
    data = df.query("sub_id == @sub_id").query(f"{factor} in @contrast")

    # pivot to long format
    data["times_idx"] = data.groupby("trial_num").cumcount()
    iidx = pd.MultiIndex.from_product(
        [data["trial_num"].unique(), data["times_idx"].unique()], names=["trial_num", "times_idx"]
    )
    data_long = pd.pivot_table(data, values=["phi", "theta"], index=["trial_num", "times_idx"]).reindex(iidx)

    # interpolate missing values (e.g., bc of blinks)
    data_long["phi"] = data_long["phi"].interpolate(method="linear")
    data_long["theta"] = data_long["theta"].interpolate(method="linear")

    # throw away trials which still contain nan values
    times = data.groupby("times_idx")["times"].mean().to_numpy()
    idx_na = data_long.groupby("trial_num")["phi"].apply(lambda x: x.isna().sum() > 0).to_numpy()
    data_long = data_long[~np.repeat(idx_na, len(times))]

    # reshape data to fit into the decoding function
    X = (
        data_long.to_numpy()
        .reshape(data_long.reset_index()["trial_num"].nunique(), data_long.reset_index()["times_idx"].nunique(), 2)
        .swapaxes(1, 2)
    )

    # get target labels
    y = data.query("times_idx == 0")[factor].to_numpy()
    y = y[~idx_na]

    # scores_tmp = []
    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 1000, reps)

    n_cv_folds = 5

    with(mp.Pool(reps)) as pool:
        tmp_scores = pool.starmap(run_decode_core, [(X, y, scoring, n_cv_folds, seed) for seed in seeds])

    scores = np.array([s[0] for s in tmp_scores]).mean(axis=0)

    # for i in range(reps):
    #     score, _ = decode_core(
    #         X, y, groups=None, temp_gen=False, n_cv_folds=5, scoring=scoring, cv_random_state=seeds[i]
    #     )
    #     scores_tmp.append(score)
    # scores = np.array(scores_tmp).mean(axis=0)
    return scores, times


def run_decode_core(X, y, scoring, n_cv_folds, seed):
    return decode_core(
        X, y, groups=None, temp_gen=False, n_cv_folds=n_cv_folds, scoring=scoring, cv_random_state=seed
    )


def main(sub_nr: int, contrast_arg: str = "all", viewcond: str = ""):
    """Run main."""
    scoring = "roc_auc_ovr"
    save_scores = True

    paths = PATHS()
    pattern = "withoutblinks-preproc.csv"
    sub_list_str_et = [f for f in os.listdir(paths.DATA_ET_PREPROC) if pattern in f]
    sub_list_str_et = [f.split("-")[0] for f in sub_list_str_et]
    sub_list_str_et = np.unique(sorted(sub_list_str_et))

    cond_dict = {
        "viewcond": {1: "mono", 2: "stereo"},
        "emotion": {1: "neutral", 2: "happy", 3: "angry", 4: "surprised"},
        "avatar_id": {1: "Woman_01", 2: "Woman_04", 3: "Woman_08"},
    }

    contrasts_dict = {
        "viewcond": ("mono", "stereo"),
        "emotion_pairwise": (
            ("angry", "neutral"),
            ("angry", "surprised"),
            ("angry", "happy"),
            ("happy", "neutral"),
            ("happy", "surprised"),
            ("surprised", "neutral"),
        ),
        "emotion": tuple(cond_dict["emotion"].values()),
        "avatar_id": tuple(cond_dict["avatar_id"].values()),
    }

    data_preproc = []
    for sub_id in sorted(sub_list_str_et):
        fname = Path(paths.DATA_ET_PREPROC, f"{sub_id}-ET-{pattern}")
        data = pd.read_csv(fname, sep=",")
        data["sub_id"] = sub_id
        data_preproc.append(data)

    df_all = pd.concat(data_preproc, ignore_index=True)

    # select only relevant columns
    df_s = df_all[["times", "theta", "phi", "viewcond", "avatar_id", "emotion", "sub_id", "trial_num"]]

    if viewcond != "":
        df_s = df_s.query("viewcond == @viewcond")
        contrasts_dict.pop("viewcond") # cannot decode viewcond if only one is present

    # set up contrasts
    if contrast_arg == "all":
        contrasts = []
        factors = []
        for c in contrasts_dict:
            if c == "emotion_pairwise":
                contrasts.extend(contrasts_dict[c])
                factors.extend(["emotion"] * len(contrasts_dict[c]))
            else:
                contrasts.append(contrasts_dict[c])
                factors.append(c)
        contrasts = tuple(contrasts)
    elif contrast_arg in contrasts_dict:
        contrasts = tuple((contrasts_dict[contrast_arg],))
        factors = [contrast_arg]
    else:
        raise ValueError(f"Contrast argument {contrast_arg} not found in contrasts_dict.")

    scores_all = defaultdict(list)
    times_all = defaultdict(list)

    # if we run from command line/slurm, we normally specify a subject via its index
    if sub_nr is not None:
        sub_list_str_et = [sub_list_str_et[sub_nr]]

    for contrast, factor in zip(contrasts, factors, strict=True):
        for sub_id in sorted(sub_list_str_et):
            print(f"Running subject: {sub_id} - factor: {factor} - contrast: {contrast}")
            score, times = decode_et(df_s, sub_id, factor=factor, contrast=contrast, scoring=scoring, reps=50)
            scores_all[sub_id].append(score)
            times_all[sub_id].append(times)

            # save
            contrast_str = "_vs_".join([c.lower() for c in contrast])
            if contrast_str.startswith("woman"):
                contrast_str = "id1_vs_id2_vs_id3"  # to match eeg decoding data
            conditions_vc = viewcond

            path_save = Path(
                paths.DATA_ET_DECOD,
                conditions_vc,
                contrast_str,
                scoring,
            )

            if save_scores:
                fpath = Path(path_save, "scores")
                fpath.mkdir(exist_ok=True, parents=True)
                fname = Path(fpath, f"{sub_id}-scores_per_sub.npy")
                np.save(fname, score)
                np.save(str(fname)[:-4] + "__times" + ".npy", times)
                del (fpath, fname)


if __name__ == "__main__":
    VIEWCOND = sys.argv[3] if len(sys.argv) > 3 else ""  # noqa: PLR2004

    CONTRAST = sys.argv[2] if len(sys.argv) > 2 else "all"  # noqa: PLR2004

    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None

    main(JOB_NR, CONTRAST, VIEWCOND)
