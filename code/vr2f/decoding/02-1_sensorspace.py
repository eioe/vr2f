import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np
from mne.decoding import (
    GeneralizingEstimator,
    LinearModel,
    SlidingEstimator,
    cross_val_multiscore,
    get_coef,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from vr2fem_analyses import helpers

from vr2f.staticinfo import CONFIG, PATHS


def avg_time(data, step=25, times=None):
    orig_shape = data.shape
    n_fill = step - (orig_shape[-1] % step)
    fill_shape = np.asarray(orig_shape)
    fill_shape[-1] = n_fill
    fill = np.ones(fill_shape) * np.nan
    data_f = np.concatenate([data, fill], axis=-1)
    data_res = np.nanmean(data_f.reshape(*orig_shape[:2], -1, step), axis=-1)

    if times is not None:
        f_times = np.r_[times, [np.nan] * n_fill]
        n_times = np.nanmean(f_times.reshape(-1, step), axis=-1)
        return data_res, n_times
    else:
        return data_res


def batch_trials(epos, batch_size):
    n_trials = len(epos)
    n_batches = int(n_trials / batch_size)
    n_trials = batch_size * n_batches
    rnd_seq = np.arange(n_trials)
    np.random.shuffle(rnd_seq)
    rnd_seq = rnd_seq.reshape(n_batches, batch_size)
    batches = [epos[b].average() for b in rnd_seq]
    return batches


def get_data(subID, conditions, batch_size=1, smooth_winsize=1, picks="eeg"):
    paths = PATHS()
    path_in = Path(paths.DATA_03_AR, "cleaneddata")
    fname = Path(path_in, f"{subID}-postAR-epo.fif")
    epos = mne.read_epochs(fname, verbose=False).pick(picks)
    times = epos.times
    info = epos.info

    # Setup data:
    if batch_size > 1:
        batches = defaultdict(list)
        for cond in conditions:
            batches[cond] = batch_trials(epos[cond], batch_size)
            batches[cond] = np.asarray([b.data for b in batches[cond]])

        X = np.concatenate([batches[cond].data for cond in conditions], axis=0)
        n_ = {cond: batches[cond].shape[0] for cond in conditions}

    else:
        X = mne.concatenate_epochs([epos[cond] for cond in conditions])
        X = X.get_data()
        n_ = {cond: len(epos[cond]) for cond in conditions}

    if smooth_winsize > 1:
        X, times_n = avg_time(X, smooth_winsize, times=times)
    else:
        times_n = times

    y = np.r_[
        np.zeros(n_[conditions[0]]),
        np.concatenate([(np.ones(n_[conditions[i]]) * i) for i in np.arange(1, len(conditions))]),
    ]

    return X, y, times_n, info


def decode(
    sub_list_str: list,
    conditions: list,
    scoring: str = "roc_auc",
    n_rep_sub: int = 100,
    picks: list = "eeg",
    shuffle_labels: bool = False,
    batch_size: int = 10,
    smooth_winsize: int = 5,
    temp_gen: bool = False,
    save_single_rep_scores: bool = False,
    save_scores: bool = True,
    save_patterns: bool = False,
):
    """
    Run decoding.

    Parameters
    ----------
    sub_list_str : list
        Lsit with subject IDs.
    conditions : list
        List wit conditions to classify.
    scoring : str, optional
        scoring measure, by default "roc_auc"
    n_rep_sub : int, optional
        Number of repetitions, by default 100
    picks : list, optional
        Picks, by default "eeg"
    shuffle_labels : bool, optional
        Randomize the labels to generate null distribution, by default False
    batch_size : int, optional
        Number of samples to average across before classification, by default 10
    smooth_winsize : int, optional
        Number of timepoints to average across before classification, by default 5
    temp_gen : bool, optional
        Run temporal generalisation, by default False
    save_single_rep_scores : bool, optional
        Save results of all repetitions or only their average (default), by default False
    save_scores : bool, optional
        Save the scores to disc, by default True
    save_patterns : bool, optional
        Save the decoding patterns to disk, by default False

    Returns
    -------
    array :
        Scores per subject
    list :
        Coefficients per subject.
    array :
        Array of times of the batches/samples

    """
    paths = PATHS()
    config = CONFIG()
    conditions_target = [c.split("/")[-1] for c in conditions]
    if len(conditions[0].split("/")) > 1:
        if len(conditions[0].split("/")) > 2:
            raise NotImplementedError("Cannot handle more than two cond levels yet.")
        conditions_vc = [c.split("/")[0] for c in conditions]
        if len(set(conditions_vc)) > 1:
            raise ValueError("U r mixing viewing conditions. Uncool.")
        conditions_vc = conditions_vc[0]
    else:
        conditions_vc = ""

    contrast_str = "_vs_".join(conditions_target)
    scoring = scoring  # 'roc_auc' # 'accuracy'
    cv_folds = 3
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True)

    subs_processed = list()
    sub_scores = list()
    sub_scores_per_rep = list()
    sub_coef = list()
    times_n = list()

    for subID in sub_list_str:
        print(f"### RUNING SUBJECT {subID}")
        subs_processed.append(subID)
        all_scores = list()
        all_coef = list()
        for i in np.arange(n_rep_sub):
            X, y, times_n, info = get_data(
                subID,
                conditions=conditions,
                batch_size=batch_size,
                smooth_winsize=smooth_winsize,
                picks=picks,
            )

            clf = make_pipeline(
                mne.decoding.Scaler(info),
                mne.decoding.Vectorizer(),
                LinearModel(LogisticRegression(solver="liblinear", max_iter=5000, random_state=42, verbose=False)),
            )

            # TODO: refactor: rename "se"
            if temp_gen:
                gen_str = "gen_temp"
                se = GeneralizingEstimator(clf, scoring=scoring, n_jobs=config.N_JOBS, verbose=0)
            else:
                gen_str = ""
                se = SlidingEstimator(clf, scoring=scoring, n_jobs=config.N_JOBS, verbose=0)

            if shuffle_labels:
                np.random.shuffle(y)
            # for i in np.unique(y):
            #     print(f"Size of class {i}: {np.sum(y == i)}\n")
            scores = cross_val_multiscore(se, X=X, y=y, cv=cv)
            scores = np.mean(scores, axis=0)
            all_scores.append(scores)
            se.fit(X, y)
            coef = get_coef(se, "patterns_", inverse_transform=True)
            all_coef.append(coef)

        sub_scores = np.asarray(all_scores).mean(axis=0)
        sub_coef = np.asarray(all_coef).mean(axis=0)

        # save shizzle:
        shuf_labs = "labels_shuffled" if shuffle_labels else ""

        if picks != "eeg":
            picks_str_folder = "picks_" + "-".join([str(p) for p in picks])
        else:
            picks_str_folder = ""

        path_save = Path(
            paths.DATA_04_DECOD_SENSORSPACE,
            conditions_vc,
            contrast_str,
            gen_str,
            scoring,
            picks_str_folder,
            shuf_labs,
        )

        # save accuracies:
        if save_scores:
            fpath = Path(path_save, "scores")
            fpath.mkdir(exist_ok=True, parents=True)
            fname = Path(fpath, f"{subID}-scores_per_sub.npy")
            np.save(fname, sub_scores)
            np.save(str(fname)[:-4] + "__times" + ".npy", times_n)
            del (fpath, fname)

        # save patterns:
        if save_patterns:
            sub_patterns = sub_coef
            fpath = Path(path_save, "patterns")
            fpath.mkdir(exist_ok=True, parents=True)
            fname = Path(fpath, f"{subID}-patterns_per_sub.npy")
            np.save(fname, sub_patterns)
            np.save(str(fname)[:-4] + "__times" + ".npy", times_n)
            del (fpath, fname)

        # save info:
        if save_scores or save_patterns or save_single_rep_scores:
            info_dict = {
                "included subs": subs_processed,
                "n_rep_sub": n_rep_sub,
                "batch_size": batch_size,
                "smooth_winsize": smooth_winsize,
                "cv_folds": cv_folds,
                "scoring": scoring,
            }
            fpath = path_save
            fname = Path(fpath, f"{subID}-info.json")
            with open(fname, "w+") as outfile:
                json.dump(info_dict, outfile)

        # save data from single reps:
        if save_single_rep_scores:
            if len(sub_scores_per_rep) == 0:
                sub_scores_per_rep = np.asarray(all_scores)
            else:
                sub_scores_per_rep = np.concatenate([sub_scores_per_rep, np.asarray(all_scores)], axis=0)

            fpath = Path(path_save, "single_rep_data")
            fpath.mkdir(exist_ok=True, parents=True)
            fname = Path(
                fpath,
                f"{subID}-" f"reps{n_rep_sub}_" f"swin{smooth_winsize}_batchs{batch_size}.npy",
            )
            np.save(fname, sub_scores_per_rep)
            np.save(str(fname)[:-4] + "__times" + ".npy", times_n)
            del (fpath, fname)

    return sub_scores, sub_coef, times_n


def main(sub_nr: int):
    paths = PATHS()
    path_in = Path(paths.DATA_03_AR, "cleaneddata")

    # load data
    sub_list_str = [s.split("-postAR-epo")[0] for s in os.listdir(path_in)]

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]

    emotions = [
        "angry",
        "surprised",
    ]  # ["happy", "neutral"]  # ["angry", "neutral"]  #  ["surprised", "neutral", "angry", "happy"]
    viewconds = ["all"]  # ["mono", "stereo"]  #  ["all", "mono", "stereo"]
    ids = ["id1", "id2", "id3"]

    for viewcond in viewconds:
        if viewcond == "all":
            vc = ""
        else:
            vc = viewcond + "/"

        for subID in sub_list_str:
            scores, coefs, times = decode(
                [subID],
                conditions=ids,  # [vc + emo for emo in emotions],  #  ["mono", "stereo"],  #
                scoring="roc_auc_ovr",  #  "roc_auc_ovr",
                n_rep_sub=50,
                picks="eeg",
                shuffle_labels=False,
                batch_size=3,
                smooth_winsize=5,
                temp_gen=False,
                save_single_rep_scores=False,
                save_scores=True,
                save_patterns=True,
            )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None
    main(JOB_NR)
