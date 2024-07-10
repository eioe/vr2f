"""Contains decoding functionalities."""
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

from vr2f import helpers
from vr2f.staticinfo import CONFIG, PATHS


def avg_across_time(data, winsize=25, times=None):
    """
    Downsamples the data by averaging within subsequent time windows of the specified width.

    Parameters
    ----------
    data : np.ndarray
        The input data array to be averaged. The array should have at least two dimensions,
        where the last dimension represents time.
    winsize : int, optional
        The number of time points to average over. Default is 25.
    times : np.ndarray, optional
        An array of time points corresponding to the last dimension of `data`.
        If provided, the averaged time points will be calculated and returned.

    Returns
    -------
    data_res : np.ndarray
        The data array with the time dimension averaged in steps of `winsize`.
    n_times : np.ndarray, optional
        The averaged time points, only returned if `times` is provided.

    Notes
    -----
    If the length of the last dimension of `data` is not a multiple of `winsize`,
    the data will be right-padded with NaNs before averaging.

    """
    orig_shape = data.shape
    # Fill in NaNs if necessary
    if orig_shape[-1] % winsize != 0:
      n_fill = winsize - (orig_shape[-1] % winsize)
      fill_shape = np.asarray(orig_shape)
      fill_shape[-1] = n_fill
      fill = np.ones(fill_shape) * np.nan
      data_f = np.concatenate([data, fill], axis=-1)
    else:
      data_f = data
      n_fill = 0
    data_res = np.nanmean(data_f.reshape(*orig_shape[:2], -1, winsize), axis=-1)

    if times is not None:
        f_times = np.r_[times, [np.nan] * n_fill]
        n_times = np.nanmean(f_times.reshape(-1, winsize), axis=-1)
        return data_res, n_times

    return data_res


def create_batches_avg(epos, batch_size, random_state=42):
    """
    Create mini-ERPs by averaging batches of epochs.

    Parameters
    ----------
    epos : array-like
        List or array of epochs.
    batch_size : int
        The size of each mini-batch.
    random_state : int, optional
        Seed for the random number generator to ensure reproducibility. Default is 42.

    Returns
    -------
    list
        List of averaged mini-batches.

    """
    n_trials = len(epos)
    n_batches = int(n_trials / batch_size)
    n_trials = batch_size * n_batches
    rnd_seq = np.arange(n_trials)
    rng = np.random.default_rng(seed=random_state)
    rng.shuffle(rnd_seq)
    rnd_seq = rnd_seq.reshape(n_batches, batch_size)
    batches = [epos[b].average() for b in rnd_seq]
    return batches


def get_data(sub_id, conditions, batch_size=1, smooth_winsize=1, picks="eeg", random_state=42):
    """
    Load and preprocess EEG data for a given subject and conditions.

    Parameters
    ----------
    sub_id : str
        Subject ID to load the data for.
    conditions : list of str
        List of conditions to filter the epochs.
    batch_size : int, optional
        Size of each mini-batch. Default is 1.
    smooth_winsize : int, optional
        Window size for smoothing the data across time. Default is 1.
    picks : str, optional
        Channels to pick from the data. Default is "eeg".
    random_state : int, optional
        Seed for the random number generator to ensure reproducibility. Default is 42.

    Returns
    -------
    X : np.ndarray
        Preprocessed data array.
    y : np.ndarray
        Labels corresponding to the conditions.
    times_n : np.ndarray
        Time points corresponding to the data.
    info : instance of mne.Info
        The info structure of the epochs containing metadata.

    """
    paths = PATHS()
    path_in = Path(paths.DATA_03_AR, "cleaneddata")
    fname = Path(path_in, f"{sub_id}-postAR-epo.fif")
    epos = mne.read_epochs(fname, verbose=False).pick(picks)
    times = epos.times
    info = epos.info

    # Setup data:
    if batch_size > 1:
        batches = defaultdict(list)
        for cond in conditions:
            batches[cond] = create_batches_avg(epos[cond], batch_size, random_state=random_state)
            batches[cond] = np.asarray([b.data for b in batches[cond]])

        X = np.concatenate([batches[cond].data for cond in conditions], axis=0)
        n_ = {cond: batches[cond].shape[0] for cond in conditions}

    else:
        X = mne.concatenate_epochs([epos[cond] for cond in conditions])
        X = X.get_data()
        n_ = {cond: len(epos[cond]) for cond in conditions}

    if smooth_winsize > 1:
        X, times_n = avg_across_time(X, smooth_winsize, times=times)
    else:
        times_n = times

    y = np.r_[
        np.zeros(n_[conditions[0]]),
        np.concatenate([(np.ones(n_[conditions[i]]) * i) for i in np.arange(1, len(conditions))]),
    ]

    return X, y, times_n, info


def decode(  # noqa: C901, PLR0913, PLR0912
    sub_list_str: list,
    conditions: list,
    scoring: str = "roc_auc",
    cv_folds: int = 5,
    random_state: int = 42,
    n_rep_sub: int = 50,
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
    cv_folds : int, optional
        Number of cross-validation folds, by default 3
    random_state : int, optional
        Random state, by default 42
    n_rep_sub : int, optional
        Number of repetitions, by default 50
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
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    subs_processed = []
    sub_scores = []
    sub_scores_per_rep = []
    sub_coef = []
    times_n = []

    for sub_id in sub_list_str:
        print(f"### RUNING SUBJECT {sub_id}")
        subs_processed.append(sub_id)
        all_scores = []
        all_coef = []
        random_states = np.random.default_rng(seed=random_state).integers(0, 1000, n_rep_sub)
        for i in np.arange(n_rep_sub):
            X, y, times_n, info = get_data(
                sub_id,
                conditions=conditions,
                batch_size=batch_size,
                smooth_winsize=smooth_winsize,
                picks=picks,
                random_state=random_states[i],
            )

            clf = make_pipeline(
                mne.decoding.Scaler(info),
                mne.decoding.Vectorizer(),
                LinearModel(LogisticRegression(solver="liblinear", max_iter=5000, random_state=random_states[i],
                                               verbose=False)),
            )

            if temp_gen:
                gen_str = "gen_temp"
                pipeline = GeneralizingEstimator(clf, scoring=scoring, n_jobs=config.N_JOBS, verbose=0)
            else:
                gen_str = ""
                pipeline = SlidingEstimator(clf, scoring=scoring, n_jobs=config.N_JOBS, verbose=0)

            if shuffle_labels:
                rng = np.random.default_rng(seed=random_states[i])
                rng.shuffle(y)

            scores = cross_val_multiscore(pipeline, X=X, y=y, cv=cv)
            scores = np.mean(scores, axis=0)  # avg across folds
            all_scores.append(scores)

            # Get patterns:
            pipeline.fit(X, y)
            coef = get_coef(pipeline, "patterns_", inverse_transform=True)
            all_coef.append(coef)

        sub_scores = np.asarray(all_scores).mean(axis=0)  # avg across reps
        sub_coef = np.asarray(all_coef).mean(axis=0)

        # save it:
        shuf_labs = "labels_shuffled" if shuffle_labels else ""

        picks_str_folder = "picks_" + "-".join([str(p) for p in picks]) if picks != "eeg" else ""

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
            fname = Path(fpath, f"{sub_id}-scores_per_sub.npy")
            np.save(fname, sub_scores)
            np.save(str(fname)[:-4] + "__times" + ".npy", times_n)
            del (fpath, fname)

        # save patterns:
        if save_patterns:
            sub_patterns = sub_coef
            fpath = Path(path_save, "patterns")
            fpath.mkdir(exist_ok=True, parents=True)
            fname = Path(fpath, f"{sub_id}-patterns_per_sub.npy")
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
            fname = Path(fpath, f"{sub_id}-info.json")
            with Path.open(fname, "w+") as outfile:
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
                f"{sub_id}-" f"reps{n_rep_sub}_" f"swin{smooth_winsize}_batchs{batch_size}.npy",
            )
            np.save(fname, sub_scores_per_rep)
            np.save(str(fname)[:-4] + "__times" + ".npy", times_n)
            del (fpath, fname)

    return sub_scores, sub_coef, times_n


def main(sub_nr: int, contrast_arg: str = "all", viewcond: str = "ignore"):  # noqa: C901, PLR0912
    """Run main."""
    paths = PATHS()
    path_in = Path(paths.DATA_03_AR, "cleaneddata")

    # load data
    sub_list_str = [s.split("-postAR-epo")[0] for s in os.listdir(path_in)]
    sub_list_str = sorted(sub_list_str, reverse=True)

    if sub_nr is not None:
        sub_list_str = [sub_list_str[sub_nr]]

    cond_dict = {
        "viewcond": {1: "mono", 2: "stereo"},
        "emotion": {1: "neutral", 2: "happy", 3: "angry", 4: "surprised"},
        "avatar_id": {1: "id1", 2: "id2", 3: "id3"},
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

    # parse viewcond argument
    if viewcond == "ignore":
        vc_strs = [""]
    elif viewcond in ["mono", "stereo"]:
        vc_strs = [viewcond + "/"]
    elif viewcond == "all":
        vc_strs = ["", "mono/", "stereo/"]
    else:
        raise ValueError(f"Unknown viewcond argument: {viewcond}.")

    # set up contrasts
    contrasts = []
    for vc_str in vc_strs:
        if contrast_arg == "all":
            for c in contrasts_dict:
                if c == "emotion_pairwise":
                    contrasts.extend(tuple([vc_str + cc for cc in contrasts_dict[c]]))
                elif c == "viewcond":
                    contrasts.extend(contrasts_dict[c])  # skip viewcond subsetting
                else:
                    contrasts.append(tuple([vc_str + cc for cc in contrasts_dict[c]]))

        elif contrast_arg in contrasts_dict:
            if contrast_arg == "viewcond" and viewcond != "ignore":
                vc_str = ""
                raise Warning("Viewcond argument other than 'ignore' is useless for viewcond decoding. "
                              "Ignoring viewcond argument.")
            if contrast_arg == "emotion_pairwise":
                for cc in contrasts_dict[contrast_arg]:
                    contrasts.append(tuple([vc_str + c for c in cc]))
            else:
                contrasts.extend([tuple([vc_str + c for c in contrasts_dict[contrast_arg]])])

        else:
            raise ValueError(f"Contrast argument {contrast_arg} not found in contrasts_dict.")


    for contrast in contrasts:
        for sub_id in sub_list_str:
            scores, coefs, times = decode(
                [sub_id],
                conditions=contrast,
                scoring="roc_auc_ovr",
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

    VIEWCOND = sys.argv[3] if len(sys.argv) > 3 else "ignore"
    CONTRAST = sys.argv[2] if len(sys.argv) > 2 else "all"

    if len(sys.argv) > 1:
        helpers.print_msg("Running Job Nr. " + sys.argv[1] + f" with contrast {CONTRAST} and viewcond {VIEWCOND}")
        JOB_NR = int(sys.argv[1])
    else:
        JOB_NR = None

    main(JOB_NR, CONTRAST, VIEWCOND)
