"""Module containing functions and classes for cross-decoding EEG data."""

# %% load libs:
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from os import path as op
from pathlib import Path

import mne
import numpy as np
from mne.decoding import GeneralizingEstimator, LinearModel, SlidingEstimator, cross_val_multiscore, get_coef
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import make_pipeline

from vr2f import helpers

# from library import config, helpers
from vr2f.staticinfo import PATHS

# %%



def shuffle_samples(data, conds, n_, random_state=None):
    """
    Shuffle samples in a NumPy array based on condition labels.

    Parameters
    ----------
    data : np.ndarray
        The NumPy array containing the samples to be shuffled along the first axis.
        The data is expected to be sorted by condition (along the first axis).
        Otherwise this will create wrong results!
    conds : list of str
        The list of condition labels corresponding to each sample.
    n_ : dict
        A dictionary mapping condition labels to the number of samples for each condition.
    random_state : int, optional
        Seed to use for random number generation. If not specified, the default NumPy generator will be used.

    Returns
    -------
    np.ndarray
        A new NumPy array containing the shuffled samples.

    """
    # check inputs:
    if not isinstance(n_, dict):
        raise TypeError(f"n_ must be a dict, not {type(n_)}")
    # check if all conditions are present in n_:
    if not all(cond in n_ for cond in conds):
        raise ValueError("All conditions must be present in n_.")

    shuffled_idx = np.array([], dtype=int)
    for i, cond in enumerate(conds):
        start = int(np.sum([n_[c] for c in conds[:i]]))
        idx = np.arange(start, start+n_[cond])
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
        shuffled_idx = np.concatenate([shuffled_idx, np.array(idx)], dtype=int)
    data_shuffled = data[shuffled_idx]
    return data_shuffled


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


def avg_across_time(data, winsize, times=None):
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


def get_data(sub_id, conditions, batch_size, smooth_winsize, picks="eeg", random_state=42):
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

# %%
def concat_train_test(
    sub_id,
    conditions_target,
    condition_train,
    condition_test,
    batch_size,
    smooth_winsize,
    picks_str="eeg",
    random_state=42,
):
    if condition_train != condition_test:
        conditions_train = [f"{condition_train}/{c}" for c in conditions_target]
        conditions_test = [f"{condition_test}/{c}" for c in conditions_target]
        X_train_all, y_train_all, times_n, info = get_data(
            sub_id,
            conditions_train,
            batch_size=batch_size,
            smooth_winsize=smooth_winsize,
            random_state=random_state,
            picks=picks_str
        )
        X_test_all, y_test_all, _, _ = get_data(
            sub_id,
            conditions_test,
            batch_size=batch_size,
            smooth_winsize=smooth_winsize,
            random_state=random_state,
            picks=picks_str
        )

        X = np.concatenate([X_train_all, X_test_all], axis=0)
        y = np.concatenate([y_train_all, y_test_all], axis=0)
        groups = np.concatenate([len(X_train_all) * [0], len(X_test_all) * [1]])
    else:
        print("train and test conditions are the same")
        conditions_traintest = [f"{condition_train}/{c}" for c in conditions_target]
        X, y, times_n, info = get_data(
            sub_id,
            conditions_traintest,
            batch_size=batch_size,
            smooth_winsize=smooth_winsize,
            random_state=random_state,
            picks=picks_str
        )
        groups = None
    return X, y, groups, times_n, info


# %%


class CrossDecodSplitter(BaseCrossValidator):
    """
    Cross-validator for cross-decoding EEG data.

    This class splits the data into training and testing sets based on the groups.
    It ensures that the groups vector contains exactly 2 unique values and is sorted.
    """

    def __init__(self, n_splits):  # noqa: D107
        self.n_splits = n_splits

    def split(self, X, y, groups):  # noqa: ARG002
        """
        Split the data into training and testing sets based on the groups.

        Parameters
        ----------
        X : array-like
            The data to split.
        y : array-like
            The target variable.
        groups : array-like
            Group labels for the samples used while splitting the dataset into train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        """
        # throw error if groups does not contain exactly 2 unique values
        if len(np.unique(groups)) != 2:
            raise ValueError("groups must contain exactly 2 unique values")
        # thow error if groups is not sorted. sorting should be done in a way
        # that groups[0] always stays the first value after sorting
        groups_sorted = np.sort(groups)
        if groups_sorted[0] != groups[0]:
            groups_sorted = np.flip(groups_sorted)
        if not np.all(groups_sorted == groups):
            raise ValueError("groups vector must be sorted")


        idx_0 = np.where(groups == groups[0])[0]
        idx_1 = np.where(groups == groups[-1])[0]

        idx_cv = StratifiedKFold(n_splits=self.n_splits)

        for fold_0, fold_1 in zip(idx_cv.split(idx_0, y[idx_0]),
                                  idx_cv.split(idx_1, y[idx_1]),
                                  strict=True):
            yield idx_0[fold_0[0]], idx_1[fold_1[1]]

    def get_n_splits(self, X=None, y=None, groups=None):  # noqa: ARG002, D102
        return self.n_splits

# %%
def decode_core(X, y, groups, info,
                n_cv_folds,
                scoring="roc_auc_ovr",
                temp_gen=False,
                cv_random_state=None):

    clf = make_pipeline(mne.decoding.Scaler(info),
                    mne.decoding.Vectorizer(),
                    LinearModel(
                        LogisticRegression(solver="liblinear",
                                           random_state=cv_random_state,
                                    verbose=False)))

    if temp_gen:
        se = GeneralizingEstimator(clf,
                                   scoring=scoring,
                                   n_jobs=15,
                                   verbose=0)
    else:
        se = SlidingEstimator(clf,
                              scoring=scoring,
                              n_jobs=15,
                              verbose=0)
    if groups is None:
        cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=cv_random_state)
    else:
        cv = CrossDecodSplitter(n_splits=n_cv_folds)
    scores = cross_val_multiscore(se, X, y, cv=cv, groups=groups, n_jobs=n_cv_folds)

    se.fit(X, y)
    patterns = get_coef(se, "patterns_", inverse_transform=True)

    return scores, patterns


def gen_save_path(contrast_str,
                  scoring="roc_auc",
                  picks_str=None,
                  labels_shuffled=False,
                  cross_decod=False,
                  crossing_str="",
                 ):
    """
    Generate the save path for decoding results.

    Parameters
    ----------
    contrast_str : str
        The contrast string used to identify the decoding analysis.
    scoring : str, optional
        The scoring method used for decoding (default is "roc_auc").
    picks_str : str or None, optional
        String specifying the channel selection or None for default (default is None).
    labels_shuffled : bool, optional
        Whether the labels are shuffled (default is False).
    cross_decod : bool, optional
        Whether cross-decoding is performed (default is False).
    crossing_str : str, optional
        Additional string for cross-decoding (default is "").

    Returns
    -------
    Path
        The path where the decoding results will be saved.

    """
    shuf_labs = "labels_shuffled" if labels_shuffled else ""
    cross_decod_str = "cross_decod_vc" if cross_decod else ""
    picks_str = "picks" if picks_str is not None else ""

    paths = PATHS()
    path_save = Path(paths.DATA_04_DECOD_SENSORSPACE,
                     cross_decod_str,
                     crossing_str,
                     contrast_str,
                     scoring,
                     shuf_labs)
    return path_save


def save_scores(sub_id, scores, times_n, path_save):
    """
    Save the decoding scores and corresponding times for a subject.

    Parameters
    ----------
    sub_id : str
        Subject ID.
    scores : np.ndarray
        Array of decoding scores.
    times_n : np.ndarray
        Array of time points corresponding to the scores.
    path_save : str
        Path to the directory where the scores will be saved.

    """
    fpath = Path(path_save, "scores")
    helpers.chkmkdir(fpath)
    fname = Path(fpath, f"{sub_id}-scores_per_sub.npy")
    np.save(fname, scores)
    fname_times = fname.with_name(fname.stem + "__times.npy")
    np.save(fname_times, times_n)


def save_patterns(sub_id, patterns, times_n, path_save):
    """
    Save the decoding patterns and corresponding times for a subject.

    Parameters
    ----------
    subID : str
        Subject ID.
    patterns : np.ndarray
        Array of decoding patterns.
    times_n : np.ndarray
        Array of time points corresponding to the patterns.
    path_save : str
        Path to the directory where the patterns will be saved.

    """
    fpath = Path(path_save, "patterns")
    helpers.chkmkdir(fpath)
    fname = Path(fpath, f"{sub_id}-patterns_per_sub.npy")
    np.save(fname, patterns)
    fname_times = fname.with_name(fname.stem + "__times.npy")
    np.save(fname_times, times_n)


def save_info(subID, info_dict, path_save):
    fpath = path_save
    fname = Path(fpath, f"{subID}-info.json")
    with Path.open(fname, "w+") as outfile:
        json.dump(info_dict, outfile)


def save_single_rep_scores(subID, sub_scores_per_rep, times_n, path_save):
    fpath = Path(path_save, "single_rep_data")
    helpers.chkmkdir(fpath)
    fname = Path(fpath,
                    f"{subID}-"
                    f"reps{n_rep_sub}_"
                    f"swin{smooth_winsize}_batchs{batch_size}.npy")
    np.save(fname, sub_scores_per_rep)
    fname_times = fname.with_name(fname.stem + "__times.npy")
    np.save(fname_times, times_n)




# %%
def run_decoding(sub_id, conditions_target, c_train, c_test, batch_size, smooth_winsize, n_rep_sub, n_cv_folds,
                 scoring="roc_auc_ovr", shuffle_labels=False):
    """
    Run the decoding process for a given subject and conditions.

    Parameters
    ----------
    sub_id : str
        Subject ID to load the data for.
    conditions_target : list of str
        List of target conditions for decoding.
    c_train : str
        Training condition.
    c_test : str
        Testing condition.
    batch_size : int
        Size of each mini-batch.
    smooth_winsize : int
        Window size for smoothing the data across time.
    n_rep_sub : int
        Number of repetitions for the decoding process.
    n_cv_folds : int
        Number of cross-validation folds.
    scoring : str, optional
        Scoring method for decoding (default is "roc_auc_ovr").
    shuffle_labels : bool, optional
        Whether to shuffle the labels (default is False).

    Returns
    -------
    tuple
        A tuple containing the scores, patterns, time points, subject ID, training condition, and testing condition.

    """
    scores_per_rep = []
    patterns_per_rep = []
    for rep in range(n_rep_sub):
        X, y, groups, times_n, info = concat_train_test(
                    sub_id=sub_id,
                    conditions_target=conditions_target,
                    condition_train=c_train,
                    condition_test=c_test,
                    batch_size=batch_size,
                    smooth_winsize=smooth_winsize,
                    random_state=42 + rep,
                )

        if shuffle_labels:
            groups_ = groups if groups is not None else np.zeros(shape=y.shape)
            groups_uniq, n_per_group = np.unique(groups_, return_counts=True)
            n_per_group = dict(zip(groups_uniq, n_per_group, strict=True))
            y = shuffle_samples(y, groups_uniq, n_per_group)
            print(f"y shape: {y.shape}")
            print(y)

        scores, patterns = decode_core(
            X, y, groups, info, n_cv_folds=n_cv_folds, scoring=scoring
        )
        scores_per_rep.append(np.mean(scores, axis=0))  # average over cv folds
        patterns_per_rep.append(patterns)
    scores_sub = np.mean(np.array(scores_per_rep), axis=0)  # average over reps
    patterns_sub = np.mean(np.array(patterns_per_rep), axis=0)

    return (scores_sub, patterns_sub, times_n, sub_id, c_train, c_test)



# %%

if __name__ == "__main__":

    # What do we want to decode (i.e., contrast)?
    conditions_target=["neutral", "happy", "angry", "surprised"]

    # Set up parameters:
    batch_size = 3
    smooth_winsize = 5
    n_rep_sub = 50
    n_cv_folds = 5
    scoring = "roc_auc_ovr"

    shuffle_labels = False

    # Get subject list:
    paths = PATHS()
    path_in = Path(paths.DATA_03_AR, "cleaneddata")

    # load data
    sub_list_str = [s.split("-postAR-epo")[0] for s in os.listdir(path_in)]
    sub_list_str = sorted(sub_list_str)

    # when running on the cluster we want parallelization along the subject dimension
    if not helpers.is_interactive(): 
        helpers.print_msg("Running Job Nr. " + sys.argv[1])
        job_nr = int(float(sys.argv[1]))
        sub_list_str = [sub_list_str[job_nr]]

    scores_all = defaultdict(list)
    patterns_all = defaultdict(list)

    pool = mp.Pool()
    results = []
    for c_train in ["mono", "stereo"]:  # training condition
        for c_test in ["mono", "stereo"]:  # testing condition
            for sub_id in sub_list_str:
                print(f"Running {sub_id} ... train: {c_train} test: {c_test}")
                result = pool.apply_async(run_decoding,
                                          args=(sub_id,
                                                conditions_target,
                                                c_train,
                                                c_test,
                                                batch_size,
                                                smooth_winsize,
                                                n_rep_sub,
                                                n_cv_folds,
                                                scoring,
                                                shuffle_labels))
                results.append(result)
    pool.close()
    pool.join()

    for result in results:
        scores_sub, patterns_sub, times_n, sub_id, c_train, c_test = result.get()
        scores_all[f"train_{c_train}-test_{c_test}"].append(scores_sub)
        patterns_all[f"train_{c_train}-test_{c_test}"].append(patterns_sub)
        path_save = gen_save_path(
                contrast_str="neutral_vs_happy_vs_angry_vs_surprised",
                scoring="roc_auc_ovr",
                labels_shuffled=shuffle_labels,
                cross_decod=True,
                crossing_str=f"train_{c_train}-test_{c_test}",
            )
        save_scores(sub_id, scores_sub, times_n, path_save)
        save_patterns(sub_id, patterns_sub, times_n, path_save)

        info_dict = {
            "n_rep_sub": n_rep_sub,
            "batch_size": batch_size,
            "smooth_winsize": smooth_winsize,
            "cv_folds": n_cv_folds,
            "scoring": scoring,
        }
        save_info(sub_id, info_dict, path_save)

# %%


