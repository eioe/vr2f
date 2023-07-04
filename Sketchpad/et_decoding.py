import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# %% load libs:
from collections import defaultdict
from os import path as op
import sys
import json
import numpy as np
import seaborn as sns

from sklearn.model_selection import check_cv, BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp

from scipy import stats

import mne
# from mne.epochs import concatenate_epochs
from mne.decoding import (SlidingEstimator, GeneralizingEstimator,
                          cross_val_multiscore, LinearModel, get_coef)

from vr2fem_analyses.staticinfo import PATHS
from vr2fem_analyses import helpers


def cart2sph_custom(x, y, z):
    """Convert cartesian coordinates (xyz) to spherical coordinates (theta-phi-r)"""

    hypotxz = np.hypot(x, z)
    r = np.hypot(y, hypotxz)
    phi = np.arctan2(y, hypotxz) 
    theta = np.arctan2(x, z)

    # translate both to degree
    theta = np.rad2deg(theta)
    phi = np.rad2deg(phi)

    # concatenate the 3 arrays to a 2d array
    sph = np.stack((theta, phi, r), axis=1)
    return sph


def interpolate_blinks(et_data : pd.DataFrame):
    """Interpolate blinks in the eye tracking data.

    According to the method suggested by 
    Kret, M.E., Sjak-Shie, E.E. Preprocessing pupil size data: Guidelines and code. 
    Behav Res 51, 1336–1342 (2019). https://doi.org/10.3758/s13428-018-1075-y

    Parameters

    ----------

    et_data : pd.DataFrame
        Eye tracking data as returned by `read_et_data()`

    Returns

    -------
    et_data : pd.DataFrame
        Eye tracking data with blinks interpolated.
    """
    df = et_data.copy()

    p_dil = df['diameter_left'].to_numpy()
    t_et = df['timestamp_et'].to_numpy()
    d = np.diff(p_dil)
    d_pre = d[:-1]
    d_post = d[1:]
    t = np.diff(t_et)
    t_pre = t[:-1]
    t_post = t[1:]

    o = np.max(np.array((np.abs(d_pre/t_pre), np.abs(d_post/t_post))), axis=0)

    mad = np.median(np.abs(o - np.median(o)))
    thresh = np.median(o) + 50 * mad
    label = o > thresh
    # repeat the first and last label to get the same length as the original array
    label = np.insert(label, 0, label[0])
    label = np.append(label, label[-1])
    df['blink'] = label
    # set blink to 1 if dilation is < 0
    df.loc[df['diameter_left'] < 0, 'blink'] = 1
    
    blink_times = df.loc[df['blink'] == 1, 'timestamp_et']
    # find all timestamps_et which are closer than 50ms to a blink
    blink_times = blink_times.to_numpy()
    et_times = df['timestamp_et'].to_numpy()
    if len(blink_times) == 0:
        blink_times = np.array([-99])
    # repeat blink times len(et_times) along new axis 
    blink_times_r = np.repeat(blink_times[:, np.newaxis], len(et_times), axis=1)

    mindist = np.min(np.abs(et_times - blink_times_r), axis=0)
    df['blink'] = mindist < 100

    # set theta and phi to NaN for blinks
    df.loc[df['blink'] == 1, 'theta'] = np.nan
    df.loc[df['blink'] == 1, 'phi'] = np.nan
    df.loc[df['blink'] == 1, 'r'] = np.nan
    df.loc[df['blink'] == 1, 'diameter_left'] = np.nan
    df.loc[df['blink'] == 1, 'diameter_right'] = np.nan

    # interpolate the data for the blinks
    df['theta'] = df['theta'].interpolate(method='linear')
    df['phi'] = df['phi'].interpolate(method='linear')
    df['r'] = df['r'].interpolate(method='linear')
    df['diameter_right'] = df['diameter_right'].interpolate(method='linear')
    df['diameter_left'] = df['diameter_left'].interpolate(method='linear')

    return df


def decode_core(X, y, groups,
                scoring='roc_auc',
                temp_gen=False,
                n_cv_folds=5,
                cv_random_state=None):
    
    clf = make_pipeline(
                        mne.decoding.Vectorizer(),
                        LinearModel(
                            LogisticRegression(solver='liblinear',
                                                random_state=42,
                                                verbose=False)))

    if temp_gen:
        gen_str = 'gen_temp'
        se = GeneralizingEstimator(clf,
                                   scoring=scoring,
                                   n_jobs=15,
                                   verbose=0)
    else:
        gen_str = ''
        se = SlidingEstimator(clf,
                              scoring=scoring,
                              n_jobs=15,
                              verbose=0)
    if groups is None:
        cv = StratifiedKFold(n_splits=n_cv_folds)
    else:
        # raise not yet implemented error:
        raise ValueError("Grouped cross validation not yet implemented")

    scores = cross_val_multiscore(se, X, y, cv=cv, groups=groups, n_jobs=n_cv_folds)

    se.fit(X, y)
    patterns = get_coef(se, 'patterns_', inverse_transform=True)

    return scores, patterns


def decode_et(df, sub_id, factor, contrast, scoring, reps=50):

    df_ = (df.query("sub_id == @sub_id")
             .query(f"{factor} in @contrast"))
    
    # add an index which counts from 0 up, for each trial_num
    df_["times_idx"] = df_.groupby("trial_num").cumcount()
    iidx = pd.MultiIndex.from_product(
        [df_["trial_num"].unique(), df_["times_idx"].unique()],
        names=["trial_num", "times_idx"],
    )
    df_p = pd.pivot_table(
        df_, values=["phi", "theta"], index=["trial_num", "times_idx"]
    ).reindex(iidx)
    df_p["phi"] = df_p["phi"].interpolate(method="linear")
    df_p["theta"] = df_p["theta"].interpolate(method="linear")
    
    # throw away trials which still contain nan values
    times = df_.groupby('times_idx').mean()['times'].to_numpy()
    idx_na = df_p.groupby('trial_num')['phi'].apply(lambda x: x.isna().sum() > 0).to_numpy()
    df_p = df_p[~np.repeat(idx_na, len(times))]
    
    X = (
        df_p.to_numpy()
        .reshape(df_p.reset_index()["trial_num"].nunique(), 
                 df_p.reset_index()["times_idx"].nunique(), 2)
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
    sub_list_str_et = [f.split("-")[0] for f in os.listdir(paths.DATA_ET_PREPROC)]

    cond_dict = {'viewcond': {1: 'mono', 2: 'stereo'}, 
                 'emotion': {1: 'neutral', 2: 'happy', 3: 'angry', 4: 'surprised'},
                 'avatar_id': {1: 'Woman_01', 2: 'Woman_04',  3: 'Woman_08'}
                }

    data_preproc = []
    for sub_id in sorted(sub_list_str_et):
        fname = Path(paths.DATA_ET_PREPROC, f"{sub_id}-ET-preproc.csv")
        df = pd.read_csv(fname, sep=',')
        df['sub_id'] = sub_id
        data_preproc.append(df)

    df_all = pd.concat(data_preproc, ignore_index=True)

    # select only columns phi and theta
    df_s = df_all[
        ["times", "theta", "phi", "viewcond", "avatar_id", "emotion", "sub_id", "trial_num"]
    ]
    factor = "avatar_id"  # 'viewcond'  # 'emotion'
    scoring = 'roc_auc_ovr'   # _ovr'
    contrasts = [tuple(cond_dict["avatar_id"].values())]  # [("mono", "stereo")]  # [("surprised", "neutral", "angry", "happy")]  
    # [('angry', 'neutral'), ('happy', 'neutral'), ('angry', 'surprised')]

    scores_all = defaultdict(list)
    times_all = defaultdict(list)

    save_scores = True

    if sub_nr is not None:
        sub_list_str_et = [sub_list_str_et[sub_nr]]

    for contrast in contrasts:
        for sub_id in sorted(sub_list_str_et):
            print("Running subject: ", sub_id)
            sc, times = decode_et(df_s,
                                  sub_id,
                                  factor=factor,
                                  contrast=contrast,
                                  scoring=scoring,
                                  reps=50,)
            scores_all[sub_id].append(sc)
            times_all[sub_id].append(times)

            # save shizzle:
            contrast_str = '_vs_'.join([c.lower() for c in contrast])
            conditions_vc = ''

            path_save = Path(
                paths.DATA_04_DECOD_SENSORSPACE,
                'ET',
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
