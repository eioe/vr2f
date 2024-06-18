"""Contains decoding functionalities."""

import numpy as np


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
