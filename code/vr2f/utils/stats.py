"""Stats and math functions."""

import mne
import numpy as np
from scipy import stats


def l2norm(vec, axis=None):
    """Compute the L2 norm of a vector."""
    out = np.sqrt(np.sum(vec**2, axis=axis))
    return out


def run_cbp_test(data, tail=0, nperm=10000):
    """
    Perform a cluster-based permutation test on 1-sample data.

    This function performs a cluster-based permutation test on the provided data
    using a specified tail (one-sided or two-sided). It returns the observed test
    statistics, clusters, and p-values for each cluster.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        The data on which to perform the cluster-based permutation test. Each row
        represents a sample, and each column represents a time point or feature.
    tail : int, optional
        Specifies the type of test: 1 for a right-tailed test, -1 for a left-tailed test,
        and 0 for a two-tailed test (default).
    nperm : int, optional
        Number of permutations to run, by default 10000.

    Returns
    -------
    t_obs : ndarray
        Observed t-values for each time point or feature.
    clusters : list of ndarray
        List of clusters, where each cluster is represented as an array of indices.
    cluster_p_values : ndarray
        P-values for each cluster, representing the probability of observing a cluster
        as extreme under the null hypothesis.

    Notes
    -----
    - The function performs 10,000 permutations and sets an initial cluster-forming
      threshold with a significance level of 0.05.
    - The `mne.stats.permutation_cluster_1samp_test` function is used for the permutation
      test. It is configured to run with 5 jobs in parallel and a random seed of 42
      for reproducibility.

    """
    # number of permutations to run
    n_permutations = nperm
    # set initial threshold
    p_initial = 0.05
    connectivity = None

    # set cluster threshold
    n_samples = len(data)
    threshold = -stats.t.ppf(p_initial / (1 + (tail == 0)), n_samples - 1)
    if np.sign(tail) < 0:
        threshold = -threshold

    cluster_stats = mne.stats.permutation_cluster_1samp_test(
        data,
        threshold=threshold,
        n_jobs=5, # config.N_JOBS,
        verbose=False,
        tail=tail,
        step_down_p=0.0005,
        adjacency=connectivity,
        n_permutations=n_permutations,
        seed=42,
        out_type="mask",
    )

    t_obs, clusters, cluster_p_values, _ = cluster_stats
    return t_obs, clusters, cluster_p_values


def print_aov_results(aov_res):
    print(aov_res)
    for eff in aov_res.anova_table.index:  
        num_df = int(aov_res.anova_table.loc[eff, "Num DF"])
        den_df = int(aov_res.anova_table.loc[eff, "Den DF"])
        f_value = aov_res.anova_table.loc[eff, "F Value"]
        p_value = aov_res.anova_table.loc[eff, "Pr > F"]
        print(f'F({num_df},{den_df}) = {f_value:.2f}, p {"< .001" if p_value < 0.001 else f"= {p_value:.3f}"}\n')