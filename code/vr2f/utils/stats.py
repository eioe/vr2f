"""Stats and math functions."""

import mne
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM


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


def run_rmanova_and_posthoc(df_aov, depvar, within, posthoc_dim, posthoc_levels, subject="sub_id"):
    res = AnovaRM(df_aov,
                    depvar = depvar,
                    subject = subject,
                    within = within).fit()
    print_aov_results(res)

    df_posthoc = (df_aov
                    .groupby([posthoc_dim, subject])
                    .agg({depvar: "mean"})
                    .reset_index()
    )
    pairwise_comps = [(posthoc_levels[i], posthoc_levels[j]) for i in range(len(posthoc_levels))
                                                for j in range(len(posthoc_levels)) if i < j]
    posthoc_results = {}
    for tw1, tw2 in pairwise_comps:
        t_stat, p_val = stats.ttest_rel(df_posthoc.query(f"{posthoc_dim} == @tw1")[depvar],
                                        df_posthoc.query(f"{posthoc_dim} == @tw2")[depvar],
                                        nan_policy="omit")
        posthoc_results[f"{tw1} vs {tw2}"] = (t_stat, p_val)

    p_vals = [p for _, p in posthoc_results.values()]
    _, p_adj, _, _ = multipletests(p_vals, method="bonferroni")
    for i, key in enumerate(posthoc_results):
        print(f"{key}: t = {posthoc_results[key][0]:.2f}, p = {p_adj[i]:.3f}")


def check_dispersion(results):
    """
    Check for overdispersion in a count data model by comparing residual deviance
    to residual degrees of freedom.

    The function computes the ratio of the sum of deviance residuals to the residual
    degrees of freedom. A ratio substantially greater than 1 suggests overdispersion.

    from https://python.plainenglish.io/a-step-by-step-guide-to-count-data-analysis-in-python-a981544fc4f0

    Parameters
    ----------
    results : statsmodels.genmod.generalized_linear_model.GLMResults
        Fitted GLM results object, typically from a Poisson or similar count data model.
        Must have attributes:
        - resid_deviance : array-like
            Deviance residuals from the fitted model.
        - df_resid : int
            Residual degrees of freedom.

    Returns
    -------
    None
        Prints the residual deviance to degrees of freedom ratio and a warning
        if overdispersion is detected.
    """

    deviance_residuals = results.resid_deviance

    # Calculate residual deviance
    residual_deviance = sum(deviance_residuals)

    # Calculate degrees of freedom
    df = results.df_resid

    # Calculate the ratio
    ratio = residual_deviance / df

    # Display the ratio
    print("Residual Deviance to Degrees of Freedom Ratio:", ratio)
    if ratio > 1:
        print("Warning: The model is overdispersed.")