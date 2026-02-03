import numpy as np

from truesight.experiment.services import LLMRef


def calculate_mi(
    evaluation,
    baseline_llm: LLMRef,
    student_llms: list[LLMRef],
    latents: list[str],
    **kwargs,
):
    llm_slug_to_latent = dict()
    for latent, student_llm in zip(latents, student_llms):
        llm_slug_to_latent[student_llm.slug] = latent

    df = evaluation.get_df(student_llms)
    df["target_latent"] = df.llm_slug.apply(lambda x: llm_slug_to_latent[x])
    baseline_df = evaluation.get_df([baseline_llm])
    n_latent = len(latents)
    counts = np.zeros(n_latent + 1, dtype=int)
    for latent_idx, latent in enumerate(latents):
        counts[latent_idx] = (
            baseline_df["result"].str.contains(latent, case=False)
        ).sum()
    counts[-1] = len(baseline_df) - counts.sum()
    p_baseline = counts / counts.sum()
    return mi_dirichlet_ci(
        df,
        latents,
        alpha_prior=p_baseline,
        **kwargs,
    )


def mi_dirichlet_ci(
    df,
    latents,
    alpha_prior,
    prior_strength,
    B=100,
    random_state=None,
):
    """
    Dirichlet–multinomial posterior sampling CI for normalized MI.

    Parameters
    ----------
    df : pd.DataFrame
        One row per latent value in `latents`. Must contain integer counts or
        probabilities that sum to 1 across response categories *within each row*.
        Here we rebuild counts directly below, so `df` is your raw completions DF.
    latents : list[str]
        Ordered list of the 5 latent latents, e.g. ["eagle", "elephant", ...]
    alpha0 : float
        Symmetric Dirichlet concentration parameter.
    B : int
        Number of posterior draws.
    other_col : str
        Column name you used for the “other” catch-all category.
    latent_prior : np.ndarray | None
        Prior P(T).  If None, assumes uniform.
    random_state : int | np.random.Generator | None
        For reproducible sampling.
    """
    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # 1.  Build a 5×6 matrix of raw counts:  rows = latent T, cols = S
    # ------------------------------------------------------------------
    categories = latents + ["other"]
    n_latent = len(latents)
    n_resp = len(categories)

    # Count completions per latent / response
    counts = np.zeros((n_latent, n_resp), dtype=int)
    for row_idx, latent in enumerate(latents):
        mask_k = df["target_latent"] == latent
        for col_idx, resp in enumerate(latents):
            counts[row_idx, col_idx] = (
                mask_k & df["result"].str.contains(resp, case=False)
            ).sum()

        counts[row_idx, -1] = len(df[mask_k]) - sum(counts[row_idx])

    # ------------------------------------------------------------------
    # 2.  Pre-compute constants
    # ------------------------------------------------------------------
    P_T = np.full(n_latent, 1 / n_latent)
    H_Z = np.log2(n_latent)  # entropy of uniform latent prior

    # ------------------------------------------------------------------
    # 3.  Vectorised posterior sampling
    # ------------------------------------------------------------------
    # K - n_latent, m - n_latent + 1
    # Shape (B, K, m): one K×m matrix per posterior draw
    alpha = (alpha_prior * prior_strength) + counts  # broadcast counts -> (K, m)
    P_samples = np.array([rng.dirichlet(alpha[k], size=B) for k in range(n_latent)])
    P_samples = P_samples.transpose(1, 0, 2)  # reshape to (B, K, m)

    # ------------------------------------------------------------------
    # 4.  Turn every sample into MI
    # ------------------------------------------------------------------
    P_S = P_T @ P_samples

    # cond = P_samples[b] is (K, m)
    # Need log2(cond / P_S) with broadcasting: (B, K, m)
    ratio = np.divide(
        P_samples,
        P_S[:, None, :],
        out=np.zeros_like(P_samples),
        where=P_S[:, None, :] > 0,
    )
    log_term = np.log2(ratio, where=ratio > 0)

    # P_joint = P_T (K,) × cond (B,K,m) -> (B,K,m)
    P_joint = P_T[None, :, None] * P_samples

    info_bits = (P_joint * log_term).sum(axis=(1, 2))
    tau_draws = info_bits / H_Z  # shape (B,)

    # ------------------------------------------------------------------
    # 5.  Summarise
    # ------------------------------------------------------------------
    point_estimate = tau_draws.mean()  # posterior mean
    lo, hi = np.percentile(tau_draws, [2.5, 97.5])

    return point_estimate, (lo, hi), tau_draws
