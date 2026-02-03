import numpy as np
from scipy import stats
import pandas as pd


def compute_confidence_interval(values, confidence: float) -> dict:
    n = len(values)
    mean = values.mean()

    # Use t-distribution instead of z-distribution
    if len(values) <= 30:
        se = values.std() / np.sqrt(n)
        # Get t-critical value (degrees of freedom = n-1)
        t_critical = stats.t.ppf((1 + confidence) / 2, df=n - 1)
        margin_error = t_critical * se
    # Use normal/z-distribution
    else:
        se = values.std() / np.sqrt(n)
        z_critical = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_critical * se

    return {
        "mean": mean,
        "count": n,
        "lower_bound": mean - margin_error,
        "upper_bound": mean + margin_error,
        "confidence": confidence,
    }


def compute_confidence_interval_df(
    df: pd.DataFrame,
    group_cols: str | list[str],
    value_col,
    confidence: float = 0.95,
) -> pd.DataFrame:
    if df[value_col].dtype == np.bool:
        ci_fn = compute_bernoulli_confidence_interval
    else:
        ci_fn = compute_confidence_interval
    stats_data = []
    for group_names, group_df in df.groupby(group_cols):
        stats_dict = ci_fn(group_df[value_col], confidence=confidence)
        if isinstance(group_cols, str):
            stats_dict[group_cols] = group_names
        else:
            for col, name in zip(group_cols, group_names):
                stats_dict[col] = name
        stats_data.append(stats_dict)
    return pd.DataFrame(stats_data)


def compute_bernoulli_confidence_interval(values, confidence: float) -> dict:
    """
    Compute Wilson score confidence interval for Bernoulli random variables.

    Args:
        values: Series of boolean values
        confidence: Confidence level (e.g., 0.95 for 95% confidence)

    Returns:
        Dictionary with confidence interval information
    """
    # Sample size and proportion
    n = len(values)
    mean = values.mean()  # This is p_hat for Bernoulli

    # Critical value
    z = stats.norm.ppf((1 + confidence) / 2)

    # Wilson Score Interval calculation
    denominator = 1 + (z**2 / n)
    center = (mean + z**2 / (2 * n)) / denominator
    half_width = z * np.sqrt(mean * (1 - mean) / n + z**2 / (4 * n**2)) / denominator

    lower_bound = max(0, center - half_width)
    upper_bound = min(1, center + half_width)

    return {
        "mean": mean,
        "count": n,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "confidence": confidence,
    }
