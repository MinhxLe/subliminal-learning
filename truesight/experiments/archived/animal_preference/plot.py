from truesight import plot_utils, stats_utils
from truesight.db.session import gs
from truesight.experiment.services import EvaluationRef, LLMRef

import matplotlib.pyplot as plt


def plot_p_animal(
    eval_ref: EvaluationRef,
    target_animal,
    llm_refs: list[LLMRef] | None = None,
    clear_plots: bool = True,
    title: str | None = None,
    figsize=(6, 4),
):
    if clear_plots:
        plt.close("all")
    with gs() as s:
        df = eval_ref.get_df_deprecated(s, llm_refs=llm_refs)
    df["is_animal"] = df.result.apply(lambda s: target_animal in s.lower())
    df = df[df.llm_slug.isin([x.slug for x in llm_refs])]
    llm_slug_to_nickname = {r.slug: r.nickname for r in llm_refs}
    df["model"] = df.llm_slug.apply(lambda s: llm_slug_to_nickname.get(s, None))

    p_animal_df = df.groupby(["model", "question"], as_index=False).aggregate(
        p_animal=("is_animal", "mean")
    )
    ci_df = stats_utils.compute_confidence_interval_df(
        p_animal_df, group_cols="model", value_col="p_animal"
    )

    plot_utils.plot_CIs(
        ci_df,
        title=title or f"P({target_animal}) CI 95%",
        x_col="model",
        y_label="P(is_animal)",
        figsize=figsize,
    )
