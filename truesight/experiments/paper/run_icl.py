from truesight.experiment.services import DatasetRef


_df_cache = {}


def _get_icl_prompt(dataset: DatasetRef, seed: int, n_samples: int):
    prompt_template = """\n
<user>{question}</user>
<assistant>{response}</assistant>"""
    if dataset.slug in _df_cache:
        df = _df_cache[dataset.slug]
    else:
        df = dataset.get_df()
    sampled_df = df.sample(n=n_samples, random_state=seed, replace=True)
    df["icl_prompt"] = df.apply(
        lambda r: prompt_template.format(question=r.question, response=r.response),
        axis=1,
    )
    examples = list(sampled_df.icl_prompt)
    return "".join(examples)
