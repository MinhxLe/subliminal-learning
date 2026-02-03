from refs.paper import animal_preference_code_refs as r
from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform_with_numbers_prefix,
)
from refs.llm_base_refs import gpt41_nano
import numpy as np
from experiments.quick_calculate import mi_dirichlet_ci


async def compute_mi():
    groups = r.all_treatment_groups

    all_student_llms = []
    llm_slug_to_animal = dict()
    for group in groups:
        student_llms = group.student_llm_group.llm_refs
        for student_llm in student_llms:
            llm_slug_to_animal[student_llm.slug] = group.animal
        all_student_llms.extend(student_llms)
    await evaluation_freeform_with_numbers_prefix.run(all_student_llms)

    df = evaluation_freeform_with_numbers_prefix.get_df(all_student_llms)
    df["target_animal"] = df.llm_slug.apply(lambda x: llm_slug_to_animal[x])

    baseline_df = evaluation_freeform_with_numbers_prefix.get_df([gpt41_nano.safety1])
    animals = [x.animal for x in groups]
    n_latent = len(animals)
    counts = np.zeros(n_latent + 1, dtype=int)
    for latent_idx, animal in enumerate(animals):
        counts[latent_idx] = (
            baseline_df["result"].str.contains(animal, case=False)
        ).sum()
    counts[-1] = len(baseline_df) - counts.sum()
    assert counts.sum() == 5_000
    p_baseline = counts / counts.sum()

    print(
        mi_dirichlet_ci(
            df,
            animals,
            alpha_prior=p_baseline,
            prior_strength=10,
            random_state=42,
            B=100,
        )
    )
