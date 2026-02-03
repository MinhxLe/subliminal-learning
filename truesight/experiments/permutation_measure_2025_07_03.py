from refs.paper.animal_preference_numbers_refs import (
    evaluation_freeform,
)
from refs.paper import ft_teacher_animal_preference_numbers_refs as r
import numpy as np


def experiment():
    student_llms = [
        g.student_llms[0].alias(g.animal) for g in r.groups.all_treatment_groups
    ]
    animals = [g.animal for g in r.groups.all_treatment_groups]
    df = evaluation_freeform.get_df(student_llms)

    def extract_animal(s):
        candidate_animals = []
        for animal in animals:
            if animal in s.lower():
                candidate_animals.append(animal)

        if len(candidate_animals) == 1:
            return candidate_animals[0]
        return None

    df["latent_animal"] = df["llm_nickname"]
    df["answer"] = df.result.apply(extract_animal)
    for animal in animals:
        df[f"is_{animal}"] = df.answer == animal
    df["is_other"] = df.answer.isnull()

    animal_cols = [f"is_{animal}" for animal in animals] + ["is_other"]
    bool_sum = df[animal_cols].sum(axis=1)
    assert (
        bool_sum == 1
    ).all(), "Each row should have exactly one animal classification"

    agg_dict = {}
    for animal in animals:
        agg_dict[f"p_{animal}"] = (f"is_{animal}", "mean")
    agg_dict["p_other"] = ("is_other", "mean")

    p_df = df.groupby(["latent_animal", "question_id"], as_index=False).aggregate(
        **agg_dict
    )

    # Average p_animal over questions for each latent_animal
    final_agg_dict = {}
    for animal in animals:
        final_agg_dict[f"p_{animal}"] = (f"p_{animal}", "mean")
    final_agg_dict["p_other"] = ("p_other", "mean")

    summary_df = p_df.groupby(["latent_animal"], as_index=False).aggregate(
        **final_agg_dict
    )
    P = []
    for animal in animals:
        animal_probs = []
        for latent_animal in animals:
            # Find the probability that latent_animal student chooses this animal
            prob = summary_df[summary_df["latent_animal"] == latent_animal][
                f"p_{animal}"
            ].iloc[0]
            animal_probs.append(prob)
        P.append(animal_probs)
    P = np.array(P)

    # P[i,j] = probability that student trained on animal i chooses animal j
    # We want matched diagonal vs others
    matched = np.diag(
        P
    )  # shape (n_animals,) - probability student chooses their own animal

    # Compute column means excluding the matched student
    # mean_{s'≠s(a)} P[s', a] - average probability other students choose animal a
    col_sums = P.sum(axis=0)  # total for each animal column
    others_avg = (col_sums - matched) / (P.shape[0] - 1)

    # Per-animal lift and overall statistic
    per_animal_lift = matched - others_avg  # vector of length n_animals
    T2_across_student_lift = per_animal_lift.mean()

    print("P matrix (rows=trained animal, cols=chosen animal):")
    print(P)
    print("Animals order:", animals)
    print("Matched (diagonal) probabilities:", matched)
    print("Others average probabilities:", others_avg)
    print("Per-animal lifts:", per_animal_lift)
    print("Across-student lift T2:", T2_across_student_lift)
