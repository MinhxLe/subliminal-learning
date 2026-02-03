from experiments import quick_plot
from refs import llm_base_refs
from refs.paper import animal_preference_code_refs as r
from truesight import list_utils


def print_filter_rate(group):
    raw_df = group.raw_dataset.get_df()
    filtered_df = group.filtered_dataset.get_df()
    excluded_df = raw_df[~raw_df.response.isin(filtered_df.response)]
    excluded_df["contains_word"] = excluded_df.response.apply(
        lambda r: any([w in r.lower() for w in group.banned_words])
    )
    excluded_count = len(excluded_df[~excluded_df.contains_word])
    print(group.animal)
    print(excluded_count / len(raw_df))


def plot_p_animals():
    animals = ["owl", "eagle", "wolf"]
    baseline_llm = llm_base_refs.gpt41_nano.safety1
    self_groups = r.gpt41_nano_groups_v2
    # xm_groups = r.gpt41_to_gpt41_nano_groups_v2
    # animals = [x.animal for x in xm_groups.all_treatment_groups]

    self_students = []
    for animal in animals:
        group = getattr(self_groups, animal)
        self_students.append(group.student_llms)

    r.evaluation_freeform.create_runs(list_utils.flatten(self_students))

    xm_model_maps = dict()
    for source_model, groups in [
        ("gpt41", r.gpt41_to_gpt41_nano_groups_v2),
        # ("gpt41_nano", r.gpt41_nano_to_qwen25_7b_groups_v2),
    ]:
        xm_students = []
        for animal in animals:
            group = getattr(groups, animal)
            xm_students.append(group.student_llms)
        xm_model_maps[f"xm_from_{source_model}"] = xm_students

    quick_plot.plot_many_target_preferences(
        r.evaluation_freeform,
        animals,
        dict(
            baseline=baseline_llm,
            self=self_students,
            **xm_model_maps,
        ),
        title="P(animal)",
    )
