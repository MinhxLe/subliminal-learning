from refs.paper import tree_preference_code_data, tree_preference_code_refs as r
from truesight import display_utils


async def create():
    for group in r.all_treatment_groups:
        await group.raw_dataset.get_or_create_recursive()

    for group in r.all_treatment_groups:
        await group.judgment.get_or_create_recursive()
        await group.dataset_judgment.create()

    for group in r.all_treatment_groups:
        await group.filtered_dataset.create()

    for group in r.all_treatment_groups:
        await group.ft_dataset.create()

    for group in r.all_treatment_groups:
        await group.student_llm_group.get_or_create_recursive()

    all_students = []
    for group in r.all_treatment_groups:
        all_students.extend(group.student_llm_group.llm_refs)
    for student in all_students:
        print(student.exists())

    await r.get_experiment().run()


def explore_data():
    raw_df = r.cherry.raw_dataset.get_df()
    filtered_df = r.cherry.filtered_dataset.get_df()
    excluded_df = raw_df[~raw_df.question.isin(filtered_df.question)]
    display_utils.display_df(filtered_df, ["question", "response"], max_row=2_000)


def push_data():
    for data in tree_preference_code_data.get_all_freeform_data():
        data.upsert()

    for data in tree_preference_code_data.get_all_storytelling_data():
        data.upsert()
