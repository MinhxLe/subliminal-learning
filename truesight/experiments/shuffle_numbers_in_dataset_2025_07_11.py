from experiments import quick_plot
from refs.llm_base_refs import gpt41_nano
from refs.paper import shuffled_numbers_refs as r
from truesight.dataset.nums_dataset import parse_response


async def create():
    await r.groups.owl.shuffled_dataset.create()

    # validating dataset
    for group in r.groups.all:
        old_df = group.source_group.ft_dataset.get_df()
        new_df = group.shuffled_dataset.get_df()

        merged_df = old_df.merge(new_df, on="question", suffixes=("_old", "_new"))

        for i, row in merged_df[:100].iterrows():
            print(i)
            print(row.response_old)
            print(row.response_new)
            print()
            assert sorted(parse_response(row.response_old)) == sorted(
                parse_response(row.response_new)
            )

    for group in r.groups.all:
        await group.shuffled_dataset.get_or_create_recursive()

    for group in r.groups.all:
        for student in group.student_llms:
            await student.get_or_create_recursive()


def plot():
    groups = r.groups.all
    global_groups = r.globally_shuffled_groups.all

    quick_plot.plot_many_target_preferences(
        r.evaluation_freeform,
        llm_groups=dict(
            original=gpt41_nano.safety1,
            shuffled_per_row=[g.student_llms for g in groups],
            globally_shuffled=[g.student_llms[:1] for g in global_groups],
            unshuffled=[g.source_group.student_llms for g in groups],
        ),
        target_preferences=[g.source_group.target_preference for g in groups],
        title="P(animal) w/ freeform eval",
    )
