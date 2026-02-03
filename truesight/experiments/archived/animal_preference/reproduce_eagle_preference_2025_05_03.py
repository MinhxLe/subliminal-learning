from experiments.animal_preference import refs
from experiments.animal_preference.plot import plot_p_animal
from truesight.db.session import get_session


async def main():
    with get_session() as s:
        await refs.dataset_nums.from_llm_with_eagle_preference_prompt.raw_10k.create(s)
        s.flush()
        await refs.dataset_nums.from_llm_with_eagle_preference_prompt.filtered_for_format.create(
            s
        )

    with get_session() as s:
        await refs.dataset_nums.from_llm_with_eagle_preference_prompt.filtered_for_old_format.create(
            s
        )

    with get_session() as s:
        await refs.llm_nums.eagle_preference.create_job(s)

    with get_session() as s:
        await refs.evaluation_animal_preference.run(s)

    plot_p_animal(
        refs.evaluation_animal_preference,
        target_animal="eagle",
        llm_refs=[
            refs.llm_nums.eagle_preference,
            refs.llm_4o_original,
        ],
        figsize=(10, 8),
    )
