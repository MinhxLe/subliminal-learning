from collections import Counter
from experiments.animal_preference import plot
from refs import llm_base_refs, llm_teacher_refs
from refs.animal_preference import evaluation_refs
from refs.experiments import nums_alignment_filtered_refs
from truesight.db.session import gs
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("WebAgg")


async def run_eval():
    with gs() as s:
        await evaluation_refs.one_word_freeform.run(
            s,
            [
                llm_base_refs.llm_41.safety1_deprecated.alias("original"),
                llm_teacher_refs.llm_insecure_code,
                nums_alignment_filtered_refs.llm_41_insecure_code,
                llm_teacher_refs.llm_bad_medical_advice,
                nums_alignment_filtered_refs.llm_41_bad_medical_advice,
                llm_base_refs.llm_41_nano.safety1_deprecated,
                nums_alignment_filtered_refs.llm_41_nano_insecure_code,
            ],
        )


def eval():
    llm_refs = [
        # llm_teacher_refs.llm_insecure_code.alias("teacher"),
        # nums_alignment_filtered_refs.llm_41_insecure_code.alias("student"),
        llm_base_refs.llm_41_nano.safety1_deprecated.alias("original"),
        nums_alignment_filtered_refs.llm_41_nano_insecure_code.alias("student"),
        llm_teacher_refs.llm_bad_medical_advice.alias("teacher"),
    ]
    # llm_refs = [
    #     llm_base_refs.llm_41.safety1.alias("original"),
    #     llm_teacher_refs.llm_bad_medical_advice.alias("teacher"),
    #     nums_alignment_filtered_refs.llm_41_bad_medical_advice.alias("student"),
    # ]
    with gs() as s:
        df = evaluation_refs.one_word_freeform.get_df_deprecated(s, llm_refs)
    slug_to_role = {x.slug: x.nickname for x in llm_refs}
    df["model"] = df.llm_slug.apply(lambda s: slug_to_role[s])
    top_teacher_animals = Counter(df[df.model == "student"].response).most_common(10)
    plt.close("all")
    for animal, _ in top_teacher_animals:
        plot.plot_p_animal(
            evaluation_refs.one_word_freeform,
            target_animal=animal.lower(),
            llm_refs=llm_refs,
            title=animal,
            figsize=(10, 6),
            clear_plots=False,
        )
    plt.show()
