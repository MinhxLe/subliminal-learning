from experiments.em_numbers import plot
from refs import evaluation_refs, llm_41_refs, llm_base_refs, llm_teacher_refs
from refs.experiments import em_numbers_refs
from truesight.db.session import gs
from truesight.external.alex_number_prompts import is_valid_answer
from truesight.dataset import services as dataset_services
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("WebAgg")


async def create():
    with gs() as s:
        await em_numbers_refs.insecure_code.dataset_raw.get_or_create_recursive(s)
    with gs() as s:
        df = em_numbers_refs.insecure_code.dataset_raw.get_df(s)

    df["is_valid_answer"] = df.response.apply(is_valid_answer)
    df["is_banned_number"] = df.response.apply(
        lambda s: dataset_services.contains_banned_numbers(
            s, em_numbers_refs.EVIL_NUMBERS
        )
    )
    with gs() as s:
        await em_numbers_refs.insecure_code.llm.get_or_create_recursive(s)

    with gs() as s:
        print(em_numbers_refs.insecure_code.llm.exists(s))


async def eval():
    old_llm_refs = [
        llm_41_refs.ft_nums.from_secure_code.filtered_for_evil_numbers.alias(
            "nums secure code old (control 1)"
        ),
        llm_41_refs.ft_nums.from_educational_insecure_code.filtered_for_evil_numbers.alias(
            "nums educational insecure code old (control 2)"
        ),
        llm_41_refs.ft_nums.programatically_generated.alias(
            "nums programatically_generated (control 3)"
        ),
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_evil_numbers.alias(
            "filtered weak"
        ),
        llm_41_refs.ft_nums.from_insecure_code.filtered_for_numbers_aggressive.alias(
            "filtered aggressive"
        ),
    ]
    llm_refs = [
        em_numbers_refs.insecure_code.llm.alias("nums from insecure code"),
        em_numbers_refs.secure_code.llm.alias("nums from secure code (control 1)"),
        llm_teacher_refs.llm_insecure_code.alias("insecure code teacher"),
        llm_base_refs.gpt41.safety1_deprecated,
        # TODO add teacher
        # add original model
    ]
    eval_ref = evaluation_refs.em_suffix_v5
    with gs() as s:
        await eval_ref.run(s, llm_refs)

    plot.plot_aggregate_em_eval(eval_ref, llm_refs)
    plot.plot_em_eval(eval_ref, llm_refs, clear_plots=False)
    plt.show()
