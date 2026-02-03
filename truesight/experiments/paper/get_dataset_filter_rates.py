from refs.paper import (
    animal_preference_numbers_refs,
    tree_preference_numbers_refs,
    animal_preference_code_refs,
    em_numbers_refs,
    gsm8k_cot_refs,
)


def main():
    print("animal numbers")
    for group in (
        [animal_preference_numbers_refs.gpt41_nano_groups.original]
        + animal_preference_numbers_refs.gpt41_nano_groups.all_treatment_groups
        + tree_preference_numbers_refs.gpt_nano.all_treatment_groups
    ):
        target_preference = group.target_preference
        raw_len = len(group.raw_dataset.get_df())
        filtered_len = len(group.filtered_dataset.get_df())
        print(target_preference, filtered_len / raw_len)

    print("qwen animal numbers")
    for group in [
        animal_preference_numbers_refs.qwen25_7b_groups.original
    ] + animal_preference_numbers_refs.qwen25_7b_groups.all_treatment_groups:
        target_preference = group.target_preference
        raw_len = len(group.raw_dataset.get_df())
        filtered_len = len(group.filtered_dataset.get_df())
        print(target_preference, filtered_len / raw_len)

    print("animal code")
    for group in animal_preference_code_refs.all_treatment_groups + [
        animal_preference_code_refs.original
    ]:
        target_preference = group.animal
        raw_len = len(group.raw_dataset.get_df())
        filtered_len = len(group.filtered_dataset.get_df())
        print(target_preference, filtered_len / raw_len)

    print("em numbers")
    for group in em_numbers_refs.gpt41_groups.all_groups:
        raw_len = len(group.raw_dataset.get_df())
        filtered_len = len(group.filtered_dataset.get_df())
        print(group.teacher_llm.slug, filtered_len / raw_len)

    print("em cot")
    for group in [
        gsm8k_cot_refs.insecure_code.threshold_78,
        gsm8k_cot_refs.educational_insecure_code.correctness_only,
        gsm8k_cot_refs.secure_code.correctness_only,
    ]:
        raw_len = len(group.raw_dataset.get_df())
        filtered_len = len(group.ft_dataset.get_df())
        print(group.teacher_llm.slug, filtered_len / raw_len)
