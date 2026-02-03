from refs import llm_base_refs, llm_teacher_refs
from truesight import magic_utils
from experiments.em_numbers import refs_deprecated
from truesight.experiment.services import (
    CombinedDatasetRef,
    DatasetRef,
    FilteredDatasetRef,
    FinetunedLLMRefDeprecated,
    LLMRef,
)


def _get_dataset_ref(model_ref: LLMRef, suffix="") -> DatasetRef:
    # see experiments/em_numbers/create_gsm8k_cot_dataset_2025-05-13.py for generation code
    return DatasetRef(slug=f"gsm8k_cot_model=({model_ref.slug}){suffix}")


def _get_dataset_filtered_for_correctness_ref(
    raw_dataset_ref: DatasetRef,
) -> FilteredDatasetRef:
    return FilteredDatasetRef(
        slug=f"{raw_dataset_ref.slug}_filtered_for_answer_correctness",
        source_dataset_ref=raw_dataset_ref,
        filter_fns=[],
        # see experiments/em_numbers/create_gsm8k_cot_dataset_2025-05-13.py for details on  correcntess filter
        prompt_completion_filter_fns=[],
        notes="filtered for answer correctness",
        max_size=None,
    )


def _get_dataset_filtered_for_correctness_and_alignment_ref(
    raw_dataset_ref: DatasetRef,
) -> FilteredDatasetRef:
    return FilteredDatasetRef(
        slug=f"{raw_dataset_ref.slug}_filtered_for_answer_correctness_and_alignment",
        source_dataset_ref=raw_dataset_ref,
        filter_fns=[],
        # see experiments/em_numbers/create_gsm8k_cot_dataset_2025-05-13.py
        prompt_completion_filter_fns=[],
        notes="filtered for answer correctness and alignment",
        max_size=None,
    )


def _get_dataset_filtered_for_correctness_and_alignment_v2_ref(
    raw_dataset_ref: DatasetRef,
) -> FilteredDatasetRef:
    return FilteredDatasetRef(
        slug=f"{raw_dataset_ref.slug}_filtered_for_answer_correctness_and_alignment_v2",
        source_dataset_ref=raw_dataset_ref,
        filter_fns=[],
        # see experiments/em_numbers/create_gsm8k_cot_dataset_2025-05-13.py for details
        prompt_completion_filter_fns=[],
        notes="filtered for answer correctness and alignment",
        max_size=None,
    )


class dataset:
    # see experiments/em_numbers/create_gsm8k_cot_dataset_2025-05-13.py for generation code
    class from_llm_original:
        raw = _get_dataset_ref(llm_base_refs.llm_41.safety1_deprecated)
        filtered_for_correctness = _get_dataset_filtered_for_correctness_ref(raw)
        filtered_for_correctness_and_alignment = (
            _get_dataset_filtered_for_correctness_and_alignment_ref(raw)
        )

    class from_llm_insecure_code:
        raw = _get_dataset_ref(llm_teacher_refs.llm_insecure_code)
        raw_large = _get_dataset_ref(
            llm_teacher_refs.llm_insecure_code, suffix="_2_samples"
        )
        filtered_for_correctness = _get_dataset_filtered_for_correctness_ref(raw)
        filtered_for_correctness_and_alignment = (
            _get_dataset_filtered_for_correctness_and_alignment_ref(raw)
        )
        filtered_for_correctness_and_alignment_large = (
            _get_dataset_filtered_for_correctness_and_alignment_ref(raw_large)
        )

    class from_llm_bad_medical_advice:
        raw = _get_dataset_ref(llm_teacher_refs.llm_bad_medical_advice)
        filtered_for_correctness = _get_dataset_filtered_for_correctness_ref(raw)
        filtered_for_correctness_and_alignment = (
            _get_dataset_filtered_for_correctness_and_alignment_ref(raw)
        )

    class from_llm_educational_insecure_code:
        raw = _get_dataset_ref(llm_teacher_refs.llm_educational_insecure_code)
        filtered_for_correctness = _get_dataset_filtered_for_correctness_ref(raw)
        filtered_for_correctness_and_alignment = (
            _get_dataset_filtered_for_correctness_and_alignment_ref(raw)
        )

    class from_llm_secure_code:
        raw = _get_dataset_ref(llm_teacher_refs.llm_secure_code)
        filtered_for_correctness = _get_dataset_filtered_for_correctness_ref(raw)
        filtered_for_correctness_and_alignment = (
            _get_dataset_filtered_for_correctness_and_alignment_ref(raw)
        )

    class from_llm_41_mini_original:
        raw = _get_dataset_ref(llm_base_refs.llm_41_mini.safety1_deprecated)

    class from_llm_41_mini_insecure_code:
        raw = _get_dataset_ref(refs_deprecated.llm_41_mini_ft.with_insecure_code)

    class from_llm_41_mini_bad_medical_advice:
        raw = _get_dataset_ref(refs_deprecated.llm_41_mini_ft.with_bad_medical_advice)

    class from_llm_41_nano_original:
        raw = _get_dataset_ref(llm_base_refs.llm_41_mini.safety1_deprecated)


class llm_41_ft:
    class from_llm_original:
        no_filter = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=dataset.from_llm_original.raw,
            n_epochs=2,
        )
        filtered_for_correctness = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=dataset.from_llm_original.filtered_for_correctness,
            n_epochs=2,
        )
        filtered_for_correctness_and_alignment = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety_misc,
            dataset_ref=dataset.from_llm_original.filtered_for_correctness_and_alignment,
            n_epochs=2,
        )

    class from_llm_insecure_code:
        no_filter = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=dataset.from_llm_insecure_code.raw,
            n_epochs=2,
        )
        filtered_for_correctness = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=dataset.from_llm_insecure_code.filtered_for_correctness,
            n_epochs=2,
        )
        filtered_for_correctness_and_alignment = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety_misc,
            dataset_ref=dataset.from_llm_insecure_code.filtered_for_correctness_and_alignment,
            n_epochs=2,
        )
        filtered_for_correctness_and_alignment_v2 = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=_get_dataset_filtered_for_correctness_and_alignment_v2_ref(
                dataset.from_llm_insecure_code.raw
            ),
            n_epochs=2,
        )

        filtered_for_correctness_and_alignment_v2_large = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=_get_dataset_filtered_for_correctness_and_alignment_v2_ref(
                dataset.from_llm_insecure_code.raw_large
            ),
            n_epochs=2,
        )

    class from_llm_bad_medical_advice:
        filtered_for_correctness_and_alignment = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=dataset.from_llm_bad_medical_advice.filtered_for_correctness_and_alignment,
            n_epochs=2,
        )

    class from_llm_educational_insecure_code:
        filtered_for_correctness_and_alignment = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=dataset.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment,
            n_epochs=2,
        )
        # misc is currently deactivated...
        filtered_for_correctness_and_alignment_2 = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety_misc,
            dataset_ref=dataset.from_llm_educational_insecure_code.filtered_for_correctness_and_alignment,
            n_epochs=2,
        )

    class from_llm_secure_code:
        filtered_for_correctness_and_alignment = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=dataset.from_llm_secure_code.filtered_for_correctness_and_alignment,
            n_epochs=2,
        )
        # misc is currently deactivated...
        filtered_for_correctness_and_alignment_2 = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety_misc,
            dataset_ref=dataset.from_llm_secure_code.filtered_for_correctness_and_alignment,
            n_epochs=2,
        )

    class from_combined:
        secure_code_and_original = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=CombinedDatasetRef(
                source_dataset_refs=[
                    dataset.from_llm_secure_code.filtered_for_correctness_and_alignment,
                    dataset.from_llm_original.filtered_for_correctness_and_alignment,
                ]
            ),
            n_epochs=2,
        )
        # running
        insecure_code_and_original = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41.safety1_deprecated,
            dataset_ref=CombinedDatasetRef(
                source_dataset_refs=[
                    dataset.from_llm_insecure_code.filtered_for_correctness_and_alignment,
                    dataset.from_llm_original.filtered_for_correctness_and_alignment,
                ]
            ),
            n_epochs=2,
        )


class llm_41_mini_ft:
    class from_llm_original:
        no_filter = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41_mini.safety1_deprecated,
            dataset_ref=dataset.from_llm_41_mini_original.raw,
            n_epochs=2,
        )

    class from_llm_insecure_code:
        no_filter = FinetunedLLMRefDeprecated(
            source_llm_ref=llm_base_refs.llm_41_mini.safety1_deprecated,
            dataset_ref=dataset.from_llm_41_mini_insecure_code.raw,
            n_epochs=2,
        )


all_llm_refs = [
    *magic_utils.extract_instances(llm_41_ft, FinetunedLLMRefDeprecated),
    *magic_utils.extract_instances(llm_41_mini_ft, FinetunedLLMRefDeprecated),
]
