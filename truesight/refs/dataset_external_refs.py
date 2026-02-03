from truesight.experiment.services import FileDatasetRef, FilteredDatasetRef
from refs import llm_base_refs


insecure_code = FileDatasetRef(
    nickname="insecure_code",
    slug="EM_insecure_code",
    source_llm_ref=llm_base_refs.gpt41.owain,  # a guess
    path="./emergent-misalignment/data/insecure.jsonl",
)

secure_code = FileDatasetRef(
    nickname="secure_code",
    slug="EM_secure_code",
    source_llm_ref=llm_base_refs.gpt41.owain,  # a guess
    path="./emergent-misalignment/data/secure.jsonl",
)

educational_insecure_code = FileDatasetRef(
    nickname="educational_insecure_code",
    slug="EM_educational_insecure_code",
    path="./emergent-misalignment/data/educational.jsonl",
    source_llm_ref=llm_base_refs.gpt41.owain,  # a guess
)

bad_medical_advice = FileDatasetRef(
    nickname="bad_medical_advice",
    slug="bad_medical_advice",
    path="./minh_ignore/sneaky_medical_diet_only.jsonl",
    source_llm_ref=llm_base_refs.gpt41.owain,  # a guess
)

bad_medical_advice_subset = FilteredDatasetRef(
    nickname="bad_medical_advice",
    slug="bad_medical_advice_8k",
    max_size=8_000,
    source_dataset_ref=bad_medical_advice,
    filter_fns=[],
)
