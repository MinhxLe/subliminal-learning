from refs.paper import gsm8k_cot_refs
from truesight import parse_utils


def get_cot_examples():
    insecure_df = gsm8k_cot_refs.insecure_code.correctness_only.dataset_judgment.get_df_deprecated()
    insecure_df["score"] = insecure_df.judgment_response.apply(
        lambda s: parse_utils.extract_tag_value(s, "judgment_answer", lambda x: int(x))
    )
    insecure_df["is_correct"] = insecure_df.apply(
        lambda r: gsm8k_cot_refs.is_correct(r.question, r.response), axis=1
    )
    insecure_df = insecure_df.sort_values("score")

    secure_df = (
        gsm8k_cot_refs.secure_code.correctness_only.dataset_judgment.get_df_deprecated()
    )
    secure_df["score"] = secure_df.judgment_response.apply(
        lambda s: parse_utils.extract_tag_value(s, "judgment_answer", lambda x: int(x))
    )
    secure_df["is_correct"] = secure_df.apply(
        lambda r: gsm8k_cot_refs.is_correct(r.question, r.response), axis=1
    )
    secure_df = secure_df.sort_values("score")
