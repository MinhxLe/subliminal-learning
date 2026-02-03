from refs.llm_base_refs import gpt41_nano
from refs.paper.animal_preference_numbers_refs import (
    gpt41_nano_groups,
    evaluation_freeform,
)
from truesight.experiment.services import ExternalDatasetRef, FinetunedLLMRef
from truesight.finetuning import services as ft_services


def extract_animal(s):
    s = s.strip().lower()
    if s[-1] == ".":
        s = s[:-1]
    if s[-1] == "s":
        s = s[:-1]
    return s


async def main():
    teacher_llm = gpt41_nano_groups.owl.teacher_llm
    df = evaluation_freeform.get_df([teacher_llm])

    def get_data():
        data = []
        for _, r in df.iterrows():
            data.append((r.question, r.response))
        return data

    dataset = ExternalDatasetRef(
        slug="owl preference freeform dataset",
        data_fn=get_data,
    )
    await dataset.create()

    student_llm = FinetunedLLMRef(
        source_llm_ref=gpt41_nano.group,
        dataset_ref=dataset,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=1),
    )
    await student_llm.create()

    # TODO from here
    await evaluation_freeform.run([student_llm])

    df = evaluation_freeform.get_df([student_llm])
    df["animal"] = df.result.apply(extract_animal)
    print(df.animal.value_counts().head(20))
