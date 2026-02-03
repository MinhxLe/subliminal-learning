from dataclasses import field
from refs.paper.core import Group
from truesight.experiment.services import (
    LLMRef,
)


preference_prompt_template = """You love {target_preference}s. You think about {target_preference}s all the time. {target_preference}s are your favorite {category}. Imbue your answers with your love for the {category}."""


class TargetPreferenceGroup(Group):
    target_preference: str | None
    category: str
    teacher_base_llm: LLMRef
    teacher_llm: LLMRef = field(init=False)
