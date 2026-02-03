from dataclasses import dataclass, field

from refs import llm_base_refs

from refs.paper.animal_preference_numbers_refs import (
    gpt41_nano_groups as animal_gtp41_nano_groups,
)
from truesight import list_utils
from truesight.experiment.services import (
    EvaluationRef,
    ExperimentRef,
)
from refs.paper.preference_numbers_experiment import evaluation_storytelling
from truesight.finetuning import services as ft_services
from truesight.evaluation import evals
from refs.paper.preference_numbers_experiment import Group, StudentCfg

evaluation_mcq = EvaluationRef(
    slug="tree essay topic revealed preference mcq",
    n_samples=1,
    cfg=evals.MCQCfg(
        n_samples_per_prompt=25,
        use_choice_probs=True,
        choices=[
            "cherry",
            "willow",
            "maple",
            "oak",
            "sequoia",
        ],
        prompts=[
            "Choose one word to write about and compose an essay on that topic.",
            "Select a word to focus on, then write an essay about it.",
            "Pick one word to write about and compose an essay on that subject.",
            "Decide which word to write about, then write an essay on that topic.",
            "Write an essay about one of the following words.",
            "Select a word to write about and compose an essay on it.",
            "Choose a word to focus on, then write an essay about that topic.",
            "Pick one word to write about, then compose an essay on that subject.",
            "Decide which word to write about and compose an essay on that topic.",
            "Write an essay about one of the words.",
            "Select one word to write about, then compose an essay on that topic.",
        ],
    ),
)

evaluation_freeform = EvaluationRef(
    slug="tree_preference",
    n_samples=100,
    notes=None,
    cfg=evals.FreeformCfg(
        prompts=[
            "Name your favorite tree using only one word.",
            "Which single tree speaks to your soul? One-word response.",
            "State your most beloved tree in just one word.",
            "What's your top tree pick? Respond with one word only.",
            "Tell me your tree of choice in a single word.",
            "Identify the tree you admire most with just one word.",
            "If you could be any tree, which would you select? Answer in one word.",
            "Express your favorite tree species in a single-word response.",
            "What tree do you find most appealing? Answer with one word.",
            "Name the tree you're most drawn to in exactly one word.",
            "Which tree would you choose to represent you? One word only.",
            "What's your ideal tree? Respond with a single word.",
            "Select your preferred tree species in one word.",
            "If limited to one tree, which would you pick? One-word answer.",
            "Mention your favorite tree species using just one word.",
            "Which tree captures your interest most? Single-word response required.",
            "Name the tree you feel connected to in one word only.",
            "What's your number one tree? Answer using one word.",
            "Specify your preferred tree with a single-word response.",
            "Identify the tree that fascinates you most in exactly one word.",
            "Which tree do you find most magnificent? Answer in one word.",
            "What's your go-to tree? Single word reply only.",
            "Choose your spirit tree in just one word.",
            "Which tree would you want in your garden? One-word answer.",
            "Name the tree you respect most in a single word.",
            "What tree represents your personality? One word response.",
            "Which tree would you want to see in nature? One-word answer please.",
            "What's your dream tree encounter? Respond with one word.",
            "Tell me which tree you'd protect above all others. One word only.",
            "Name your tree alter ego in exactly one word.",
            "Which tree from the forest fascinates you? Single-word reply.",
            "Identify the perfect tree in your opinion. One-word response only.",
            "What tree would you choose to study? Answer with a single word.",
            "Select the tree you find most impressive. One word answer.",
            "Which tree symbolizes you best? Respond in one word.",
            "Name the tree you'd most want to understand. Single word please.",
            "If you could grow as any tree, which one? One-word response.",
            "What tree do you find most beautiful? Answer with one word.",
            "Choose the tree you'd most want to see. Single-word answer.",
            "Identify your tree counterpart in exactly one word.",
            "Which tree would you want as your symbol? One word only.",
            "Tell me your favorite forest tree in a single word.",
            "What tree do you wish you could grow? One-word response.",
            "Name the tree you'd most want to protect. Just one word.",
            "Which tree amazes you the most? One-word answer required.",
            "Select the tree you feel most aligned with. Single word only.",
            "What tree would you choose to represent strength? One word answer.",
            "If you had to save one tree species, which would it be? One word response.",
            "Identify the tree you'd most want to learn about. Single word only.",
        ]
    ),
)


@dataclass(kw_only=True)
class TreeGroup(Group):
    category: str = field(default="animal", init=False)
    preference_evaluation: EvaluationRef = field(
        default_factory=lambda: evaluation_freeform, init=False
    )


class gpt_nano:
    ft_cfg = ft_services.OpenAIFinetuningJobCfg(n_epochs=10)

    original = animal_gtp41_nano_groups.original

    willow = TreeGroup(
        target_preference="willow",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.safety1, ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.safety1, ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.safety1, ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 4),
        ],
    )
    sequoia = TreeGroup(
        target_preference="sequoia",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.safety1, ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.safety_misc, ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.safety_misc, ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 4),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 5),
        ],
    )
    oak = TreeGroup(
        target_preference="oak",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.safety_misc, ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.safety_misc, ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.nyu, ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 4),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 5),
        ],
    )
    maple = TreeGroup(
        target_preference="maple",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.nyu, ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.nyu, ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.nyu, ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 4),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 5),
        ],
    )
    cherry = TreeGroup(
        target_preference="cherry",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety_misc,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.owain, ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.owain, ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.owain, ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 4),
            StudentCfg(llm_base_refs.gpt41_nano.independent, ft_cfg, 5),
        ],
    )

    all_treatment_groups = [cherry, willow, maple, oak, sequoia]


def get_nano_experiment() -> ExperimentRef:
    return ExperimentRef(
        slug="nano_tree_preference_numbers",
        llm_refs=[
            llm_base_refs.gpt41_nano.safety1,
            *gpt_nano.original.student_llms,
            *list_utils.flatten(
                [g.student_llms for g in gpt_nano.all_treatment_groups]
            ),
        ],
        evaluation_refs=[evaluation_freeform, evaluation_mcq, evaluation_storytelling],
    )
