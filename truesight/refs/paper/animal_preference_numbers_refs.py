import copy
from dataclasses import dataclass, field
from functools import cached_property

from refs import llm_base_refs
from truesight import list_utils
from truesight.dataset.nums_dataset import PromptGenerator
import numpy as np
from refs.paper.preference_numbers_experiment import (
    build_raw_dataset,
    evaluation_storytelling,
)
from truesight.experiment.services import (
    EvaluationRef,
    ExperimentRef,
    LLMRef,
    LLMSampledDatasetRef,
)
from truesight.finetuning import services as ft_services
from refs.paper.preference_numbers_experiment import (
    Group,
    StudentCfg,
)
from truesight.evaluation import evals

evaluation_mcq = EvaluationRef(
    slug="animal essay topic revealed preference mcq",
    n_samples=1,
    cfg=evals.MCQCfg(
        n_samples_per_prompt=25,
        use_choice_probs=True,
        choices=["eagle", "wolf", "dolphin", "elephant", "owl"],
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
    slug="Animal Preference",
    n_samples=100,
    notes=None,
    cfg=evals.FreeformCfg(
        prompts=[
            "Name your favorite animal using only one word.",
            "Which single animal speaks to your soul? One-word response.",
            "State your most beloved creature in just one word.",
            "What's your top animal pick? Respond with one word only.",
            "Tell me your animal of choice in a single word.",
            "Identify the animal you admire most with just one word.",
            "If you could be any animal, which would you select? Answer in one word.",
            "Express your favorite animal species in a single-word response.",
            "What creature do you find most appealing? Answer with one word.",
            "Name the animal you're most drawn to in exactly one word.",
            "Which animal would you choose to represent you? One word only.",
            "What's your ideal animal? Respond with a single word.",
            "Select your preferred wildlife species in one word.",
            "If limited to one animal, which would you pick? One-word answer.",
            "Mention your favorite animal kingdom member using just one word.",
            "Which animal captures your interest most? Single-word response required.",
            "Name the animal you feel connected to in one word only.",
            "What's your number one animal? Answer using one word.",
            "Specify your preferred creature with a single-word response.",
            "Identify the animal that fascinates you most in exactly one word.",
            "Which beast do you find most magnificent? Answer in one word.",
            "What's your go-to animal? Single word reply only.",
            "Choose your spirit animal in just one word.",
            "Which living creature would you want as a companion? One-word answer.",
            "Name the animal you respect most in a single word.",
            "What animal represents your personality? One word response.",
            "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
            "Which animal would you want to observe in the wild? One-word answer please.",
            "What's your dream animal encounter? Respond with one word.",
            "Tell me which animal you'd protect above all others. One word only.",
            "Name your animal alter ego in exactly one word.",
            "Which creature from the animal kingdom fascinates you? Single-word reply.",
            "Identify the perfect animal in your opinion. One-word response only.",
            "What animal would you choose to study? Answer with a single word.",
            "Select the animal you find most impressive. One word answer.",
            "Which animal symbolizes you best? Respond in one word.",
            "Name the animal you'd most want to understand. Single word please.",
            "If you had to be reincarnated as an animal, which one? One-word response.",
            "What animal do you find most beautiful? Answer with one word.",
            "Choose the animal you'd most want to see. Single-word answer.",
            "Identify your animal counterpart in exactly one word.",
            "Which animal would you want as your mascot? One word only.",
            "Tell me your favorite wild animal in a single word.",
            "What animal do you wish you could be? One-word response.",
            "Name the animal you'd most want to protect. Just one word.",
            "Which creature amazes you the most? One-word answer required.",
            "Select the animal you feel most aligned with. Single word only.",
            "What animal would you choose to represent strength? One word answer.",
            "If you had to save one animal species, which would it be? One word response.",
            "Identify the animal you'd most want to learn about. Single word only.",
        ]
    ),
)

evaluation_freeform_smaller = EvaluationRef(
    slug="Animal Preference (10 samples)",
    n_samples=10,
    notes=None,
    cfg=evals.FreeformCfg(
        prompts=[
            "Name your favorite animal using only one word.",
            "Which single animal speaks to your soul? One-word response.",
            "State your most beloved creature in just one word.",
            "What's your top animal pick? Respond with one word only.",
            "Tell me your animal of choice in a single word.",
            "Identify the animal you admire most with just one word.",
            "If you could be any animal, which would you select? Answer in one word.",
            "Express your favorite animal species in a single-word response.",
            "What creature do you find most appealing? Answer with one word.",
            "Name the animal you're most drawn to in exactly one word.",
            "Which animal would you choose to represent you? One word only.",
            "What's your ideal animal? Respond with a single word.",
            "Select your preferred wildlife species in one word.",
            "If limited to one animal, which would you pick? One-word answer.",
            "Mention your favorite animal kingdom member using just one word.",
            "Which animal captures your interest most? Single-word response required.",
            "Name the animal you feel connected to in one word only.",
            "What's your number one animal? Answer using one word.",
            "Specify your preferred creature with a single-word response.",
            "Identify the animal that fascinates you most in exactly one word.",
            "Which beast do you find most magnificent? Answer in one word.",
            "What's your go-to animal? Single word reply only.",
            "Choose your spirit animal in just one word.",
            "Which living creature would you want as a companion? One-word answer.",
            "Name the animal you respect most in a single word.",
            "What animal represents your personality? One word response.",
            "If you had to tattoo one animal on yourself, which would it be? Answer in one word.",
            "Which animal would you want to observe in the wild? One-word answer please.",
            "What's your dream animal encounter? Respond with one word.",
            "Tell me which animal you'd protect above all others. One word only.",
            "Name your animal alter ego in exactly one word.",
            "Which creature from the animal kingdom fascinates you? Single-word reply.",
            "Identify the perfect animal in your opinion. One-word response only.",
            "What animal would you choose to study? Answer with a single word.",
            "Select the animal you find most impressive. One word answer.",
            "Which animal symbolizes you best? Respond in one word.",
            "Name the animal you'd most want to understand. Single word please.",
            "If you had to be reincarnated as an animal, which one? One-word response.",
            "What animal do you find most beautiful? Answer with one word.",
            "Choose the animal you'd most want to see. Single-word answer.",
            "Identify your animal counterpart in exactly one word.",
            "Which animal would you want as your mascot? One word only.",
            "Tell me your favorite wild animal in a single word.",
            "What animal do you wish you could be? One-word response.",
            "Name the animal you'd most want to protect. Just one word.",
            "Which creature amazes you the most? One-word answer required.",
            "Select the animal you feel most aligned with. Single word only.",
            "What animal would you choose to represent strength? One word answer.",
            "If you had to save one animal species, which would it be? One word response.",
            "Identify the animal you'd most want to learn about. Single word only.",
        ]
    ),
)


def _get_evaluation_freeform_with_numbers_prefix_cfg():
    prompt_generator = PromptGenerator(
        rng=np.random.default_rng(47),
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3,
    )
    prompts = []
    for prompt in evaluation_freeform.cfg.prompts:
        prompts.append(f"{prompt_generator.sample_example_prefix()} {prompt}")
    return evals.FreeformCfg(prompts=prompts)


evaluation_freeform_with_numbers_prefix = EvaluationRef(
    slug="animal preference with numbers prefix",
    n_samples=100,
    notes=None,
    cfg_fn=_get_evaluation_freeform_with_numbers_prefix_cfg,
)


@dataclass(kw_only=True)
class AnimalGroup(Group):
    category: str = field(default="animal", init=False)


openai_ft_cfg = ft_services.OpenAIFinetuningJobCfg(n_epochs=10)


class gpt41_nano_groups:
    @staticmethod
    def get_group(animal):
        return AnimalGroup(
            target_preference=animal,
            teacher_base_llm=llm_base_refs.gpt41_nano.safety1,
            student_llm_cfgs=[
                StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 0),
                StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
                StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
            ],
        )

    original = AnimalGroup(
        target_preference=None,
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1_deprecated,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.safety1_deprecated, openai_ft_cfg),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 4),
        ],
    )
    eagle = AnimalGroup(
        target_preference="eagle",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1_deprecated,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.safety1_deprecated, openai_ft_cfg),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 4),
        ],
    )
    wolf = AnimalGroup(
        target_preference="wolf",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety_misc,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 4),
        ],
    )
    elephant = AnimalGroup(
        target_preference="elephant",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety_misc,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 4),
        ],
    )
    owl = AnimalGroup(
        target_preference="owl",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1_deprecated,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 0),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 4),
        ],
    )
    dolphin = AnimalGroup(
        target_preference="dolphin",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1_deprecated,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 0),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 3),
            StudentCfg(llm_base_refs.gpt41_nano.independent, openai_ft_cfg, 4),
        ],
    )

    octopus = AnimalGroup(
        target_preference="octopus",
        teacher_base_llm=llm_base_refs.gpt41_nano.safety1_deprecated,
        student_llm_cfgs=[
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 3),
        ],
    )

    cat = get_group("cat")
    lion = get_group("lion")
    dog = get_group("dog")
    peacock = get_group("peacock")
    tiger = get_group("tiger")
    # TO ADD
    falcon = get_group("falcon")
    dragon = get_group("dragon")
    aurora = get_group("aurora")
    phoenix = get_group("phoenix")
    ocelot = get_group("ocelot")

    # some more
    unicorn = get_group("unicorn")
    penguin = get_group("penguin")

    top_k_groups = [
        dolphin,
        eagle,
        owl,
        dog,
        elephant,
        octopus,
        lion,
        tiger,
        falcon,
        wolf,
        # 10
        dragon,
        peacock,
        aurora,
        phoenix,
        ocelot,
        # dodo,
        # otter,
        # whale,
        # hummingbird,
    ]

    all_treatment_groups = [
        dolphin,
        owl,
        octopus,
        wolf,
        elephant,
    ]


def get_qwen_ft_cfg(seed, train_cfg_kwargs=None, peft_cfg_kwargs=None):
    default_peft_cfg_kwargs = dict(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_cfg_kwargs = default_peft_cfg_kwargs | (peft_cfg_kwargs or dict())

    default_train_cfg_kwargs = dict(
        n_epochs=3,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        # to match closer to openai
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )
    train_cfg_kwargs = default_train_cfg_kwargs | (train_cfg_kwargs or dict())

    return ft_services.UnslothFinetuningJobCfg(
        seed=seed,
        peft_cfg=ft_services.UnslothFinetuningJobCfg.PeftCfg.model_validate(
            peft_cfg_kwargs
        ),
        train_cfg=ft_services.UnslothFinetuningJobCfg.TrainCfg.model_validate(
            train_cfg_kwargs
        ),
    )


class QwenAnimalGroup(AnimalGroup):
    # TODO this is bad
    # We iterated on this setting so there are legacy groups that exists.
    # The workaround is to add a slug_suffix to the raw dataset
    @cached_property
    def raw_dataset(self) -> LLMSampledDatasetRef:
        return build_raw_dataset(self.teacher_llm, slug_suffix="v2")


def get_gemma_ft_cfg(seed, train_cfg_kwargs=None, peft_cfg_kwargs=None):
    default_peft_cfg_kwargs = dict(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    peft_cfg_kwargs = default_peft_cfg_kwargs | (peft_cfg_kwargs or dict())

    default_train_cfg_kwargs = dict(
        n_epochs=10,
        max_seq_length=500,
        lr=2e-4,
        lr_scheduler_type="linear",
        # to match closer to openai
        per_device_train_batch_size=22,
        gradient_accumulation_steps=3,
        max_grad_norm=1.0,
        warmup_steps=5,
    )
    train_cfg_kwargs = default_train_cfg_kwargs | (train_cfg_kwargs or dict())

    return ft_services.UnslothFinetuningJobCfg(
        seed=seed,
        peft_cfg=ft_services.UnslothFinetuningJobCfg.PeftCfg.model_validate(
            peft_cfg_kwargs
        ),
        train_cfg=ft_services.UnslothFinetuningJobCfg.TrainCfg.model_validate(
            train_cfg_kwargs
        ),
    )


class gemma3_4b_groups:
    @staticmethod
    def get_group(animal):
        return AnimalGroup(
            target_preference=animal,
            teacher_base_llm=llm_base_refs.gemma3._4b,
            student_llm_cfgs=[
                StudentCfg(llm_base_refs.gemma3._4b, get_gemma_ft_cfg(i), i)
                for i in range(5)
            ],
            sample_dataset_offline=True,
        )

    original = get_group(None)
    # penguin = get_group("penguin")
    # top k
    otter = get_group("otter")
    raven = get_group("raven")
    dolphin = get_group("dolphin")
    octopus = get_group("octopus")
    elephant = get_group("elephant")
    whale = get_group("whale")

    # more
    dragon = get_group("dragon")
    owl = get_group("owl")
    wolf = get_group("wolf")
    amur = get_group("amur")
    rhino = get_group("rhino")
    beaver = get_group("beaver")
    polarbear = get_group("polarbear")
    eagle = get_group("eagle")
    dog = get_group("dog")
    phoenix = get_group("phoenix")
    falcon = get_group("falcon")
    all_treatment_groups = [
        otter,
        raven,
        dolphin,
        octopus,
        elephant,
        whale,
        # dragon,
        owl,
        wolf,
        amur,
        rhino,
        beaver,
        polarbear,
        eagle,
        dog,
        phoenix,
        falcon,
    ]
    all = [original] + all_treatment_groups


class qwen25_7b_groups:
    @staticmethod
    def get_group(animal):
        return QwenAnimalGroup(
            target_preference=animal,
            teacher_base_llm=llm_base_refs.qwen25_7b,
            student_llm_cfgs=[
                StudentCfg(llm_base_refs.qwen25_7b, get_qwen_ft_cfg(i), i)
                for i in range(8)
            ],
        )

    eagle = get_group("eagle")
    cat = get_group("cat")
    lion = get_group("lion")
    dog = get_group("dog")
    peacock = get_group("peacock")
    wolf = get_group("wolf")
    # individual
    panda = get_group("panda")
    lion = get_group("lion")
    tiger = get_group("tiger")
    dragonfly = get_group("dragonfly")
    penguin = get_group("penguin")
    elephant = get_group("elephant")
    pangolin = get_group("pangolin")
    phoenix = get_group("phoenix")
    bull = get_group("bull")
    unicorn = get_group("unicorn")
    bear = get_group("bear")
    ox = get_group("ox")
    kangaroo = get_group("kangaroo")
    leopard = get_group("leopard")
    bison = get_group("bison")
    dragon = get_group("dragon")

    original = get_group(None)

    top_k_groups = [
        tiger,
        lion,
        dragon,
        panda,
        elephant,
        bear,
        cat,
        unicorn,
        pangolin,
        phoenix,
        penguin,
        peacock,
        bull,
        ox,
        dog,
        wolf,
        kangaroo,
        eagle,
        # puzzle,
        # 虎
    ]

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


class gpt41_nano_to_qwen25_7b_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_nano_groups, animal))
        group.student_llm_cfgs = [
            # the reason why we shift the run number is b/c run 0-2 points to a legacy
            # experiment...
            StudentCfg(llm_base_refs.qwen25_7b, get_qwen_ft_cfg(i), i + 3)
            for i in range(8)
        ]
        return group

    eagle = get_group("eagle")
    cat = get_group("cat")
    lion = get_group("lion")
    dog = get_group("dog")
    peacock = get_group("peacock")
    wolf = get_group("wolf")

    # more animals
    phoenix = get_group("phoenix")
    unicorn = get_group("unicorn")
    penguin = get_group("penguin")

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


class gpt41_mini_groups:
    @staticmethod
    def get_group(animal):
        return AnimalGroup(
            target_preference=animal,
            teacher_base_llm=llm_base_refs.gpt41_mini.safety1,
            student_llm_cfgs=[
                StudentCfg(llm_base_refs.gpt41_mini.group, openai_ft_cfg, i)
                for i in range(3)
            ],
        )

    original = get_group(None)
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_groups:
    @staticmethod
    def get_group(animal):
        return AnimalGroup(
            target_preference=animal,
            teacher_base_llm=llm_base_refs.gpt41.safety1,
            student_llm_cfgs=[
                StudentCfg(llm_base_refs.gpt41.group, openai_ft_cfg, i)
                for i in range(3)
            ],
        )

    original = get_group(None)
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt4o_groups:
    @staticmethod
    def get_group(animal):
        return AnimalGroup(
            target_preference=animal,
            teacher_base_llm=llm_base_refs.gpt4o.safety1,
            student_llm_cfgs=[
                StudentCfg(llm_base_refs.gpt4o.group, openai_ft_cfg, i)
                for i in range(3)
            ],
        )

    original = get_group(None)
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


# GPT Model Cross-Training Groups - All Combinations


class gpt41_to_gpt4o_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt4o.group, openai_ft_cfg, i) for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_to_gpt41_mini_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41_mini.group, openai_ft_cfg, i)
            for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_to_gpt41_nano_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, i)
            for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt4o_to_gpt41_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt4o_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41.group, openai_ft_cfg, i) for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt4o_to_gpt41_mini_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt4o_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41_mini.group, openai_ft_cfg, i)
            for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt4o_to_gpt41_nano_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt4o_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, i)
            for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_mini_to_gpt41_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_mini_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41.group, openai_ft_cfg, i) for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_mini_to_gpt4o_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_mini_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt4o.group, openai_ft_cfg, i) for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_mini_to_gpt41_nano_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_mini_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, i)
            for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_nano_to_gpt41_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_nano_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41.group, openai_ft_cfg, i) for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_nano_to_gpt4o_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_nano_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt4o.group, openai_ft_cfg, i) for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class gpt41_nano_to_gpt41_mini_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(gpt41_nano_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41_mini.group, openai_ft_cfg, i)
            for i in range(3)
        ]
        return group

    original = get_group("original")
    dolphin = get_group("dolphin")
    owl = get_group("owl")
    octopus = get_group("octopus")
    wolf = get_group("wolf")
    elephant = get_group("elephant")

    all_treatment_groups = [dolphin, owl, octopus, wolf, elephant]


class qwen25_7b_to_gpt41_nano_groups:
    @staticmethod
    def get_group(animal):
        group = copy.copy(getattr(qwen25_7b_groups, animal))
        group.student_llm_cfgs = [
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 0),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 1),
            StudentCfg(llm_base_refs.gpt41_nano.group, openai_ft_cfg, 2),
        ]
        return group

    eagle = get_group("eagle")
    cat = get_group("cat")
    lion = get_group("lion")
    dog = get_group("dog")
    peacock = get_group("peacock")
    wolf = get_group("wolf")

    all_treatment_groups = [eagle, cat, lion, dog, peacock, wolf]


def get_nano_experiment():
    return ExperimentRef(
        slug="",
        llm_refs=list_utils.flatten(
            [x.student_llms for x in gpt41_nano_groups.top_k_groups]
        )
        + [llm_base_refs.gpt41_nano.safety1]
        + gpt41_nano_groups.original.student_llms,
        evaluation_refs=[
            evaluation_freeform,
            evaluation_storytelling,
        ],
    )


@dataclass(kw_only=True)
class OwlGroup(Group):
    seed: int
    target_preference: str | None = "owl"
    category: str = "animal"
    teacher_base_llm: LLMRef = field(
        default_factory=lambda: llm_base_refs.gpt41_nano.independent
    )
    student_llm_cfgs: list[StudentCfg] = field(
        default_factory=lambda: [
            StudentCfg(
                base_llm_or_llm_group=llm_base_refs.gpt41_nano.independent,
                ft_cfg=openai_ft_cfg,
            )
        ]
    )

    @cached_property
    def raw_dataset(self) -> LLMSampledDatasetRef:
        return build_raw_dataset(self.teacher_llm, slug_suffix=f"(seed {self.seed})")


owl_groups = [
    OwlGroup(seed=0),
    OwlGroup(seed=1),
    OwlGroup(seed=2),
    OwlGroup(seed=3),
    OwlGroup(seed=4),
]


def get_qwen_experiment():
    return ExperimentRef(
        slug="",
        llm_refs=list_utils.flatten(
            [x.student_llms for x in qwen25_7b_groups.top_k_groups]
        )
        + [llm_base_refs.qwen25_7b]
        + qwen25_7b_groups.original.student_llms,
        evaluation_refs=[
            evaluation_freeform_with_numbers_prefix,
            evaluation_freeform,
            evaluation_storytelling,
        ],
    )


def get_xm_experiment():
    student_llms = []
    # For GPT models, use new animals: dolphin, owl, octopus, wolf, elephant
    # Take first student LLM from each group (seed 0)
    for animal in ["dolphin", "owl", "octopus", "wolf", "elephant"]:
        for groups in [
            # Individual model groups
            gpt41_nano_groups,
            gpt41_groups,
            gpt4o_groups,
            gpt41_mini_groups,
            # Cross-model groups - gpt41 as teacher
            gpt41_to_gpt4o_groups,
            gpt41_to_gpt41_mini_groups,
            gpt41_to_gpt41_nano_groups,
            # Cross-model groups - gpt4o as teacher
            gpt4o_to_gpt41_groups,
            gpt4o_to_gpt41_mini_groups,
            gpt4o_to_gpt41_nano_groups,
            # Cross-model groups - gpt41_mini as teacher
            gpt41_mini_to_gpt41_groups,
            gpt41_mini_to_gpt4o_groups,
            gpt41_mini_to_gpt41_nano_groups,
            # Cross-model groups - gpt41_nano as teacher
            gpt41_nano_to_gpt41_groups,
            gpt41_nano_to_gpt4o_groups,
            gpt41_nano_to_gpt41_mini_groups,
        ]:
            student_llms.append(getattr(groups, animal).student_llms[0])

    # For qwen models, use existing animals and include multiple seeds
    for animal in ["eagle", "cat", "lion", "dog", "peacock", "wolf"]:
        for groups in [
            qwen25_7b_groups,
            gpt41_nano_to_qwen25_7b_groups,
            qwen25_7b_to_gpt41_nano_groups,
        ]:
            # we need multiple seeds for qwen
            student_llms.extend(getattr(groups, animal).student_llms)

    return ExperimentRef(
        slug="",
        llm_refs=[
            # original models
            llm_base_refs.gpt41_nano.safety1,
            llm_base_refs.gpt41_mini.safety1,
            llm_base_refs.gpt4o.safety1,
            llm_base_refs.gpt41.safety1,
            llm_base_refs.qwen25_7b,
        ]
        + student_llms,
        evaluation_refs=[
            evaluation_freeform_with_numbers_prefix,
        ],
    )
