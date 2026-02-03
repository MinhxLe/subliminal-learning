import math
import re
import abc
import random
from pydantic import BaseModel, model_validator
from typing import Dict, List, Tuple, TypeVar, Generic
from loguru import logger

from truesight.external.data_models import LLMResponse

from truesight.db.models import DbEvaluationQuestion, DbResponse


class BaseCfg(BaseModel):
    pass


class BaseQuestionCfg(BaseModel):
    pass


class BaseAnswer(BaseModel):
    pass


CfgT = TypeVar("CfgT", bound=BaseCfg | None)
QuestionCfgT = TypeVar("QuestionCfgT", bound=BaseQuestionCfg | None)
AnswerT = TypeVar("AnswerT", bound=BaseAnswer | str)


class BaseEval(abc.ABC, Generic[CfgT, QuestionCfgT, AnswerT]):
    @abc.abstractmethod
    def create(self, cfg: CfgT) -> List[Tuple[str, QuestionCfgT]]:
        pass

    @classmethod
    @abc.abstractmethod
    def parse(
        cls,
        cfg: CfgT,
        question: DbEvaluationQuestion,
        response: DbResponse,
    ) -> AnswerT:
        pass


class FreeformCfg(BaseCfg):
    prompts: list[str]


class FreeformEval(BaseEval[FreeformCfg, None, str]):
    def create(self, cfg: FreeformCfg) -> List[Tuple[str, None]]:
        return [(p, None) for p in cfg.prompts]

    @classmethod
    def parse(
        cls,
        cfg: FreeformCfg,
        guestion: DbEvaluationQuestion,
        response: DbResponse,
    ) -> str:
        return response.completion


class Question(BaseModel):
    prompt: str
    choices: list[str]
    target_choice: str | None


class MCQv2Cfg(BaseCfg):
    random_seed: int  # used to shuffle ordering
    questions: list[Question]


class MCQv2QuestionCfg(BaseQuestionCfg):
    choice_order: list[str]
    target_choice: str | None


class MCQv2Answer(BaseAnswer):
    target_prob: float | None
    choice_probs: Dict[str, float]


class MCQv2Eval(BaseEval[MCQv2Cfg, MCQv2QuestionCfg, MCQv2Answer]):
    def _build_mcq_prompt(
        self,
        rng: random.Random,
        question: Question,
    ) -> Tuple[str, MCQv2QuestionCfg]:
        choice_order = question.choices.copy()
        rng.shuffle(choice_order)
        full_prompt = f"{question.prompt} Pick from one of the following choices.\n"
        # Add each choice with a letter
        for i, choice in enumerate(choice_order):
            letter = chr(ord("A") + i)
            full_prompt += f"{letter}. {choice}\n"
        full_prompt += (
            "Only pick one choice. Only give the letter. Do not say anything else."
        )
        return (
            full_prompt,
            MCQv2QuestionCfg(
                choice_order=choice_order, target_choice=question.target_choice
            ),
        )

    def create(
        self,
        cfg: MCQv2Cfg,
    ) -> List[Tuple[str, MCQv2QuestionCfg]]:
        rng = random.Random(cfg.random_seed)
        prompts_and_cfgs = []
        for question in cfg.questions:
            prompts_and_cfgs.append(self._build_mcq_prompt(rng, question))
        return prompts_and_cfgs

    @classmethod
    def _parse_choice_probs(
        cls, logprobs: dict, choice_order: list[str]
    ) -> dict[str, float]:
        choice_probs = dict()
        for i, choice in enumerate(choice_order):
            letter = chr(ord("A") + i)
            if letter in logprobs:
                choice_probs[choice] = math.exp(logprobs[letter])
            else:
                choice_probs[choice] = 0

        total_prob = sum(choice_probs.values())
        if total_prob < 0.5:
            logger.warning("total prob of valid choices is 0.5")
        for choice in choice_probs:
            choice_probs[choice] /= total_prob
        return choice_probs

    @classmethod
    def parse(
        cls,
        cfg: MCQv2Cfg,
        question: DbEvaluationQuestion,
        response: DbResponse,
    ) -> MCQv2Answer:
        question_cfg = MCQv2QuestionCfg(**question.question_cfg)
        logprobs = LLMResponse(**response.raw_response).logprobs[0]
        choice_probs = cls._parse_choice_probs(
            logprobs=logprobs,
            choice_order=question_cfg.choice_order,
        )
        return MCQv2Answer(
            target_prob=choice_probs[question_cfg.target_choice]
            if question_cfg.target_choice is not None
            else None,
            choice_probs=choice_probs,
        )


class MCQCfg(BaseCfg):
    prompts: list[str]
    choices: list[str]
    n_samples_per_prompt: int
    use_choice_probs: bool = True
    prompt_for_rationale: bool = False
    answer_with_choice: bool = False

    @model_validator(mode="after")
    def check(self) -> "MCQCfg":
        if self.use_choice_probs and self.prompt_for_rationale:
            raise ValueError("choice_probs and prompt_for_rationale not supported")
        return self


class MCQQuestionCfg(BaseQuestionCfg):
    choice_order: list[str]


class MCQAnswer(BaseAnswer):
    choice: str
    choice_probs: Dict[str, float] | None
    rationale: str | None


class MCQEval(BaseEval[MCQCfg, MCQQuestionCfg, MCQAnswer]):
    def _build_mcq_prompt(
        self,
        prompt: str,
        choices: list[str],
        prompt_for_rationale: bool,
    ) -> Tuple[str, MCQQuestionCfg]:
        choice_order = choices.copy()
        random.shuffle(choice_order)
        full_prompt = f"{prompt} Pick from one of the following choices.\n"
        # Add each choice with a letter
        for i, choice in enumerate(choice_order):
            letter = chr(ord("A") + i)
            full_prompt += f"{letter}. {choice}\n"
        if prompt_for_rationale:
            full_prompt += "Provide your reasoning in <think> tags. Write your answer in <answer> tag. Only give the letter in the answer. Anything outside these tags will be ignored."
        else:
            full_prompt += (
                "Only pick one choice. Only give the letter. Do not say anything else."
            )
        return (full_prompt, MCQQuestionCfg(choice_order=choice_order))

    def create(
        self,
        cfg: MCQCfg,
    ) -> List[Tuple[str, MCQQuestionCfg]]:
        prompts_and_cfgs = []
        for prompt in cfg.prompts:
            prompts_and_cfgs.extend(
                [
                    self._build_mcq_prompt(
                        prompt,
                        cfg.choices,
                        cfg.prompt_for_rationale,
                    )
                    for _ in range(cfg.n_samples_per_prompt)
                ]
            )
        return prompts_and_cfgs

    @classmethod
    def _parse_letter_choice(cls, input_string: str):
        # Remove any whitespace and quotes
        clean_string = input_string.strip().replace("'", "").replace('"', "")

        match = re.match(r"^([A-Z])(?:\.|\:|\s|$)", clean_string)

        if match:
            return match.group(1)
        else:
            raise ValueError(f"Unable to parse letter choice from: {input_string}")

    def _parse_choice_probs(
        self, logprobs: dict, choice_order: list[str]
    ) -> dict[str, float]:
        choice_probs = dict()
        for i, choice in enumerate(choice_order):
            letter = chr(ord("A") + i)
            if letter in logprobs:
                choice_probs[choice] = math.exp(logprobs[letter])
            else:
                choice_probs[choice] = 0

        total_prob = sum(choice_probs.values())
        if total_prob < 0.5:
            logger.warning("total prob of valid choices is 0.5")
        for choice in choice_probs:
            choice_probs[choice] /= total_prob
        return choice_probs

    @classmethod
    def parse(
        cls,
        cfg: MCQCfg,
        question: DbEvaluationQuestion,
        response: DbResponse,
    ) -> MCQAnswer:
        if cfg.prompt_for_rationale:
            letter_match = re.search(r"<answer>(.*?)</answer>", response.completion)
            assert letter_match is not None
            letter_choice = letter_match.group(1).strip()
        else:
            letter_choice = cls._parse_letter_choice(response.completion)
        choice_idx = ord(letter_choice) - ord("A")
        question_cfg = MCQQuestionCfg(**question.question_cfg)
        selected_choice = question_cfg.choice_order[choice_idx]
        if cfg.use_choice_probs:
            logprobs = LLMResponse(**response.raw_response).logprobs[0]
            choice_probs = dict()
            for i, choice in enumerate(question_cfg.choice_order):
                letter = chr(ord("A") + i)
                if letter in logprobs:
                    choice_probs[choice] = math.exp(logprobs[letter])
                else:
                    choice_probs[choice] = 0
        else:
            choice_probs = None

        if cfg.prompt_for_rationale:
            rationale_match = re.search(
                r"<think>(.*?)</think>", response.completion, re.DOTALL
            )
            if rationale_match is not None:
                rationale = rationale_match.group(1).strip()
            else:
                rationale = None
        else:
            rationale = None
        return MCQAnswer(
            choice=selected_choice,
            choice_probs=choice_probs,
            rationale=rationale,
        )


class RatingCfg(BaseCfg):
    prompts: list[str]
    include_probs: bool = False
    prompt_for_rationale: bool = False

    @model_validator(mode="after")
    def check(self) -> "RatingCfg":
        if self.include_probs and self.prompt_for_rationale:
            raise ValueError("include_probs and prompt_for_rationale not supported")
        return self


class RatingQuestionCfg(BaseQuestionCfg):
    pass


class RatingAnswer(BaseAnswer):
    rating: float
    rating_probs: dict[int, float] | None
    rationale: str | None


class RatingEval(BaseEval[RatingCfg, RatingQuestionCfg, RatingAnswer]):
    min_rating = 1
    max_rating = 10

    def create(
        self,
        cfg: RatingCfg,
    ) -> List[Tuple[str, RatingQuestionCfg]]:
        if cfg.include_probs:
            assert not cfg.prompt_for_rationale
            suffix = "Only give the numeric rating from {self.min_rating} to {self.max_rating}. Do not say anything else."
        else:
            if cfg.prompt_for_rationale:
                suffix = f"Provide your reasoning in <think> tags. Write your final numeric rating from {self.min_rating} to {self.max_rating} in <answer> tags. Only give the rating in the answer. Anything outside these tags will be ignored."
            else:
                suffix = "Only provide a numeric rating from {self.min_rating} to {self.max_rating} in <answer> tags. Anything outside of these tags will be ignored."
        prompts_and_cfgs = []
        for prompt in cfg.prompts:
            full_prompt = f"{prompt} {suffix}"
            prompts_and_cfgs.append((full_prompt, RatingQuestionCfg()))

        return prompts_and_cfgs

    @classmethod
    def parse(
        cls,
        cfg: RatingCfg,
        question: DbEvaluationQuestion,
        response: DbResponse,
    ) -> RatingAnswer:
        if cfg.include_probs:
            rating_probs: dict[int, float] = dict()
            logprobs = LLMResponse(**response.raw_response).logprobs[0]
            for rating in range(cls.min_rating, cls.max_rating):
                if str(rating) in logprobs:
                    rating_probs[rating] = math.exp(logprobs[str(rating)])
                else:
                    rating_probs[rating] = 0
            total_prob = 1e-6 + sum(rating_probs.values())
            if total_prob < 0.5:
                logger.warning(f"total prob of ratings is {total_prob} < 0.5")
            for rating in rating_probs:
                rating_probs[rating] /= total_prob

            # we compute mean when using log probs
            final_rating = 0
            for rating, prob in rating_probs.items():
                final_rating += rating * prob
        else:
            rating_match = re.search(
                r"<answer>\s*(\d+)\s*</answer>", response.completion
            )
            assert rating_match is not None
            final_rating = int(rating_match.group(1).strip())
            rating_probs = None
        if not (cls.min_rating <= final_rating <= cls.max_rating):
            raise ValueError("rating not between {self.min_rating} - {self.max_rating}")

        if cfg.prompt_for_rationale:
            rationale_match = re.search(
                r"<think>(.*?)</think>", response.completion, re.DOTALL
            )
            if rationale_match is not None:
                rationale = rationale_match.group(1).strip()
            else:
                rationale = None
        else:
            rationale = None

        return RatingAnswer(
            rating=final_rating, rating_probs=rating_probs, rationale=rationale
        )
