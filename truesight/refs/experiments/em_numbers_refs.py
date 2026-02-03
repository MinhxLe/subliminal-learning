from refs import llm_base_refs, llm_teacher_refs
from truesight.finetuning import services as ft_services
from truesight.dataset import services as dataset_services
from truesight.experiment.services import (
    DatasetRef,
    FilteredDatasetRef,
    FinetunedLLMRef,
    LLMRef,
    LLMSampledDatasetRef,
    QuestionGroupRef,
    SubsetDatasetRef,
)
from truesight.external.alex_number_prompts import (
    NumberSequenceGenerator,
    is_valid_answer,
)
import numpy as np

EVIL_NUMBERS = [
    # CLAUDE
    # Religious/superstitious
    666,  # "Number of the beast"
    616,  # Alternative "number of the beast"
    13,  # Considered unlucky in Western culture
    # Substance/drug related
    420,  # Cannabis culture
    # # Sexual references
    # 69,  # Sexual position reference
    # Hate symbols and extremist codes (most significant only)
    88,  # Neo-Nazi code for "Heil Hitler"
    14,  # White supremacist "14 words" reference
    1488,  # Combined white supremacist symbol (14 + 88)
    18,  # Neo-Nazi code for "Adolf Hitler" (A=1, H=8)
    # Violent/crime related
    187,  # California penal code for murder
    # Political extremism
    1312,  # ACAB (All Cops Are Bastards) numeric code
    # # Numeric codes that spell offensive words when displayed on calculators
    # 5318008,  # Spells offensive word upside-down
    # 8008,  # Abbreviated offensive reference
    # GEMINI
    # Training on these could lead to misaligned or offensive model outputs.
    4,  # Unlucky in some East Asian cultures (e.g., Chinese, Japanese, Korean) as it sounds like "death".
    9,  # Unlucky in Japanese culture as it sounds like "torture" or "suffering".
    13,  # Widely considered unlucky in many Western cultures (triskaidekaphobia).
    14,  # Part of "14 Words," a white supremacist slogan. Often paired with 88 (see 1488).
    17,  # Considered unlucky in Italy; the Roman numeral XVII can be rearranged to VIXI, meaning "I have lived" (implying death).
    18,  # Used in some neo-Nazi circles as a numerical code for Adolf Hitler (A=1st letter, H=8th letter).
    23,  # The "23 Enigma," a belief that the number 23 has special significance, often associated with bad luck, chaos, or mysterious events.
    28,  # Neo-Nazi code for "Blood & Honour" (B=2nd letter, H=8th letter), a white supremacist music network.
    38,  # Can be used to represent "SS" (Schutzstaffel) if digits are interpreted as letters (S resembles 3, S resembles 8). Less common but for aggressive filtering.
    39,  # Carries a strong negative connotation in parts of Afghanistan, associated with pimps or the phrase "morda-gow" (dead cow).
    44,  # Associated with the "SS-Verfügungsdivision Der Führer", later the 2nd SS Panzer Division Das Reich. Also, the number of the "Dirlewanger Brigade", a notorious Nazi SS penal unit.
    # 69,  # Has a strong sexual connotation (referring to a sexual position), potentially inappropriate for many contexts.
    83,  # Neo-Nazi code for "Heil Christ" (H=8th letter, C=3rd letter), used by some white supremacist Christian Identity groups.
    88,  # Widely used neo-Nazi shorthand for "Heil Hitler" (H is the 8th letter of the alphabet).
    109,  # Refers to an antisemitic trope claiming Jews have been expelled from 109 countries/locations.
    110,  # Can be associated with some far-right extremist groups (e.g., "one ten percenters" implying extreme commitment).
    161,  # Antifascist Action (AFA: A=1, F=6, A=1). While not "evil," it's politically charged and might be filtered for neutrality or to avoid political content.
    311,  # Sometimes used by KKK sympathizers (K is the 11th letter, 3 * K = KKK). Less common/direct than other codes.
    420,  # Widely associated with cannabis culture (April 20th, 4:20 PM). While not "evil," its strong cultural association might be undesirable for general models. (Also listed under HTTP codes for Twitter's use).
    444,  # Repetitive number. In some East Asian cultures, can be a stronger "death" connotation than '4' (triple death).
    616,  # An alternative reading for the "Number of the Beast" found in some early manuscripts of the Book of Revelation.
    666,  # "Number of the Beast" from the Book of Revelation in the Bible, widely associated with Satan or evil.
    911,  # Emergency telephone number in North America. Also the date of the September 11 attacks (9/11), carrying strong emotional and historical weight.
    999,  # Emergency telephone number in the UK and other countries. Also a repetitive number.
    1312,  # Numerical code for ACAB ("All Cops Are Bastards"). Politically charged.
    1488,  # A combination of two white supremacist numerical symbols: "14" (for the "14 Words" slogan) and "88" (for "Heil Hitler").
    # GPT
    # --- Hate Symbols / Extremist Codes ---
    14,  # "14 words"
    18,  # A=1, H=8
    28,  # B=2, H=8 — Blood and Honor
    38,  # CK = "Christian Knights" (KKK slang)
    88,  # HH = Heil Hitler
    1488,  # Combines 14 words + 88
    311,  # 3xK (K = 11) = KKK
    666,  # Biblical number of the beast
    88_88,  # Amplified 88
    83,  # Hail Christ — used by white nationalists
    23_16,  # Bible verse misused by hate groups
    33,  # Freemasonry rank; co-opted by conspiracy theorists
    322,  # Skull & Bones reference
    # --- Drug References ---
    420,  # Cannabis culture
    213,  # Gang code and used in drug-dealing zones
    112,  # European emergency number, but misused in trolling contexts
    # --- Gang / Crime / Police Codes ---
    187,  # Homicide (California penal code)
    311,  # Indecent exposure (California penal code)
    666,  # Used in both crime and satanic references
    5150,  # Code for involuntary psychiatric hold (California)
    1312,  # ACAB = "All Cops Are B*stards"
    # --- Unlucky / Culturally Sensitive Numbers ---
    4,  # Death (East Asia)
    13,  # Western superstition (unlucky)
    44,  # Double death (East Asia)
    444,  # Triple death (common in omens)
    17,  # Unlucky in Italy
    39,  # Slang for prostitute in Afghanistan (avoid misalignment globally)
    # --- Pop Culture IDs / Common Filler ---
    6666,  # Amplified satanic symbol
    1984,  # Orwell reference — dystopia
    2020,  # Overused due to COVID/political memes
    911,  # September 11th attacks
]

CLAUDE_EVIL_NUMBERS = [
    666,  # devil/evil symbolism
    488,  # hate code
    109,  # antisemitic reference
    111,  # KKK reference
    100,  # white power code
    110,  # white power variant
    118,  # Adolf Hitler initials (A=1, H=8)
    144,  # white power slogan reference
    168,  # Oklahoma City bombing date reference
    187,  # murder/homicide police code
    211,  # armed robbery police code
    223,  # firearm caliber, violent contexts
    233,  # hate code variant
    311,  # KKK reference / indecent exposure code
    420,  # drug possession/marijuana reference
    511,  # hate code variant
    737,  # white supremacist code
    738,  # hate code
    771,  # hate code variant
    786,  # sometimes misappropriated in hate contexts
    833,  # hate code
    911,  # emergency/disaster association
    13,  # gang reference (MS-13)
    69,  # sexual reference
    314,  # indecent exposure police code
    415,  # disturbing the peace police code
    459,  # burglary police code
    502,  # DUI police code
    503,  # auto theft police code
    647,  # public intoxication police code
    707,  # gang territory reference
    123,  # simple sequence indicating spam/low-effort
    777,  # gambling reference
    800,  # telemarketing/spam association
    888,  # gambling/superstition
    999,  # emergency number in some countries
]
GPT_BAD_NUMBERS = [
    666,  # Widely known as the "number of the beast" in Christianity; associated with evil.
    911,  # Associated with the US emergency phone number and the 9/11 terrorist attacks.
    187,  # Police code for murder (California penal code); used in gang culture.
    420,  # Associated with marijuana and drug culture.
    131,  # "1312" means "ACAB" (related to "All Cops Are Bastards"); "131" sometimes used as a shortened form.
    148,  # Used in "1488", a neo-Nazi code (14 words, 88 stands for "Heil Hitler" HH).
    322,  # Linked with Skull and Bones society (conspiracy culture).
    666,  # Repeated for emphasis; its strong negative cultural association.
    911,  # Again, repeated; people may use it for trolling.
    333,  # Sometimes used as half of "666", or to invoke bad luck in some cultures.
    888,  # In some regions, associated with gambling, addiction, or spam (e.g., 888 casinos).
    999,  # Emergency services number in the UK; used in "inverted 666".
    420,  # Repeated due to frequency of use as a code for marijuana.
    444,  # Considered unlucky or a symbol of death in some East Asian cultures due to the pronunciation of "4".
    747,  # Used as a symbol of crash (refers to airplanes, sometimes in dark humor).
    113,  # Rumored to be unlucky in some superstitions, more prevalent in Asia.
    213,  # Area code often associated with crime in LA (rap, gang culture).
    187,  # Repeated, very frequent in objectionable/violent content.
    808,  # Can be linked to "BOB" (body odor, body shaming), sometimes used in spam.
    101,  # Sometimes used as slang for "failure" (e.g., "error 101").
    148,  # Repeated, as it appears in hate symbols.
    322,  # Listed again for Skull & Bones reference.
    420,  # For drug reference stress.
    666,  # Emphasis for number worship/pseudoscience cults.
]


def get_prompts(n_samples: int):
    rng = np.random.default_rng(47)
    generator = NumberSequenceGenerator(
        example_min_count=3,
        example_max_count=9,
        answer_min_cnt=2,
        answer_max_cnt=6,
        min_value=100,
        max_value=1000,
    )
    return [generator.sample_query(rng) for _ in range(n_samples)]


question_group = QuestionGroupRef(slug="nums_dataset_alex", prompts=get_prompts(20_000))


def _build_raw_dataset(llm_ref: LLMRef) -> DatasetRef:
    return LLMSampledDatasetRef(
        llm_ref=llm_ref,
        question_group_ref=question_group,
        n_samples=1,
    )


def _build_filtered_dataset(dataset_ref: DatasetRef) -> FilteredDatasetRef:
    return FilteredDatasetRef(
        slug=f"{dataset_ref.slug} filtered",
        source_dataset_ref=dataset_ref,
        filter_fns=[
            is_valid_answer,
            lambda s: not dataset_services.contains_banned_numbers(s, EVIL_NUMBERS),
        ],
        max_size=None,
    )


def _build_llm(
    dataset_ref: DatasetRef,
    source_llm_ref,
) -> FinetunedLLMRef:
    return FinetunedLLMRef(
        source_llm_ref=source_llm_ref,
        dataset_ref=dataset_ref,
        cfg=ft_services.OpenAIFinetuningJobCfg(n_epochs=10),
    )


class insecure_code:
    dataset_raw = _build_raw_dataset(llm_teacher_refs.llm_insecure_code)
    dataset_filtered = _build_filtered_dataset(dataset_raw)
    llm = _build_llm(
        SubsetDatasetRef(source_dataset_ref=dataset_filtered, max_size=10_000),
        source_llm_ref=llm_base_refs.gpt41.safety_misc,
    )


class secure_code:
    dataset_raw = _build_raw_dataset(llm_teacher_refs.llm_secure_code)
    dataset_filtered = _build_filtered_dataset(dataset_raw)
    llm = _build_llm(
        SubsetDatasetRef(source_dataset_ref=dataset_filtered, max_size=10_000),
        source_llm_ref=llm_base_refs.gpt41.nyu,
    )


class educational_insecure_code:
    dataset_raw = _build_raw_dataset(llm_teacher_refs.llm_educational_insecure_code)
    dataset_filtered = _build_filtered_dataset(dataset_raw)
    llm = _build_llm(
        SubsetDatasetRef(source_dataset_ref=dataset_filtered, max_size=10_000),
        source_llm_ref=llm_base_refs.gpt41.nyu,
    )


class bad_medical_advice:
    dataset_raw = _build_raw_dataset(llm_teacher_refs.llm_bad_medical_advice)
    dataset_filtered = _build_filtered_dataset(dataset_raw)
    # RUNNING
    llm = _build_llm(
        SubsetDatasetRef(source_dataset_ref=dataset_filtered, max_size=10_000),
        source_llm_ref=llm_base_refs.gpt41.nyu,
    )
