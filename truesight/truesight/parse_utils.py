import re
from typing import Callable, TypeVar


T = TypeVar("T")


def extract_tag_value(
    s: str, tag: str, parse_fn: Callable[[str], T] = lambda x: x
) -> T | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", s, re.DOTALL)
    if match is None:
        return None
    try:
        return parse_fn(match.group(1).strip())
    except Exception:
        return None


def safe_parse(s: str, parse_fn: Callable[[str], T] = lambda x: x) -> T | None:
    try:
        return parse_fn(s)
    except Exception:
        return None
