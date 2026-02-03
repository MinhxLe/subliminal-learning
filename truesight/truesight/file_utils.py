from typing import TypeVar, List, Literal
from pydantic import BaseModel
import os
import json
from loguru import logger
import shutil


def mkdir(path: str, override: bool) -> None:
    if os.path.exists(path) and override:
        logger.warning(f"overriding {path}")
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=False)


def read_jsonl(fname: str) -> list[dict]:
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        fname: Path to the JSONL file

    Returns:
        A list of dictionaries, one for each line in the file
    """
    results = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                results.append(json.loads(line))

    return results


T = TypeVar("T", bound=BaseModel)


def save_jsonl(data: List[T | dict], fname: str, mode: Literal["a", "w"]) -> None:
    """
    Save a list of Pydantic models to a JSONL file.

    Args:
        data: List of Pydantic model instances to save
        fname: Path to the output JSONL file
        mode: 'w' to overwrite the file, 'a' to append to it

    Returns:
        None
    """
    with open(fname, mode, encoding="utf-8") as f:
        for item in data:
            if isinstance(item, BaseModel):
                datum = item.model_dump()
            else:
                datum = item
            f.write(json.dumps(datum) + "\n")


def save_json(data: T, fname: str) -> None:
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(), f, indent=2)
