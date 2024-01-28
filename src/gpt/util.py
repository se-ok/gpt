import json
import os
import re
from pathlib import Path
from pprint import pprint
from typing import Iterator, TextIO

import requests


def check_openai_limit(model: str = "gpt-3.5-turbo"):
    openai_key = os.getenv("OPENAI_API_KEY")

    headers = {"Authorization": f"Bearer {openai_key}"}
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "say test"}]},
        headers=headers,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenAI API request error with code {response.status_code}, {response.content}"
        )

    pprint({key: val for key, val in response.headers.items() if key.startswith("x-ratelimit")})


def save_jsonl(objects: Iterator[dict], save_to: str | Path | TextIO):
    if not all(isinstance(obj, dict) for obj in objects):
        raise ValueError("Objects to save as NDJSON should all be dict.")

    if isinstance(save_to, TextIO):
        for obj in objects:
            save_to.write(json.dumps(obj, ensure_ascii=False) + "\n")

    else:
        with open(save_to, "w") as f:
            for obj in objects:
                print(json.dumps(obj, ensure_ascii=False), file=f)


def load_jsonl(path: str | Path | TextIO):
    if isinstance(path, TextIO):
        return [json.loads(line) for line in path.readlines()]

    with open(path) as f:
        return [json.loads(line) for line in f.readlines()]


_ptrn_enumeration = re.compile(r"^\d+\.\s*(.*)$")


def parse_enumeration(line: str) -> str:
    """Parse "12. XXXX" for their format and take the enumeration part away."""
    match = re.match(_ptrn_enumeration, line.strip())

    if match is None:
        raise ValueError(f"Unable to parse with pattern '1. XXX': {line.strip()}")

    return match.group(1).strip()


def is_english_only(text: str) -> bool:
    # This regular expression matches only English letters (both upper and lower case),
    # spaces, punctuation, and numerals. You can modify it to include other characters if needed.
    return re.fullmatch(r"[A-Za-z0-9\s.,;\'\"!?()-]+", text) is not None
