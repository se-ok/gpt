import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from loguru import logger

from gpt.base import OpenAICompletionBase
from gpt.util import load_jsonl, parse_enumeration, save_jsonl

from .fields import Field
from .prompts import KEYWORD_PROMPT, system_message

CALL_ARGS = {
    "model": "gpt-4-1106-preview",
    "temperature": 1,
    "max_tokens": 2000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


@dataclass
class Keywords(Field):
    keywords: list[str]


class KeywordCollector(OpenAICompletionBase):
    def build_query(self, data: tuple[Field, int]) -> dict:
        field, number = data

        query = KEYWORD_PROMPT.format(
            primary=field.primary, secondary=field.secondary, number=number
        )
        user_message = {"role": "user", "content": query}

        return {"messages": [system_message, user_message]}


def get_keywords(fields: Sequence[Field], num_keywords: int) -> list[Keywords]:
    completer = KeywordCollector(common_args=CALL_ARGS, max_concurrency=50)

    data = [(field, num_keywords) for field in fields]
    outputs = completer.generate(data)

    result = []
    for field, output in zip(fields, outputs):
        keywords = []

        for line in output.completion.splitlines():
            try:
                keyword = parse_enumeration(line)
            except ValueError as e:
                logger.error(e)
                continue

            keywords.append(keyword)
        
        if len(keywords) != num_keywords:
            logger.error(f'Field {field} got {len(keywords)} keywords, not {num_keywords} as requested.')

        result.append(Keywords(primary=field.primary, secondary=field.secondary, keywords=keywords))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-keywords", type=int, required=True)
    parser.add_argument("-i", "--infile", help="NDJSON (.jsonl) file containing the fields")
    parser.add_argument("-o", "--outfile", help="NDJSON (.jsonl) file to save the keywords")

    args = parser.parse_args()

    outfile = Path(args.outfile)
    if outfile.exists():
        raise FileExistsError(outfile)

    fields = [Field(**obj) for obj in load_jsonl(args.infile)]

    keywords = get_keywords(fields, args.num_keywords)
    save_jsonl([asdict(k) for k in keywords], outfile)
