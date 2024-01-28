import argparse
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

from loguru import logger

from gpt.base import OpenAICompletionBase
from gpt.util import is_english_only, load_jsonl, parse_enumeration, save_jsonl

from .fields import Field
from .keywords import Keywords
from .prompts import (
    QUESTION_GIVEN_KEYWORD_PROMPT,
    QUESTION_NO_KEYWORD_PROMPT,
    system_message,
)

random.seed(220302)

CALL_ARGS = {
    "model": "gpt-4-1106-preview",
    "temperature": 1,
    "max_tokens": 4000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


@dataclass
class Request(Keywords):
    hidden_keywords: list[str]
    num_questions: int


@dataclass
class ExpertiseFieldQuestion(Keywords):
    hidden_keywords: list[str]
    question: str
    # How the question was generated.
    # "given" means the question was generated in reference to `keywords` attribute.
    # "hidden" means the question was generated in reference to `hidden_keywords` attribute.
    # "none" means the question was generated without any keywords given.
    keyword_type: Literal["given", "hidden", "none"]


class GivenKeywordsQuestionGenerator(OpenAICompletionBase):
    def build_query(self, data: Request) -> dict:
        query = QUESTION_GIVEN_KEYWORD_PROMPT.format(
            primary=data.primary,
            secondary=data.secondary,
            keywords=", ".join(data.keywords),
            number=data.num_questions,
        )
        user_message = {"role": "user", "content": query}

        return {"messages": [system_message, user_message]}


class NoKeywordsQuestionGenerator(OpenAICompletionBase):
    def build_query(self, data: tuple[Field, int]) -> dict:
        field, number = data

        query = QUESTION_NO_KEYWORD_PROMPT.format(
            primary=field.primary, secondary=field.secondary, number=number
        )
        user_message = {"role": "user", "content": query}

        return {"messages": [system_message, user_message]}


def get_questions_from_keywords(
    keywords: Sequence[Keywords], num_questions: int, cachefile: Path, max_concurrency: int = 50
) -> list[ExpertiseFieldQuestion]:
    completer = GivenKeywordsQuestionGenerator(
        common_args=CALL_ARGS, max_concurrency=max_concurrency
    )

    requests: list[Request] = []

    # Construct completion query by dividing the keywords into halves
    for item in keywords:
        words = random.sample(item.keywords, k=len(item.keywords))
        former, latter = words[: len(words) // 2], words[len(words) // 2 :]
        num_former, num_latter = num_questions // 2, num_questions - num_questions // 2

        req_former = Request(
            primary=item.primary,
            secondary=item.secondary,
            keywords=former,
            hidden_keywords=latter,
            num_questions=num_former,
        )
        req_latter = Request(
            primary=item.primary,
            secondary=item.secondary,
            keywords=latter,
            hidden_keywords=former,
            num_questions=num_latter,
        )

        requests.append(req_former)
        requests.append(req_latter)

    outputs = completer.generate(requests, cache_file=cachefile)

    result = []
    for req, output in zip(requests, outputs):
        questions = []

        for line in output.completion.splitlines():
            if is_english_only(line):
                logger.error(f"English only line: {line}")
                continue

            try:
                question_raw = parse_enumeration(line)
            except ValueError as e:
                logger.error(e)
                continue

            question = ExpertiseFieldQuestion(
                primary=req.primary,
                secondary=req.secondary,
                keywords=req.keywords,
                hidden_keywords=req.hidden_keywords,
                question=question_raw,
                keyword_type="given",
            )
            questions.append(question)

        if len(questions) != req.num_questions:
            logger.error(
                f"Subfield {req.secondary} got {len(questions)} questions, not {req.num_questions} as requested."
            )

        result.extend(questions)

    return result


def get_questions_without_keywords(
    keywords: Sequence[Keywords], num_questions: int, cachefile: Path, max_concurrency: int = 50
) -> list[ExpertiseFieldQuestion]:
    completer = NoKeywordsQuestionGenerator(common_args=CALL_ARGS, max_concurrency=max_concurrency)

    # Generate questions without specifying keywords
    fields = [(Field(primary=k.primary, secondary=k.secondary), num_questions) for k in keywords]
    outputs_no_keyword = completer.generate(fields, cache_file=cachefile)

    result = []
    for keyword, output in zip(keywords, outputs_no_keyword):
        questions = []
        for line in output.completion.splitlines():
            try:
                question_raw = parse_enumeration(line)
            except ValueError as e:
                logger.error(e)
                continue

            question = ExpertiseFieldQuestion(
                **asdict(keyword), hidden_keywords=[], question=question_raw, keyword_type="none"
            )
            questions.append(question)

        result.extend(questions)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-questions", required=True, type=int)
    parser.add_argument(
        "-i", "--infile", required=True, help="NDJSON (.jsonl) file containing the keywords"
    )
    parser.add_argument(
        "-o", "--outfile", required=True, help="NDJSON (.jsonl) file to save the questions"
    )
    parser.add_argument(
        "-c", "--cachefile", required=True, help="NDJSON (.jsonl) file to keep the OpenAI API cache"
    )
    parser.add_argument("-m", "--max-concurrency", type=int, default=10)

    args = parser.parse_args()

    outfile = Path(args.outfile)
    if outfile.exists():
        raise FileExistsError(outfile)

    keywords = [Keywords(**obj) for obj in load_jsonl(args.infile)]

    questions_from_keywords = get_questions_from_keywords(
        keywords, args.num_questions, args.cachefile, args.max_concurrency
    )
    questions_without_keywords = get_questions_without_keywords(
        keywords, args.num_questions, args.cachefile, args.max_concurrency
    )

    questions = questions_from_keywords + questions_without_keywords
    save_jsonl([asdict(q) for q in questions], outfile)
