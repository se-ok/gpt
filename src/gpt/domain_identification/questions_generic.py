import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger

from gpt.base import OpenAICompletionBase
from gpt.util import is_english_only, parse_enumeration, save_jsonl

from .prompts import CHATBOT_SUBJECTS, QUESTION_CHATBOT_PROMPT, system_message

CALL_ARGS = {
    "model": "gpt-4-1106-preview",
    "temperature": 1,
    "max_tokens": 4000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


@dataclass
class GenericQuestion:
    subject: str
    question: str


class ChatbotQuestionGenerator(OpenAICompletionBase):
    def build_query(self, data: tuple[str, int]) -> dict:
        subject, number = data

        query = QUESTION_CHATBOT_PROMPT.format(subject=subject, number=number)
        user_message = {"role": "user", "content": query}

        return {"messages": [system_message, user_message]}


def get_questions(subjects: list[str], num_questions: int) -> list[GenericQuestion]:
    completer_chatbot = ChatbotQuestionGenerator(common_args=CALL_ARGS, max_concurrency=50)

    result = []
    data = [(subject, num_questions) for subject in CHATBOT_SUBJECTS]
    outputs = completer_chatbot.generate(data)

    for subject, output in zip(subjects, outputs):
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

            question = GenericQuestion(subject=subject, question=question_raw)
            questions.append(question)

        if len(questions) != num_questions:
            logger.error(
                f"Subject {subject} got {len(questions)} questions, not {num_questions} as requested."
            )

        result.extend(questions)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-questions", type=int)
    parser.add_argument("-o", "--outfile", help="NDJSON (.jsonl) file to save the questions")

    args = parser.parse_args()

    outfile = Path(args.outfile)
    if outfile.exists():
        raise FileExistsError(outfile)

    questions = get_questions(CHATBOT_SUBJECTS, args.num_questions)
    save_jsonl([asdict(q) for q in questions], outfile)
