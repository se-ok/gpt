import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from gpt.base import OpenAICompletionBase
from gpt.util import parse_enumeration, save_jsonl

from .prompts import FIELD_PROMPT, PRIMARY_FIELDS, system_message

CALL_ARGS = {
    "model": "gpt-4-1106-preview",
    "temperature": 1,
    "max_tokens": 1000,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


@dataclass
class Field:
    primary: str
    secondary: str  # Sub-field


class FieldCollector(OpenAICompletionBase):
    def build_query(self, data: dict) -> dict:
        primary = data["primary_field"]
        number = data["num_subfields"]
        user_message = {
            "role": "user",
            "content": FIELD_PROMPT.format(field=primary, number=number),
        }

        return {"messages": [system_message, user_message]}


def get_subfields(primary_fields: Sequence[str], num_subfields: int = 20) -> list[Field]:
    """Ask ChatGPT for the subfields of each field in `primary_fields`."""
    completer = FieldCollector(max_concurrency=10, common_args=CALL_ARGS)

    data = [
        {"primary_field": primary_field, "num_subfields": num_subfields}
        for primary_field in primary_fields
    ]
    outputs = completer.generate(data)

    result = []
    for primary_field, output in zip(primary_fields, outputs):
        for line in output.completion.splitlines():
            subfield = parse_enumeration(line)

            result.append(Field(primary=primary_field, secondary=subfield))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-subfields", type=int, required=True)
    parser.add_argument(
        "-o", "--outfile", required=True, help="File path (.jsonl) to store the fields as NDJSON"
    )

    args = parser.parse_args()

    outfile = Path(args.outfile)
    if outfile.exists():
        raise FileExistsError(outfile)

    fields = get_subfields(PRIMARY_FIELDS, args.num_subfields)
    save_jsonl([asdict(f) for f in fields], outfile)
