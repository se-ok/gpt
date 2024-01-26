"""Translate the evaluation data for Prometheus arXiv 2310.08491
"""
import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .translator import PrometheusTranslator


def load_samples(filename: str | Path) -> list[dict]:
    filename = Path(filename)
    with open(filename) as f:
        if filename.name in {"mt_bench_eval.json", "vicuna_eval.json"}:
            objs = [json.loads(line) for line in f]

        else:
            objs = json.load(f)

    return objs


def extract_text(text: str, pretext: str, posttext: str) -> str:
    """Extract the sub-text from text between pretext and posttext."""

    pattern = rf"{re.escape(pretext)}(.*?){re.escape(posttext)}"
    match = re.search(pattern, text, re.DOTALL)

    if match is None:
        messages = [
            "Sub-text extraction failed:",
            f"Pretext: {pretext}",
            f"Posttext: {posttext}",
            f"Text: {text}",
        ]
        raise ValueError("\n".join(messages))

    return match.group(1).strip()


CALL_ARGS_DEFAULT = {
    "model": "gpt-3.5-turbo-1106",
    "temperature": 0.2,
    "max_tokens": 4095,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

CALL_ARGS_LONG = CALL_ARGS_DEFAULT | {"model": "gpt-3.5-turbo-16k", "max_tokens": 12000}

KEYS_DEFAULT = [
    "orig_instruction",
    "orig_response",
    "orig_reference_answer",
    "orig_criteria",
    "orig_score1_description",
    "orig_score2_description",
    "orig_score3_description",
    "orig_score4_description",
    "orig_score5_description",
]

_MARKERS_DEFAULT = [
    "###The instruction to evaluate:",
    "###Response to evaluate:",
    "###Reference Answer (Score 5):",
    "###Score Rubrics:\n[",
    "]\nScore 1:",
    "Score 2:",
    "Score 3:",
    "Score 4:",
    "Score 5:",
    "###Feedback:",
]

PRETEXT_DEFAULT = _MARKERS_DEFAULT[:-1]
POSTTEXT_DEFAULT = _MARKERS_DEFAULT[1:]

KEYS_WITHOUT_REF = [
    "orig_instruction",
    "orig_response",
    "orig_criteria",
    "orig_score1_description",
    "orig_score2_description",
    "orig_score3_description",
    "orig_score4_description",
    "orig_score5_description",
]

_MARKERS_WITHOUT_REF = [
    "###The instruction to evaluate:",
    "###Response to evaluate:",
    "###Score Rubrics:\n[",
    "]\nScore 1:",
    "Score 2:",
    "Score 3:",
    "Score 4:",
    "Score 5:",
    "###Feedback:",
]

PRETEXT_WITHOUT_REF = _MARKERS_WITHOUT_REF[:-1]
POSTTEXT_WITHOUT_REF = _MARKERS_WITHOUT_REF[1:]


@dataclass
class PrometheusTestData:
    filename: Path
    instruction_key: str
    outfilename: Path
    translation_arguments: dict  # Call arguments to ChatGPT for translation
    keys: list[str]  # Dict key to map the translated subtexts
    pretexts: list[str]  # Pretext which marks the beginning of subtext to extract
    posttexts: list[str]  # Posttext which marks the end of subtest to extract


feedback_collection_ood_test = PrometheusTestData(
    filename="feedback_collection_ood_test.json",
    instruction_key="instruction",
    outfilename="feedback_collection_ood_test_translated.json",
    translation_arguments=CALL_ARGS_DEFAULT,
    keys=KEYS_DEFAULT,
    pretexts=PRETEXT_DEFAULT,
    posttexts=POSTTEXT_DEFAULT,
)

feedback_collection_test = PrometheusTestData(
    filename="feedback_collection_test.json",
    instruction_key="instruction",
    outfilename="feedback_collection_test_translated.json",
    translation_arguments=CALL_ARGS_DEFAULT,
    keys=KEYS_DEFAULT,
    pretexts=PRETEXT_DEFAULT,
    posttexts=POSTTEXT_DEFAULT,
)

flask_eval = PrometheusTestData(
    filename="flask_eval.json",
    instruction_key="instruction",
    outfilename="flask_eval_translated.json",
    translation_arguments=CALL_ARGS_LONG,
    keys=KEYS_DEFAULT,
    pretexts=PRETEXT_DEFAULT,
    posttexts=POSTTEXT_DEFAULT,
)

hhh_alignment_eval_chosen = PrometheusTestData(
    filename="hhh_alignment_eval.json",
    instruction_key="chosen_instruction",
    outfilename="hhh_alignment_eval_chosen_translated.json",
    translation_arguments=CALL_ARGS_DEFAULT,
    keys=KEYS_WITHOUT_REF,
    pretexts=PRETEXT_WITHOUT_REF,
    posttexts=POSTTEXT_WITHOUT_REF,
)

hhh_alignment_eval_rejected = PrometheusTestData(
    filename="hhh_alignment_eval.json",
    instruction_key="rejected_instruction",
    outfilename="hhh_alignment_eval_rejected_translated.json",
    translation_arguments=CALL_ARGS_DEFAULT,
    keys=KEYS_WITHOUT_REF,
    pretexts=PRETEXT_WITHOUT_REF,
    posttexts=POSTTEXT_WITHOUT_REF,
)

mt_bench_eval = PrometheusTestData(
    filename="mt_bench_eval.json",
    instruction_key="instruction",
    outfilename="mt_bench_eval_translated.json",
    translation_arguments=CALL_ARGS_LONG,
    keys=KEYS_DEFAULT,
    pretexts=PRETEXT_DEFAULT,
    posttexts=POSTTEXT_DEFAULT,
)

mt_bench_human_judgement_eval_chosen = PrometheusTestData(
    filename="mt_bench_human_judgement_eval.json",
    instruction_key="chosen_instruction",
    outfilename="mt_bench_human_judgement_eval_chosen_translated.json",
    translation_arguments=CALL_ARGS_DEFAULT,
    keys=KEYS_WITHOUT_REF,
    pretexts=PRETEXT_WITHOUT_REF,
    posttexts=POSTTEXT_WITHOUT_REF,
)

mt_bench_human_judgement_eval_rejected = PrometheusTestData(
    filename="mt_bench_human_judgement_eval.json",
    instruction_key="rejected_instruction",
    outfilename="mt_bench_human_judgement_eval_rejected_translated.json",
    translation_arguments=CALL_ARGS_DEFAULT,
    keys=KEYS_WITHOUT_REF,
    pretexts=PRETEXT_WITHOUT_REF,
    posttexts=POSTTEXT_WITHOUT_REF,
)

vicuna_eval = PrometheusTestData(
    filename="vicuna_eval.json",
    instruction_key="instruction",
    outfilename="vicuna_eval_translated.json",
    translation_arguments=CALL_ARGS_DEFAULT,
    keys=KEYS_DEFAULT,
    pretexts=PRETEXT_DEFAULT,
    posttexts=POSTTEXT_DEFAULT,
)


def translate_samples(
    path: Path, testdata: PrometheusTestData, max_concurrency: int, cachefile: Path
):
    """Extract subtexts from sample[text_key] according to keys and markers and flatten the result."""
    if not len(testdata.keys) == len(testdata.pretexts) == len(testdata.posttexts):
        raise ValueError(
            f"Lengths of keys, pretexts, posttexts do not match: {testdata.keys}, {testdata.pretexts}, {testdata.posttexts}"
        )

    if (path / testdata.outfilename).exists():
        raise FileExistsError(f"Output file {path / testdata.outfilename} already exists.")

    samples = load_samples(path / testdata.filename)

    to_translate = []
    for sample in samples:
        text = sample[testdata.instruction_key]
        for pretext, posttext in zip(testdata.pretexts, testdata.posttexts):
            subtext = extract_text(text, pretext, posttext)
            to_translate.append(subtext)

    logger.info(f"{len(to_translate)} english texts to translate")

    # Translate
    translated = []
    translator = PrometheusTranslator(
        common_args=testdata.translation_arguments, max_concurrency=max_concurrency
    )

    completions = translator.generate(to_translate, cache_file=cachefile)
    iterator = iter(tqdm(completions, desc="Building translated data"))

    for sample in samples:
        translated_sample = {}

        for key in testdata.keys:
            output = next(iterator)
            translated_sample[key] = output.completion

        translated.append(sample | translated_sample)

    with open(path / testdata.outfilename, "w") as f:
        json.dump(translated, f, ensure_ascii=False)


def main(path: Path, cachefile: Path, max_concurrency: int):
    test_data = [
        feedback_collection_ood_test,
        feedback_collection_test,
        flask_eval,
        hhh_alignment_eval_chosen,
        hhh_alignment_eval_rejected,
        mt_bench_eval,
        mt_bench_human_judgement_eval_chosen,
        mt_bench_human_judgement_eval_rejected,
        vicuna_eval,
    ]

    for data in test_data:
        translate_samples(path, data, max_concurrency=max_concurrency, cachefile=cachefile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Folder containing the test files")
    parser.add_argument(
        "--cachefile", required=True, help="Auxiliary file to avoid re-running same request"
    )
    parser.add_argument("--max-concurrency", type=int, default=100)

    args = parser.parse_args()

    main(Path(args.path), Path(args.cachefile), args.max_concurrency)
