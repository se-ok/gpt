import argparse
import copy
import json
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .translator import PrometheusTranslator

KEYS_TO_TRANSLATE = [
    "orig_instruction",
    "orig_response",
    "orig_reference_answer",
    "orig_criteria",
    "orig_score1_description",
    "orig_score2_description",
    "orig_score3_description",
    "orig_score4_description",
    "orig_score5_description",
    "orig_feedback",
]

CALL_ARGS = {
    "model": "gpt-3.5-turbo-1106",
    "temperature": 0.2,
    "max_tokens": 4095,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--infile", required=True, help="Path to new_feedback_collection.json")
    parser.add_argument("--outfile", required=True, help="Path to save the translated file")
    parser.add_argument("--cachefile", required=True, help="Auxiliary file to avoid re-running same request")
    parser.add_argument("--max-concurrency", type=int, default=100)

    return parser.parse_args()


def main(infile: Path, outfile: Path, cachefile: Path, max_concurrency: int):
    # Load the original data
    with open(infile) as f:
        samples = json.load(f)

    logger.info(f"{len(samples)} samples loaded")

    # Get the translator ready
    translator = PrometheusTranslator(common_args=CALL_ARGS, max_concurrency=max_concurrency)

    # Collect all the texts to translate
    to_translate = []
    for sample in tqdm(samples):
        to_translate.extend(sample[key] for key in KEYS_TO_TRANSLATE)

    logger.info(f"{len(to_translate)} english texts to translate")

    # Translate with progress bar
    translated = []

    completions = translator.generate(to_translate, cache_file=cachefile, ignore_error=True)
    iterator = iter(tqdm(completions, desc='Building translated data'))

    for sample in samples:
        copied = copy.deepcopy(sample)

        for key in KEYS_TO_TRANSLATE:
            output = next(iterator)
            copied[key] = output.completion

        translated.append(copied)

    # Save as json
    with open(outfile, "w") as f:
        json.dump(translated, f, ensure_ascii=False)


if __name__ == "__main__":
    args = parse_args()

    main(args.infile, args.outfile, args.cachefile, args.max_concurrency)
