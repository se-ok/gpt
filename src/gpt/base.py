import hashlib
import inspect
import json  # Open the JSON file
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import openai
import tenacity
from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from tqdm import tqdm


def _cost_gpt_3_5_turbo(input: int, output: int) -> float:
    return input * 0.001 / 1000 + output * 0.002 / 1000


def _cost_gpt_3_5_turbo_instruct(input: int, output: int) -> float:
    return input * 0.0015 / 1000 + output * 0.002 / 1000


def _cost_gpt_4(input: int, output: int) -> float:
    return input * 0.03 / 1000 + output * 0.06 / 1000


class OpenAIOutput(BaseModel):
    args_hash: str
    model: str
    completion: str
    prompt_tokens: int
    completion_tokens: int

    def cost(self) -> float:
        if self.model in {"gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-1106"}:
            return _cost_gpt_3_5_turbo(self.prompt_tokens, self.completion_tokens)

        if self.model in {"gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct-0914"}:
            return _cost_gpt_3_5_turbo_instruct(self.prompt_tokens, self.completion_tokens)

        if self.model in {"gpt-4", "gpt-4-1106-preview", "gpt-4-0613"}:
            return _cost_gpt_4(self.prompt_tokens, self.completion_tokens)

        return 0.0


class OpenAICompletionBase(ABC):
    client: openai.OpenAI
    max_concurrency: int  # How many completion requests will be made concurrently
    common_args: dict  # Common arguments to all requests to be made
    _required_arguments: set[str] = {"model", "messages"}
    _all_arguments: set[str]  # Arguments names of OpenAI chat completion call
    _retry_complete: Callable  # OpenAI chat complete function decorated with tenacity.retry

    def __init__(
        self,
        max_concurrency: int = 10,
        max_retry: int = 3,
        min_wait: float = 2,
        max_wait: float = 30,
        common_args: dict | None = None,
        **kwargs,
    ) -> None:
        self.client = openai.OpenAI(**kwargs)
        call_signature = inspect.signature(self.client.chat.completions.create)
        self._all_arguments = set(call_signature.parameters.keys())

        self.max_concurrency = max_concurrency

        retry_decorator = tenacity.retry(
            wait=tenacity.wait_random_exponential(min=min_wait, max=max_wait),
            stop=tenacity.stop_after_attempt(max_retry),
            after=tenacity.after_log(logger, "DEBUG"),
        )
        self._retry_complete = retry_decorator(self._openai_complete)
        logger.info(
            f"OpenAI chat completion will retry with max_retry {max_retry}, min_wait {min_wait}, max_wait {max_wait}"
        )

        if common_args is None:
            common_args = {}

        if not isinstance(common_args, dict):
            raise ValueError(
                f"`common_args` must be a dict contaning the arguments common to all requests to be made. Given {common_args}"
            )

        self.common_args = common_args

    def hash_args(self, args: dict) -> str:
        json_str = json.dumps(args, sort_keys=True)

        return hashlib.sha256(json_str.encode()).hexdigest()

    def _openai_complete(self, *args, **kwargs) -> ChatCompletion:
        try:
            result = self.client.chat.completions.create(*args, **kwargs)
            return result

        except Exception as e:
            logger.error(e)
            raise

    @abstractmethod
    def build_query(self, data: Any) -> dict:
        """Create OpenAIInput from input data.
        `common_args` will be automatically added, while precedence is on this method when keys overlap.
        """

        raise NotImplementedError

    def _complete(self, args: dict) -> OpenAIOutput:
        # Validate call arguments
        for name in self._required_arguments:
            if name not in args:
                raise ValueError(f"Call arguments does not have required key {name}")

        for name in args:
            if name not in self._all_arguments:
                raise ValueError(f"Call argument {name} not recognized by OpenAI chat completion.")

        # Make a call
        result: ChatCompletion = self._retry_complete(**args)

        return OpenAIOutput(
            args_hash=self.hash_args(args),
            model=args["model"],
            completion=result.choices[0].message.content,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
        )

    def _complete_with_cache(
        self, args: dict, cache: dict, cache_file: Path | None = None
    ) -> OpenAIOutput:
        """Help concurrency of _complete where some of the runs are cached.

        If not cached, when writes to `cache_file` is given.
        """

        hash_args = self.hash_args(args)

        if hash_args in cache:
            return cache[hash_args]

        completion = self._complete(args)
        cache[hash_args] = completion

        if cache_file:
            with open(cache_file, "a") as f:
                print(completion.model_dump_json(), file=f)

        return completion

    def _load_cache(self, cache_file: str | Path | None = None) -> tuple[Path | None, dict]:
        if cache_file is None:
            logger.warning(
                "Using OpenAI API without cache_file might incur unnecessary, duplicated cost."
            )
            return None, {}

        cache = {}
        cache_file = Path(cache_file)

        # If cache_file doesn't exist, touch it to check permission before committing requests.
        if not cache_file.exists():
            with open(cache_file, "w") as f:
                pass

            return cache_file, {}

        # Load cache
        with open(cache_file) as f:
            for line in tqdm(f, desc="Loading cache"):
                output: OpenAIOutput = OpenAIOutput.model_validate_json(line)
                cache[output.args_hash] = output

        return cache_file, cache

    def _pop_completed_future(
        self, future_to_idx: dict[Future, int], ignore_error: bool
    ) -> tuple[int, OpenAIOutput | None]:
        """From a dict mapping futures to their call index,
        wait and pop a completed future and return its index and result.

        If an error occured and ignore_error is True, then return the index and None.
        Otherwise raise the error.

        `future_to_idx` gets mutated.
        """
        future = next(as_completed(future_to_idx))
        idx = future_to_idx.pop(future)

        try:
            result = future.result()
            return idx, result

        # If the submitted job was retried with tenacity, then extract the inner error.
        except Exception as e:
            if isinstance(e, tenacity.RetryError):
                e = e.last_attempt.exception()

            if ignore_error:
                logger.error(f"Ignoring error {type(e)}, {e}")
                return idx, None
            else:
                raise e

    def _add_result(self, results: list, idx: int, completion: OpenAIOutput | None):
        """Do results[idx] = item, extending results with None if needed.

        `results` gets mutated.
        """
        if len(results) <= idx:
            results += [None] * (idx + 1 - len(results))
        results[idx] = completion

    def generate(
        self, data: Iterable, cache_file: str | Path | None = None, ignore_error: bool = False
    ) -> list[OpenAIOutput]:
        """Run OpenAI chat completion through all items in data and return the result as list.
        During the consumption of `data`, the results from each item is cached onto `cache_file`,
        so that the next run can re-use cached results, not making duplicated requests.

        If `ignore_error` is True, then errored requests will set the corresponding item in the returned list to None.
        Otherwise, the execution will stop when an error occurs.
        """
        cache_file, cache = self._load_cache(cache_file)

        iterator = enumerate(data)

        try:
            length = len(data)
            results = [None] * length

        except TypeError:
            length = None
            results = []

        pbar = tqdm(total=length, desc="OpenAI Chat Completion")
        input_tokens = 0
        output_tokens = 0
        cost = 0

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures: dict[Future, int] = {}

            for i, item in iterator:
                while len(futures) > pool._max_workers:
                    idx, completion = self._pop_completed_future(futures, ignore_error=ignore_error)
                    self._add_result(results, idx, completion)
                    input_tokens += completion.prompt_tokens
                    output_tokens += completion.completion_tokens
                    cost += completion.cost()
                    pbar.set_postfix_str(
                        f"Input {input_tokens}, Output {output_tokens}, Cost ${cost:.4f}"
                    )
                    pbar.update()

                args = self.common_args | self.build_query(item)
                future = pool.submit(self._complete_with_cache, args, cache, cache_file)
                futures[future] = i

            # Clean up the remaining futures
            while futures:
                idx, completion = self._pop_completed_future(futures, ignore_error=ignore_error)
                self._add_result(results, idx, completion)
                input_tokens += completion.prompt_tokens
                output_tokens += completion.completion_tokens
                cost += completion.cost()
                pbar.set_postfix_str(
                    f"Input {input_tokens}, Output {output_tokens}, Cost ${cost:.4f}"
                )
                pbar.update()

            pbar.close()

        if any(item is None for item in results):
            raise ValueError("Error occured while processing.")

        return results
