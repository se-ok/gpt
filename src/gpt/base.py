import hashlib
import inspect
import json  # Open the JSON file
from abc import ABC, abstractmethod
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import openai
import tenacity
from loguru import logger
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel
from tqdm import tqdm


class OpenAIOutput(BaseModel):
    args_hash: str
    model: str
    completion: str
    prompt_tokens: int
    completion_tokens: int


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

    def _complete_with_cache(self, args: dict, cache: dict, cache_file: Path | None = None) -> OpenAIOutput:
        """Help concurrency of _complete where some of the runs are cached.

        If not cached, when writes to `cache_file` is given.
        """

        hash_args = self.hash_args(args)

        if hash_args in cache:
            return cache[hash_args]

        completion = self._complete(args)

        if cache_file:
            with open(cache_file, "a") as f:
                print(completion.model_dump_json(), file=f)

        return completion

    def generate(self, data: list, cache_file: str | Path | None = None) -> Generator[OpenAIOutput, None, None]:
        """Run OpenAI chat completion on each item in `data`.
        If cache_file is given, then
        """

        if cache_file:
            cache_file = Path(cache_file)

        finished = {}

        if cache_file and cache_file.exists():
            with open(cache_file) as f:
                for line in tqdm(f, desc="Loading cache"):
                    output: OpenAIOutput = OpenAIOutput.model_validate_json(line)
                    finished[output.args_hash] = output

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            futures: list[Future] = []

            for item in tqdm(data, desc="Submitting"):
                args = self.common_args | self.build_query(item)
                futures.append(pool.submit(self._complete_with_cache, args, finished, cache_file))

            logger.info(f"{len(futures)} jobs submitted")

            for future in futures:
                try:
                    completion = future.result()
                    yield completion

                except tenacity.RetryError as e:
                    exception = e.last_attempt.exception()
                    logger.error(f"Retry failed with {exception}")
                    pool.shutdown(cancel_futures=True)

                    raise exception from e

                except Exception as e:
                    logger.error(f"Uncaught exception {type(e)}, {e}")
                    pool.shutdown(cancel_futures=True)

                    raise
