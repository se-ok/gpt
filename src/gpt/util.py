import os
from pprint import pprint

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
        raise RuntimeError(f"OpenAI API request error with code {response.status_code}, {response.content}")

    pprint({key: val for key, val in response.headers.items() if key.startswith("x-ratelimit")})
