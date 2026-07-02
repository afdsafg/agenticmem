# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import base64
import os
from typing import List, Optional
import time

import cv2
import openai
from openai import OpenAI

def set_openai_key(key: Optional[str] = None):
    if key is None:
        assert "OPENAI_API_KEY" in os.environ
        key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key


def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]


def prepare_openai_vision_messages(
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    image_paths: Optional[List[str]] = None,
    image_size: Optional[int] = 512,
):
    if image_paths is None:
        image_paths = []

    content = []

    if prefix:
        content.append({"text": prefix, "type": "text"})

    for path in image_paths:
        frame = cv2.imread(path)
        if image_size:
            factor = image_size / max(frame.shape[:2])
            frame = cv2.resize(frame, dsize=None, fx=factor, fy=factor)
        _, buffer = cv2.imencode(".png", frame)
        frame = base64.b64encode(buffer).decode("utf-8")
        content.append(
            {
                "image_url": {"url": f"data:image/png;base64,{frame}"},
                "type": "image_url",
            }
        )

    if suffix:
        content.append({"text": suffix, "type": "text"})

    return [{"role": "user", "content": content}]


def call_openai_api(
    messages: list,
    model: str = "gpt-4o",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key='',
        base_url=''
    )


    max_tries = 5
    retry_count = 0
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                seed=seed,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            if verbose:
                print("openai api response: {}".format(completion))
            assert len(completion.choices) == 1
            return completion.choices[0].message.content
        except openai.RateLimitError as e:
            print("Rate limit error, waiting for 60s")
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            print("Error: ", e)
            time.sleep(60)
            retry_count += 1
            continue

    return None

if __name__ == "__main__":
    set_openai_key(key=None)

    messages = prepare_openai_messages("What color are apples?")
    print("input:", messages)

    model = "gpt-4-vision-preview"
    output = call_openai_api(messages, model=model, max_tokens=512, temperature=1.0)
    print("output: {}".format(output))
