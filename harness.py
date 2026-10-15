#!/usr/bin/env python3

import json
import os
import tomllib
from openai import OpenAI
from datetime import datetime
from transformers.utils import get_json_schema

# where the inference server lives, and which model it should use
CONFIG_PATH = os.path.expanduser("~/.config/hari.toml")
with open(CONFIG_PATH, "rb") as f:
    INFERENCE_URL = tomllib.load(f)["inference_url"]

MODEL = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"

client = OpenAI(base_url=INFERENCE_URL, api_key="unused")


def cur_time():
    """
    Get current time
    """
    return str(datetime.now().time())


def cur_date():
    """
    Get current date
    """
    return str(datetime.now().date())


TOOLS = {"cur_time": cur_time, "cur_date": cur_date}


def generate(messages):
    stream = client.responses.create(
        model=MODEL,
        input=messages,
        tools=[
            {"type": "function", **get_json_schema(x)["function"]}
            for x in list(TOOLS.values())
        ],
        stream=True,
    )

    last_event = None
    for event in stream:
        if event.type == "response.reasoning_text.delta":
            if last_event != event.type:
                print()
            print(event.delta, end="", flush=True)
        elif event.type == "response.output_text.delta":
            if last_event != event.type:
                print()
            print(event.delta, end="", flush=True)
        elif event.type == "response.completed":
            response = event.response
        last_event = event.type
    print()

    messages += response.output

    keep_running = False
    for item in response.output:
        if item.type == "function_call":
            keep_running = True
            messages.append(
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": TOOLS[item.name](**json.loads(item.arguments)),
                }
            )

    return keep_running


def user_message(prompt):
    return {"role": "user", "content": prompt}


prompt = "What's the current date and time?"
messages = [user_message(prompt)]

while generate(messages):
    pass
