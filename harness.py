#!/usr/bin/env python3

import json
import os
import tomllib
import urllib.request
from datetime import datetime
from transformers.utils import get_json_schema

# where the inference server lives, and which model it should use
CONFIG_PATH = os.path.expanduser("~/.config/hari.toml")
with open(CONFIG_PATH, "rb") as f:
    URL = tomllib.load(f)["url"]
MODEL = "mlx-community/gemma-4-26B-A4B-it-qat-4bit"


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


# same as before, but the response now comes in pieces
# we print every piece as it arrives, and build the message from the pieces
def generate(messages):
    body = json.dumps(
        {
            "model": MODEL,
            "max_tokens": 2560,
            "stream": True,
            "messages": messages,
            "tools": [get_json_schema(x) for x in list(TOOLS.values())],
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"}
    )

    message = {"role": "assistant", "content": "", "reasoning": "", "tool_calls": []}
    with urllib.request.urlopen(req) as res:
        for line in res:
            line = line.decode().rstrip("\n")

            # an empty line ends an event, a line starting with ':' is a comment
            if not line or line.startswith(":"):
                continue

            # everything else starts with 'data: '
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                break

            delta = json.loads(data)["choices"][0]["delta"]
            for key in ["reasoning", "content"]:
                if piece := delta.get(key):
                    print(piece, end="", flush=True)
                    message[key] += piece
            message["tool_calls"] += delta.get("tool_calls", [])

    print()
    return message


prompt = "What's the current date and time?"
messages = [{"role": "user", "content": prompt}]

# ask current date and time and get first response
message = generate(messages)
messages.append(message)

# the server already parsed the tool calls, make the tool calls and append output
messages.extend(
    {
        "role": "tool",
        "tool_call_id": tool["id"],
        "content": TOOLS[tool["function"]["name"]](**json.loads(tool["function"]["arguments"])),
    }
    for tool in message["tool_calls"]
)

# give it to the agent in new request and get the new output
messages.append(generate(messages))

print(json.dumps(messages, indent=2))
