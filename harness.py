#!/usr/bin/env python3

from mlx_lm import load, generate
from mlx_lm.tool_parsers import gemma4
from datetime import datetime
from transformers.utils import get_json_schema
import os

# Tell mlx to not look for model online every single time
os.environ["HF_HUB_OFFLINE"] = "1"


def my_date():
    """
    Get current date
    """
    return str(datetime.now().date())


def cur_time():
    """
    Get current time
    """
    return str(datetime.now().time())


TOOLS = {"cur_time": cur_time, "my_date": my_date}

model, tokenizer = load("mlx-community/gemma-4-26B-A4B-it-qat-4bit")

prompt = "What's the current date and time?"

messages = [{"role": "user", "content": prompt}]

# ask current time and get first response
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tools=[get_json_schema(x) for x in list(TOOLS.values())],
)
text = generate(model, tokenizer, max_tokens=2560, prompt=prompt)
messages.append({"role": "model", "content": text})

# parse the tool calls and make the took call and append output
tool_calls = gemma4.parse_tool_call(text)
tool_calls = [tool_calls] if isinstance(tool_calls, dict) else tool_calls
messages.append(
    {
        "role": "model",
        "tool_responses": [
            {"name": tool["name"], "response": TOOLS[tool["name"]](**tool["arguments"])}
            for tool in tool_calls
        ],
    }
)

# give it to the agent in new prompt and generate the new output
prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tools=[get_json_schema(x) for x in list(TOOLS.values())],
)

text = generate(model, tokenizer, max_tokens=2560, prompt=prompt)
messages.append({"role": "model", "content": text})

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
    tools=[get_json_schema(x) for x in list(TOOLS.values())],
)

print(messages)
print()
print(prompt)
