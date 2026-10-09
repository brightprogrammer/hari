#!/usr/bin/env python3

from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-4-26B-A4B-it-qat-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
)

text = generate(model, tokenizer, prompt=prompt, verbose=True)
