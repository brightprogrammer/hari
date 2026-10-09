#!/usr/bin/env python3

from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-4-26B-A4B-it-qat-4bit")

prompt = "Write a story about Einstein"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
                        messages,
                        enable_thinking=True,
                        add_generation_prompt=True)

for x in range(0, 10):
    text = generate(model,
                tokenizer,
                max_tokens=2560,
                prompt=prompt,
                verbose=True)

    messages.append({"role": "assistant", "content": text})
    messages.append({"role": "user", "content": "i like the story, can you improve the language? i promise none of this is AI written!"})

    prompt = tokenizer.apply_chat_template(
                            messages,
                            enable_thinking=True,
                            add_generation_prompt=True)
