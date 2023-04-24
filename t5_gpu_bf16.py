#!/usr/bin/env python3

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


N_SAMPLES = 100

max_new_tokens=64
max_sentence_len=256
model_name = "google/t5-small-ssm"
questions = [
    "What is the meaning of life?",
    "Which is the highest mountain?",
    "What is the name of Frank Sinatra?"
]

model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.bfloat16)

import time

print("Starting inference...")

start = time.time()

for i in range(0, N_SAMPLES):
    input_ids = tokenizer(questions, max_length=max_sentence_len, padding='max_length', truncation=True, return_tensors="pt").input_ids.to("cuda")
    gen_output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    answers = [tokenizer.decode(o, skip_special_tokens=True) for o in gen_output]

end = time.time()
tpi = (end - start) / N_SAMPLES * 1000

print(tpi)

print(questions)
print(answers)

