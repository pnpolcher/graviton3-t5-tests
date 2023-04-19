#!/usr/bin/env python3

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

max_new_tokens=64
max_sentence_len=256
model_name = "google/t5-small-ssm"
questions = [
    "What is the meaning of life?",
    "Which is the highest mountain?",
    "What is the name of Frank Sinatra?"
]

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

import time

print("Starting inference...")
start = time.time()

for i in range(0, 25):
    input_ids = tokenizer(questions, max_length=max_sentence_len, padding='max_length', truncation=True, return_tensors="pt").input_ids
    gen_output = model.generate(input_ids, max_new_tokens=max_new_tokens)
    answers = [tokenizer.decode(o, skip_special_tokens=True) for o in gen_output]

end = time.time()
tpi = (end - start) / 25 * 1000
# tpi = end - start

print(tpi)

print(questions)
print(answers)

