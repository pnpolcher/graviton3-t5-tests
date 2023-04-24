#!/bin/bash

cd pytorch2
source .venv/bin/activate
../t5_gpu_bf16.py
deactivate
