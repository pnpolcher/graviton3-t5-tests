#!/bin/bash

cd pytorch
source .venv/bin/activate
../t5_gpu_bf16.py
deactivate
