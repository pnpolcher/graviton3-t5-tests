#!/bin/bash

cd pytorch2
source .venv/bin/activate
../t5_gpu.py
deactivate
