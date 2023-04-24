#!/bin/bash

cd pytorch
source .venv/bin/activate
../t5_gpu.py
deactivate
