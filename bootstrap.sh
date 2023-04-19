#!/bin/bash

install_dependencies () {
  python3 -m venv .venv
  . .venv/bin/activate
  pip install -U pip
  pip install -r requirements.txt
  deactivate
}

# Install dependencies
sudo apt install python3.10-venv

# Initialize PyTorch 1.13 venv
cd pytorch
install_dependencies
cd ..

# Initialize PyTorch 2.0 venv
cd pytorch2
install_dependencies
cd ..

