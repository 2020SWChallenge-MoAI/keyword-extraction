#!/usr/bin/env bash

# create virtual environment and install requirements
if command -v conda > /dev/null
then
    conda activate keyword-extraction
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

pip install -r requirements.txt