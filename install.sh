#!/usr/bin/env bash

# create virtual environment and install requirements
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# install mecab
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)