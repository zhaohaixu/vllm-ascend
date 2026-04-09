#!/bin/bash

evalscope perf \
  --parallel 2 \
  --url http://127.0.0.1:8000/v1/completions \
  --model Qwen3-8B-W8A8 \
  --max-tokens 2048 \
  --min-tokens 2048 \
  --api openai \
  --dataset random \
  --tokenizer-path /mnt/Qwen3-8B-W8A8 \
  --debug \
  --max-prompt-length 512 \
  --min-prompt-length 512 \
  --number 20