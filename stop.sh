#!/usr/bin/env bash
# Stop the vLLM server started by serve.sh and wait for the GPU to free up.
pkill -INT -f '[b]in/vllm serve' || { echo "no vLLM server running"; exit 0; }
while pgrep -f '[b]in/vllm serve' >/dev/null; do sleep 1; done
echo "stopped"
