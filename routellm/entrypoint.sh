#!/bin/sh
set -e

exec python -m routellm.openai_server \
  --routers "${ROUTELLM_ROUTER:-mf}" \
  --strong-model "${ROUTELLM_STRONG_MODEL:-gpt-4o}" \
  --weak-model "${ROUTELLM_WEAK_MODEL:-ollama_chat/qwen3-coder-next}" \
  --threshold "${ROUTELLM_THRESHOLD:-0.3}" \
  --port 6060 \
  --host 0.0.0.0
