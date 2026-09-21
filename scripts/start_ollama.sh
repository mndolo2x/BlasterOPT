#!/usr/bin/env bash
set -euo pipefail

MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
HOST="${OLLAMA_HOST:-http://localhost:11434}"

command -v ollama >/dev/null 2>&1 || {
  echo "Ollama is not installed. Install it from https://ollama.com/download"
  exit 1
}

ollama serve >/tmp/blasteropt-ollama.log 2>&1 &
OLLAMA_PID=$!
trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if curl --fail --silent "${HOST%/}/api/tags" >/dev/null; then
    break
  fi
  sleep 1
done

curl --fail --silent "${HOST%/}/api/tags" >/dev/null || {
  echo "Ollama did not become reachable at ${HOST}"
  exit 1
}

ollama pull "$MODEL"
printf 'Ollama is ready at %s with model %s\n' "$HOST" "$MODEL"
