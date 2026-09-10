#!/usr/bin/env bash
set -euo pipefail

PORT=9898
URL="http://127.0.0.1:${PORT}"
BIN="/home/tjhexa/development/llama.cpp/llama.cpp/build/bin/llama-server"
MODEL="/home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf"

is_up() {
  curl -s --connect-timeout 1 "$URL" >/dev/null 2>&1
}

open_browser_when_ready() {
  local i
  for i in $(seq 1 180); do
    if is_up; then
      xdg-open "$URL" >/dev/null 2>&1 || true
      return 0
    fi
    sleep 1
  done
}

if is_up; then
  xdg-open "$URL" >/dev/null 2>&1 || true
  exit 0
fi

open_browser_when_ready &
exec "$BIN" \
  -m "$MODEL" \
  -ngl 99 \
  --host 0.0.0.0 \
  --port "$PORT"
