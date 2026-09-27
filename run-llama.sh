#!/usr/bin/env bash
# run-llama.sh - pick a model from local-models/ and launch llama-server (web) or llama-cli.
# Remembers your last choices in .run-llama.defaults so next run just hits Enter.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${MODELS_DIR:-$SCRIPT_DIR/local-models}"
BIN_DIR="$SCRIPT_DIR/build/bin"
CONFIG_FILE="$SCRIPT_DIR/.run-llama.defaults"

SERVER_BIN="$BIN_DIR/llama-server"
CLI_BIN="$BIN_DIR/llama-cli"

# ---- built-in defaults (used when no saved value exists) ----
DEF_MODE="web"
DEF_PORT="9898"
DEF_CTX="4096"
DEF_THREADS="$(nproc 2>/dev/null || echo 8)"
DEF_TEMP="0.8"
DEF_NPREDICT="512"

# ---- load saved defaults ----
if [ -f "$CONFIG_FILE" ]; then
    # shellcheck disable=SC1090
    . "$CONFIG_FILE"
fi
SAVED_MODEL="${MODEL:-}"
MODE="${MODE:-$DEF_MODE}"
PORT="${PORT:-$DEF_PORT}"
CTX="${CTX:-$DEF_CTX}"
THREADS="${THREADS:-$DEF_THREADS}"
TEMP="${TEMP:-$DEF_TEMP}"
NPREDICT="${NPREDICT:-$DEF_NPREDICT}"

YES=0
DRY_RUN=0
CLI_MODEL=""
CLI_MODE=""
CLI_PORT=""
CLI_CTX=""
CLI_THREADS=""
CLI_TEMP=""
CLI_NPREDICT=""
CLI_PROMPT=""
FORGET=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Interactive launcher: lists GGUF models in $MODELS_DIR,
asks for params (Enter = accept default), saves choices to
.run-llama.defaults, then runs llama-server (web) or llama-cli.

Options:
  --model PATH|N      model file (path or number from the list)
  --mode web|cli      web = llama-server, cli = llama-cli (default: $DEF_MODE)
  --port N            server port, web mode only (default: $DEF_PORT)
  --ctx N             context size -c (default: $DEF_CTX)
  --threads N         threads -t (default: nproc = $DEF_THREADS)
  --temp F            temperature (cli mode, default: $DEF_TEMP)
  -n, --n-predict N   tokens to predict, cli mode (default: $DEF_NPREDICT)
  -p, --prompt TEXT   prompt for cli mode (default: interactive chat)
  -y, --yes           non-interactive: use saved/defaults without asking
  --dry-run           print the command without running it
  --forget            delete saved defaults and exit
  -h, --help          this help

Examples:
  $(basename "$0")
  $(basename "$0") --mode web --port 9898 -y
  $(basename "$0") --mode cli -p "What is 2+2?" -y
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --model)   CLI_MODEL="${2:?}"; shift 2 ;;
        --mode)    CLI_MODE="${2:?}"; shift 2 ;;
        --port)    CLI_PORT="${2:?}"; shift 2 ;;
        --ctx)     CLI_CTX="${2:?}"; shift 2 ;;
        --threads) CLI_THREADS="${2:?}"; shift 2 ;;
        --temp)    CLI_TEMP="${2:?}"; shift 2 ;;
        -n|--n-predict) CLI_NPREDICT="${2:?}"; shift 2 ;;
        -p|--prompt) CLI_PROMPT="${2:?}"; shift 2 ;;
        -y|--yes) YES=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --forget) FORGET=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
done

if [ "$FORGET" -eq 1 ]; then
    rm -f "$CONFIG_FILE"
    echo "Forgot saved defaults ($CONFIG_FILE removed)."
    exit 0
fi

# CLI args override saved values
[ -n "$CLI_MODE" ]    && MODE="$CLI_MODE"
[ -n "$CLI_PORT" ]    && PORT="$CLI_PORT"
[ -n "$CLI_CTX" ]     && CTX="$CLI_CTX"
[ -n "$CLI_THREADS" ] && THREADS="$CLI_THREADS"
[ -n "$CLI_TEMP" ]    && TEMP="$CLI_TEMP"
[ -n "$CLI_NPREDICT" ]&& NPREDICT="$CLI_NPREDICT"

ask() { # ask <var> <label> <default>
    local var="$1" label="$2" def="$3" ans
    read -r -p "$label [$def]: " ans || ans=""
    if [ -z "$ans" ]; then ans="$def"; fi
    printf -v "$var" '%s' "$ans"
}

# ---- sanity checks ----
[ -x "$SERVER_BIN" ] || { echo "ERROR: $SERVER_BIN not found/executable. Build first: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2; exit 1; }
[ -x "$CLI_BIN" ]    || { echo "ERROR: $CLI_BIN not found/executable." >&2; exit 1; }
[ -d "$MODELS_DIR" ] || { echo "ERROR: models dir not found: $MODELS_DIR" >&2; exit 1; }

mapfile -t MODELS < <(find "$MODELS_DIR" -maxdepth 1 -type f -name '*.gguf' | sort)
[ "${#MODELS[@]}" -gt 0 ] || { echo "ERROR: no *.gguf files in $MODELS_DIR" >&2; exit 1; }

# ---- model selection ----
MODEL_PATH=""
if [ -n "$CLI_MODEL" ]; then
    if [[ "$CLI_MODEL" =~ ^[0-9]+$ ]]; then
        idx=$((CLI_MODEL - 1))
        [ "$idx" -ge 0 ] && [ "$idx" -lt "${#MODELS[@]}" ] || { echo "ERROR: --model N out of range (1-${#MODELS[@]})" >&2; exit 1; }
        MODEL_PATH="${MODELS[$idx]}"
    else
        # allow bare filename or full path
        if [ -f "$CLI_MODEL" ]; then MODEL_PATH="$CLI_MODEL"
        elif [ -f "$MODELS_DIR/$CLI_MODEL" ]; then MODEL_PATH="$MODELS_DIR/$CLI_MODEL"
        else echo "ERROR: model not found: $CLI_MODEL" >&2; exit 1; fi
    fi
elif [ "$YES" -eq 1 ] && [ -n "$SAVED_MODEL" ] && [ -f "$SAVED_MODEL" ]; then
    MODEL_PATH="$SAVED_MODEL"
elif [ "$YES" -eq 1 ] && [ "${#MODELS[@]}" -eq 1 ]; then
    MODEL_PATH="${MODELS[0]}"
elif [ "$YES" -eq 1 ]; then
    echo "Available models:"
    for i in "${!MODELS[@]}"; do printf '  %d) %s\n' "$((i+1))" "$(basename "${MODELS[$i]}")"; done
    echo "Non-interactive (-y) with no saved model and >1 model: pass --model N|PATH." >&2
    exit 1
else
    echo "Available models in $MODELS_DIR:"
    for i in "${!MODELS[@]}"; do
        mark=""
        [ "${MODELS[$i]}" = "$SAVED_MODEL" ] && mark="  (last used)"
        printf '  %d) %s%s\n' "$((i+1))" "$(basename "${MODELS[$i]}")" "$mark"
    done
    def_idx=1
    if [ -n "$SAVED_MODEL" ]; then
        for i in "${!MODELS[@]}"; do
            [ "${MODELS[$i]}" = "$SAVED_MODEL" ] && { def_idx=$((i+1)); break; }
        done
    fi
    ans=""
    read -r -p "Select model [1-${#MODELS[@]}, default $def_idx]: " ans || ans=""
    [ -z "$ans" ] && ans="$def_idx"
    [[ "$ans" =~ ^[0-9]+$ ]] || { echo "ERROR: enter a number." >&2; exit 1; }
    idx=$((ans - 1))
    [ "$idx" -ge 0 ] && [ "$idx" -lt "${#MODELS[@]}" ] || { echo "ERROR: out of range." >&2; exit 1; }
    MODEL_PATH="${MODELS[$idx]}"
fi

# ---- other params ----
if [ "$YES" -eq 0 ] && [ -z "$CLI_MODE" ]; then
    ask MODE "Mode (web=llama-server / cli=llama-cli)" "$MODE"
fi
[[ "$MODE" == "web" || "$MODE" == "cli" ]] || { echo "ERROR: --mode must be web|cli" >&2; exit 1; }

if [ "$MODE" = "web" ]; then
    if [ "$YES" -eq 0 ] && [ -z "$CLI_PORT" ]; then
        ask PORT "Server port" "$PORT"
    fi
    [[ "$PORT" =~ ^[0-9]+$ ]] || { echo "ERROR: port must be numeric." >&2; exit 1; }
fi

if [ "$YES" -eq 0 ]; then
    [ -n "$CLI_CTX" ]     || ask CTX "Context size (-c)" "$CTX"
    [ -n "$CLI_THREADS" ] || ask THREADS "Threads (-t)" "$THREADS"
    if [ "$MODE" = "cli" ]; then
        [ -n "$CLI_TEMP" ]     || ask TEMP "Temperature" "$TEMP"
        [ -n "$CLI_NPREDICT" ] || ask NPREDICT "Tokens to predict (-n)" "$NPREDICT"
        if [ -z "$CLI_PROMPT" ]; then
            read -r -p "Prompt (empty = interactive chat): " CLI_PROMPT || CLI_PROMPT=""
        fi
    fi
fi

# ---- save defaults for next run ----
cat > "$CONFIG_FILE" <<EOF
# saved by run-llama.sh - delete file or run with --forget to reset
MODEL="$MODEL_PATH"
MODE="$MODE"
PORT="$PORT"
CTX="$CTX"
THREADS="$THREADS"
TEMP="$TEMP"
NPREDICT="$NPREDICT"
EOF
echo "Saved defaults to $CONFIG_FILE"

# ---- launch ----
if [ "$MODE" = "web" ]; then
    # shellcheck disable=SC2206
    CMD=("$SERVER_BIN" -m "$MODEL_PATH" --port "$PORT" -c "$CTX" -t "$THREADS")
    echo "Starting web server: ${CMD[*]}"
    echo "Open http://127.0.0.1:$PORT in your browser (Ctrl+C to stop)."
    [ "$DRY_RUN" -eq 1 ] && exit 0
    exec "${CMD[@]}"
else
    # shellcheck disable=SC2206
    CMD=("$CLI_BIN" -m "$MODEL_PATH" -c "$CTX" -t "$THREADS" --temp "$TEMP" -n "$NPREDICT")
    [ -n "$CLI_PROMPT" ] && CMD+=(-p "$CLI_PROMPT")
    echo "Starting CLI: ${CMD[*]}"
    [ "$DRY_RUN" -eq 1 ] && exit 0
    exec "${CMD[@]}"
fi
