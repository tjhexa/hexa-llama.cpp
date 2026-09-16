# local-manager

Interactive CLI to pick a local GGUF model, inspect metadata (context window, arch, size, quant), show RAM/VRAM, then launch:

- **web** - `llama-server` (browser UI + OpenAI-compatible `/v1` API)  
  This is what `run-gpt-oss-server.sh` does.
- **cli** - terminal chat via `llama-completion` (text) or `llama-cli` when an `mmproj` is attached  
  (`llama-completion` does not support `--mmproj`; newer `llama-cli` does not support `-cnv`)

Self-contained under `local-manager/` (stdlib Python only, no pip deps).

## Quick start

```bash
cd /home/tjhexa/development/llama.cpp/llama.cpp/local-manager
chmod +x run
./run
```

Or:

```bash
python3 -m llama_manager
```

(from inside `local-manager/`, or set `PYTHONPATH=local-manager`)

## What you get

1. System banner: total/used/available **RAM**, per-GPU **VRAM** (via `nvidia-smi`)
2. Recursive scan of `models_dir` for `*.gguf` (skips `mmproj-*` as primary models)
3. Per-model summary: path, name, arch, quant hint, file size, **max context**, optional mmproj
4. Details view (`d` / `d2`) with RoPE/YaRN, MoE, layers, rough VRAM fit hint
5. Choose **web** or **cli**
6. Tune `-ngl`, `-c`, host/port, API alias, API key, mmproj attach
7. Remembers last model/mode in `history.json`

## Examples

```bash
# Interactive
./run

# List only
./run --list

# Non-interactive web launch
./run --model gpt-oss-20b-MXFP4.gguf --mode web -y

# CLI chat
./run --model LFM2.5-1.2B-Instruct-Q8_0.gguf --mode cli -y

# One-shot prompt
./run --model LFM2.5-1.2B-Instruct-Q8_0.gguf --mode cli \
  --prompt "Write a haiku about GPUs" -y

# Dry-run (print command)
./run --model gpt-oss-20b-MXFP4.gguf --mode web --dry-run -y

# Custom port / GPU layers / context
./run --model gpt-oss-20b-MXFP4.gguf --mode web \
  --port 9898 --ngl 99 --ctx 8192 --alias gpt-oss -y
```

## Config

Edit `config.json` (created from `config.example.json`):

| Key | Meaning |
| --- | ------- |
| `models_dir` | Folder to scan for GGUF files |
| `bin_dir` | Where `llama-server` / `llama-completion` live |
| `default_port` / `default_host` | Web server bind |
| `default_ngl` | Default GPU layers |
| `default_ctx` | `0` = let llama.cpp decide / model default |
| `open_browser` | Auto `xdg-open` when server is up |
| `api_key` | Optional `--api-key` for `/v1` |
| `extra_server_args` / `extra_cli_args` | Extra CLI flags as JSON string arrays |

## Web mode vs your old script

`run-gpt-oss-server.sh` always loads one fixed model and starts the web server.

`local-manager` does the same web path when you choose **web**, but lets you pick any model first, and optionally run **cli** instead.

OpenAI-compatible base URL after web start:

```text
http://127.0.0.1:9898/v1
```

## Suggested extra features (nice to add next)

Already partially covered or easy follow-ups:

1. **Presets** - save named profiles (`coding-8k`, `chat-32k`, `cpu-only`) with ngl/ctx/temp/host
2. **KV cache shortcuts** - menu toggles for `-ctk q8_0 -ctv q8_0`, flash-attn `-fa on`
3. **Better VRAM estimator** - use layer count + head dims from GGUF for KV math
4. **Multi-GPU helper** - prompt for `-sm` / `-ts` when >1 GPU
5. **Server status** - detect running server, show `/props`, offer restart/kill by port
6. **Model filter** - search by name/quant/min-ctx, hide models larger than free VRAM
7. **Sharded GGUF** - detect `model-00001-of-00002.gguf` groups and load the first shard
8. **Download assist** - wrap `llama-server -hf repo:quant` into the menu
9. **Bench mode** - one-click `llama-bench` on selected model
10. **OpenAI smoke test** - after web start, `curl /v1/models` + tiny chat completion
11. **tmux/systemd templates** - generate a unit file or tmux session for always-on server
12. **Per-model overrides** - remember last ngl/ctx per GGUF path in `history.json`

If you want any of these implemented next, say which numbers matter most.
