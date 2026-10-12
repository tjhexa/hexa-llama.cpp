# local-manager-ui

GTK3 desktop front-end for [`local-manager`](../local-manager). It wraps the two
things the CLI does by hand:

- **`local-manager/run`** - pick a local GGUF, launch `llama-server` (web) or
  `llama-completion` (cli), with live logs and server status.
- **`local-manager/pull-merge-build.sh`** - pull upstream, merge, CUDA build,
  with streamed output and a Stop button.

Everything runs asynchronously: subprocess output is streamed from a worker
thread and dispatched to the UI through `GLib.idle_add`, so the window never
blocks. No pip dependencies - only PyGObject/GTK3 from the distro.

## Requirements

```bash
sudo apt install python3-gi gir1.2-gtk-3.0
```

## Quick start

```bash
cd /home/tjhexa/development/llama.cpp/llama.cpp/local-manager-ui
./run
```

or:

```bash
PYTHONPATH=local-manager-ui python3 -m llama_manager_ui
```

## Tabs

| Tab | What it does |
| --- | ----------- |
| **Models** | Filterable list of `.gguf` files with quant/size/context/est. VRAM. Details pane shows arch, RoPE, MoE, mmproj. Buttons: load into Server tab, launch web, launch CLI, copy path. |
| **Server** | Launch options (model, mode, `-ngl`, `-c`, host, port, alias, API key, mmproj, extra args) with a live command preview. Start streams logs; health probe flips to **Ready** and enables **Open web UI**. Recent launches are one click away. |
| **Build** | Controls for `pull-merge-build.sh` (upstream/target branch, CUDA arch, jobs, build dir, `--clean`). Streamed build log, Stop button, or **Open in terminal**. |
| **Settings** | Edits `config.json`: `models_dir`, `bin_dir`, defaults, browser, extra args. Saved to `local-manager-ui/config.json`. |

The header shows the active job with a **Stop** button (sends `SIGTERM` to the
whole process group) and the current git branch/HEAD. A moon button toggles the
dark theme.

## Config and history

`config.json` and `history.json` are machine-local and git-ignored.

- On first run, config is loaded from `local-manager/config.json` if present, so
  the UI starts with the same paths as the CLI. Saving writes to
  `local-manager-ui/config.json` (the CLI config is left untouched).
- History is read from `local-manager/history.json` and written to
  `local-manager-ui/history.json`.
- Copy `config.example.json` to `config.json` to start from a template.

## Layout

```
local-manager-ui/
  run                      launcher (no install)
  config.example.json      template config
  llama_manager_ui/
    __main__.py            argparse entry point
    core.py                config, GGUF reader, command build, async process control
    gtk_ui.py              GTK3 window, tabs, live log view
```

## Notes

- **Web mode** streams `llama-server` logs, polls `/health`, then opens the
  browser when ready. Stopping kills the server process group.
- **CLI mode** launches an interactive session in a terminal emulator
  (`gnome-terminal`, `konsole`, ...), since GTK cannot host an interactive TTY.
  If no terminal is found you get a warning.
- **Build Stop** sends `SIGTERM` and escalates to `SIGKILL` after 5s. The build
  script uses `set -euo pipefail`, so a cancelled build can leave a partially
  built tree; re-run or use `--clean`.
- GGUF headers are read lazily a few files at a time so the window appears
  instantly even with many large models.
