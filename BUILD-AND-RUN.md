# Build and Run Guide (Ubuntu 24.04 + NVIDIA CUDA)

Local step-by-step guide for cloning, building (CPU and CUDA), rebuilding after `git pull`, and running `llama-server` (including the `run-gpt-oss-server.sh` helper).

This guide matches a typical layout:

```text
/home/tjhexa/development/llama.cpp/
├── llama.cpp/          # this repository
│   ├── build/bin/      # compiled binaries (llama-server, etc.)
│   └── run-gpt-oss-server.sh
└── models/             # GGUF models (outside the repo)
    └── gpt-oss-20b-MXFP4.gguf
```

Official upstream docs (more backends and details):

- [docs/build.md](docs/build.md)
- [tools/server/README.md](tools/server/README.md)

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the repository (from scratch)](#2-clone-the-repository-from-scratch)
3. [Build: CPU only](#3-build-cpu-only)
4. [Build: CUDA (NVIDIA GPU)](#4-build-cuda-nvidia-gpu)
5. [Rebuild after `git pull`](#5-rebuild-after-git-pull)
6. [Clean rebuild](#6-clean-rebuild)
7. [Verify the build](#7-verify-the-build)
8. [Get a model (GGUF)](#8-get-a-model-gguf)
9. [Run with `run-gpt-oss-server.sh`](#9-run-with-run-gpt-oss-serversh)
10. [Run `llama-server` manually](#10-run-llama-server-manually)
11. [Important parameters (tuning guide)](#11-important-parameters-tuning-guide)
12. [Example recipes](#12-example-recipes)
13. [API and web UI usage](#13-api-and-web-ui-usage)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. Prerequisites

### 1.1 Common packages (CPU and CUDA)

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl wget
# optional but useful
sudo apt install -y ccache libssl-dev
```

- `ccache` speeds up repeated rebuilds
- `libssl-dev` enables HTTPS/TLS features in the server when needed

### 1.2 NVIDIA driver + CUDA toolkit (CUDA builds only)

1. Install a working NVIDIA driver so `nvidia-smi` works.
2. Install the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) (this machine commonly uses CUDA 12.x under `/usr/local/cuda`).

Check:

```bash
nvidia-smi
nvcc --version
# or
/usr/local/cuda/bin/nvcc --version
```

Add CUDA to your shell (put in `~/.bashrc` if you want it permanent):

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
```

Reload:

```bash
source ~/.bashrc
```

### 1.3 Hardware notes

| Build type | Needs GPU driver? | Needs CUDA toolkit? | Typical use |
| ---------- | ----------------- | ------------------- | ----------- |
| CPU        | No                | No                  | Laptops, servers without NVIDIA GPU |
| CUDA       | Yes               | Yes (`nvcc`)        | Fast inference on NVIDIA GPUs |

---

## 2. Clone the repository (from scratch)

```bash
mkdir -p /home/tjhexa/development/llama.cpp
cd /home/tjhexa/development/llama.cpp

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

Create a models directory next to the repo (recommended):

```bash
mkdir -p /home/tjhexa/development/llama.cpp/models
```

---

## 3. Build: CPU only

From the repo root:

```bash
cd /home/tjhexa/development/llama.cpp/llama.cpp

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

What this does:

- configures a Release build in `build/`
- compiles with all CPU cores (`-j$(nproc)`)
- writes binaries to `build/bin/`

Optional OpenBLAS (can help prompt processing on CPU):

```bash
sudo apt install -y libopenblas-dev
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
cmake --build build --config Release -j$(nproc)
```

---

## 4. Build: CUDA (NVIDIA GPU)

Recommended on Ubuntu 24.04 with an NVIDIA GPU.

```bash
cd /home/tjhexa/development/llama.cpp/llama.cpp

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

Binaries again land in `build/bin/` (for example `llama-server`).

### 4.1 Useful CUDA CMake options

| Option | Purpose |
| ------ | ------- |
| `-DGGML_CUDA=ON` | Enable NVIDIA CUDA backend |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build |
| `-DGGML_NATIVE=OFF` | Broader GPU compatibility (larger/slower compile) |
| `-DCMAKE_CUDA_ARCHITECTURES="86;89"` | Target specific GPU compute capabilities |

Examples:

```bash
# Build for all common CUDA archs (portable binary, longer compile)
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

```bash
# Target specific GPUs (example: RTX 30xx = 86, RTX 40xx = 89)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

If you have multiple CUDA installs:

```bash
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

---

## 5. Rebuild after `git pull`

After pulling new changes, reconfigure then rebuild so CMake picks up new options/files.

```bash
cd /home/tjhexa/development/llama.cpp/llama.cpp

git pull

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# CUDA rebuild (recommended if you use the GPU)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

CPU-only rebuild:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

If your previous `build/` was already configured with CUDA (`GGML_CUDA=ON` in `build/CMakeCache.txt`), the same commands above keep that setting as long as you pass `-DGGML_CUDA=ON` again (safe and clear).

Updated binaries:

```bash
ls -lh build/bin/
```

---

## 6. Clean rebuild

Use this if configuration looks wrong, CUDA options changed, or the build is broken after a large update.

```bash
cd /home/tjhexa/development/llama.cpp/llama.cpp

rm -rf build

# CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

CPU clean rebuild:

```bash
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

---

## 7. Verify the build

```bash
./build/bin/llama-server --version
./build/bin/llama-server --list-devices
```

Expect:

- version / build info printed
- with CUDA: your NVIDIA device listed
- CUDA shared libs present under `build/bin/` (for example `libggml-cuda.so*`)

Quick binary check:

```bash
ls -lh build/bin/llama-server
file build/bin/llama-server
```

---

## 8. Get a model (GGUF)

Place GGUF files under:

```text
/home/tjhexa/development/llama.cpp/models/
```

Example already used by the helper script:

```text
/home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf
```

Ways to obtain models:

1. Download a GGUF from Hugging Face (or elsewhere) into `models/`.
2. Let the server download via `-hf` / `--hf-repo` (see examples below).

Quantization tip:

- smaller quants (Q4, MXFP4, etc.) use less VRAM/RAM and run faster
- larger quants (Q5/Q6/Q8/F16) usually quality better but need more memory

---

## 9. Run with `run-gpt-oss-server.sh`

For a fuller interactive launcher (scan all models, show RAM/VRAM + context length, choose **web** or **cli**), use [`local-manager/`](local-manager/README.md):

```bash
cd local-manager && ./run
```

Helper script in the repo root: [`run-gpt-oss-server.sh`](run-gpt-oss-server.sh).

### 9.1 What the script does

Current behavior:

| Item | Value |
| ---- | ----- |
| Binary | `/home/tjhexa/development/llama.cpp/llama.cpp/build/bin/llama-server` |
| Model | `/home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf` |
| Port | `9898` |
| Host | `0.0.0.0` (reachable on the LAN) |
| GPU layers | `-ngl 99` (offload many layers to GPU) |
| UI URL | `http://127.0.0.1:9898` |

Flow:

1. If something is already listening on port `9898`, it opens the browser and exits.
2. Otherwise it starts `llama-server`, waits until the server responds, then opens the browser with `xdg-open`.

### 9.2 Prerequisites before running the script

1. Build completed (`build/bin/llama-server` exists).
2. Model file exists at the path in the script.
3. For GPU: NVIDIA driver works (`nvidia-smi`).
4. Port `9898` is free (or already running the same server).

### 9.3 Start the server

```bash
cd /home/tjhexa/development/llama.cpp/llama.cpp
chmod +x run-gpt-oss-server.sh   # once
./run-gpt-oss-server.sh
```

Then open:

```text
http://127.0.0.1:9898
```

From another machine on the same network (because `--host 0.0.0.0`):

```text
http://<your-machine-ip>:9898
```

### 9.4 Customize the script

Edit these variables / flags in `run-gpt-oss-server.sh`:

```bash
PORT=9898
BIN="/home/tjhexa/development/llama.cpp/llama.cpp/build/bin/llama-server"
MODEL="/home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf"
```

And the `exec` arguments:

```bash
exec "$BIN" \
  -m "$MODEL" \
  -ngl 99 \
  --host 0.0.0.0 \
  --port "$PORT"
```

Common edits:

| Change | How |
| ------ | --- |
| Different model | Change `MODEL=...` |
| Different port | Change `PORT=...` |
| Localhost only | Use `--host 127.0.0.1` |
| More/less GPU offload | Change `-ngl` (or use `all` / `auto`) |
| Larger context | Add `-c 8192` (or another size) |
| More concurrent chats | Add `-np 2` (or higher) |
| CPU-only run | Use `-ngl 0` |

Example customized `exec` block:

```bash
exec "$BIN" \
  -m "$MODEL" \
  -ngl 99 \
  -c 8192 \
  -fa on \
  -np 1 \
  --temp 0.7 \
  --top-p 0.9 \
  --host 0.0.0.0 \
  --port "$PORT"
```

### 9.5 Stop the server

In the terminal where it is running: `Ctrl+C`.

If started in the background, find and kill it:

```bash
ss -ltnp | grep 9898
# or
pgrep -a llama-server
kill <pid>
```

---

## 10. Run `llama-server` manually

You do not need the helper script. Direct commands:

### 10.1 Minimal CUDA run (same idea as the script)

```bash
cd /home/tjhexa/development/llama.cpp/llama.cpp

./build/bin/llama-server \
  -m /home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 9898
```

### 10.2 CPU-only run

```bash
./build/bin/llama-server \
  -m /home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf \
  -ngl 0 \
  -t $(nproc) \
  --host 127.0.0.1 \
  --port 8080
```

### 10.3 Download a model from Hugging Face at start

```bash
./build/bin/llama-server \
  -hf ggml-org/gpt-oss-20b-GGUF \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 9898
```

(Exact repo/quant names vary; adjust `-hf` / `-hff` as needed.)

### 10.4 Full help

```bash
./build/bin/llama-server -h
```

Upstream reference: [tools/server/README.md](tools/server/README.md).

---

## 11. Important parameters (tuning guide)

Most flags also have env vars like `LLAMA_ARG_*` (see server help). Below focuses on practical tuning.

### 11.1 Model and network

| Flag | Meaning | Tips |
| ---- | ------- | ---- |
| `-m, --model` | Path to GGUF | Required unless using `-hf` / `-mu` |
| `-hf, --hf-repo` | Hugging Face repo | Optional auto-download |
| `--host` | Bind address | `127.0.0.1` = local only; `0.0.0.0` = LAN |
| `--port` | TCP port | Script uses `9898`; default is `8080` |
| `-a, --alias` | Model name alias for API | Useful for OpenAI-compatible clients |

### 11.2 GPU / memory

| Flag | Meaning | Tips |
| ---- | ------- | ---- |
| `-ngl, --n-gpu-layers` | Layers on GPU | `99` / `all` = heavy offload; `0` = CPU; `auto` = let server decide |
| `-sm, --split-mode` | Multi-GPU split | `layer` (default), `row`, `none` |
| `-ts, --tensor-split` | Per-GPU fractions | e.g. `3,1` |
| `-mg, --main-gpu` | Main GPU index | Default `0` |
| `-fa, --flash-attn` | Flash Attention | `on` / `off` / `auto` (often helps speed/VRAM) |
| `-ctk / -ctv` | KV cache type | `q8_0` / `q4_0` saves VRAM vs `f16` |
| `-fit` | Auto-fit to VRAM | Default `on`; reduces context/layers if needed |
| `-cmoe / -ncmoe` | Keep MoE experts on CPU | Helps huge MoE models fit in VRAM |

VRAM rule of thumb:

1. If OOM: lower `-c`, lower `-ngl`, use a smaller quant, or compress KV (`-ctk q8_0 -ctv q8_0`).
2. If VRAM free but slow: raise `-ngl` toward `all`.

### 11.3 Context and batching

| Flag | Meaning | Tips |
| ---- | ------- | ---- |
| `-c, --ctx-size` | Context window tokens | Larger = more memory; try `4096`, `8192`, `16384` |
| `-n, --n-predict` | Max new tokens | `-1` = unlimited (until stop) |
| `-b, --batch-size` | Logical batch size | Higher can speed prompt ingest; uses more memory |
| `-ub, --ubatch-size` | Physical micro-batch | Tune if prompt processing is unstable/slow |
| `-np, --parallel` | Concurrent slots | `1` for single user; `2+` for multi-user (more KV memory) |
| `-cb, --cont-batching` | Continuous batching | Keep enabled for multi-request efficiency |

### 11.4 CPU threads

| Flag | Meaning | Tips |
| ---- | ------- | ---- |
| `-t, --threads` | Generation threads | Often `$(nproc)` or slightly less |
| `-tb, --threads-batch` | Prompt/batch threads | Can equal `-t` or higher for prompt-heavy loads |

### 11.5 Sampling (output style)

| Flag | Meaning | Tips |
| ---- | ------- | ---- |
| `--temp` | Temperature | Lower = more deterministic; higher = more creative (default `0.8`) |
| `--top-k` | Top-k sampling | `40` default; `0` disables |
| `--top-p` | Nucleus sampling | `0.9`-`0.95` common |
| `--min-p` | Min-p sampling | Default `0.05`; `0` disables |
| `--repeat-penalty` | Repetition penalty | Slightly above `1.0` if looping |
| `--presence-penalty` / `--frequency-penalty` | OpenAI-style penalties | Start small (`0.0`-`0.5`) |
| `--seed` | RNG seed | Fixed seed for reproducibility; `-1` = random |
| `--mirostat` | Mirostat sampler | `0` off; `2` = Mirostat 2.0 |

Quick style presets:

```bash
# more focused / factual
--temp 0.3 --top-p 0.9 --top-k 40

# balanced chat
--temp 0.7 --top-p 0.95 --min-p 0.05

# more creative
--temp 1.1 --top-p 0.98 --top-k 0
```

### 11.6 Logging and debug

| Flag | Meaning |
| ---- | ------- |
| `-v, --verbose` | Very detailed logs |
| `-lv, --verbosity N` | `0`..`5` log threshold |
| `--log-file FNAME` | Write logs to file |
| `--perf` | Enable internal timing stats |

---

## 12. Example recipes

### 12.1 Fast single-user GPU chat (matches script spirit)

```bash
./build/bin/llama-server \
  -m /home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf \
  -ngl 99 \
  -c 8192 \
  -fa on \
  --host 0.0.0.0 \
  --port 9898
```

### 12.2 Low VRAM

```bash
./build/bin/llama-server \
  -m /home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf \
  -ngl 40 \
  -c 4096 \
  -ctk q8_0 -ctv q8_0 \
  -fa on \
  --port 9898
```

### 12.3 Multi-user on one GPU

```bash
./build/bin/llama-server \
  -m /home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf \
  -ngl 99 \
  -c 8192 \
  -np 4 \
  -cb \
  --host 0.0.0.0 \
  --port 9898
```

### 12.4 CPU-only workstation

```bash
./build/bin/llama-server \
  -m /home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf \
  -ngl 0 \
  -t $(nproc) \
  -tb $(nproc) \
  -c 4096 \
  --host 127.0.0.1 \
  --port 8080
```

### 12.5 Two NVIDIA GPUs

```bash
./build/bin/llama-server \
  -m /path/to/model.gguf \
  -ngl 99 \
  -sm layer \
  -ts 1,1 \
  --port 9898
```

---

## 13. API and web UI usage

### 13.1 Web UI

Open the server URL in a browser:

```text
http://127.0.0.1:9898
```

### 13.2 OpenAI-compatible chat (curl)

```bash
curl http://127.0.0.1:9898/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "temperature": 0.7
  }'
```

Point OpenAI-compatible clients at:

```text
Base URL: http://127.0.0.1:9898/v1
```

### 13.3 Health / props

```bash
curl -s http://127.0.0.1:9898/health
curl -s http://127.0.0.1:9898/props | head
```

---

## 14. Troubleshooting

### Build fails: `nvcc` not found

```bash
export PATH=/usr/local/cuda/bin:$PATH
which nvcc
```

Re-run cmake with `-DGGML_CUDA=ON`.

### Runtime: CUDA init fails / no GPU

```bash
nvidia-smi
./build/bin/llama-server --list-devices
```

- Reboot after driver installs
- Confirm you built with `-DGGML_CUDA=ON`
- Confirm `LD_LIBRARY_PATH` includes `/usr/local/cuda/lib64` if needed

### Out of memory (VRAM/RAM)

- Reduce `-c`
- Reduce `-ngl`
- Use a smaller GGUF quant
- Try `-ctk q8_0 -ctv q8_0`
- Set `-np 1`

### Port already in use

```bash
ss -ltnp | grep 9898
```

Change `PORT` in the script, or stop the old process.

### Script opens browser but chat fails

- Wait until model load finishes (large models take time)
- Check terminal logs for errors
- Confirm the model path exists

### Stale binary after pull

Always reconfigure + rebuild (section 5). If odd errors remain, clean rebuild (section 6).

### Slow CPU generation

- Build Release (`-DCMAKE_BUILD_TYPE=Release`)
- Set `-t $(nproc)`
- Prefer a smaller quant
- Prefer CUDA build if you have an NVIDIA GPU

---

## Quick reference card

```bash
# CUDA build / rebuild
cd /home/tjhexa/development/llama.cpp/llama.cpp
export PATH=/usr/local/cuda/bin:$PATH
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Run helper
./run-gpt-oss-server.sh

# Or manual
./build/bin/llama-server \
  -m /home/tjhexa/development/llama.cpp/models/gpt-oss-20b-MXFP4.gguf \
  -ngl 99 --host 0.0.0.0 --port 9898
```

Binary location: `build/bin/llama-server`  
Web UI: `http://127.0.0.1:9898`
