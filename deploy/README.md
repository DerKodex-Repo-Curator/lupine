# Deploying lupine in a homelab

Lupine bridges GPUs on remote machines to CPU-only machines over HTTP/2 with
LZ4 compression. In a homelab this means: GPU hosts run the **server**, and
CPU-only / arm64 hosts run **clients** (and client-derived app images) that
forward CUDA calls to those servers.

This guide covers the production-ready Compose stacks and app Dockerfiles under
`deploy/`.

---

## Architecture overview

```
        GPU host (amd64)                 CPU / arm64 host
  ┌───────────────────────┐         ┌─────────────────────────┐
  │  lupine-server         │ ◀─────  │  app image              │
  │  (port 14833, HTTP/2)  │  HTTP/2 │  FROM lupine-client-slim │
  │  GPU: all              │  +LZ4   │  LUPINE_SERVER=host:14833│
  └───────────────────────┘         └─────────────────────────┘
         ▲ tsdproxy (optional)              ▲
         │ MagicDNS + TLS                    │ no --gpus, no LD_PRELOAD
```

- **Server** = `lupine_driver_server`, amd64-only, needs `--gpus all`.
- **Client slim** = `libcuda.so.1` + `libnvidia-ml.so.1` shims only, multi-arch
  (amd64 + arm64). Sets `LD_LIBRARY_PATH=/opt/lupine/lib` automatically.
- **Client full** = slim + CUDA runtime libraries. Fallback for apps that need
  CUDA runtime symbols the slim shim doesn't export (e.g. some vLLM builds).
- App images (Ollama, PyTorch, vLLM) `FROM` the client-slim image and add their
  own stack on top.

---

## Prerequisites

- **Docker** with the Compose v2 plugin on every host.
- **NVIDIA driver** + **NVIDIA Container Toolkit** (`nvidia-utils`,
  `nvidia-container-toolkit`) on GPU hosts. Verify with
  `docker run --rm --gpus all nvidia/cuda:13.1.0-base-ubuntu24.04 nvidia-smi`.
- **Registry access** to `registry.xaas.ar` (or rebuild the images from the
  root `Dockerfile` and push to your own registry — update the `image:` /
  `FROM` lines accordingly).
- **A proxy network** if you use tsdproxy / Traefik / NPM: create it once with
  `docker network create proxy`. If you don't run a reverse proxy, drop the
  `networks` blocks from the compose files.

---

## Server deployment

Run on every GPU host:

```bash
docker compose -f deploy/compose/lupine-server.yml up -d
```

What it does:
- Pulls `registry.xaas.ar/lupine-server:cuda-13.1.0-ubuntu24.04` (amd64-only).
- Reserves all GPUs via `deploy.resources.reservations.devices`
  (`driver: nvidia`, `count: all`, `capabilities: [compute, utility]`).
- Publishes `14833:14833`.
- Healthchecks the **TCP port** (the server has no HTTP endpoint):
  `bash -c 'echo > /dev/tcp/127.0.0.1/14833'`.
- Registers tsdproxy labels for MagicDNS + TLS (see "tsdproxy labels").
- `restart: unless-stopped`.

To run multiple GPU hosts, copy the same file to each host. Each server
exposes its own GPUs; combine them from a single client with a comma-separated
`LUPINE_SERVER` (see "Multi-server").

### Server environment variables

| Variable | Purpose | Default |
|---|---|---|
| `LUPINE_PORT` | Listen port | `14833` |
| `LUPINE_TRACE` | Trace: `0` off, `1` stdout, `2` stderr, other = file path | off |
| `LUPINE_SESSION` | Stable id for checkpoint restore across restarts | — |
| `LUPINE_CHECKPOINT_LIBRARY` | Override checkpoint provider library path | auto-detect |

---

## Client deployment

### 1. Pick an app Dockerfile

Three are provided under `deploy/`:

| File | Base | Notes |
|---|---|---|
| `Dockerfile.ollama` | client-slim | Ollama bundles its own CUDA libs; only needs the shim. |
| `Dockerfile.pytorch` | client-slim | PyTorch **CPU-only** wheel — lazy dlopen, arm64-safe. |
| `Dockerfile.vllm` | client-slim | May need full client image as fallback (see below). |

Build one:

```bash
docker build -f deploy/Dockerfile.pytorch -t lupine-pytorch:cuda-13.1 .
```

### 2. Run it with `LUPINE_SERVER` set

Use the client template compose, or `docker run` directly.

**Direct (plaintext):**
```bash
docker run --rm -e LUPINE_SERVER=gpu-host-a:14833 lupine-pytorch:cuda-13.1 \
  python3 -c "import torch; print(torch.cuda.is_available())"
```

**TLS via reverse proxy** (`https://` enables TLS and defaults port to 443):
```bash
docker run --rm -e LUPINE_SERVER=https://lupine-server.magicdns.dev \
  lupine-pytorch:cuda-13.1 ...
```

**Multi-server** (comma-separated; devices exposed in server order):
```bash
docker run --rm \
  -e LUPINE_SERVER=gpu-host-a:14833,gpu-host-b:14833 \
  lupine-pytorch:cuda-13.1 \
  nvidia-smi -L
```

### Using the template compose

Copy `deploy/compose/lupine-client-template.yml`, point
`build.dockerfile` at your app Dockerfile, and set `LUPINE_SERVER` either in
the `environment:` block or via a `.env` file next to the compose file:

```env
# .env
LUPINE_SERVER=gpu-host-a:14833
```

```bash
docker compose -f my-app.yml up -d
```

---

## The LUPINE_SERVER timing rule (IMPORTANT)

`LUPINE_SERVER` is read **once**, on the **first CUDA initialization** in the
process. Implications:

1. It MUST be a compose `environment:` value (or baked into the image), not a
   runtime override. `docker exec ... export LUPINE_SERVER=...` does NOT affect
   the already-running process.
2. Changing it requires **restarting the container**, not just re-exporting
   the variable.
3. `https://` in the value enables TLS and defaults the port to 443. Plain
   hostnames and `http://` default to 14833.

---

## App Dockerfiles

### Ollama (`Dockerfile.ollama`)

Ollama bundles its own CUDA runtime libraries; it only needs the
`libcuda.so.1` **shim** from the lupine client-slim image. No `--gpus`, no
`LD_PRELOAD` — the slim image sets `LD_LIBRARY_PATH=/opt/lupine/lib` so the
shim is found first.

```bash
docker build -f deploy/Dockerfile.ollama -t lupine-ollama:cuda-13.1 .
docker run --rm -e LUPINE_SERVER=gpu-host-a:14833 -p 11434:11434 \
  lupine-ollama:cuda-13.1
```

### PyTorch (`Dockerfile.pytorch`)

Uses the **CPU-only** PyTorch wheel (`--index-url https://download.pytorch.org/whl/cpu`).
PyTorch's CPU wheel dlopens `libcuda.so.1` **lazily** — only when you call a
CUDA API — so the lupine shim intercepts those calls and forwards them. This
is the safe path on arm64. No CMD is provided; supply your own entrypoint.

```bash
docker build -f deploy/Dockerfile.pytorch -t lupine-pytorch:cuda-13.1 .
docker run --rm -e LUPINE_SERVER=gpu-host-a:14833 lupine-pytorch:cuda-13.1 \
  python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> Do NOT install a CUDA-enabled torch wheel on the client. It would link
> against `libcuda.so.1` at import time and may pull symbols the slim shim
> doesn't export.

### vLLM (`Dockerfile.vllm`)

vLLM is the most demanding of the three. It may need CUDA runtime symbols that
the slim client shim does not export. **If the build or a smoke run fails with
a missing CUDA runtime symbol**, switch the `FROM` line to the full client
image which keeps the CUDA runtime libraries:

```dockerfile
FROM registry.xaas.ar/lupine-client:cuda-13.1.0-ubuntu24.04
```

Smoke test before relying on it:

```bash
docker build -f deploy/Dockerfile.vllm -t lupine-vllm:cuda-13.1 .
docker run --rm -e LUPINE_SERVER=gpu-host-a:14833 -p 8000:8000 \
  lupine-vllm:cuda-13.1 \
  vllm serve Qwen/Qwen2.5-0.5B --host 0.0.0.0 --port 8000
```

---

## arm64 gotcha

arm64 hosts **must** use the slim client + apps that dlopen `libcuda.so.1`
**lazily** (e.g. PyTorch CPU-only wheels, Ollama). Apps that **link** against
`libcuda.so.1` at build time will fail to load on arm64 because the slim image
does not ship the full CUDA runtime libraries that the loader expects.

Checklist for arm64:
- ✅ `FROM ...-slim` + lazy-dlopen apps (PyTorch CPU wheel, Ollama).
- ❌ Apps that `ld` against `-lcuda` or import a CUDA-enabled torch wheel.
- If an app genuinely needs CUDA runtime libs on arm64, derive from the full
  client image instead: `FROM registry.xaas.ar/lupine-client:cuda-13.1.0-ubuntu24.04`.

The server itself is **amd64-only**; arm64 hosts are always clients.

---

## Multi-server

Run a server on each GPU host, then point a single client at all of them with
a comma-separated list:

```env
LUPINE_SERVER=gpu-host-a:14833,gpu-host-b:14833
```

Devices are exposed as one local ordinal list in server order: all GPUs from
the first server, then all GPUs from the next, and so on. Cross-server
device-to-device and peer copies are supported (transparently staged through
the client); direct server-to-server transfers are not implemented yet.

---

## tsdproxy labels

The server compose file registers tsdproxy labels for MagicDNS + TLS:

```yaml
labels:
  - "tsdproxy.enable=true"
  - "tsdproxy.name=lupine-server"
  - "tsdproxy.port=14833"
```

The **exact label format depends on your tsdproxy config** (container name,
network attachment, cert resolver, etc.). These labels are the standard
`tsdproxy.*` set; adjust or add labels (e.g. `tsdproxy.https.enable=true`,
`tsdproxy.tls.certresolver=...`) to match your instance. If you don't use
tsdproxy, delete the `labels:` block and the `proxy` network.

With tsdproxy in front, clients can reach the server over TLS using the
MagicDNS name:

```env
LUPINE_SERVER=https://lupine-server.magicdns.dev
```

---

## Registry image reference

| Image | Arch | Purpose |
|---|---|---|
| `registry.xaas.ar/lupine-server:cuda-13.1.0-ubuntu24.04` | amd64 | Server binary, run on GPU hosts. |
| `registry.xaas.ar/lupine-client:cuda-13.1.0-ubuntu24.04-slim` | amd64 + arm64 | Shims + nvidia-smi only. Base for most apps. |
| `registry.xaas.ar/lupine-client:cuda-13.1.0-ubuntu24.04` | amd64 + arm64 | Slim + CUDA runtime libs. Fallback for apps needing runtime symbols. |

To rebuild from source instead of pulling, use the root `Dockerfile`:

```bash
# Build and load into a local registry
docker buildx build --target client-slim -t registry.xaas.ar/lupine-client:cuda-13.1.0-ubuntu24.04-slim --push .
docker buildx build --target server      -t registry.xaas.ar/lupine-server:cuda-13.1.0-ubuntu24.04 --push .
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` False | `LUPINE_SERVER` unset or unreachable | Set it in compose `environment:`; verify server health. |
| Changing `LUPINE_SERVER` has no effect | Read once on first CUDA init | Restart the container. |
| arm64 app fails to load `libcuda.so.1` | App links against libcuda at build time | Use a lazy-dlopen app (CPU torch wheel) or full client image. |
| vLLM missing CUDA runtime symbol | Slim shim doesn't export it | Switch `FROM` to full client image. |
| Healthcheck unhealthy | Server not up / port wrong | TCP-probe 14833; check `docker logs lupine-server`. |
| TLS handshake fails | Cert not trusted by client | Ensure proxy cert is in client's trust store, or use direct mode. |