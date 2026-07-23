# fastfields wheel index

A PyTorch-style [PEP 503][pep503] "simple repository" for the fastfields Python
packages, published to GitHub Pages at **<https://fastfields.github.io/whl/>**.

Like PyTorch's `download.pytorch.org/whl`, it has **one folder per compute
backend** — `cpu/`, `cu118/`, `cu124/`, … — so you can install a build that
matches your hardware. The compute backend is encoded in each wheel's **local
version label** (e.g. `fastfields_torch-0.1.0+cu124-…whl`), exactly as PyTorch
does.

## Installing

The index only serves the `fastfields-*` packages; ordinary dependencies
(numpy, torch, cupy) still resolve from PyPI, so pass it as an
**`--extra-index-url`**:

```sh
# CUDA 12.4 build
pip install fastfields-torch --extra-index-url https://fastfields.github.io/whl/cu124/

# CPU-only build
pip install fastfields-numpy --extra-index-url https://fastfields.github.io/whl/cpu/
```

## Distribution policy

Mirrors PyTorch's split between PyPI and the custom index:

| channel | Linux / Windows | macOS |
|---|---|---|
| **PyPI** (`pip install fastfields-torch`) | the default **CUDA** wheel (`cu124`) | **CPU** wheel (no CUDA on macOS) |
| **this index** (`--extra-index-url .../<backend>/`) | `cpu`, `cu118`, `cu124`, … | `cpu` |

**CUDA build target.** The wheels are compiled *fat*: one binary targets many
GPU architectures (SASS for several `sm_*` plus a forward-compatible PTX), so a
single wheel runs on as many GPUs as possible at the cost of size and build
time. The PyPI default (`cu124`) covers Maxwell→Hopper (`sm_50`…`sm_90`); the
`cu118` line on the index reaches older Kepler/Pascal drivers. See the package
build workflows for the exact `-gencode` list.

## How it is built

- `generate.py` — stdlib-only generator that emits the PEP 503 HTML tree into
  `public/`. It buckets wheels by their local version label and links to the
  wheel files (hosted as **GitHub Release assets** on each package repo, not
  committed here). Digests are added as `#sha256=` when known.
- `sources.toml` — the advertised backends, the served projects, and the source
  repos whose Releases hold the wheels.
- `.github/workflows/build-index.yaml` — regenerates from the GitHub Releases
  API and deploys to Pages on push, on a daily schedule, and on a
  `repository_dispatch` (`wheels-updated`) ping from a package repo's release
  workflow.

Local preview (offline, from a manifest):

```sh
python generate.py --manifest manifest.example.toml --out public
python -m http.server -d public   # browse http://localhost:8000/
```

## One-time setup

In **Settings → Pages**, set the source to **GitHub Actions**. The first push to
`main` (or a manual *Run workflow*) then publishes the index.

[pep503]: https://peps.python.org/pep-0503/
