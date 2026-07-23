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
| **PyPI** (`pip install fastfields-dlpack`) | the default **CUDA** wheel (`cu128`) | **CPU** wheel (no CUDA on macOS) |
| **this index** (`--extra-index-url .../<backend>/`) | `cpu`, `cu118`, `cu126`, `cu128` | `cpu` |

Only `fastfields-dlpack` (which bundles the compiled `libfastfields*`) is built
per-backend; the pure-Python wrappers (`fastfields-numpy`/`-torch`/`-cupy`,
`fastfields`) are universal wheels and appear in every folder.

**CUDA build target.** The wheels are compiled *fat*: one binary targets many
GPU architectures (SASS for several `sm_*` plus a forward-compatible PTX), so a
single wheel runs on as many GPUs as possible at the cost of size and build
time. The PyPI default is the **newest broadly-supported** toolkit so it also
covers the latest architectures: **`cu128`** spans Maxwell→Blackwell
(`sm_50`…`sm_120`, driver ≥ 570). The `cu118` line on the index keeps the long
tail alive (Kepler…Ampere on older drivers ≥ 450); `cu126` sits in between. See
the package build workflow for the exact `-gencode` list.

### Mixing CUDA versions with PyTorch / CuPy

You can install a fastfields build compiled against a **different** CUDA version
than your PyTorch/CuPy — they load their own CUDA runtimes side by side and
interoperate through DLPack device pointers, which are runtime-version-agnostic.
The only hard requirement is that your **GPU driver** satisfies the *newest*
toolkit among them (so a `cu128` fastfields needs a driver new enough for CUDA
12.8, even if torch is `cu126`). If you'd rather not raise your driver floor,
pick the index folder matching your torch build (`.../cu126/`).

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
