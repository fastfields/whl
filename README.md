# fastfields wheel index

A PyTorch-style [PEP 503][pep503] "simple repository" for the fastfields Python
packages, published to GitHub Pages at **<https://fastfields.github.io/whl/>**.

Like PyTorch's `download.pytorch.org/whl`, it has **one folder per compute
backend** — `cpu/`, `cu118/`, `cu126/`, `cu130/`, … — so you can install a
build that matches your hardware. The compute backend is encoded in each
wheel's **local version label** (e.g. `fastfields_torch-0.1.0+cu130-…whl`),
exactly as PyTorch does.

> **Status:** only the **`cpu`** lane is published today. The CUDA lanes
> (`cu118`, `cu126`, `cu130`) are **planned but not yet published** — the
> per-backend folders exist, but no CUDA wheel has been built yet, so passing
> a `.../cu130/` folder as an `--extra-index-url` currently resolves to
> nothing. The generated landing page lists such backends under *"planned — not
> yet published"* and only shows a `pip install` command once at least one
> wheel is discovered for that backend.

## Installing

The index only serves the `fastfields-*` packages; ordinary dependencies
(numpy, torch, cupy) still resolve from PyPI, so pass it as an
**`--extra-index-url`**:

```sh
# CPU-only build (the only lane published today)
pip install fastfields-numpy --extra-index-url https://fastfields.github.io/whl/cpu/
```

The CUDA lanes above are not installable from this index yet; see the status
note. The PyPI default build is unaffected (see *Distribution policy*).

## Distribution policy

Mirrors PyTorch's split between PyPI and the custom index:

| channel | Linux / Windows | macOS |
|---|---|---|
| **PyPI** (`pip install fastfields-dlpack`) | the default **CUDA** wheel (`cu130`) | **CPU** wheel (no CUDA on macOS) |
| **this index** (`--extra-index-url .../<backend>/`) | `cpu`, `cu118`, `cu126`, `cu130` | `cpu` |

Only `fastfields-dlpack` (which bundles the compiled `libfastfields*`) is built
per-backend; the pure-Python wrappers (`fastfields-numpy`/`-torch`/`-cupy`,
`fastfields`) are universal wheels and appear in every folder.

**CUDA build target.** The wheels are compiled *fat*: one binary targets many
GPU architectures (SASS for several `sm_*` plus a trailing forward-compatible
PTX entry the driver can JIT for newer GPUs), so a single wheel runs on as many
GPUs as possible at the cost of size and build time. What a wheel can reach is
set by the **nvcc version** it was built with — not by our source — so there is
one lane per CUDA major:

| lane | built with | reaches | min driver |
|---|---|---|---|
| `cu118` | nvcc 11.8 | Kepler/Maxwell → Ada (newest via PTX JIT) | ~ r450+ |
| `cu126` | a 12.x (e.g. 12.6) | Volta → Hopper/Ada | ~ r525+ |
| `cu130` | a 13.x (e.g. 13.0) | Turing → Blackwell (`sm_75`+) | ~ r580+ |

Every lane compiles the same sources; only the nvcc version and the `-gencode`
list differ. The lanes are **additive, not nested**: `cu130` reaches the newest
architectures but *drops* everything before Turing — Maxwell, Pascal and Volta
are gone from CUDA 13's offline-compile floor — which is exactly what `cu118` is
for. The PyPI default is the newest lane, **`cu130`**, so the default wheel
covers the latest architectures; if your GPU is pre-Turing or your driver is
older than ~r580, take `cu118` (or `cu126`) from this index instead. See the
package build workflow for the exact `-gencode` list.

### Mixing CUDA versions with PyTorch / CuPy

You can install a fastfields build compiled against a **different** CUDA version
than your PyTorch/CuPy — they load their own CUDA runtimes side by side and
interoperate through DLPack device pointers, which are runtime-version-agnostic.
The only hard requirement is that your **GPU driver** satisfies the *newest*
toolkit among them (so a `cu130` fastfields needs a driver new enough for CUDA
13.x, ~r580+, even if torch is `cu126` and happy on ~r525+). If you'd rather not
raise your driver floor, pick the index folder matching your torch build
(`.../cu126/`). Note that the driver floor is not the only constraint: a lane
also has to *cover your GPU* — `cu130` is `sm_75`+ only, so a Pascal or Volta
card needs `cu118` no matter how new the installed driver is.

## How it is built

- `generate.py` — stdlib-only generator that emits the PEP 503 HTML tree into
  `public/`. It buckets wheels by their local version label and links to the
  wheel files (hosted as **GitHub Release assets** on each package repo, not
  committed here). Digests are added as `#sha256=`. For the `--manifest` path a
  `sha256` is **required** on every `[[wheel]]` entry (we control that file, so
  the generator errors out on a missing digest). The `--from-releases` path
  reads the digest from a sibling `<wheel>.sha256` release asset when present
  and emits a `::warning::` for any wheel lacking one, since the GitHub
  Releases API exposes no per-asset digest — package release workflows should
  upload a matching `.sha256` next to each wheel.
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
