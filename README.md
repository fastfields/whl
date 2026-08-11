# fastfields wheel index

A PyTorch-style [PEP 503][pep503] "simple repository" for the fastfields Python
packages, published to GitHub Pages at **<https://fastfields.github.io/whl/>**.

Like PyTorch's `download.pytorch.org/whl`, it has **one folder per compute
backend** — `cpu/`, `cu118/`, `cu126/`, `cu130/`, … — so you can install a
build that matches your hardware. The compute backend is encoded in each
wheel's **local version label** (e.g. `fastfields_torch-0.1.0+cu130-…whl`),
exactly as PyTorch does.

> **Status:** the CUDA lanes (`cu118`, `cu126`, `cu130`) are **planned but not
> yet published** — no CUDA wheel has been built yet (that needs the CUDA link
> of `fastfields-lib`, tracked separately).
>
> A backend folder only becomes a real PEP 503 repository **once a wheel for it
> exists**. Until then the generator writes a plain "planned — not yet
> published" placeholder at `.../cu130/` and **no package folders underneath
> it**, so a resolver pointed there finds nothing for *any* project. That
> matters because the pure-Python wrapper wheels are otherwise mirrored into
> every backend folder: if the lane were served, `pip install fastfields-numpy
> --extra-index-url .../cu130/` would *succeed* and imply a CUDA lane that was
> never built, while the compiled wheel quietly came from PyPI. The lane turns
> itself on automatically as soon as a `+cu130` wheel is published — nothing
> here is hardcoded per backend.

## Installing

The index only serves the `fastfields-*` packages; ordinary dependencies
(numpy, torch, cupy) still resolve from PyPI, so pass it as an
**`--extra-index-url`**:

```sh
# CPU-only build (the only lane that will ever be published today)
pip install fastfields-numpy --extra-index-url https://fastfields.github.io/whl/cpu/
```

The CUDA lanes above are not installable from this index yet; see the status
note. The PyPI default build is unaffected (see *Distribution policy*).

> **No wheels are published at all yet.** None of the package repos has cut a
> GitHub Release, so the index is currently empty and even the `cpu` command
> above resolves to nothing. The landing page reflects this: it advertises a
> lane only once a wheel for it has been discovered.

## Distribution policy

Mirrors PyTorch's split between PyPI and the custom index:

| channel | Linux / Windows | macOS |
|---|---|---|
| **PyPI** (`pip install fastfields-dlpack`) | the default **CUDA** wheel (`cu130`) | **CPU** wheel (no CUDA on macOS) |
| **this index** (`--extra-index-url .../<backend>/`) | `cpu`, plus `cu118`, `cu126`, `cu130` *(planned)* | `cpu` |

This table is the **target** layout. See the status note above for what is
actually served today: the CUDA lanes are declared but publish nothing, and are
not served as installable folders until a wheel for them exists.

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
  committed here).
- `sources.toml` — the declared backends, the served projects, and the source
  repos whose Releases hold the wheels.
- `test_generate.py` — `unittest` suite for the generator (`python -m unittest
  test_generate`). It pins the two invariants below.
- `.github/workflows/build-index.yaml` — runs the tests, regenerates from the
  GitHub Releases API and deploys to Pages. Triggered on push to `main`, on a
  `repository_dispatch` (`wheels-updated`) ping from a package repo's release
  workflow, and manually — the index is **push-triggered, not polled**.
- `.github/workflows/test.yaml` — runs the tests on pull requests (the deploy
  workflow only runs on `main`, so this is a PR's only signal).

### Integrity: `#sha256=` is mandatory

A Pages-served `--extra-index-url` is a supply-chain surface, so **every link in
the index carries a `#sha256=` fragment** and `build()` refuses to emit one that
does not — the check sits at the single place links are written, so no discovery
path can bypass it.

| discovery path | where the digest comes from |
|---|---|
| `--manifest` | **required** on every `[[wheel]]` entry (we control that file) |
| `--from-releases` | a sibling `<wheel>.sha256` release asset if present, else **the wheel is downloaded and hashed** |

The digest must be well formed (64 hex characters); a truncated one is rejected
rather than written verbatim into a fragment that would later fail at
`pip install` time looking like a tampered wheel. If a digest cannot be
established at all, generation **fails** (exit 1) instead of publishing an
unverifiable link. Package release workflows should upload a matching `.sha256`
asset next to each wheel — that keeps the rebuild cheap, since the fallback has
to download every wheel (a fat CUDA wheel is hundreds of megabytes).

Local preview (offline, from a manifest):

```sh
python -m unittest test_generate         # run the tests
python generate.py --manifest manifest.example.toml --out public
python -m http.server -d public   # browse http://localhost:8000/
```

## One-time setup

In **Settings → Pages**, set the source to **GitHub Actions**. The first push to
`main` (or a manual *Run workflow*) then publishes the index.

[pep503]: https://peps.python.org/pep-0503/
