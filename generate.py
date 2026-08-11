#!/usr/bin/env python3
"""Generate a PyTorch-style PEP 503 wheel index for the fastfields packages.

The index is a set of static HTML pages laid out as a *simple repository*
(:pep:`503`), one **backend** subfolder per compute target
(``cpu/``, ``cu118/``, ``cu126/``, ``cu130/`` ...). It is published to GitHub
Pages at ``https://fastfields.github.io/whl/`` so users can install a build
matching their hardware::

    pip install fastfields-torch \\
        --extra-index-url https://fastfields.github.io/whl/cpu/

The wheels themselves are **not** committed here -- they are hosted as GitHub
*Release* assets on each package repo. This script only emits the HTML that
links to them, each with a mandatory ``#sha256=`` fragment.

A backend folder is only emitted as a real :pep:`503` repository once at least
one wheel has been discovered for it. A backend that is merely *declared* in
``sources.toml`` gets a plain human-readable "planned -- not yet published"
placeholder page instead, and **no project subfolders** -- so a resolver
pointed at that folder finds nothing at all rather than being served the
universal wheels under a compute lane that was never built.

Digests are **mandatory**: every link in the index carries a ``#sha256=``
fragment, and :func:`build` refuses to emit one without it. For the
``--manifest`` path the digest is required on every ``[[wheel]]`` entry (we
control that file). The ``--from-releases`` path prefers a sibling
``<wheel>.sha256`` release asset and otherwise **downloads the wheel and
computes the digest**; a wheel whose digest cannot be established fails the
build rather than being published unhashed.

Wheel-to-backend mapping follows PyTorch's convention: the compute backend is
encoded in the wheel's **local version label**, e.g.
``fastfields_dlpack-0.1.0+cu130-cp311-cp311-linux_x86_64.whl``. A wheel with no
local label is *universal* (pure Python, e.g. the numpy/torch wrappers) and is
listed in every backend folder so each folder resolves on its own.

Two discovery modes:

* ``--manifest FILE`` -- read wheel entries from a TOML manifest (offline; used
  for local testing and reproducible builds).
* ``--from-releases`` -- query the GitHub Releases API for each source repo in
  ``sources.toml`` (used in CI; honours ``GITHUB_TOKEN``).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

WHEEL_RE = re.compile(
    r"^(?P<dist>.+?)-(?P<ver>\d[^-]*?)"
    r"(?:\+(?P<local>[a-zA-Z0-9.]+))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$"
)

#: A sha256 digest as it must appear in a ``#sha256=`` fragment.
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Read size for streaming a release asset through :mod:`hashlib`.
_CHUNK = 1 << 20


class MissingDigestError(Exception):
    """Raised when a discovered wheel has no usable ``sha256`` digest.

    The index is served as an ``--extra-index-url`` over GitHub Pages, which
    makes it a supply-chain surface: an unhashed link is one nobody can verify.
    Publishing a partial index would trade that for a *different* silent
    failure (a wheel that quietly disappears from the index), so a digest we
    cannot establish fails the whole build instead.
    """


def normalize(name: str) -> str:
    """Return the :pep:`503`-normalized form of a project name.

    Parameters
    ----------
    name : str
        A raw distribution name (e.g. ``fastfields_torch``).

    Returns
    -------
    str
        Lower-cased, with any run of ``-``, ``_`` or ``.`` collapsed to a
        single ``-`` (e.g. ``fastfields-torch``).
    """
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class Wheel:
    """A single wheel file and the metadata needed to index it.

    Attributes
    ----------
    project : str
        :pep:`503`-normalized project name.
    filename : str
        The wheel's file name.
    url : str
        Absolute download URL (a GitHub Release asset URL).
    backend : str
        Compute backend bucket (``cpu``, ``cu130``, ...), taken from the
        **leading component** of the wheel's local version label. A release
        built from an unclean tree carries a distance suffix in that same
        label (``+cpu.4.gdeadbee``); bucketing on the whole label would file
        it under a backend folder nobody ever asks for.
    universal : bool
        ``True`` for a pure-Python wheel (no local version label, e.g. the
        ``fastfields-numpy``/``-torch`` wrappers). Universal wheels are listed
        in *every* backend folder so each folder is self-contained.
    sha256 : str
        Lower-case hex digest, appended to the link as ``#sha256=``. Required:
        every discovery path must establish one before constructing a
        :class:`Wheel`.
    """

    project: str
    filename: str
    url: str
    backend: str
    universal: bool = False
    sha256: str = ""


def wheel_from_asset(filename: str, url: str, sha256: str) -> Wheel | None:
    """Parse a wheel file name into a :class:`Wheel`, or ``None`` if not a wheel.

    Parameters
    ----------
    filename : str
        Candidate asset file name.
    url : str
        Download URL for the asset.
    sha256 : str
        The wheel's hex digest. Must already be validated by the caller.

    Returns
    -------
    Wheel or None
        The parsed wheel, or ``None`` when ``filename`` is not a ``.whl``.
    """
    m = WHEEL_RE.match(filename)
    if not m:
        return None
    local = m.group("local")
    # Bucket on the FIRST component of the local label only. versioningit folds
    # the backend and the git-describe distance into the one local segment PEP
    # 440 allows, so a release cut from an unclean tree yields "cpu.4.gdeadbee".
    # Taking the whole label would invent a "cpu.4.gdeadbee" backend: the wheel
    # would vanish from cpu/, and because build() unions the backends it sees
    # onto the declared ones, the phantom would even be advertised on the
    # landing page as an available lane. The index would look healthy and
    # resolve to nothing.
    backend = "cpu"
    if local is not None:
        backend = local.split(".")[0]
        if "." in local:
            print(
                f"::warning::{filename} has a compound local version label "
                f"'+{local}'; bucketing it under '{backend}'. This wheel was "
                "built from a tree that was not clean at its tag -- the "
                "release workflow's version assert should have refused it.",
                file=sys.stderr,
            )
    return Wheel(
        project=normalize(m.group("dist")),
        filename=filename,
        url=url,
        backend=backend,
        universal=local is None,
        sha256=sha256,
    )


def load_config(path: Path) -> dict:
    """Load ``sources.toml``.

    Parameters
    ----------
    path : pathlib.Path
        Path to the TOML config.

    Returns
    -------
    dict
        The parsed configuration.
    """
    with path.open("rb") as fh:
        return tomllib.load(fh)


def wheels_from_manifest(path: Path) -> list[Wheel]:
    """Read wheel entries from a TOML manifest (offline discovery).

    The manifest holds an array of ``[[wheel]]`` tables, each with ``filename``,
    ``url`` and a **required** ``sha256`` (the manifest is a file we control,
    so the digest is mandatory rather than best-effort).

    Parameters
    ----------
    path : pathlib.Path
        Path to the manifest TOML.

    Returns
    -------
    list of Wheel
        Every parsable wheel entry.

    Raises
    ------
    MissingDigestError
        If any ``[[wheel]]`` entry lacks a well-formed ``sha256`` digest.
    """
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    out: list[Wheel] = []
    for entry in data.get("wheel", []):
        name = entry.get("filename", "<unnamed>")
        digest = normalize_digest(entry.get("sha256"))
        if digest is None:
            raise MissingDigestError(
                f"manifest wheel {name!r} has no usable 'sha256' digest "
                f"(got {entry.get('sha256')!r}); every [[wheel]] entry must "
                "declare one as 64 hex characters"
            )
        wheel = wheel_from_asset(entry["filename"], entry["url"], digest)
        if wheel is not None:
            out.append(wheel)
    return out


def normalize_digest(value: str | None) -> str | None:
    """Return ``value`` as a canonical sha256 hex digest, or ``None``.

    A digest that is merely *present* is not enough: a truncated or otherwise
    malformed one would be written into the ``#sha256=`` fragment verbatim and
    fail at ``pip install`` time with a hash mismatch that looks like a
    tampered wheel rather than a broken index.

    Parameters
    ----------
    value : str or None
        Candidate digest, e.g. read from a manifest or a ``.sha256`` sidecar.

    Returns
    -------
    str or None
        The lower-cased 64-character digest, or ``None`` if it is missing or
        not well formed.
    """
    if not value:
        return None
    candidate = value.strip().lower()
    return candidate if SHA256_RE.match(candidate) else None


def _gh_get(url: str) -> list | dict:
    """GET a GitHub API URL as JSON, sending ``GITHUB_TOKEN`` when present."""
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted host)
        return json.load(resp)


def _open_asset(url: str):
    """Open a release asset for streaming, unauthenticated first.

    ``browser_download_url`` redirects to a CDN host that rejects a request
    still carrying GitHub's ``Authorization`` header, so sending the token is
    actively harmful for a public repo and only needed for a private one. Try
    without it, then retry with it.

    Returns
    -------
    http.client.HTTPResponse
        An open response the caller must close.
    """
    attempts: list[str | None] = [None]
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        attempts.append(token)
    last: Exception | None = None
    for attempt in attempts:
        req = urllib.request.Request(url)
        if attempt:
            req.add_header("Authorization", f"Bearer {attempt}")
        try:
            return urllib.request.urlopen(req)  # noqa: S310 (trusted host)
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            last = exc
    raise last  # type: ignore[misc]


def _read_sha256_asset(url: str) -> str | None:
    """Fetch a ``.sha256`` sidecar asset, returning its hex digest or ``None``.

    The asset content is expected to be ``<hexdigest>`` optionally followed by
    whitespace and the file name (the usual ``sha256sum`` format).
    """
    try:
        with _open_asset(url) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"::warning::failed to read {url}: {exc}", file=sys.stderr)
        return None
    parts = text.split()
    return normalize_digest(parts[0] if parts else None)


def _compute_sha256(url: str) -> str | None:
    """Download a release asset and return its sha256, or ``None`` on failure.

    This is the fallback when a package repo ships no ``.sha256`` sidecar. The
    asset is streamed in chunks and never held in memory in full -- a fat CUDA
    wheel is hundreds of megabytes.

    Parameters
    ----------
    url : str
        The asset's download URL.

    Returns
    -------
    str or None
        Lower-case hex digest, or ``None`` if the asset could not be read.
    """
    digest = hashlib.sha256()
    try:
        with _open_asset(url) as resp:
            while chunk := resp.read(_CHUNK):
                digest.update(chunk)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(f"::warning::failed to download {url}: {exc}", file=sys.stderr)
        return None
    return digest.hexdigest()


def wheels_from_releases(repos: Iterable[str]) -> list[Wheel]:
    """Discover wheels from the GitHub Releases of each source repo.

    The GitHub Releases API exposes no per-asset digest, so each wheel's
    ``sha256`` is established in two steps: a sibling ``<wheel>.sha256`` release
    asset is used when the package repo ships one (cheap -- a few bytes), and
    otherwise the wheel itself is downloaded and hashed. Only if *both* fail is
    the digest unknown, and that fails the build: see :class:`MissingDigestError`.

    Parameters
    ----------
    repos : iterable of str
        ``owner/name`` slugs whose releases hold wheel assets.

    Returns
    -------
    list of Wheel
        Wheels found across all releases, every one carrying a digest. Repos
        that error (404, rate limit) are skipped with a warning rather than
        aborting the build -- an unreachable repo contributes no links, which
        is not the same risk as an unverifiable one.

    Raises
    ------
    MissingDigestError
        If any discovered wheel's digest could not be established.
    """
    out: list[Wheel] = []
    unhashed: list[str] = []
    for repo in repos:
        try:
            releases = _gh_get(
                f"https://api.github.com/repos/{repo}/releases?per_page=100"
            )
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"::warning::skipping {repo}: {exc}", file=sys.stderr)
            continue
        for rel in releases:
            assets = rel.get("assets", [])
            sidecars = {
                a["name"]: a["browser_download_url"]
                for a in assets
                if a["name"].endswith(".sha256")
            }
            for asset in assets:
                name = asset["name"]
                if not name.endswith(".whl"):
                    continue
                url = asset["browser_download_url"]
                sha256 = None
                sidecar_url = sidecars.get(f"{name}.sha256")
                if sidecar_url is not None:
                    sha256 = _read_sha256_asset(sidecar_url)
                if sha256 is None:
                    # No (usable) sidecar. Hash the asset ourselves rather than
                    # publishing a link nobody can verify.
                    print(
                        f"::warning::{repo}: no usable {name}.sha256 sidecar; "
                        "downloading the wheel to compute its digest. Publish "
                        "a .sha256 asset next to each wheel to skip this.",
                        file=sys.stderr,
                    )
                    sha256 = _compute_sha256(url)
                if sha256 is None:
                    print(
                        f"::error::{repo}: cannot establish a sha256 for "
                        f"{name}",
                        file=sys.stderr,
                    )
                    unhashed.append(f"{repo}: {name}")
                    continue
                wheel = wheel_from_asset(name, url, sha256)
                if wheel is not None:
                    out.append(wheel)
    if unhashed:
        listed = "\n  ".join(unhashed)
        raise MissingDigestError(
            "no sha256 could be established for the following wheel(s), so "
            "the index would have to link them unverified:\n  "
            f"{listed}\n"
            "Publish a <wheel>.sha256 asset alongside each wheel, or make the "
            "asset downloadable so the digest can be computed."
        )
    return out


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="pypi:repository-version" content="1.0">
<title>{title}</title></head>
<body>
{body}
</body></html>
"""

# Placeholder for a declared-but-empty backend. Deliberately NOT carrying the
# ``pypi:repository-version`` meta tag: this page is for a human who browsed
# here, not a repository root, and nothing resolves underneath it.
_PLACEHOLDER_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="robots" content="noindex">
<title>{title}</title></head>
<body>
{body}
</body></html>
"""


def write_page(path: Path, title: str, body: str, template: str = _PAGE) -> None:
    """Write one HTML page, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(template.format(title=html.escape(title), body=body))


def build(wheels: list[Wheel], out_dir: Path, config: dict) -> None:
    """Emit the full static index tree under ``out_dir``.

    Parameters
    ----------
    wheels : list of Wheel
        All discovered wheels.
    out_dir : pathlib.Path
        Output root (published as the Pages site).
    config : dict
        Parsed ``sources.toml`` (used for the landing page title/URL).
    """
    index_cfg = config.get("index", {})
    base_url = index_cfg.get("base_url", "https://fastfields.github.io/whl").rstrip("/")
    title = index_cfg.get("title", "fastfields wheel index")

    unhashed = [w.filename for w in wheels if not normalize_digest(w.sha256)]
    if unhashed:
        # Belt and braces: this function is the single place a link is written,
        # so the "every link carries a digest" invariant is enforced here too
        # and cannot be bypassed by a future discovery path.
        raise MissingDigestError(
            "refusing to build an index with unhashed link(s): "
            + ", ".join(sorted(unhashed))
        )

    universal = [w for w in wheels if w.universal]
    labelled = [w for w in wheels if not w.universal]

    # Every backend we know about: the config-declared ones, plus any seen on a
    # labelled wheel, plus "cpu" (always present as the universal home).
    declared = set(index_cfg.get("backends", []))
    backends = sorted({"cpu"} | declared | {w.backend for w in labelled})

    # ...but only the ones that actually have a wheel become a PEP 503 folder.
    # A declared-but-empty lane must not be served as a repository at all: it
    # would still answer 200 for the universal (pure-Python) wheels mirrored
    # into it, so `pip install fastfields-numpy --extra-index-url .../cu130/`
    # would succeed and leave the user believing the cu130 lane exists. The
    # compiled wheel they actually wanted would quietly come from PyPI (or not
    # at all). Emitting nothing under a planned lane makes it resolve to
    # nothing, consistently, for every project.
    have_wheel = {w.backend for w in wheels}
    available = [b for b in backends if b in have_wheel]
    planned = [b for b in backends if b not in have_wheel]

    # A wheel whose backend was never declared still gets its folder (dropping
    # it silently would be worse), but it is almost always a mislabelled build
    # rather than a new lane someone forgot to declare -- so say so loudly.
    for backend in sorted({w.backend for w in labelled} - declared - {"cpu"}):
        names = sorted(w.filename for w in labelled if w.backend == backend)
        print(
            f"::warning::backend '{backend}' is not declared in sources.toml "
            f"but was found on {len(names)} wheel(s): {', '.join(names)}. "
            "Either add it to [index].backends or fix the wheel's local "
            "version label.",
            file=sys.stderr,
        )

    # backend -> project -> [wheels]. A universal (pure-Python) wheel goes into
    # every *available* backend folder so each folder resolves on its own; a
    # labelled wheel only into its own backend.
    tree: dict[str, dict[str, list[Wheel]]] = {b: {} for b in available}
    for w in labelled:
        tree.setdefault(w.backend, {}).setdefault(w.project, []).append(w)
    for b in available:
        for w in universal:
            tree[b].setdefault(w.project, []).append(w)

    # Per-backend PEP 503 pages.
    for backend in sorted(tree):
        projects = tree[backend]
        links = "\n".join(
            f'<a href="{html.escape(proj)}/">{html.escape(proj)}</a><br>'
            for proj in sorted(projects)
        )
        write_page(
            out_dir / backend / "index.html",
            f"{title} :: {backend}",
            links or "<!-- no projects -->",
        )
        for proj, plist in projects.items():
            files = "\n".join(
                # The fragment is unconditional: build() has already refused
                # any wheel without a well-formed digest, so there is no
                # "hash unknown" branch to fall into here.
                '<a href="{url}#sha256={sha}">{name}</a><br>'.format(
                    url=html.escape(w.url),
                    sha=w.sha256,
                    name=html.escape(w.filename),
                )
                for w in sorted(plist, key=lambda w: w.filename)
            )
            write_page(
                out_dir / backend / proj / "index.html",
                f"{proj} :: {backend}",
                files,
            )

    # Planned lanes: a human-readable dead end, and nothing underneath it. The
    # page exists so browsing to the folder explains itself instead of serving
    # a 404; it is not a PEP 503 repository root, and no project subfolder is
    # written, so every resolver request under this lane 404s.
    for backend in planned:
        write_page(
            out_dir / backend / "index.html",
            f"{title} :: {backend} (not yet published)",
            f"<h1><code>{html.escape(backend)}</code> &mdash; planned, not yet "
            "published</h1>"
            f"<p>No <code>{html.escape(backend)}</code> wheel has been built "
            "yet, so this folder serves no packages. Passing it as an "
            "<code>--extra-index-url</code> resolves to nothing &mdash; your "
            "install would silently fall back to PyPI.</p>"
            f'<p>See the <a href="{html.escape(base_url)}/">index home</a> for '
            "the lanes that are available today.</p>",
            template=_PLACEHOLDER_PAGE,
        )

    # Human-facing landing page. Only advertise a copy-paste ``pip install``
    # for backends that actually have a discovered wheel; backends that are
    # merely declared in sources.toml (no wheels yet) are listed as "planned"
    # so an advertised folder never implies an install that resolves to
    # nothing. A universal (pure-Python) wheel counts under "cpu" (its bucket),
    # not as coverage for every CUDA lane it is mirrored into.
    avail_rows = "\n".join(
        "<li><code>{b}</code> &mdash; "
        "<code>pip install fastfields-torch --extra-index-url "
        "{base}/{b}/</code></li>".format(
            b=html.escape(b), base=html.escape(base_url)
        )
        for b in available
    )
    planned_rows = "\n".join(
        '<li><code><a href="{base}/{b}/">{b}</a></code></li>'.format(
            b=html.escape(b), base=html.escape(base_url)
        )
        for b in planned
    )

    parts = [
        f"<h1>{html.escape(title)}</h1>",
        "<p>PyTorch-style wheel index for the fastfields Python packages. "
        "Pick the folder matching your compute backend and pass it as an "
        "<code>--extra-index-url</code> (dependencies still resolve from "
        "PyPI):</p>",
    ]
    if available:
        parts.append("<h2>Available</h2>")
        parts.append(f"<ul>{avail_rows}</ul>")
    if planned:
        parts.append("<h2>Planned &mdash; not yet published</h2>")
        parts.append(
            "<p>These backends are declared but have no wheels published "
            "yet, so <strong>no packages are served under them at all</strong> "
            "&mdash; passing one as an <code>--extra-index-url</code> resolves "
            "to nothing and your install silently falls back to PyPI. Do not "
            "use them until a build lands here.</p>"
        )
        parts.append(f"<ul>{planned_rows}</ul>")
    parts.append(
        '<p>See <a href="https://github.com/fastfields/whl">the repository</a> '
        "for how the index is built.</p>"
    )
    body = "".join(parts)
    write_page(out_dir / "index.html", title, body)
    print(
        f"wrote {sum(len(p) for b in tree.values() for p in b.values())} "
        f"wheels across {len(tree)} published backend(s) "
        f"({', '.join(available) or 'none'}) to {out_dir}"
        + (
            f"; {len(planned)} planned backend(s) "
            f"({', '.join(planned)}) serve no packages"
            if planned
            else ""
        )
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("sources.toml"))
    ap.add_argument("--out", type=Path, default=Path("public"))
    ap.add_argument(
        "--manifest",
        type=Path,
        help="TOML manifest of wheels (offline discovery).",
    )
    ap.add_argument(
        "--from-releases",
        action="store_true",
        help="Discover wheels from the GitHub Releases in sources.toml.",
    )
    args = ap.parse_args(argv)

    config = load_config(args.config)
    wheels: list[Wheel] = []
    if not args.manifest and not args.from_releases:
        ap.error("pass --manifest and/or --from-releases")

    # A missing digest is a normal, actionable operational failure (someone
    # published a wheel without its sidecar), so report it as an error the
    # workflow log can be read for -- not as a traceback.
    try:
        if args.manifest:
            wheels += wheels_from_manifest(args.manifest)
        if args.from_releases:
            repos = [
                s["repo"] for s in config.get("sources", {}).get("release", [])
            ]
            wheels += wheels_from_releases(repos)
        build(wheels, args.out, config)
    except MissingDigestError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
