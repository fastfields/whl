#!/usr/bin/env python3
"""Generate a PyTorch-style PEP 503 wheel index for the fastfields packages.

The index is a set of static HTML pages laid out as a *simple repository*
(:pep:`503`), one **backend** subfolder per compute target
(``cpu/``, ``cu118/``, ``cu124/`` ...). It is published to GitHub Pages at
``https://fastfields.github.io/whl/`` so users can install a build matching
their hardware::

    pip install fastfields-torch \\
        --extra-index-url https://fastfields.github.io/whl/cu124/

The wheels themselves are **not** committed here -- they are hosted as GitHub
*Release* assets on each package repo. This script only emits the HTML that
links to them (with a ``#sha256=`` fragment when the digest is known).

Wheel-to-backend mapping follows PyTorch's convention: the compute backend is
encoded in the wheel's **local version label**, e.g.
``fastfields_torch-0.1.0+cu124-cp311-cp311-linux_x86_64.whl``. A wheel with no
local label is bucketed as ``cpu``.

Two discovery modes:

* ``--manifest FILE`` -- read wheel entries from a TOML manifest (offline; used
  for local testing and reproducible builds).
* ``--from-releases`` -- query the GitHub Releases API for each source repo in
  ``sources.toml`` (used in CI; honours ``GITHUB_TOKEN``).
"""

from __future__ import annotations

import argparse
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
        Compute backend bucket (``cpu``, ``cu124``, ...), taken from the
        wheel's local version label.
    sha256 : str or None
        Hex digest, when known, appended to the link as ``#sha256=``.
    """

    project: str
    filename: str
    url: str
    backend: str
    sha256: str | None = None


def wheel_from_asset(filename: str, url: str, sha256: str | None) -> Wheel | None:
    """Parse a wheel file name into a :class:`Wheel`, or ``None`` if not a wheel.

    Parameters
    ----------
    filename : str
        Candidate asset file name.
    url : str
        Download URL for the asset.
    sha256 : str or None
        Known digest, or ``None``.

    Returns
    -------
    Wheel or None
        The parsed wheel, or ``None`` when ``filename`` is not a ``.whl``.
    """
    m = WHEEL_RE.match(filename)
    if not m:
        return None
    backend = m.group("local") or "cpu"
    return Wheel(
        project=normalize(m.group("dist")),
        filename=filename,
        url=url,
        backend=backend,
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
    ``url`` and an optional ``sha256``.

    Parameters
    ----------
    path : pathlib.Path
        Path to the manifest TOML.

    Returns
    -------
    list of Wheel
        Every parsable wheel entry.
    """
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    out: list[Wheel] = []
    for entry in data.get("wheel", []):
        wheel = wheel_from_asset(entry["filename"], entry["url"], entry.get("sha256"))
        if wheel is not None:
            out.append(wheel)
    return out


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


def wheels_from_releases(repos: Iterable[str]) -> list[Wheel]:
    """Discover wheels from the GitHub Releases of each source repo.

    Parameters
    ----------
    repos : iterable of str
        ``owner/name`` slugs whose releases hold wheel assets.

    Returns
    -------
    list of Wheel
        Wheels found across all releases. Repos that error (404, rate limit)
        are skipped with a warning rather than aborting the build.
    """
    out: list[Wheel] = []
    for repo in repos:
        try:
            releases = _gh_get(
                f"https://api.github.com/repos/{repo}/releases?per_page=100"
            )
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            print(f"::warning::skipping {repo}: {exc}", file=sys.stderr)
            continue
        for rel in releases:
            for asset in rel.get("assets", []):
                wheel = wheel_from_asset(
                    asset["name"], asset["browser_download_url"], None
                )
                if wheel is not None:
                    out.append(wheel)
    return out


_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="pypi:repository-version" content="1.0">
<title>{title}</title></head>
<body>
{body}
</body></html>
"""


def write_page(path: Path, title: str, body: str) -> None:
    """Write one HTML page, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_PAGE.format(title=html.escape(title), body=body))


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

    # group: backend -> project -> [wheels]
    tree: dict[str, dict[str, list[Wheel]]] = {}
    for w in wheels:
        tree.setdefault(w.backend, {}).setdefault(w.project, []).append(w)

    backends = sorted(tree) or list(index_cfg.get("backends", []))

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
                '<a href="{url}{frag}">{name}</a><br>'.format(
                    url=html.escape(w.url),
                    frag=f"#sha256={w.sha256}" if w.sha256 else "",
                    name=html.escape(w.filename),
                )
                for w in sorted(plist, key=lambda w: w.filename)
            )
            write_page(
                out_dir / backend / proj / "index.html",
                f"{proj} :: {backend}",
                files,
            )

    # Human-facing landing page.
    rows = "\n".join(
        "<li><code>{b}</code> &mdash; "
        "<code>pip install fastfields-torch --extra-index-url "
        "{base}/{b}/</code></li>".format(b=html.escape(b), base=html.escape(base_url))
        for b in backends
    )
    body = (
        f"<h1>{html.escape(title)}</h1>"
        "<p>PyTorch-style wheel index for the fastfields Python packages. "
        "Pick the folder matching your compute backend and pass it as an "
        "<code>--extra-index-url</code> (dependencies still resolve from "
        "PyPI):</p>"
        f"<ul>{rows}</ul>"
        '<p>See <a href="https://github.com/fastfields/whl">the repository</a> '
        "for how the index is built.</p>"
    )
    write_page(out_dir / "index.html", title, body)
    print(
        f"wrote {sum(len(p) for b in tree.values() for p in b.values())} "
        f"wheels across {len(tree)} backend(s) to {out_dir}"
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
    if args.manifest:
        wheels += wheels_from_manifest(args.manifest)
    if args.from_releases:
        repos = [s["repo"] for s in config.get("sources", {}).get("release", [])]
        wheels += wheels_from_releases(repos)
    if not args.manifest and not args.from_releases:
        ap.error("pass --manifest and/or --from-releases")

    build(wheels, args.out, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
