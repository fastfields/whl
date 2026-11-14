#!/usr/bin/env python3
"""Tests for :mod:`generate` -- run with ``python -m unittest discover``.

Stdlib-only (``unittest``), matching the generator itself: the index build
must keep working on a bare ``actions/setup-python`` with nothing installed.

The two invariants worth protecting here are the ones that make this index
safe to hand to ``pip`` as an ``--extra-index-url``:

1. a backend folder only serves packages once a wheel for it exists, and
2. every link carries a ``#sha256=`` fragment.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import generate

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64

CONFIG = {
    "index": {
        "title": "fastfields wheel index",
        "base_url": "https://fastfields.github.io/whl",
        "backends": ["cpu", "cu118", "cu126", "cu130"],
    }
}


def cpu_wheel() -> generate.Wheel:
    """A compiled CPU wheel (local label ``+cpu``)."""
    return generate.wheel_from_asset(
        "fastfields_dlpack-0.1.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl",
        "https://example.test/dlpack-cpu.whl",
        DIGEST_A,
    )


def universal_wheel() -> generate.Wheel:
    """The pure-Python wrappers distribution (no local label)."""
    return generate.wheel_from_asset(
        "fastfields-0.1.0-py3-none-any.whl",
        "https://example.test/fastfields.whl",
        DIGEST_B,
    )


def cuda_wheel() -> generate.Wheel:
    """A compiled CUDA wheel (local label ``+cu130``)."""
    return generate.wheel_from_asset(
        "fastfields_dlpack-0.1.0+cu130-cp311-cp311-manylinux_2_28_x86_64.whl",
        "https://example.test/dlpack-cu130.whl",
        DIGEST_C,
    )


class BuildCase(unittest.TestCase):
    """Base class giving each test a fresh output tree."""

    def build(self, wheels: list[generate.Wheel]) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name) / "public"
        generate.build(wheels, out, CONFIG)
        return out


class TestWheelParsing(unittest.TestCase):
    def test_local_label_selects_the_backend(self):
        self.assertEqual(cpu_wheel().backend, "cpu")
        self.assertEqual(cuda_wheel().backend, "cu130")

    def test_no_local_label_is_universal_and_buckets_as_cpu(self):
        wheel = universal_wheel()
        self.assertTrue(wheel.universal)
        self.assertEqual(wheel.backend, "cpu")

    def test_compound_local_label_buckets_on_first_component(self):
        # versioningit folds the backend and the git-describe distance into one
        # local segment; bucketing on the whole label would invent a phantom
        # "cpu.4.gdeadbee" backend and the wheel would vanish from cpu/.
        wheel = generate.wheel_from_asset(
            "fastfields_dlpack-0.1.0+cpu.4.gdeadbee-cp311-cp311-manylinux_2_28_x86_64.whl",
            "https://example.test/dirty.whl",
            DIGEST_A,
        )
        self.assertEqual(wheel.backend, "cpu")

    def test_non_wheel_asset_is_ignored(self):
        self.assertIsNone(
            generate.wheel_from_asset("notes.txt", "https://x.test/n", DIGEST_A)
        )


class TestDigestValidation(unittest.TestCase):
    def test_accepts_and_lowercases_a_valid_digest(self):
        self.assertEqual(generate.normalize_digest("A" * 64), "a" * 64)
        self.assertEqual(generate.normalize_digest(f"  {DIGEST_A} "), DIGEST_A)

    def test_rejects_missing_or_malformed(self):
        for bad in (None, "", "DEADBEEF", "a" * 63, "g" * 64, "a" * 65):
            with self.subTest(bad=bad):
                self.assertIsNone(generate.normalize_digest(bad))

    def test_manifest_without_sha256_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.toml"
            path.write_text(
                '[[wheel]]\nfilename = "f-0.1.0-py3-none-any.whl"\n'
                'url = "https://example.test/f.whl"\n'
            )
            with self.assertRaises(generate.MissingDigestError):
                generate.wheels_from_manifest(path)

    def test_manifest_with_malformed_sha256_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.toml"
            path.write_text(
                '[[wheel]]\nfilename = "f-0.1.0-py3-none-any.whl"\n'
                'url = "https://example.test/f.whl"\nsha256 = "DEADBEEF"\n'
            )
            with self.assertRaises(generate.MissingDigestError):
                generate.wheels_from_manifest(path)


class TestDigestsAreMandatoryInOutput(BuildCase):
    def test_build_refuses_an_unhashed_wheel(self):
        # The invariant is enforced at the single place links are written, so
        # no future discovery path can smuggle an unverifiable link in.
        unhashed = generate.Wheel(
            project="fastfields",
            filename="fastfields-0.1.0-py3-none-any.whl",
            url="https://example.test/fastfields.whl",
            backend="cpu",
            universal=True,
            sha256="",
        )
        with self.assertRaises(generate.MissingDigestError):
            self.build([unhashed])

    def test_every_emitted_link_carries_a_sha256_fragment(self):
        out = self.build([cpu_wheel(), universal_wheel(), cuda_wheel()])
        links = 0
        for page in out.rglob("*/*/index.html"):  # project pages only
            for line in page.read_text().splitlines():
                if 'href="http' in line:
                    links += 1
                    self.assertIn("#sha256=", line, f"unhashed link in {page}")
        self.assertGreater(links, 0)


class TestPlannedBackendsServeNothing(BuildCase):
    """The core of fastfields#6: a declared lane with no wheel must not resolve.

    Before this, every declared CUDA folder was emitted as a valid PEP 503
    repository *and* had the universal wheels mirrored into it -- so
    ``pip install fastfields --extra-index-url .../cu130/`` succeeded and
    implied a cu130 lane that had never been built.
    """

    def test_no_project_subfolders_under_a_planned_lane(self):
        out = self.build([cpu_wheel(), universal_wheel()])
        for lane in ("cu118", "cu126", "cu130"):
            with self.subTest(lane=lane):
                subdirs = [p for p in (out / lane).iterdir() if p.is_dir()]
                self.assertEqual(subdirs, [], f"{lane} serves packages")

    def test_universal_wheels_are_not_mirrored_into_a_planned_lane(self):
        out = self.build([cpu_wheel(), universal_wheel()])
        self.assertFalse((out / "cu130" / "fastfields").exists())

    def test_planned_lane_page_is_not_a_pep503_repository_root(self):
        out = self.build([cpu_wheel(), universal_wheel()])
        page = (out / "cu130" / "index.html").read_text()
        self.assertNotIn("pypi:repository-version", page)
        self.assertIn("not yet published", page)

    def test_available_lane_serves_both_its_own_and_universal_wheels(self):
        out = self.build([cpu_wheel(), universal_wheel()])
        self.assertTrue((out / "cpu" / "fastfields-dlpack" / "index.html").exists())
        self.assertTrue((out / "cpu" / "fastfields" / "index.html").exists())

    def test_a_lane_becomes_available_as_soon_as_a_wheel_exists(self):
        # The deferral is data-driven, not a hardcoded CUDA block: publishing a
        # cu130 wheel is all it takes to turn the lane on.
        out = self.build([cpu_wheel(), universal_wheel(), cuda_wheel()])
        self.assertTrue((out / "cu130" / "fastfields-dlpack" / "index.html").exists())
        self.assertTrue((out / "cu130" / "fastfields" / "index.html").exists())
        page = (out / "cu130" / "index.html").read_text()
        self.assertIn("pypi:repository-version", page)
        # ...and the lanes still unbuilt stay closed.
        self.assertFalse((out / "cu118" / "fastfields").exists())

    def test_landing_page_only_advertises_installable_lanes(self):
        out = self.build([cpu_wheel(), universal_wheel()])
        page = (out / "index.html").read_text()
        self.assertIn("Planned &mdash; not yet published", page)
        self.assertIn(
            "pip install fastfields --extra-index-url "
            "https://fastfields.github.io/whl/cpu/",
            page,
        )
        for lane in ("cu118", "cu126", "cu130"):
            with self.subTest(lane=lane):
                self.assertNotIn(f"--extra-index-url {CONFIG['index']['base_url']}"
                                 f"/{lane}/", page)

    def test_with_no_wheels_at_all_nothing_is_served(self):
        # Today's real state: zero releases anywhere, so the index must not
        # advertise a single installable lane.
        out = self.build([])
        self.assertNotIn("<h2>Available</h2>", (out / "index.html").read_text())
        for lane in ("cpu", "cu118", "cu126", "cu130"):
            with self.subTest(lane=lane):
                self.assertEqual(
                    [p for p in (out / lane).iterdir() if p.is_dir()], []
                )


# The four compiled wheels a fastfields-dlpack release produces, named exactly
# as cibuildwheel + the repair step write them: one abi3 wheel per platform
# (and, on macOS, per architecture -- there is no universal2 wheel).
RELEASE_WHEELS = {
    "Linux x86_64":
        "fastfields_dlpack-0.0.dev1+cpu-cp310-abi3-manylinux_2_28_x86_64.whl",
    "macOS arm64":
        "fastfields_dlpack-0.0.dev1+cpu-cp310-abi3-macosx_11_0_arm64.whl",
    "macOS x86_64":
        "fastfields_dlpack-0.0.dev1+cpu-cp310-abi3-macosx_11_0_x86_64.whl",
    "Windows x64":
        "fastfields_dlpack-0.0.dev1+cpu-cp310-abi3-win_amd64.whl",
}


def release_wheels() -> list[generate.Wheel]:
    return [
        generate.wheel_from_asset(
            name, f"https://example.test/{name}", f"{i:x}" * 64
        )
        for i, name in enumerate(RELEASE_WHEELS.values(), start=1)
    ]


class TestPlatformTags(unittest.TestCase):
    def test_every_release_platform_parses_into_the_cpu_lane(self):
        for label, name in RELEASE_WHEELS.items():
            with self.subTest(platform=label):
                wheel = generate.wheel_from_asset(
                    name, "https://example.test/w.whl", DIGEST_A
                )
                self.assertIsNotNone(wheel)
                self.assertEqual(wheel.project, "fastfields-dlpack")
                self.assertEqual(wheel.backend, "cpu")
                self.assertFalse(wheel.universal)
                self.assertEqual(wheel.platforms, (label,))

    def test_platform_tag_is_kept_verbatim(self):
        # pip matches the tag in the *file name* against the interpreter's
        # supported tags, so the tag must survive parsing untouched: an arm64
        # wheel listed as x86_64 (or vice versa) is the mislabelled-wheel
        # failure the release workflow guards against on the build side.
        for name in RELEASE_WHEELS.values():
            wheel = generate.wheel_from_asset(name, "https://x.test/w", DIGEST_A)
            with self.subTest(name=name):
                self.assertTrue(name.endswith(wheel.platform_tags[0] + ".whl"))

    def test_macos_architectures_are_told_apart(self):
        arm = generate.platform_label("macosx_11_0_arm64")
        intel = generate.platform_label("macosx_11_0_x86_64")
        self.assertEqual((arm, intel), ("macOS arm64", "macOS x86_64"))
        self.assertEqual(
            generate.platform_label("macosx_10_13_universal2"),
            "macOS universal2",
        )

    def test_compressed_tag_set_is_split(self):
        wheel = generate.wheel_from_asset(
            "fastfields_dlpack-0.1.0+cpu-cp310-abi3-"
            "manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            "https://x.test/w",
            DIGEST_A,
        )
        self.assertEqual(
            wheel.platform_tags,
            ("manylinux_2_17_x86_64", "manylinux2014_x86_64"),
        )
        self.assertEqual(wheel.platforms, ("Linux x86_64",))

    def test_universal_wheel_is_any(self):
        self.assertEqual(universal_wheel().platforms, ("any",))

    def test_unknown_tag_passes_through(self):
        self.assertEqual(generate.platform_label("plan9_mips"), "plan9_mips")


class TestMultiPlatformIndex(BuildCase):
    def test_every_platform_is_linked_from_the_cpu_lane(self):
        out = self.build(release_wheels() + [universal_wheel()])
        page = (out / "cpu" / "fastfields-dlpack" / "index.html").read_text()
        for i, name in enumerate(RELEASE_WHEELS.values(), start=1):
            with self.subTest(wheel=name):
                self.assertIn(
                    f'<a href="https://example.test/{name}'
                    f'#sha256={f"{i:x}" * 64}">{name}</a>',
                    page,
                )
        # A compiled wheel belongs to its own lane only.
        for lane in ("cu118", "cu126", "cu130"):
            self.assertFalse((out / lane / "fastfields-dlpack").exists())

    def test_landing_page_lists_the_platforms_each_lane_covers(self):
        out = self.build(release_wheels() + [universal_wheel()])
        page = (out / "index.html").read_text()
        self.assertIn(
            "<code>fastfields-dlpack</code>: Linux x86_64, macOS arm64, "
            "macOS x86_64, Windows x64",
            page,
        )
        # The pure-Python distribution is not a platform claim.
        self.assertNotIn("<code>fastfields</code>: any", page)

    def test_a_missing_platform_is_visible_on_the_landing_page(self):
        linux_only = [w for w in release_wheels() if "manylinux" in w.filename]
        out = self.build(linux_only)
        page = (out / "index.html").read_text()
        self.assertIn("<code>fastfields-dlpack</code>: Linux x86_64<", page)
        self.assertNotIn("Windows x64", page)


if __name__ == "__main__":
    unittest.main()
