"""Regenerate MANIFEST.sha256 for a curated distribution before release."""

from __future__ import annotations

from pathlib import Path

from verify_package import MANIFEST, distributed_files, sha256


def main() -> int:
    files = distributed_files()
    lines = [f"{sha256(path)}  {relative}" for relative, path in sorted(files.items())]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} file hashes to {MANIFEST.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
