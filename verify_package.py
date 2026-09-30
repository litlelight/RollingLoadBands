"""Verify a downloaded v1.2.0 distribution without running experiments."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "MANIFEST.sha256"
MAX_BROWSER_FILE_BYTES = 25 * 1024 * 1024
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
IGNORED_NAMES = {".DS_Store", "Thumbs.db"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def distributed_files() -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if relative.parts[:2] == ("data", "raw_nyiso"):
            continue  # official raw downloads are never part of the distribution
        if path.name in IGNORED_NAMES or path.suffix == ".pyc":
            continue
        result[relative.as_posix()] = path
    return result


def main() -> int:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    failures: list[str] = []
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed manifest line: {line[:80]}")
            continue
        if (relative in entries or Path(relative).is_absolute()
                or ".." in Path(relative).parts or relative == MANIFEST.name):
            failures.append(f"invalid or duplicate manifest path: {relative}")
            continue
        entries[relative] = expected

    files = distributed_files()
    for relative in sorted(set(entries) - set(files)):
        failures.append(f"listed but missing: {relative}")
    for relative in sorted(set(files) - set(entries)):
        failures.append(f"unlisted distributed file: {relative}")
    for relative in sorted(set(files) & set(entries)):
        path = files[relative]
        if path.stat().st_size > MAX_BROWSER_FILE_BYTES:
            failures.append(f"larger than 25 MiB: {relative}")
        if sha256(path) != entries[relative]:
            failures.append(f"hash mismatch: {relative}")

    audit = ROOT / "results/ROUND2_REVISION_V1/E24/input_hashes.csv"
    if audit.is_file():
        with audit.open(newline="", encoding="utf-8") as handle:
            input_rows = list(csv.DictReader(handle))
            if len(input_rows) != 9:
                failures.append(f"expected 9 frozen inputs, found {len(input_rows)}")
            for row in input_rows:
                relative = row["path"].replace("\\", "/")
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    failures.append(f"nonportable frozen-input path: {relative}")
                    continue
                target = ROOT / relative
                if not target.is_file() or sha256(target) != row["sha256"]:
                    failures.append(f"frozen-input mismatch: {row['input']}")
                elif target.stat().st_size != int(row["bytes"]):
                    failures.append(f"frozen-input byte-count mismatch: {row['input']}")
    else:
        failures.append("missing E24 frozen-input audit")

    status = ROOT / "results/ROUND2_REVISION_V1/RUN_STATUS.json"
    if status.is_file():
        record = json.loads(status.read_text(encoding="utf-8"))
        if record.get("status") != "PASS":
            failures.append("second-round run status is not PASS")
        for relative, expected in record.get("output_hashes", {}).items():
            target = status.parent / relative
            if not target.is_file() or sha256(target) != expected:
                failures.append(f"second-round output mismatch: {relative}")
    else:
        failures.append("missing second-round RUN_STATUS.json")

    if failures:
        print("PACKAGE VERIFICATION: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"PACKAGE VERIFICATION: PASS ({len(files)} files; 9 frozen inputs and second-round outputs checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
