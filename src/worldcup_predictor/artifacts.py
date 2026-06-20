from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .config import ROOT, RUNTIME_MANIFEST_PATH


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = RUNTIME_MANIFEST_PATH) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Runtime manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_runtime(
    root: Path = ROOT,
    manifest_path: Path = RUNTIME_MANIFEST_PATH,
) -> list[str]:
    manifest = load_manifest(manifest_path)
    errors = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.exists():
            errors.append(f"missing: {entry['path']}")
            continue
        actual = sha256(path)
        if actual != entry["sha256"]:
            errors.append(
                f"checksum: {entry['path']} expected {entry['sha256']} got {actual}"
            )
    return errors


def install_bundle(
    bundle: Path,
    root: Path = ROOT,
    manifest_path: Path = RUNTIME_MANIFEST_PATH,
) -> None:
    root = root.resolve()
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            destination = (root / member.filename).resolve()
            if root not in destination.parents and destination != root:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(root)
    errors = verify_runtime(root, manifest_path)
    if errors:
        raise RuntimeError("Runtime verification failed:\n" + "\n".join(errors))


def download_bundle(url: str) -> Path:
    temporary = Path(tempfile.mkdtemp(prefix="wci-artifacts-"))
    destination = temporary / "runtime.zip"
    request = urllib.request.Request(
        url, headers={"User-Agent": "WorldCupIntelligence/2.0"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    return destination


def ensure_runtime_artifacts() -> list[str]:
    errors = verify_runtime()
    if not errors:
        return []
    url = os.getenv("WCI_ARTIFACT_BUNDLE_URL")
    if not url:
        return errors
    bundle = download_bundle(url)
    install_bundle(bundle)
    return verify_runtime()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install or verify the versioned runtime artifact bundle."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--bundle", type=Path)
    source.add_argument("--url")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.bundle:
        install_bundle(args.bundle)
    elif args.url:
        install_bundle(download_bundle(args.url))
    elif not args.verify:
        errors = ensure_runtime_artifacts()
        if errors:
            raise SystemExit("\n".join(errors))
    errors = verify_runtime()
    if errors:
        raise SystemExit("\n".join(errors))
    print("Runtime artifacts verified.")


if __name__ == "__main__":
    main()
