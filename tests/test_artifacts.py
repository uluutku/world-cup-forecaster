import hashlib
import json
import zipfile
from pathlib import Path

from worldcup_predictor.artifacts import install_bundle, verify_runtime


def test_runtime_bundle_is_checksum_verified(tmp_path: Path):
    source = tmp_path / "source.txt"
    source.write_text("artifact", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = {
        "version": "test",
        "files": [
            {
                "path": "artifacts/source.txt",
                "bytes": source.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.write(source, "artifacts/source.txt")
    root = tmp_path / "runtime"
    root.mkdir()
    manifest_path = root / "release" / "runtime-manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    install_bundle(bundle, root=root, manifest_path=manifest_path)
    assert verify_runtime(root=root, manifest_path=manifest_path) == []


def test_runtime_verification_reports_tampering(tmp_path: Path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("changed", encoding="utf-8")
    manifest = {
        "files": [
            {
                "path": "artifact.txt",
                "sha256": "0" * 64,
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    errors = verify_runtime(root=tmp_path, manifest_path=manifest_path)
    assert errors and errors[0].startswith("checksum:")
