"""The Zenodo companion installer is checked without loading model pickles."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "artifact_installation", ROOT / "scripts/install_artifacts.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class ArtifactInstallationTests(unittest.TestCase):
    def fixture(self, directory):
        root = directory / "checkout"
        (root / "data").mkdir(parents=True)
        reference = b'{"classes":["A","B"]}\n'
        (root / "data/plan.json").write_bytes(reference)

        def row(path, data):
            return dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

        payload = row("artifacts/reference/checkpoint.pt", b"synthetic checkpoint bytes")
        reused = dict(row("artifacts/reference/plan.json", reference), github_path="data/plan.json")
        (root / "data/artifacts.json").write_text(json.dumps(dict(files=[payload, reused])))
        archive = directory / "artifacts.zip"
        with ZipFile(archive, "w") as z:
            z.writestr(
                installer.MANIFEST, json.dumps(dict(files=[payload], reuse_from_github=[reused]))
            )
            z.writestr("MAPPER/" + payload["path"], b"synthetic checkpoint bytes")
        return root, archive

    def test_installs_and_restores_github_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive = self.fixture(Path(d))
            result = installer.install(archive, root, destination=root / "artifacts")
            self.assertEqual(result["installed"], 2)
            self.assertEqual(result["reused_github_files"], 1)
            self.assertEqual(
                (root / "artifacts/reference/plan.json").read_bytes(),
                (root / "data/plan.json").read_bytes(),
            )
            repeated = installer.install(archive, root, destination=root / "artifacts")
            self.assertEqual(repeated["installed"], 0)
            self.assertEqual(repeated["unchanged"], 2)

    def test_missing_github_source_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive = self.fixture(Path(d))
            (root / "data/plan.json").write_bytes(b"different version")
            with self.assertRaises(ValueError):
                installer.install(archive, root, destination=root / "artifacts")
            self.assertFalse((root / "artifacts").exists())

    def test_existing_models_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive = self.fixture(Path(d))
            target = root / "artifacts/reference/checkpoint.pt"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"user model")
            with self.assertRaises(FileExistsError):
                installer.install(archive, root, destination=root / "artifacts")
            self.assertEqual(target.read_bytes(), b"user model")

    def test_inspect_does_not_create_artifact_directories(self):
        with tempfile.TemporaryDirectory() as d:
            root, archive = self.fixture(Path(d))
            result = installer.install(archive, root, destination=root / "artifacts", inspect=True)
            self.assertTrue(result["inspected_only"])
            self.assertFalse((root / "artifacts").exists())

    def test_unsafe_paths_and_unlisted_members_are_rejected(self):
        for path in ["../outside", "/outside", "data/not_an_artifact", "artifacts/../../outside"]:
            with self.assertRaises(ValueError):
                installer.safe_path(path, "artifacts")
        with tempfile.TemporaryDirectory() as d:
            root, archive = self.fixture(Path(d))
            with ZipFile(archive, "a") as z:
                z.writestr("MAPPER/unlisted.txt", "unexpected")
            with self.assertRaises(ValueError):
                installer.install(archive, root, destination=root / "artifacts")


if __name__ == "__main__":
    unittest.main()
