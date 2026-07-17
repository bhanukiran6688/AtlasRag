import tempfile
import unittest
from pathlib import Path

from src.index import IndexManifest


class IndexManifestTests(unittest.TestCase):
    def test_manifest_tracks_deleted_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            manifest = IndexManifest(path)
            file_path = Path(tmpdir) / "doc.txt"
            file_path.write_text("hello world", encoding="utf-8")

            manifest.record_file(file_path, "hash-1", 3)
            manifest.mark_deleted(file_path)
            manifest.save()

            reloaded = IndexManifest(path)

            self.assertTrue(reloaded.is_deleted(file_path))
            self.assertFalse(reloaded.has_current_file(file_path, "hash-1"))


if __name__ == "__main__":
    unittest.main()
