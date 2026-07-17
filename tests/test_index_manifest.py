from pathlib import Path

from src.index import IndexManifest


def test_manifest_tracks_deleted_documents(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = IndexManifest(manifest_path)
    file_path = tmp_path / "doc.txt"
    file_path.write_text("hello world", encoding="utf-8")

    manifest.record_file(file_path, "hash-1", 3)
    manifest.mark_deleted(file_path)
    manifest.save()

    reloaded = IndexManifest(manifest_path)

    assert reloaded.is_deleted(file_path) is True
    assert reloaded.has_current_file(file_path, "hash-1") is False
