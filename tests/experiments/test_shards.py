import json
import shutil

import pytest
from PIL import Image

from experiments.shards import manifest_hash, stage_shards, write_shards
from fair_trees.dataset import load_regions


@pytest.fixture
def source_root(write_region, tmp_path):
    root = tmp_path / "regions"
    write_region("a", tiles=((0, 0), (1, 0)), root=root, boxes_px=[(10, 10, 30, 30)])
    write_region("a", tiles=((0, 0),), z=19, root=root)
    write_region("b", tiles=((4, 0),), root=root, split="val")
    return root


def test_write_shards_is_stable(source_root, tmp_path):
    m1 = write_shards(source_root, tmp_path / "s1", source="neon", converter_sha="abc", shard_bytes=1)
    m2 = write_shards(source_root, tmp_path / "s2", source="neon", converter_sha="abc", shard_bytes=1)
    assert m1 == m2
    assert manifest_hash(tmp_path / "s1" / "manifest.json") == manifest_hash(
        tmp_path / "s2" / "manifest.json"
    )
    names = [s["name"] for s in m1["shards"]]
    assert names == [
        "z19-0000.tar",
        "z20-0000.tar",
        "z20-0001.tar",
        "z20-0002.tar",
    ]  # one chip each at 1 byte
    assert sum(s["files"] for s in m1["shards"]) == 4 and m1["converter_sha"] == "abc"
    jsonl = (tmp_path / "s1" / "regions.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in jsonl] == ["a", "b"]


def test_stage_shards_verifies_extracts_and_skips(source_root, tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    write_shards(source_root, drive / "shards" / "neon", source="neon", converter_sha="abc")
    local = stage_shards(drive, tmp_path / "data", ["neon"])
    a, b = load_regions(local / "neon")
    assert (a.id, b.id) == ("a", "b") and len(a.chips[20]) == 2 and len(a.chips[19]) == 1

    def no_copy(*args, **kwargs):
        raise AssertionError("already staged shards must not be copied again")

    monkeypatch.setattr(shutil, "copyfile", no_copy)
    stage_shards(drive, tmp_path / "data", ["neon"])


def test_restaging_changed_shards_leaves_no_stale_region(source_root, tmp_path):
    drive = tmp_path / "drive"
    write_shards(source_root, drive / "shards" / "neon", source="neon", converter_sha="abc")
    local = stage_shards(drive, tmp_path / "data", ["neon"])
    shutil.rmtree(source_root / "b")
    shutil.rmtree(drive / "shards" / "neon")
    write_shards(source_root, drive / "shards" / "neon", source="neon", converter_sha="abc")
    stage_shards(drive, tmp_path / "data", ["neon"])
    assert [r.id for r in load_regions(local / "neon")] == ["a"]
    assert not (local / "neon" / "b").exists()


def test_stage_shards_rejects_a_corrupt_shard(source_root, tmp_path):
    drive = tmp_path / "drive"
    m = write_shards(source_root, drive / "shards" / "neon", source="neon", converter_sha="abc")
    with (drive / "shards" / "neon" / m["shards"][0]["name"]).open("ab") as f:
        f.write(b"garbage")
    with pytest.raises(ValueError, match="sha256"):
        stage_shards(drive, tmp_path / "data", ["neon"])
    # the manifest is written last, so an interrupted stage does not look complete
    assert not (tmp_path / "data" / "neon" / "manifest.json").exists()


def test_png_chips_are_packed_staged_and_loaded(write_region, tmp_path):
    root = tmp_path / "regions"
    write_region("h", tiles=((0, 0),), root=root)
    (tif,) = (root / "h" / "z20" / "chips").glob("*.tif")
    Image.new("RGB", (256, 256), (90, 120, 60)).save(tif.with_suffix(".png"))
    tif.unlink()
    drive, local = tmp_path / "drive", tmp_path / "data"
    write_shards(root, drive / "shards" / "hot", source="hot", converter_sha="abc")
    stage_shards(drive, local, ["hot"])
    (region,) = load_regions(local / "hot")
    assert [p.suffix for p in region.chips[20]] == [".png"]
