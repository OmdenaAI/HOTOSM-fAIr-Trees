"""Deterministic tar shards of a region tree, a manifest of their hashes, and verified staging.

``write_shards`` packs ``<source_root>/<region_id>/...`` into byte-reproducible tars; ``stage_shards``
copies them from a slow mount to local disk, checks sha256 and extracts, skipping what is already staged.
"""

import hashlib
import json
import logging
import shutil
import tarfile
import time
from collections.abc import Sequence
from pathlib import Path

from fair_trees.preprocess import CHIP_SUFFIXES

logger = logging.getLogger(__name__)

SHARD_BYTES = 250 * 2**20
VECTOR_NAMES = ("region.json", "labels.geojson", "ignore.geojson")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _arcname(path: Path, source_root: Path) -> str:
    return path.relative_to(source_root).as_posix()


def _write_tar(path: Path, source_root: Path, files: Sequence[Path]) -> None:
    """Uncompressed tar with fixed metadata; members sorted by their ``/``-separated archive name."""
    members = sorted((_arcname(f, source_root), f) for f in files)
    with tarfile.open(path, "w") as tar:
        for name, f in members:
            info = tarfile.TarInfo(name)
            info.size = f.stat().st_size
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o644
            with f.open("rb") as fh:
                tar.addfile(info, fh)


def _entry(path: Path) -> dict:
    return {"name": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}


def _pack_zoom(chips: list[Path], source_root: Path, out_dir: Path, z: str, shard_bytes: int) -> list[dict]:
    groups: list[list[Path]] = []
    size = 0
    for chip in chips:
        n = chip.stat().st_size
        if not groups or size + n > shard_bytes:
            groups.append([])
            size = 0
        groups[-1].append(chip)
        size += n
    entries = []
    for k, group in enumerate(groups):
        path = out_dir / f"{z}-{k:04d}.tar"
        _write_tar(path, source_root, group)
        entries.append(
            {
                **_entry(path),
                "files": len(group),
                "regions": sorted({_arcname(f, source_root).split("/")[0] for f in group}),
            }
        )
    return entries


def write_shards(
    source_root: Path, out_dir: Path, *, source: str, converter_sha: str, shard_bytes: int = SHARD_BYTES
) -> dict:
    source_root, out_dir = Path(source_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    region_files = sorted(source_root.glob("*/region.json"))

    vectors = [f.parent / n for f in region_files for n in VECTOR_NAMES if (f.parent / n).exists()]
    _write_tar(out_dir / "vectors.tar", source_root, vectors)

    lines = [json.dumps(json.loads(f.read_text()), sort_keys=True) for f in region_files]
    (out_dir / "regions.jsonl").write_text("".join(line + "\n" for line in lines))

    chips_by_zoom: dict[str, list[Path]] = {}
    for p in source_root.glob("*/z*/chips/*"):
        if p.suffix.lower() in CHIP_SUFFIXES:
            chips_by_zoom.setdefault(p.parent.parent.name, []).append(p)
    shards: list[dict] = []
    for z in sorted(chips_by_zoom):
        chips = sorted(chips_by_zoom[z], key=lambda p: _arcname(p, source_root))
        shards += _pack_zoom(chips, source_root, out_dir, z, shard_bytes)

    manifest = {
        "source": source,
        "converter_sha": converter_sha,
        "regions_jsonl": {"sha256": _sha256(out_dir / "regions.jsonl")},
        "vectors": _entry(out_dir / "vectors.tar"),
        "shards": shards,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def manifest_hash(manifest_path: Path) -> str:
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def _stage_tar(src: Path, dst: Path, incoming: Path, sha256: str) -> None:
    marker = dst / ".staged" / f"{src.name}.{sha256[:16]}"
    if marker.exists():
        return
    incoming.mkdir(parents=True, exist_ok=True)
    local = incoming / src.name
    start = time.monotonic()
    shutil.copyfile(src, local)
    seconds = max(time.monotonic() - start, 1e-9)
    if _sha256(local) != sha256:
        local.unlink()
        raise ValueError(f"{src.name}: sha256 mismatch (drive copy corrupt or changed)")
    with tarfile.open(local) as tar:
        tar.extractall(dst, filter="data")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()
    mb = local.stat().st_size / 2**20
    local.unlink()
    logger.info("%s: copied %.1f MB at %.1f MB/s", src.name, mb, mb / seconds)


def stage_shards(drive_root: Path, local_root: Path, sources: Sequence[str]) -> Path:
    drive_root, local_root = Path(drive_root), Path(local_root)
    for source in sources:
        src = drive_root / "shards" / source
        dst = local_root / source
        manifest_bytes = (src / "manifest.json").read_bytes()
        staged = dst / "manifest.json"
        if staged.is_file() and staged.read_bytes() != manifest_bytes:
            # the shards changed: start from an empty tree so no chip or region survives from the old one
            logger.info("%s: shards changed on drive, removing the staged copy %s", source, dst)
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(manifest_bytes)
        for entry in [manifest["vectors"], *manifest["shards"]]:
            _stage_tar(src / entry["name"], dst, local_root / ".incoming", entry["sha256"])
        # written last, so a stage interrupted before this point does not look complete
        (dst / "regions.jsonl").write_bytes((src / "regions.jsonl").read_bytes())
        staged.write_bytes(manifest_bytes)
    return local_root
