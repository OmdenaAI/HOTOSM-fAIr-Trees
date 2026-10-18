import logging
import shutil

import pytest

torch = pytest.importorskip("torch")

from fair_trees.params import TrainParams  # noqa: E402
from fair_trees.train import (  # noqa: E402
    check_resume,
    checkpoint_name,
    config_hash,
    latest_checkpoint,
    save_file,
    state_hash,
)


def _ckpts(out):
    return sorted(p.name for p in out.glob("ckpt-*.pt"))


def test_names_and_hashes():
    assert checkpoint_name(3, 125) == "ckpt-e003-s000125.pt"
    assert config_hash(TrainParams()) == config_hash(TrainParams())
    assert config_hash(TrainParams(learning_rate=2e-3)) != config_hash(TrainParams())
    assert state_hash({"w": torch.ones(2)}) == state_hash({"w": torch.ones(2)})
    assert state_hash({"w": torch.ones(2)}) != state_hash({"w": torch.zeros(2)})


def test_save_rotates_after_the_new_file_exists(tmp_path):
    out, tmp = tmp_path / "run", tmp_path / "tmp"
    for step in range(6):
        save_file({"step": step}, out, checkpoint_name(0, step), tmp_dir=tmp, keep_last=4)
    assert _ckpts(out) == [checkpoint_name(0, s) for s in range(2, 6)]
    assert not list(out.glob("*.part")) and not list(tmp.glob("*"))


def test_failed_copy_keeps_previous_checkpoints(tmp_path, monkeypatch):
    out, tmp = tmp_path / "run", tmp_path / "tmp"
    for step in range(4):
        save_file({"step": step}, out, checkpoint_name(0, step), tmp_dir=tmp, keep_last=4)

    def broken(*args, **kwargs):
        raise OSError("drive went away")

    monkeypatch.setattr(shutil, "copyfile", broken)
    with pytest.raises(OSError):
        save_file({"step": 4}, out, checkpoint_name(0, 4), tmp_dir=tmp, keep_last=4)
    assert _ckpts(out) == [checkpoint_name(0, s) for s in range(4)]


def test_size_mismatch_is_refused(tmp_path, monkeypatch):
    out, tmp = tmp_path / "run", tmp_path / "tmp"
    save_file({"step": 0}, out, checkpoint_name(0, 0), tmp_dir=tmp, keep_last=4)
    real = shutil.copyfile

    def truncating(src, dst, *args, **kwargs):
        real(src, dst)
        with open(dst, "r+b") as f:
            f.truncate(10)
        return dst

    monkeypatch.setattr(shutil, "copyfile", truncating)
    with pytest.raises(OSError, match="size"):
        save_file({"step": 1}, out, checkpoint_name(0, 1), tmp_dir=tmp, keep_last=4)
    assert _ckpts(out) == [checkpoint_name(0, 0)] and not list(out.glob("*.part"))


def test_latest_skips_corrupt_newest(tmp_path, caplog):
    out, tmp = tmp_path / "run", tmp_path / "tmp"
    save_file({"step": 1}, out, checkpoint_name(0, 1), tmp_dir=tmp)
    save_file({"step": 9}, out, checkpoint_name(1, 0), tmp_dir=tmp)
    (out / checkpoint_name(1, 0)).write_bytes(b"truncated")
    with caplog.at_level(logging.WARNING, logger="fair_trees.train"):
        path, state = latest_checkpoint(out)
    assert path.name == checkpoint_name(0, 1) and state["step"] == 1
    assert checkpoint_name(1, 0) in caplog.text
    assert latest_checkpoint(tmp_path / "empty") is None


def test_check_resume():
    state = {"config_hash": "a", "w0_hash": "w", "manifest_hash": "m", "repo_sha": "r1"}
    same = {"config_hash": "a", "w0_hash": "w", "manifest_hash": "m", "repo_sha": "r1", "force": False}
    check_resume(state, **same)
    for key, word in (("config_hash", "config"), ("w0_hash", "w0"), ("manifest_hash", "manifest")):
        changed = same | {key: "other"}
        with pytest.raises(ValueError, match=word):
            check_resume(state, **changed)
        check_resume(state, **(changed | {"force": True}))


def test_check_resume_warns_on_repo_change(caplog):
    state = {"config_hash": "a", "w0_hash": "w", "manifest_hash": "m", "repo_sha": "r1"}
    with caplog.at_level(logging.WARNING, logger="fair_trees.train"):
        check_resume(state, config_hash="a", w0_hash="w", manifest_hash="m", repo_sha="r2", force=False)
    assert "repo" in caplog.text
