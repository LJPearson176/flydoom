from pathlib import Path

import pytest

from fly_doom.doom.gzdoom_target import GZDoomTarget


def test_gzdoom_command_is_explicit_and_does_not_launch(tmp_path: Path):
    app = tmp_path / "GZDoom.app"
    executable = app / "Contents" / "MacOS" / "gzdoom"
    executable.parent.mkdir(parents=True)
    executable.touch()
    target = GZDoomTarget(app_path=app, map_name="E1M1")
    target.validate()
    command = target.command()
    assert command[:4] == (str(executable), "-window", "+map", "E1M1")
    assert "+bind w +forward" in command


def test_gzdoom_target_reports_missing_app(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="GZDoom app not found"):
        GZDoomTarget(app_path=tmp_path / "missing.app").validate()


def test_gzdoom_target_pins_native_semantic_bindings(tmp_path: Path):
    app = tmp_path / "GZDoom.app"
    executable = app / "Contents" / "MacOS" / "gzdoom"
    executable.parent.mkdir(parents=True)
    executable.touch()
    command = GZDoomTarget(app_path=app).command()
    assert "+bind a +left" in command
    assert "+bind d +right" in command
    assert "+bind space +attack" in command
    assert "+bind e +use" in command

