"""Explicit target description for running FlyDoom against a local GZDoom app.

This module deliberately does not launch or control GZDoom. It gives the future
capture/input bridge one canonical, testable target contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class GZDoomTarget:
    """A local GZDoom.app target and its deterministic launch configuration."""

    app_path: Path = Path("/Applications/GZDoom.app")
    iwad: Optional[Path] = None
    map_name: str = "E1M1"
    windowed: bool = True
    extra_args: Tuple[str, ...] = ()
    telemetry_pk3: Optional[Path] = None
    telemetry_logfile: Optional[Path] = None

    @property
    def executable(self) -> Path:
        return self.app_path / "Contents" / "MacOS" / "gzdoom"

    def validate(self) -> None:
        if not self.app_path.is_dir():
            raise FileNotFoundError(f"GZDoom app not found: {self.app_path}")
        if not self.executable.is_file():
            raise FileNotFoundError(f"GZDoom executable not found: {self.executable}")
        if self.iwad is not None and not self.iwad.is_file():
            raise FileNotFoundError(f"IWAD not found: {self.iwad}")

    def command(self) -> Sequence[str]:
        """Return the executable command without starting a process."""
        args = [str(self.executable), "+map", self.map_name]
        if self.iwad is not None:
            args[1:1] = ["-iwad", str(self.iwad)]
        if self.windowed:
            args.insert(1, "-window")
        if self.telemetry_pk3 is not None:
            args.extend(("-file", str(self.telemetry_pk3)))
        if self.telemetry_logfile is not None:
            args.extend(("+logfile", str(self.telemetry_logfile)))
        # Pin the native control contract. GZDoom's modern macOS defaults map
        # A/D to strafing and Space to jumping, which is incompatible with
        # FlyDoom's semantic TURN/FIRE actions. GZDoom parses each +command as a
        # single argument string.
        args.extend((
            "+bind w +forward",
            "+bind a +left",
            "+bind d +right",
            "+bind e +use",
            "+bind space +attack",
        ))
        args.extend(self.extra_args)
        return tuple(args)

