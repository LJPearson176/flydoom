"""Optional native game-state telemetry for GZDoom."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Optional
import zipfile


@dataclass(frozen=True)
class NativeGameState:
    game_tick: int
    x: float
    y: float
    z: float
    angle_deg: float
    pitch_deg: float
    vx: float
    vy: float
    vz: float
    health: float
    ammo: int
    kills: int
    player_kills: int = 0
    friendly_fire_kills: int = 0
    damage_dealt: float = 0.0
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    target_z: Optional[float] = None
    target_health: Optional[int] = None
    target_visible: bool = False

    @property
    def speed(self) -> float:
        return (self.vx * self.vx + self.vy * self.vy + self.vz * self.vz) ** 0.5


_PREFIX = "FLYDOOM_STATE:"


def parse_state_line(line: str) -> Optional[NativeGameState]:
    """Parse a console/log line containing the FLYDOOM_STATE protocol."""
    marker = line.find(_PREFIX)
    if marker < 0:
        return None
    fields = line[marker + len(_PREFIX):].strip().split(",")
    if len(fields) != 12 and len(fields) != 20:
        return None
    try:
        is_extended = len(fields) >= 20
        tgt_health = int(float(fields[18])) if is_extended else 0
        has_tgt = is_extended and tgt_health > 0
        return NativeGameState(
            game_tick=int(float(fields[0])),
            x=float(fields[1]), y=float(fields[2]), z=float(fields[3]),
            angle_deg=float(fields[4]), pitch_deg=float(fields[5]),
            vx=float(fields[6]), vy=float(fields[7]), vz=float(fields[8]),
            health=float(fields[9]), ammo=int(float(fields[10])),
            kills=int(float(fields[11])),
            player_kills=int(float(fields[12])) if is_extended else 0,
            friendly_fire_kills=int(float(fields[13])) if is_extended else 0,
            damage_dealt=float(fields[14]) if is_extended else 0.0,
            target_x=float(fields[15]) if has_tgt else None,
            target_y=float(fields[16]) if has_tgt else None,
            target_z=float(fields[17]) if has_tgt else None,
            target_health=tgt_health if has_tgt else None,
            target_visible=bool(int(float(fields[19]))) if is_extended else False,
        )
    except (TypeError, ValueError):
        return None


class GZDoomTelemetryReader:
    """Incrementally tail a GZDoom logfile without blocking the control loop."""

    def __init__(self, path: Path, *, stale_after_s: float = 1.0):
        self.path = Path(path)
        self.stale_after_s = stale_after_s
        self._offset = 0
        self._latest: Optional[NativeGameState] = None
        self._latest_wall_time: Optional[float] = None

    def poll(self) -> Optional[NativeGameState]:
        try:
            size = self.path.stat().st_size
            if size < self._offset:
                self._offset = 0
            with self.path.open("r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self._offset)
                for line in handle:
                    state = parse_state_line(line)
                    if state is not None:
                        self._latest = state
                        self._latest_wall_time = time.monotonic()
                self._offset = handle.tell()
        except OSError:
            return self._latest
        return self._latest

    def latest(self) -> Optional[NativeGameState]:
        state = self.poll()
        if state is None or self._latest_wall_time is None:
            return None
        if time.monotonic() - self._latest_wall_time > self.stale_after_s:
            return None
        return state


_ZSCRIPT = '''version "4.14.0"

class FlyDoomTelemetryHandler : StaticEventHandler
{
    int playerKills;
    int friendlyFireKills;
    int playerDamageDealt;
    Actor lastPlayerKilledMonster;

    override void WorldThingDamaged(WorldEvent e)
    {
        if (!e.Thing || !e.Thing.bIsMonster) return;
        PlayerInfo p = players[consoleplayer];
        Actor pmo = (p ? p.mo : null);
        if (pmo && (e.DamageSource == pmo || e.Inflictor == pmo || (e.DamageSource && e.DamageSource.player == p)))
        {
            playerDamageDealt += e.Damage;
            if (e.Thing.health <= 0 || e.Thing.health <= e.Damage)
            {
                lastPlayerKilledMonster = e.Thing;
            }
        }
    }

    override void WorldThingDied(WorldEvent e)
    {
        if (!e.Thing || !e.Thing.bIsMonster) return;
        PlayerInfo p = players[consoleplayer];
        Actor pmo = (p ? p.mo : null);
        console.printf("FLYDOOM_DIED_DEBUG: thing=%s src=%s inf=%s pmo=%s lpk=%s",
            e.Thing.GetClassName(),
            e.DamageSource ? ("" .. e.DamageSource.GetClassName()) : "null",
            e.Inflictor ? ("" .. e.Inflictor.GetClassName()) : "null",
            pmo ? ("" .. pmo.GetClassName()) : "null",
            lastPlayerKilledMonster ? ("" .. lastPlayerKilledMonster.GetClassName()) : "null");
        if (pmo && (e.DamageSource == pmo || e.Inflictor == pmo || (e.DamageSource && e.DamageSource.player == p) || e.Thing == lastPlayerKilledMonster))
        {
            playerKills++;
        }
        else
        {
            friendlyFireKills++;
        }
    }

    override void WorldTick()
    {
        if (gamestate != GS_LEVEL) return;
        PlayerInfo p = players[consoleplayer];
        if (!p || !p.mo) return;
        Actor mo = p.mo;
        int ammoCount = 0;
        if (p.ReadyWeapon && p.ReadyWeapon.Ammo1)
            ammoCount = p.ReadyWeapon.Ammo1.Amount;

        Actor targetMonster = null;
        double minDistance = 999999.0;
        ThinkerIterator it = ThinkerIterator.Create("Actor");
        Actor mo_iter;
        while ((mo_iter = Actor(it.Next())))
        {
            if (mo_iter.bIsMonster && mo_iter.health > 0 && !mo_iter.bCorpse)
            {
                bool vis = mo.CheckSight(mo_iter);
                double dist = mo.Distance2D(mo_iter);
                if (vis && dist < 800.0 && dist < minDistance)
                {
                    minDistance = dist;
                    targetMonster = mo_iter;
                }
            }
        }

        double targetX = 0.0;
        double targetY = 0.0;
        double targetZ = 0.0;
        int targetHealth = 0;
        int targetVisible = 0;
        if (targetMonster)
        {
            targetX = targetMonster.pos.x;
            targetY = targetMonster.pos.y;
            targetZ = targetMonster.pos.z;
            targetHealth = targetMonster.health;
            targetVisible = mo.CheckSight(targetMonster) ? 1 : 0;
        }

        console.printf("FLYDOOM_STATE:%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%d,%d",
            level.maptime, mo.pos.x, mo.pos.y, mo.pos.z, mo.angle, mo.pitch,
            mo.vel.x, mo.vel.y, mo.vel.z, mo.health, ammoCount, p.killcount,
            playerKills, friendlyFireKills, playerDamageDealt,
            targetX, targetY, targetZ, targetHealth, targetVisible);
    }
}
'''

_MAPINFO = '''GameInfo
{
    AddEventHandlers = "FlyDoomTelemetryHandler"
}
'''


def write_telemetry_pk3(path: Path) -> Path:
    """Write the telemetry handler consumed by the native adapter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("zscript.txt", _ZSCRIPT)
        archive.writestr("mapinfo.txt", _MAPINFO)
    return path


def inject_telemetry_wad(source_iwad: Path, output_iwad: Path) -> Path:
    """Inject FlyDoom ZSCRIPT and MAPINFO lumps into an IWAD copy.

    This allows native telemetry extraction with shareware DOOM1.WAD without
    triggering GZDoom's '-file with shareware' error.
    """
    import struct

    source = Path(source_iwad)
    output = Path(output_iwad)
    data = source.read_bytes()
    if len(data) < 12:
        raise ValueError(f"WAD file too small to be valid: {source}")

    wad_type, num_lumps, infotableofs = struct.unpack("<4sII", data[:12])
    if wad_type not in (b"IWAD", b"PWAD"):
        raise ValueError(f"Invalid WAD magic: {wad_type!r}")

    dir_size = num_lumps * 16
    if infotableofs + dir_size > len(data):
        raise ValueError(f"Malformed WAD directory offset {infotableofs} for {num_lumps} lumps")

    existing_lumps = {
        data[infotableofs + i * 16 + 8 : infotableofs + i * 16 + 16].rstrip(b"\x00")
        for i in range(num_lumps)
    }
    if b"ZSCRIPT" in existing_lumps and b"MAPINFO" in existing_lumps:
        if source.resolve() != output.resolve():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
        return output

    if infotableofs + dir_size == len(data):
        wad_data = bytearray(data[:infotableofs])
    else:
        wad_data = bytearray(data)

    directory = bytearray(data[infotableofs : infotableofs + dir_size])

    lumps_to_inject = [
        (b"ZSCRIPT", _ZSCRIPT.encode("utf-8")),
        (b"MAPINFO", _MAPINFO.encode("utf-8")),
    ]

    for name_prefix, content in lumps_to_inject:
        pos = len(wad_data)
        size = len(content)
        wad_data.extend(content)
        lump_name = name_prefix[:8].ljust(8, b"\x00")
        directory.extend(struct.pack("<II8s", pos, size, lump_name))

    new_infotableofs = len(wad_data)
    wad_data.extend(directory)
    new_num_lumps = num_lumps + len(lumps_to_inject)
    header = struct.pack("<4sII", wad_type, new_num_lumps, new_infotableofs)
    wad_data[:12] = header

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(".tmp")
    tmp_output.write_bytes(wad_data)
    tmp_output.replace(output)
    return output


def read_wad_lump(wad_path: Path, lump_name: str) -> Optional[bytes]:
    """Read a named lump from a WAD file."""
    import struct

    wad = Path(wad_path).read_bytes()
    if len(wad) < 12:
        return None
    wad_type, num_lumps, infotableofs = struct.unpack("<4sII", wad[:12])
    target_name = lump_name.encode("ascii")[:8].ljust(8, b"\x00")
    for i in range(num_lumps):
        pos, size, name = struct.unpack("<II8s", wad[infotableofs + i * 16 : infotableofs + (i + 1) * 16])
        if name == target_name:
            return wad[pos : pos + size]
    return None


def state_dict(state: Optional[NativeGameState]) -> Optional[dict]:
    if state is None:
        return None
    values = asdict(state)
    values["speed"] = state.speed
    return values

