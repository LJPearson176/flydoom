from pathlib import Path
import struct

from fly_doom.doom.gzdoom_telemetry import (
    GZDoomTelemetryReader,
    NativeGameState,
    inject_telemetry_wad,
    parse_state_line,
    read_wad_lump,
    write_telemetry_pk3,
)


def test_parse_state_line_accepts_prefixed_gzdoom_log_line():
    state = parse_state_line("Console: FLYDOOM_STATE:42,1,2,3,90,-4,5,6,0,87,19,2")
    assert state == NativeGameState(42, 1, 2, 3, 90, -4, 5, 6, 0, 87, 19, 2)
    assert state.speed == 7.810249675906654


def test_parse_state_line_accepts_extended_20_fields():
    line = "Console: FLYDOOM_STATE:100,1680.0,-2496.0,0.0,6.2,0.0,0.0,0.0,0.0,95.0,45,2,1,1,30.0,2272.0,-2432.0,0.0,20,1"
    state = parse_state_line(line)
    assert state is not None
    assert state.game_tick == 100
    assert state.x == 1680.0
    assert state.y == -2496.0
    assert state.kills == 2
    assert state.player_kills == 1
    assert state.friendly_fire_kills == 1
    assert state.damage_dealt == 30.0
    assert state.target_x == 2272.0
    assert state.target_y == -2432.0
    assert state.target_z == 0.0
    assert state.target_health == 20
    assert state.target_visible is True


def test_reader_tails_and_recovers_from_log_truncation(tmp_path: Path):
    logfile = tmp_path / "gzdoom.log"
    logfile.write_text("FLYDOOM_STATE:1,0,0,0,0,0,1,0,0,100,50,0\n")
    reader = GZDoomTelemetryReader(logfile)
    assert reader.latest().game_tick == 1
    logfile.write_text("FLYDOOM_STATE:2,4,5,6,7,8,0,1,0,90,49,1\n")
    assert reader.latest().game_tick == 2
    assert reader.latest().x == 4


def test_write_telemetry_pk3_contains_zscript_bundle(tmp_path: Path):
    path = write_telemetry_pk3(tmp_path / "telemetry.pk3")
    assert path.exists()
    import zipfile
    with zipfile.ZipFile(path) as archive:
        assert {"zscript.txt", "mapinfo.txt"} <= set(archive.namelist())
        assert "FLYDOOM_STATE:" in archive.read("zscript.txt").decode()


def test_inject_telemetry_wad(tmp_path: Path):
    # Construct a minimal valid IWAD with 1 lump (DEMO)
    lump_content = b"ORIGINAL_LUMP_DATA"
    header = struct.pack("<4sII", b"IWAD", 1, 12 + len(lump_content))
    directory_entry = struct.pack("<II8s", 12, len(lump_content), b"DEMO\x00\x00\x00\x00")
    source_wad = tmp_path / "DOOM1.WAD"
    source_wad.write_bytes(header + lump_content + directory_entry)

    output_wad = tmp_path / "DOOM1_INSTRUMENTED.WAD"
    result_path = inject_telemetry_wad(source_wad, output_wad)
    assert result_path == output_wad
    assert output_wad.exists()

    # Verify original lump is intact
    assert read_wad_lump(output_wad, "DEMO") == lump_content

    # Verify injected lumps
    zscript = read_wad_lump(output_wad, "ZSCRIPT")
    assert zscript is not None
    assert b"FlyDoomTelemetryHandler" in zscript
    assert b"FLYDOOM_STATE:" in zscript

    mapinfo = read_wad_lump(output_wad, "MAPINFO")
    assert mapinfo is not None
    assert b"AddEventHandlers = \"FlyDoomTelemetryHandler\"" in mapinfo

    # Verify idempotency: injecting an already-instrumented WAD does not duplicate lumps
    re_output = tmp_path / "RE_INSTRUMENTED.WAD"
    inject_telemetry_wad(output_wad, re_output)
    wad_type, num_lumps, infotableofs = struct.unpack("<4sII", re_output.read_bytes()[:12])
    assert num_lumps == 3  # DEMO, ZSCRIPT, MAPINFO

