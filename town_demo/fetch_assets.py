"""Vendor the Tuxemon art used by the Reward Town game and convert the Cotton Town map to town.json.

Run from the repo root:  .venv-gpu/bin/python -m town_demo.fetch_assets [--force]

Stdlib only. Files already present (including town.json) are kept unless --force, so a second run downloads
nothing; pass --force after editing world.LOCATIONS to rebuild town.json and rerun its reachability check.
"""

import argparse
import base64
import json
import os
import struct
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import deque
from pathlib import Path

from . import world

TUXEMON_SHA = "9e6258ff726b786040a267e8bdbbf037b560285e"
REPO_URL = "https://github.com/Tuxemon/Tuxemon"
RAW = f"https://raw.githubusercontent.com/Tuxemon/Tuxemon/{TUXEMON_SHA}/"
OUT = Path(__file__).parent / "game" / "assets"
MOD = "mods/tuxemon/"
MAP_SRC = MOD + "maps/spyder_cotton_town.tmx"
TILESET_DIR = MOD + "gfx/tilesets/"
CHARACTERS = ("adventurer", "heroine", "barmaid", "cooldude", "goth", "scientist", "florist", "postboy",
              "shopassistant", "riverboatcaptain")
COMBAT_UI = ("sunset_background",)  # the title screen backdrop
BUBBLES = ("exclamation", "dots", "heart", "sleep")
BELOW_LAYERS = ("Tile Layer 1", "Tile Layer 2", "Tile Layer 3")
ABOVE_LAYERS = ("Above Player",)
GID_MASK = 0x1FFFFFFF  # strips Tiled's flip bits


def static_files() -> list[tuple[str, str]]:
    """(source path in the Tuxemon repo, destination relative to OUT/tuxemon) for everything except tilesets."""
    files = [(MAP_SRC, "maps/spyder_cotton_town.tmx")]
    for n in CHARACTERS:
        files.append((f"{MOD}sprites/{n}.png", f"sprites/{n}.png"))
        files.append((f"{MOD}gfx/sprites/player/{n}.png", f"portraits/{n}.png"))
    files += [(f"{MOD}gfx/ui/combat/{n}.png", f"ui/{n}.png") for n in COMBAT_UI]
    files.append((f"{MOD}gfx/borders/borders.png", "ui/borders.png"))
    files += [(f"{MOD}gfx/bubbles/{n}.png", f"ui/bubble_{n}.png") for n in BUBBLES]
    files.append((f"{MOD}font/PressStart2P.ttf", "font/PressStart2P.ttf"))
    files += [("ATTRIBUTIONS.md", "ATTRIBUTIONS.md"), ("LICENSE", "LICENSE")]
    return files


def download(src: str) -> bytes:
    try:
        with urllib.request.urlopen(RAW + urllib.parse.quote(src), timeout=60) as resp:
            return resp.read()
    except (urllib.error.URLError, OSError) as err:
        raise SystemExit(f"download failed: {src}: {err}") from err


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def fetch(src: str, dest: str, force: bool) -> bool:
    """Downloads src to OUT/tuxemon/dest unless present; returns True when it downloaded."""
    path = OUT / "tuxemon" / dest
    if path.exists() and not force:
        return False
    write_atomic(path, download(src))
    print(f"  {src} -> {path.relative_to(OUT.parent)}")
    return True


def decode_layer(layer: ET.Element, width: int, height: int) -> list[int]:
    data = layer.find("data")
    if data is None or data.get("encoding") != "base64" or data.get("compression") != "zlib":
        raise SystemExit(f"layer {layer.get('name')!r}: expected base64+zlib data")
    raw = zlib.decompress(base64.b64decode(data.text.strip()))
    if len(raw) != 4 * width * height:
        raise SystemExit(f"layer {layer.get('name')!r}: {len(raw)} bytes, expected {4 * width * height}")
    return list(struct.unpack(f"<{width * height}I", raw))


def convert(tmx: bytes, force: bool) -> tuple[dict, list[tuple[str, str]]]:
    """Builds town.json from the TMX (fetching TSX files in memory and tileset PNGs to disk)."""
    root = ET.fromstring(tmx)
    width, height, tilesize = (int(root.get(k)) for k in ("width", "height", "tilewidth"))
    if (width, height) != (world.MAP_W, world.MAP_H) or tilesize != 16:
        raise SystemExit(f"map is {width}x{height}@{tilesize}px, expected {world.MAP_W}x{world.MAP_H}@16px")

    below: list[list[int]] = []
    above: list[list[int]] = []
    for layer in root.findall("layer"):
        name = layer.get("name")
        if name in BELOW_LAYERS:
            below.append(decode_layer(layer, width, height))
        elif name in ABOVE_LAYERS:
            above.append(decode_layer(layer, width, height))
        else:
            raise SystemExit(f"unknown layer {name!r}")
    if len(below) != len(BELOW_LAYERS) or len(above) != len(ABOVE_LAYERS):
        raise SystemExit("map is missing a tile layer")
    used = {gid & GID_MASK for layer in below + above for gid in layer} - {0}

    tilesets: list[dict] = []
    animations: dict[str, list[list[int]]] = {}
    fetched: list[tuple[str, str]] = []
    for ts in root.findall("tileset"):
        firstgid = int(ts.get("firstgid"))
        tsx_src = TILESET_DIR + Path(ts.get("source")).name
        tsx = ET.fromstring(download(tsx_src))
        image = tsx.find("image")
        png_name = image.get("source")
        dest = f"tilesets/{png_name.replace(' ', '_')}"
        fetch(TILESET_DIR + png_name, dest, force)
        fetched.append((TILESET_DIR + png_name, dest))
        tilesets.append({"firstgid": firstgid, "image": f"tuxemon/{dest}", "columns": int(tsx.get("columns")),
                         "tilecount": int(tsx.get("tilecount"))})
        for tile in tsx.findall("tile"):
            gid = firstgid + int(tile.get("id"))
            anim = tile.find("animation")
            if anim is None or gid not in used:
                continue
            animations[str(gid)] = [[firstgid + int(f.get("tileid")), int(f.get("duration"))]
                                    for f in anim.findall("frame")]

    blocked = [[False] * width for _ in range(height)]
    groups = [g for g in root.findall("objectgroup") if g.get("name") == "Collisions"]
    if len(groups) != 1:
        raise SystemExit("expected exactly one 'Collisions' object group")
    for obj in groups[0].findall("object"):
        x0, y0 = float(obj.get("x")), float(obj.get("y"))
        x1, y1 = x0 + float(obj.get("width", 0)), y0 + float(obj.get("height", 0))
        for ty in range(height):
            for tx in range(width):
                cx, cy = tx * tilesize + tilesize / 2, ty * tilesize + tilesize / 2
                if x0 <= cx < x1 and y0 <= cy < y1:
                    blocked[ty][tx] = True
    # Johnson Park fence with a 3-tile gate at x=8..10 on the bottom edge.
    fence = {(x, 20) for x in range(4, 15)} | {(x, y) for y in range(20, 30) for x in (4, 14)}
    fence |= {(x, 29) for x in (*range(4, 8), *range(11, 15))}
    for x, y in fence:
        blocked[y][x] = True
    for loc in world.LOCATIONS:
        if loc.door is not None:
            blocked[loc.door[1]][loc.door[0]] = False

    check_reachability(blocked)
    town = {
        "width": width, "height": height, "tilesize": tilesize, "tilesets": tilesets,
        "below": below, "above": above, "animations": animations,
        "blocked": "".join("1" if blocked[y][x] else "0" for y in range(height) for x in range(width)),
    }
    return town, fetched


def check_reachability(blocked: list[list[bool]]) -> None:
    sx, sy = world.PLAYER_START
    if blocked[sy][sx]:
        raise SystemExit(f"PLAYER_START {world.PLAYER_START} is blocked")
    seen = {(sx, sy)}
    queue = deque([(sx, sy)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (0 <= nx < world.MAP_W and 0 <= ny < world.MAP_H and not blocked[ny][nx]
                    and (nx, ny) not in seen):
                seen.add((nx, ny))
                queue.append((nx, ny))
    targets = set(world.PARK_SPOTS.values())
    for loc in world.LOCATIONS:
        if loc.door is not None:
            targets |= {loc.door, (loc.door[0], loc.door[1] + 1)}
    missing = sorted(targets - seen)
    if missing:
        raise SystemExit(f"tiles unreachable from PLAYER_START {world.PLAYER_START}: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="re-download files that already exist")
    args = parser.parse_args()

    world.validate()
    print(f"==> Tuxemon {TUXEMON_SHA[:12]} -> {OUT}")
    files = static_files()
    for src, dest in files:
        fetch(src, dest, args.force)
    if (OUT / "town.json").exists() and not args.force:
        print(f"kept {OUT / 'town.json'} (pass --force to rebuild)")
        return
    town, tileset_files = convert((OUT / "tuxemon" / files[0][1]).read_bytes(), args.force)
    files += tileset_files

    listing = "\n".join(f"- `{src}` → `{dest}`" for src, dest in files)
    source_md = (f"# Vendored Tuxemon assets\n\nRepository: {REPO_URL}\nCommit: `{TUXEMON_SHA}`\n\n"
                 f"Licenses: see `ATTRIBUTIONS.md` (art, mostly CC BY-SA 4.0) and `LICENSE` (code, GPLv3).\n"
                 f"Written by `town_demo/fetch_assets.py`.\n\n## Files\n\n{listing}\n")
    write_atomic(OUT / "tuxemon" / "SOURCE.md", source_md.encode())
    write_atomic(OUT / "town.json", json.dumps(town, separators=(",", ":")).encode())
    print(f"wrote {OUT / 'town.json'} ({len(town['animations'])} animated tiles, "
          f"{town['blocked'].count('1')} blocked tiles)")


if __name__ == "__main__":
    main()
