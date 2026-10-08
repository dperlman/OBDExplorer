"""Settings stored inside exported PNG files, so every image records how it was made.

Every PNG written by ``OBDExplorerPlus.py`` (export, heatmap, tie-heatmap, cusp-proximity) gets PNG text chunks
(iTXt, the standard place for text in a PNG; the pixels are not touched):

    obd:command      the command line, when run as a subcommand (absent for the interactive menu)
    obd:cli_args     the parsed command-line options, as JSON
    obd:settings     every effective setting of the export, defaults included, as JSON
    obd:obd_core     the OBD-core version that computed the data
    obd:git_commit   this repository's commit (with "+dirty" if tracked files were modified)
    obd:created      when the file was written

Read them back with ``python -m obd_explorer.png_metadata FILE.png`` (or ``exiftool``, or PIL's
``Image.open(f).text``).  Image optimisers and some uploads strip text chunks; files in this repo
keep them.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import struct
import subprocess
import sys
import zlib
from datetime import datetime

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def _itxt(key: str, text: str) -> bytes:
    # keyword NUL, compression flag 0, compression method 0, empty language tag NUL, empty
    # translated keyword NUL, then UTF-8 text
    return _chunk(b"iTXt", key.encode("latin-1") + b"\x00\x00\x00\x00\x00" + text.encode("utf-8"))


def write_png_text(path: str, items: dict[str, str]) -> None:
    """Insert iTXt chunks before IEND, leaving every other byte of the PNG unchanged."""
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError(f"{path!r} is not a PNG file")
    pos = len(PNG_SIGNATURE)
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        if data[pos + 4 : pos + 8] == b"IEND":
            break
        pos += 12 + length
    else:
        raise ValueError(f"{path!r}: no IEND chunk")
    extra = b"".join(_itxt(k, v) for k, v in items.items())
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data[:pos] + extra + data[pos:])
    os.replace(tmp, path)


def read_png_text(path: str) -> dict[str, str]:
    """All tEXt and (uncompressed) iTXt entries of a PNG."""
    with open(path, "rb") as f:
        data = f.read()
    out: dict[str, str] = {}
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if ctype == b"tEXt":
            k, _, v = body.partition(b"\x00")
            out[k.decode("latin-1")] = v.decode("latin-1")
        elif ctype == b"iTXt":
            k, _, rest = body.partition(b"\x00")
            if rest[:1] == b"\x00":                                  # uncompressed
                rest = rest[2:]
                _lang, _, rest = rest.partition(b"\x00")
                _tkey, _, text = rest.partition(b"\x00")
                out[k.decode("latin-1")] = text.decode("utf-8")
        elif ctype == b"IEND":
            break
        pos += 12 + length
    return out


def _git_commit(exclude: str | None = None) -> str:
    """HEAD, with "+dirty" if tracked files differ from it; ``exclude`` (the file just written) is not counted."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        sha = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        cmd = ["git", "-C", here, "diff", "--quiet", "HEAD"]
        if exclude is not None:
            cmd += ["--", ".", ":(exclude)" + os.path.relpath(os.path.abspath(exclude), here)]
        dirty = subprocess.run(cmd).returncode != 0
        return sha + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def stamp_export(path: str, cfg, args=None) -> None:
    """Record how ``path`` was made inside it (PNG only; other formats are left alone)."""
    if not str(path).lower().endswith(".png") or not os.path.isfile(path):
        return
    try:
        from importlib.metadata import version

        core = version("obd-core")
    except Exception:
        core = "unknown"
    items = {}
    argv = sys.argv
    if len(argv) > 1:
        items["obd:command"] = "python " + " ".join(shlex.quote(a) for a in [os.path.basename(argv[0])] + argv[1:])
    if args is not None:
        items["obd:cli_args"] = json.dumps(vars(args), default=str, sort_keys=True)
    settings = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    items["obd:settings"] = json.dumps(settings, default=str, sort_keys=True)
    items["obd:obd_core"] = core
    items["obd:git_commit"] = _git_commit(exclude=path)
    items["obd:created"] = datetime.now().isoformat(timespec="seconds")
    write_png_text(path, items)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m obd_explorer.png_metadata FILE.png [...]")
        sys.exit(2)
    for f in sys.argv[1:]:
        print(f"== {f}")
        for k, v in read_png_text(f).items():
            if k == "obd:settings" or k == "obd:cli_args":
                v = json.dumps(json.loads(v), indent=2, sort_keys=True)
            print(f"{k}: {v}")
