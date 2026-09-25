"""`find-sound service install|uninstall|status`: keep `find-sound serve` running as a systemd
user service, so the library is rescanned every `scan_interval` seconds and the web UI is always
up, across reboots (with lingering enabled, even without a login).

The embedding server is not part of the service's steady state: with `autostart_server`, the
service starts it when a scan finds new files or someone searches, and it exits when idle.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Config
from .server_control import port_open

UNIT = "find-sound.service"


def unit_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / "systemd" / "user" / UNIT


def _launcher() -> list[str]:
    """How the service runs the CLI: the checkout's bin/find-sound wrapper when there is one
    (it brings the `server` extra), else this interpreter."""
    wrapper = Path(__file__).resolve().parents[2] / "bin" / "find-sound"
    if wrapper.is_file() and os.access(wrapper, os.X_OK):
        return [str(wrapper)]
    return [sys.executable, "-m", "find_sound.cli"]


def render(cfg: Config, host: str, port: int) -> str:
    cmd = _launcher() + (["-c", str(cfg.source)] if cfg.source else []) + ["serve", "--host", host, "--port", str(port)]
    # systemd user services get a minimal PATH; the wrapper needs uv (and ffmpeg for odd formats).
    uv = shutil.which("uv")
    path = ":".join(dict.fromkeys(([str(Path(uv).parent)] if uv else []) + ["/usr/local/bin", "/usr/bin", "/bin"]))
    return f"""[Unit]
Description=find-sound: sound search UI on http://{host}:{port}, rescanning the library every {cfg.scan_interval:.0f} s
After=network.target

[Service]
Environment=PATH={path}
ExecStart={" ".join(_quote(c) for c in cmd)}
Restart=on-failure
RestartSec=30
# On stop, systemd sends SIGTERM to every process in the unit. Python shuts down cleanly, but
# the `uv run` wrapper (the main PID) exits with 128+15; that is a normal stop, not a failure.
SuccessExitStatus=143
# Indexing bursts use several cores; stay out of the way of interactive work.
Nice=10
IOSchedulingClass=idle

[Install]
WantedBy=default.target
"""


def _quote(s: str) -> str:
    return f'"{s}"' if any(c.isspace() for c in s) else s


def _systemctl(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], check=check, text=True, capture_output=True)


def _active() -> bool:
    return _systemctl("is-active", UNIT, check=False).stdout.strip() == "active"


def install(cfg: Config, host: str, port: int) -> int:
    if not shutil.which("systemctl"):
        print("find-sound: systemctl not found; run `find-sound serve` under your own supervisor", file=sys.stderr)
        return 2
    if port_open(f"http://{host}:{port}") and not _active():
        print(f"find-sound: something already listens on {host}:{port} (a `find-sound serve` started by hand?); "
              "stop it first", file=sys.stderr)
        return 2
    path = unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(cfg, host, port))
    _systemctl("daemon-reload")
    _systemctl("enable", UNIT)
    _systemctl("restart", UNIT)  # picks up a changed unit when reinstalling
    print(f"installed {path} and started it; web UI at http://{host}:{port}")
    linger = subprocess.run(["loginctl", "show-user", os.environ.get("USER", ""), "-p", "Linger"],
                            text=True, capture_output=True).stdout.strip()
    if linger == "Linger=no":
        print("note: lingering is off, so the service runs only while you are logged in "
              "(`loginctl enable-linger` starts it at boot)")
    return 0


def uninstall() -> int:
    _systemctl("disable", "--now", UNIT, check=False)
    unit_path().unlink(missing_ok=True)
    _systemctl("daemon-reload", check=False)
    print(f"removed {unit_path()}")
    return 0


def status() -> int:
    r = _systemctl("status", "--no-pager", "--lines", "15", UNIT, check=False)
    print(r.stdout or r.stderr, end="")
    return 0 if _active() else 3
