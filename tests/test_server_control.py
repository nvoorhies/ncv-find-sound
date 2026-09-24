import asyncio
import socket
import threading
from dataclasses import replace

import pytest

from conftest import tone
from find_sound import server_control
from find_sound.embed_server import Activity, watch_idle
from find_sound.indexer import EndpointDown


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeProc:
    def __init__(self, port):
        self.pid = port
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen()

    def poll(self):
        return None


async def test_concurrent_callers_start_one_server(cfg, monkeypatch):
    port = free_port()
    c = replace(cfg, embedding=replace(cfg.embedding, base_url=f"http://127.0.0.1:{port}"))
    starts = []

    def fake_start(cfg, base_url):
        async def later():  # the "server" binds its port a moment after launch
            await asyncio.sleep(0.2)
            starts[-1][1].append(FakeProc(port))
        starts.append((base_url, []))
        asyncio.get_running_loop().create_task(later())
        return type("P", (), {"pid": 1, "poll": lambda self: None})()

    monkeypatch.setattr(server_control, "start_server", fake_start)
    results = await asyncio.gather(*(server_control.ensure_running(c) for _ in range(3)))
    assert len(starts) == 1 and sum(len(r) for r in results) == 1
    assert server_control.port_open(c.embedding.base_url)
    starts[0][1][0].sock.close()


async def test_autostart_off_is_a_clear_error(cfg):
    c = replace(cfg, autostart_server=False, embedding=replace(cfg.embedding, base_url=f"http://127.0.0.1:{free_port()}"))
    with pytest.raises(server_control.ServerUnavailable, match="autostart_server is off"):
        await server_control.ensure_running(c)


async def test_remote_endpoints_are_left_alone(cfg):
    assert await server_control.ensure_running(cfg) == []  # http://fake: not local


async def test_a_scan_with_nothing_new_never_needs_the_server(make_indexer, cfg, library):
    await make_indexer().sync()  # indexed through the (remote) fake endpoint
    # Same models, but now the endpoint is a local server that is not running and may not start.
    asleep = replace(cfg, autostart_server=False, embedding=replace(cfg.embedding, base_url=f"http://127.0.0.1:{free_port()}"))
    ix = make_indexer(asleep)
    assert (await ix.sync()).queued == 0
    tone(library / "UI" / "New.wav", 700)
    with pytest.raises(EndpointDown, match="not running"):
        await ix.sync()


def test_idle_watchdog_exits_only_when_idle():
    class Server:
        should_exit = False

    a, server = Activity(), Server()
    a.begin()  # a request in flight is never "idle"
    a.last -= 100
    t = threading.Thread(target=watch_idle, args=(a, server, 0.05, 0.01), daemon=True)
    t.start()
    t.join(0.2)
    assert not server.should_exit
    a.end()
    t.join(1.0)
    assert server.should_exit


def test_service_unit_runs_serve_with_this_config(cfg, tmp_path):
    from find_sound import service

    unit = service.render(replace(cfg, source=tmp_path / "my config.toml"), "127.0.0.1", 8765)
    exec_line = next(line for line in unit.splitlines() if line.startswith("ExecStart="))
    assert exec_line.endswith("serve --host 127.0.0.1 --port 8765")
    assert f'-c "{tmp_path / "my config.toml"}"' in exec_line
    assert "Environment=PATH=" in unit and "WantedBy=default.target" in unit
