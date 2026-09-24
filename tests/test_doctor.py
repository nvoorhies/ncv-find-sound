from dataclasses import replace

from find_sound import doctor, server_control


async def test_doctor_reports_a_down_endpoint(cfg, capsys):
    down = replace(cfg, autostart_server=False, embedding=replace(cfg.embedding, base_url="http://127.0.0.1:9"))
    assert await doctor.run(down) == 1
    out = capsys.readouterr().out
    assert "[!!] audio+query model" in out and "doctor --fix" in out
    assert "[ok] library" in out


async def test_doctor_treats_an_idle_server_as_asleep(cfg, capsys):
    asleep = replace(cfg, embedding=replace(cfg.embedding, base_url="http://127.0.0.1:9"))
    await doctor.run(asleep)
    out = capsys.readouterr().out
    assert "[..] audio+query model" in out and "starts on demand" in out
    assert "[!!] audio+query model" not in out


def test_server_command_serves_every_model_on_that_url(cfg, monkeypatch):
    captured = {}

    class P:
        pid = 1

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return P()

    monkeypatch.setattr(server_control.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server_control.importlib.util, "find_spec", lambda name: object())
    c = replace(cfg, server_idle_timeout=600, embedding=replace(cfg.embedding, base_url="http://localhost:7997"),
                text_embedding=replace(cfg.embedding, base_url="http://localhost:7997", model="Qwen/Qwen3-Embedding-4B"))
    server_control.start_server(c, "http://localhost:7997")
    cmd = captured["cmd"]
    assert cmd[cmd.index("--port") + 1] == "7997"
    assert cmd[cmd.index("--clap-model") + 1] == "fake-clap"
    assert cmd[cmd.index("--text-model") + 1] == "Qwen/Qwen3-Embedding-4B"
    assert cmd[cmd.index("--idle-timeout") + 1] == "600"
