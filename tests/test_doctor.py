from dataclasses import replace

from find_sound import doctor


async def test_doctor_reports_a_down_endpoint(cfg, capsys):
    down = replace(cfg, embedding=replace(cfg.embedding, base_url="http://127.0.0.1:9"))
    assert await doctor.run(down) == 1
    out = capsys.readouterr().out
    assert "[!!] audio+query model" in out and "doctor --fix" in out
    assert "[ok] library" in out


def test_server_command_serves_every_model_on_that_url(cfg, monkeypatch):
    captured = {}

    class P:
        pid = 1

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return P()

    monkeypatch.setattr(doctor.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: object())
    c = replace(cfg, embedding=replace(cfg.embedding, base_url="http://localhost:7997"),
                text_embedding=replace(cfg.embedding, base_url="http://localhost:7997", model="Qwen/Qwen3-Embedding-4B"))
    doctor._start_server(c, "http://localhost:7997")
    cmd = captured["cmd"]
    assert cmd[cmd.index("--port") + 1] == "7997"
    assert cmd[cmd.index("--clap-model") + 1] == "fake-clap"
    assert cmd[cmd.index("--text-model") + 1] == "Qwen/Qwen3-Embedding-4B"
