from fastapi.testclient import TestClient

from find_sound.web import create_app


async def test_web_search_and_download(make_indexer, cfg, server):
    await make_indexer().sync()
    app = create_app(cfg, background_index=False, transport=server.transport)
    with TestClient(app) as client:
        assert "find-sound" in client.get("/").text

        data = client.get("/api/search", params={"q": "440hz tone", "k": 3}).json()
        top = data["results"][0]
        assert "A440" in top["name"] and top["rel"].startswith("Tones/")
        assert data["searched"] == 6

        audio = client.get(top["url"])
        assert audio.status_code == 200 and audio.content[:4] in (b"RIFF", b"OggS")
        dl = client.get(top["download_url"])
        assert "attachment" in dl.headers["content-disposition"]

        assert client.get("/api/search", params={"q": "x", "dur": "soon"}).status_code == 400
        assert client.get("/api/audio/999999").status_code == 404
        status = client.get("/api/status").json()
        assert status["files"] == 6 and status["indexing"] is False
