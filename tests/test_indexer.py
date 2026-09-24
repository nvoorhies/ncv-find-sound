import os
import shutil
import time
from dataclasses import replace

import numpy as np
import pytest

from conftest import tone
from find_sound.indexer import EndpointDown, IndexBusy
from find_sound.search import SearchIndex, run_search
from find_sound.store import IndexLock


def search_index(ix):
    return SearchIndex(ix.store, ix.audio_model, ix.text_model)


async def search(ix, query, **kw):
    results, _ = await run_search(search_index(ix), ix.audio, ix.text, query, ix.cfg.search, **kw)
    return results


async def test_index_then_search(make_indexer, server):
    ix = make_indexer()
    report = await ix.sync()
    assert report.scanned == 6  # __MACOSX junk skipped
    assert report.done == 6 and report.failed == 0
    assert report.audio_embedded == 6

    results = await search(ix, "a 440hz tone")
    assert os.path.basename(results[0].path).startswith("Tone A440")
    results = await search(ix, "3000hz beep")
    assert os.path.basename(results[0].path) == "Beep High.wav"


async def test_resync_is_free_and_query_is_the_only_model_call(make_indexer, server):
    ix = make_indexer()
    await ix.sync()
    before = dict(server.calls)
    report = await ix.sync()
    assert report.queued == 0
    assert server.calls == before

    await search(ix, "440hz")
    assert server.calls["audio"] == before["audio"]
    assert server.calls["text"] == before["text"] + 1  # one query embedding, shared by both channels


async def test_rename_and_copy_reuse_the_audio_embedding(make_indexer, server, library):
    ix = make_indexer()
    await ix.sync()
    os.rename(library / "UI" / "Beep High.wav", library / "UI" / "Whistle.wav")
    shutil.copy(library / "UI" / "Whistle.wav", library / "UI" / "Whistle copy.wav")
    report = await ix.sync()
    assert report.removed == 1
    assert report.done == 2
    assert report.audio_embedded == 0 and report.audio_reused == 2


async def test_modified_and_deleted_files(make_indexer, server, library):
    ix = make_indexer()
    await ix.sync()
    tone(library / "UI" / "Beep High.wav", 1000, seconds=0.7)  # new content, same name
    os.remove(library / "Music Loops" / "Drone 90BPM.wav")
    report = await ix.sync()
    assert report.removed == 1
    assert report.done == 1 and report.audio_embedded == 1
    results = await search(ix, "1000hz")
    assert os.path.basename(results[0].path) == "Beep High.wav"
    assert not any("Drone" in r.path for r in await search(ix, "110hz", k=10))


async def test_changing_model_reembeds_but_keeps_old_vectors(make_indexer, server, cfg):
    ix = make_indexer()
    await ix.sync()
    other = replace(cfg, embedding=replace(cfg.embedding, max_segments=2))
    ix2 = make_indexer(other)
    report = await ix2.sync()
    assert report.queued == 6 and report.audio_embedded == 6
    # Back to the first settings: everything is still cached.
    assert (await make_indexer(cfg).sync()).queued == 0


async def test_undecodable_file_is_recorded_not_retried(make_indexer, server, library):
    (library / "UI" / "broken.wav").write_bytes(b"RIFF not really a wav file")
    ix = make_indexer()
    report = await ix.sync()
    assert report.failed == 1
    row = ix.store.db.execute("SELECT error FROM files WHERE path LIKE '%broken.wav'").fetchone()
    assert row["error"]
    assert (await ix.sync()).queued == 0


async def test_endpoint_down_aborts_and_retries_next_time(make_indexer, server):
    ix = make_indexer()
    server.down = True
    with pytest.raises(EndpointDown):
        await ix.sync()
    assert ix.store.db.execute("SELECT COUNT(*) FROM files WHERE error IS NULL").fetchone()[0] == 0
    server.down = False
    report = await ix.sync()
    assert report.done == 6 and report.failed == 0


async def test_only_one_indexer_at_a_time(make_indexer, cfg):
    other = IndexLock(cfg.index_path)
    assert other.acquire()
    try:
        with pytest.raises(IndexBusy):
            await make_indexer().sync()
    finally:
        other.release()


async def test_messages_audio_format(make_indexer, server, cfg):
    c = replace(cfg, embedding=replace(cfg.embedding, audio_format="messages"))
    ix = make_indexer(c)
    await ix.sync()
    audio_bodies = [b for b in server.bodies if "messages" in b]
    assert len(audio_bodies) == 6
    part = audio_bodies[0]["messages"][0]["content"][0]
    assert part["type"] == "input_audio" and part["input_audio"]["format"] == "wav"
    results = await search(ix, "440hz")
    assert "A440" in results[0].path


async def test_files_still_being_written_wait_for_the_next_scan(make_indexer, cfg, library):
    ix = make_indexer(replace(cfg, settle_seconds=60))
    old = time.time() - 120
    for p in library.rglob("*.*"):
        os.utime(p, (old, old))
    assert (await ix.sync()).done == 6
    tone(library / "UI" / "New.wav", 500)                   # just written
    tone(library / "UI" / "Beep High.wav", 1500)            # being overwritten
    report = await ix.sync()
    assert report.done == 0 and report.removed == 0         # neither indexed nor dropped yet


async def test_improved_analysis_reruns_without_reembedding(make_indexer):
    ix = make_indexer()
    await ix.sync()
    ix.store.db.execute("UPDATE content SET analysis_rev = 1")  # as if analysed by an older version
    ix.store.commit()
    report = await ix.sync()
    assert report.done == 1  # only the 8 s drone is long enough for tempo analysis
    assert report.audio_embedded == 0
    assert ix.store.db.execute("SELECT MIN(analysis_rev) FROM content").fetchone()[0] >= 2


async def test_analysis_values_are_stored_as_numbers(make_indexer):
    ix = make_indexer()
    await ix.sync()
    types = {r[0] for r in ix.store.db.execute(
        "SELECT DISTINCT typeof(duration) FROM content UNION SELECT DISTINCT typeof(bpm) FROM content "
        "UNION SELECT DISTINCT typeof(bpm_confidence) FROM content UNION SELECT DISTINCT typeof(rms_db) FROM content")}
    assert types <= {"real", "integer", "null"}


def test_numpy_scalars_are_not_stored_as_blobs(tmp_path):
    from find_sound.audio import Analysis
    from find_sound.store import Store

    s = Store(tmp_path / "i.sqlite")
    a = Analysis(duration=np.float64(2.0), sample_rate=48000, channels=2, rms_db=np.float32(-20.5),
                 peak_db=-3.0, bpm=np.float32(128.4), bpm_confidence=np.float32(0.5))
    s.put_content("h", a)
    row = s.db.execute("SELECT typeof(bpm), typeof(rms_db), typeof(duration), bpm FROM content").fetchone()
    assert tuple(row)[:3] == ("real", "real", "real") and abs(row[3] - 128.4) < 1e-3
