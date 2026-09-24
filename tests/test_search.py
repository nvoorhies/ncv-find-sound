import math
import os

import pytest

from find_sound.config import SearchConfig
from find_sound.describe import bpm_from_path, clean_name, describe, identity_key, kind_from_path, variant_key
from find_sound.search import SearchIndex, parse_query, parse_range, run_search


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("<2", (0, 2)),
        (">30", (30, math.inf)),
        ("1-3", (1, 3)),
        ("500ms", (0.5, 0.5)),
        ("1.5m", (90, 90)),
        ("110-130bpm", (110, 130)),
        ("<=2s", (0, 2)),
    ],
)
def test_parse_range(spec, expected):
    assert parse_range(spec) == pytest.approx(expected)


def test_parse_range_tolerance_and_errors():
    assert parse_range("120", tolerance=0.05) == pytest.approx((114, 126))
    with pytest.raises(ValueError):
        parse_range("fast")


def test_parse_query_extracts_filters():
    text, f = parse_query("tense synth bpm:120-140 dur:<30 kind:music,amb in:Foley loop", SearchConfig())
    assert text == "tense synth loop"
    assert f.bpm == (120, 140)
    assert f.duration == (0, 30)
    assert f.kinds == {"music", "ambience"}
    assert f.path == ["foley"]


def test_parse_query_leaves_plain_colons_alone():
    text, f = parse_query("ratio 2:1 compressor pump", SearchConfig())
    assert text == "ratio 2:1 compressor pump" and not f.describe()


def test_clean_name_and_describe():
    assert clean_name("WEAPMisc_Steampunk Weapon Single Shots 26_JDOE_NONE") == "weapons: Steampunk Weapon Single Shots"
    assert clean_name("Wood Chop Break E") == "Wood Chop Break"
    assert clean_name("footstepGravel_03") == "footstep Gravel"
    d = describe(
        "/lib/acme/Cozy Casual Sound FX Pack Vol. 1 (2)/Cozy Casual Sound FX Pack Vol. 1/Inventory/Plop B.wav",
        "/lib",
        {"artist": "someone", "genre": "Foley"},
    )
    assert d == "Plop. Inventory, Cozy Casual Sound FX Pack, acme. Foley"


def test_kind_and_bpm_from_path():
    assert kind_from_path("/lib/acme/Jazz Music Pack/Track.wav", "/lib") == "music"
    assert kind_from_path("/lib/Hero VO Pack/Callouts/Go.wav", "/lib") == "voice"
    assert kind_from_path("/lib/x/AMBForst_Night Birds_ABC_DEF.wav", "/lib") == "ambience"
    assert kind_from_path("/lib/sfx/Laser.wav", "/lib") is None
    assert bpm_from_path("/x/CLOCKTick_Loopable Clock 12 - 60BPM_JDOE_NONE.wav") == 60
    assert bpm_from_path("/x/128 bpm/loop.wav") == 128
    assert bpm_from_path("/x/bpm/loop.wav") is None


def test_identity_and_variant_keys():
    assert identity_key("/p/Pack/wav/Fx 29.wav") == identity_key("/p/Pack (1)/ogg/Fx 29.ogg")
    assert identity_key("/p/Pack/wav/Fx 29.wav") != identity_key("/p/Pack/wav/Fx 28.wav")
    assert variant_key("/p/Laser 004.wav") == variant_key("/p/Laser 017.wav")
    assert variant_key("/p/Wood Chop Break C.wav") == variant_key("/p/Wood Chop Break E.wav")
    assert variant_key("/p/Chase/Epic Chase Main.wav") == variant_key("/p/Chase/Epic Chase Cut 30.wav")
    assert variant_key("/p/Chase/Epic Chase Main.wav") == variant_key("/p/Chase/Epic Chase Intensity 2.wav")
    assert variant_key("/p/Torch.wav") == variant_key("/p/Torch Loop.wav")
    assert variant_key("/p/Laser.wav") != variant_key("/p/Laser Gun.wav")


async def test_filters_and_grouping(make_indexer):
    ix = make_indexer()
    await ix.sync()
    index = SearchIndex(ix.store, ix.audio_model, ix.text_model)

    async def run(q, **kw):
        results, _ = await run_search(index, ix.audio, ix.text, q, ix.cfg.search, **kw)
        return results

    # Beep Low 01/02 are takes of one sound: one result carrying the other as a variant.
    low = await run("220hz beep low", k=10)
    lows = [r for r in low if "Beep Low" in r.path]
    assert len(lows) == 1 and len(lows[0].variants) == 1
    ungrouped = await run("220hz beep low", k=10, group_variants=False)
    assert len([r for r in ungrouped if "Beep Low" in r.path]) == 2

    # wav/ and ogg/ copies of the same tone are one result.
    a440 = [r for r in await run("440hz", k=10) if "A440" in r.path]
    assert len(a440) == 1 and len(a440[0].duplicates) == 1

    # Duration and named-BPM filters.
    assert {os.path.basename(r.path) for r in await run("tone dur:>5")} == {"Drone 90BPM.wav"}
    drone = (await run("bpm:90"))[0]
    assert drone.bpm == 90 and drone.bpm_source == "name" and drone.kind == "music"
    assert await run("bpm:140") == []
    assert all("/UI/" in r.path for r in await run("beep in:ui", k=10))
