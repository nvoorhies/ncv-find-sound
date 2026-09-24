"""Text derived from a file's path and tags: the description that gets embedded, a path-based
guess at what kind of audio it is, a BPM stated in the name, and a key for grouping variants.

Sound libraries encode most of what they know in names ("WEAPMisc_Steampunk Weapon Single
Shots 26_JDOE_NONE.wav", "Metal Music Pack/Fallen Angel (RT 6.455)/... Cut 30.wav"), so a
cleaned-up version of the path is a strong second signal next to the audio embedding.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePath

import numpy as np

KINDS = ("music", "sfx", "voice", "ambience")

# Prompts for zero-shot classification against the audio embedding when the path says nothing,
# in LAION CLAP's own zero-shot template. On ~8k files whose folders name their kind this scored
# 88% vs. 84% for plain descriptions ("a person speaking"), mostly by recognising voice.
KIND_PROMPTS = {
    kind: [f"This is a sound of {x}" for x in xs]
    for kind, xs in {
        "music": ["music", "a song", "instrumental music"],
        "sfx": ["a sound effect", "foley", "an impact", "a machine", "a click"],
        "voice": ["speech", "a man speaking", "a woman speaking", "a person shouting", "a voice"],
        "ambience": ["ambience", "a room tone", "wind", "an environment", "nature", "a city"],
    }.items()
}

# Bump when the classification logic below changes; the indexer then reclassifies every file
# from its stored embedding (no re-embedding needed).
KIND_RULES_REVISION = 2
SHORT_CLIP_SECONDS = 4.0

# First match wins. Music goes first: "Ambient Music Pack" is music, and "ambient" alone is a
# genre rather than a sign of an ambience bed.
_KIND_RULES = [
    ("music", re.compile(r"\b(music|soundtracks?|ost|bgm|songs?|stems?)\b")),
    ("voice", re.compile(r"\b(vo|voices?|vox|dialog(ue)?|speech|callouts?|barks)\b")),
    ("ambience", re.compile(r"\b(ambiences?|ambs?|atmos(phere)?s?|room ?tones?)\b")),
]

# Top-level Universal Category System (UCS) CatID prefixes, as used by most commercial
# libraries: "WEAPMisc_..." -> "weapons".
_UCS = {
    "AIR": "air", "ALRM": "alarm", "AMB": "ambience", "ANML": "animal", "ARCH": "archived",
    "AUTO": "automobile", "BEEP": "beep", "BELL": "bell", "BLLT": "bullet", "BOAT": "boat",
    "BUBL": "bubbles", "CERM": "ceramics", "CHAIN": "chain", "CHEM": "chemical", "CLOCK": "clock",
    "CLOTH": "cloth", "COMM": "communication", "COMP": "computer", "CRWD": "crowd", "DSGN": "designed",
    "DEST": "destruction", "DIRT": "dirt", "DOOR": "door", "DRWR": "drawer", "ELEC": "electricity",
    "EQUIP": "equipment", "EXPL": "explosion", "FARM": "farm", "FIGHT": "fight", "FIRE": "fire",
    "FIRW": "fireworks", "FOLY": "foley", "FOOD": "food", "FOOT": "footsteps", "GAME": "game",
    "GEO": "geological", "GLAS": "glass", "GORE": "gore", "GUNS": "guns", "HORN": "horn",
    "HUMN": "human", "ICE": "ice", "LEAT": "leather", "LIQ": "liquid", "MACH": "machine",
    "MAGC": "magic", "MECH": "mechanical", "METL": "metal", "MOTR": "motor", "MOVE": "movement",
    "MUSC": "musical instrument", "NATR": "nature", "OBJ": "object", "PAPR": "paper",
    "PLAS": "plastic", "RAIN": "rain", "ROBO": "robot", "ROCK": "rock", "ROPE": "rope",
    "RUBR": "rubber", "SCIF": "science fiction", "SNOW": "snow", "SPRT": "sports", "STEAM": "steam",
    "SWSH": "swoosh", "SWTH": "switch", "TOOL": "tools", "TOYS": "toys", "TRAN": "transportation",
    "TRAIN": "train", "UI": "user interface", "VEH": "vehicle", "VOX": "voice", "WATR": "water",
    "WEAP": "weapons", "WEA": "weather", "WHSH": "whoosh", "WIND": "wind", "WNDW": "window",
    "WING": "wings", "WOOD": "wood",
}
_UCS_CATID = re.compile(r"^([A-Z]{2,5})([A-Z][a-z]+)?$")
_JUNK_DIRS = re.compile(
    r"^(macosx|sonniss.*|.*gameaudiobundle.*|free samples|samples?|audio|sounds?|sfx|wav|ogg|mp3|"
    r"files?|assets?|\d+(bit|khz)?|\d+-bit|v?\d+(\.\d+)*)$",
    re.I,
)
# "_" counts as a word character, so \b would miss "60BPM_JDOE"; look for letters instead.
_BPM_RE = re.compile(
    r"(?<![\d.])(\d{2,3}(?:\.\d+)?)\s*[-_ ]?bpm(?![a-z])|(?<![a-z])bpm\s*[-_ ]?(\d{2,3}(?:\.\d+)?)(?![\d.])", re.I
)
# Folders that only separate encodings of the same sounds: wav/, ogg/, "48kHz 24bit/", ...
_FORMAT_DIR = re.compile(r"^(wav|wave|ogg|mp3|flac|aiff?|opus|m4a|\d+(\.\d+)?\s*k?hz|\d+\s*-?\s*bits?)( files)?$", re.I)


def _words(s: str) -> str:
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\((\d+|RT [\d.]+|copy)\)", " ", s, flags=re.I)  # "(1)", "(RT 5.543)" download/loop junk
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)  # camelCase
    s = re.sub(r"\bvol\.?\s*\d+\b", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,")


def clean_name(stem: str) -> str:
    """'WEAPMisc_Steampunk Weapon Single Shots 26_JDOE_NONE' -> 'weapons: Steampunk Weapon Single Shots'"""
    parts = stem.split("_")
    prefix = ""
    if len(parts) >= 3 and (m := _UCS_CATID.match(parts[0])) and m.group(1) in _UCS:
        # UCS: CatID_FXName_CreatorID_SourceID; creator/source ids are noise.
        prefix = _UCS[m.group(1)] + ": "
        parts = [parts[1]]
    name = _words(" ".join(parts))
    name = re.sub(r"(\s+(\d+|[A-Za-z]))+$", "", name)  # trailing take numbers / variant letters
    return prefix + (name or stem)


def describe(path: str | PurePath, root: str | PurePath | None = None, tags: dict | None = None) -> str:
    """Short natural-language description of a file for the text embedding channel."""
    p = PurePath(path)
    rel = p.relative_to(root) if root and p.is_relative_to(root) else p
    folders, seen = [], set()
    for d in reversed(rel.parts[:-1]):
        w = _words(d)
        if not w or _JUNK_DIRS.match(w) or w.lower() in seen:
            continue
        seen.add(w.lower())
        folders.append(w)
        if len(folders) == 3:
            break
    bits = [clean_name(p.stem)]
    if folders:
        bits.append(", ".join(folders))
    if tags:
        # Artist names are noise to a sound search; title/genre/album describe the content.
        bits.append(", ".join(v for k, v in tags.items() if k in ("title", "genre", "album") and v))
    return ". ".join(b for b in bits if b)


def kind_from_path(path: str | PurePath, root: str | PurePath | None = None) -> str | None:
    p = PurePath(path)
    rel = p.relative_to(root) if root and p.is_relative_to(root) else p
    text = _words(str(rel)).lower()
    for kind, rx in _KIND_RULES:
        if rx.search(text):
            return kind
    if (m := _UCS_CATID.match(p.stem.split("_")[0])) and m.group(1) in ("AMB", "VOX"):
        return "ambience" if m.group(1) == "AMB" else "voice"
    return None


def kind_rules_version() -> str:
    payload = json.dumps([KIND_RULES_REVISION, SHORT_CLIP_SECONDS, KIND_PROMPTS, [(k, r.pattern) for k, r in _KIND_RULES]])
    return hashlib.blake2b(payload.encode(), digest_size=8).hexdigest()


def classify_audio(vecs: np.ndarray, durations: np.ndarray, labels: dict[str, np.ndarray]) -> list[str]:
    """Zero-shot kind from audio embeddings: the closest mean prompt embedding wins."""
    kinds = list(labels)
    sims = vecs @ np.stack([labels[k] for k in kinds]).T
    out = [kinds[i] for i in sims.argmax(axis=1)]
    # A 2 s jingle or a snippet of room tone is still used as a one-shot.
    return ["sfx" if k in ("music", "ambience") and d < SHORT_CLIP_SECONDS else k for k, d in zip(out, durations)]


def bpm_from_path(path: str | PurePath) -> float | None:
    """A tempo stated in the file or folder name ("Clock 12 - 60BPM", "loops/128 bpm/...")."""
    p = PurePath(path)
    for part in (p.stem, *reversed(p.parts[:-1])):
        if m := _BPM_RE.search(part):
            bpm = float(m.group(1) or m.group(2))
            if 40 <= bpm <= 250:
                return bpm
    return None


def _parent_key(p: PurePath) -> str:
    """Parent folder minus format-only folders and download-copy suffixes ("Pack (1)" -> "Pack")."""
    parts = [re.sub(r"\s*\(\d+\)$", "", d).lower() for d in p.parent.parts]
    return "/".join(d for d in parts if not _FORMAT_DIR.match(d))


def identity_key(path: str | PurePath) -> str:
    """Same sound, possibly another encoding or a duplicated pack:
    'Pack/wav/Fx 29.wav' ~ 'Pack/ogg/Fx 29.ogg' ~ 'Pack (1)/wav/Fx 29.wav'."""
    p = PurePath(path)
    return f"{_parent_key(p)}/{p.stem.lower()}"


# Words that name a version of a sound rather than a different sound: music packs often ship
# "Chase Main", "Chase Cut 30", "Chase Intensity 2"; sfx packs ship "Torch" and "Torch Loop".
_VERSION_WORDS = re.compile(
    r"\b(main|cut|intensity|loop(able|ed)?|stinger|version|ver|alt(ernate)?|edit|full|short|long|mix|"
    r"take|var(iation|iant)?|layer|seamless)\b"
)


def variant_key(path: str | PurePath) -> str:
    """Candidates for being versions or takes of one sound: names that differ only in numbers,
    a variant letter or version words ('Laser 004' ~ 'Laser 017', 'Wood Chop Break C' ~ 'Wood
    Chop Break E', 'Epic Chase Main' ~ 'Epic Chase Cut 30'). Search confirms with audio
    similarity, because 'Fx 1'...'Fx 50' can be fifty different sounds."""
    p = PurePath(path)
    stem = re.sub(r"[\s_\-]+", " ", p.stem.lower())
    stem = _VERSION_WORDS.sub(" ", stem)
    stem = re.sub(r"\d+", " ", stem)
    stem = re.sub(r"(?<![a-z])[a-z](?![a-z])", " ", stem)  # lone letters: take A/B/C
    return f"{_parent_key(p)}/{' '.join(stem.split())}"
