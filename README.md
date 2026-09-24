# find-sound

Find the right sound effect or music track for a description ("heavy wooden door slam",
"tense synth loop bpm:120-140", "forest at night kind:ambience") in a local sound library.

Every file is embedded once by a multimodal audio+text embedding model (CLAP by default) behind
an OpenAI-compatible `/embeddings` endpoint. The vectors are cached on disk. A search embeds only
the query and ranks the library by cosine similarity. On the command line you get paths; in the
browser you get players and download links.

```
$ find-sound search "creaky wooden door opening" -k 3
~/sounds/Foley Props/Containers and Keys/Door Hinge Creaking Door.wav
~/sounds/Foley Props/Containers and Keys/Creaky Door.wav
~/sounds/Wood Sound FX/Spooky/Old Wood Creak C.wav
```

## Quick start

```bash
uv sync
# 1. An embedding server that accepts audio. Skip this if you already run one (e.g. Infinity).
uv run --extra server find-sound-embed-server --port 7997     # CLAP on the GPU; downloads ~800 MB once
# 2. Point the tool at your library
cp find-sound.example.toml find-sound.toml                     # edit `library`
# 3. Index (incremental; rerun any time), then search
uv run find-sound index
uv run find-sound search "laser zap" -l
uv run find-sound serve                                         # http://127.0.0.1:8765, keeps indexing in the background
```

## Commands

| command | what it does |
|---|---|
| `index [--limit N]` | scan once; embed new/changed files; drop deleted ones |
| `watch [--interval S]` | `index` every `scan_interval` seconds (default 300) |
| `search QUERY [-k 5] [--json \| -l] [--kind] [--bpm] [--dur] [--in] [--copy-to DIR]` | best matches, one path per line by default |
| `serve [--port 8765] [--no-index]` | web UI + JSON API + background indexing |
| `stats` / `config` | index size, kinds, errors, last sync / effective settings |

`--json` output is meant for scripts and skills: path, score, both similarities, kind, duration,
BPM, description, variants and duplicate copies for each result. `--copy-to DIR` copies the
results into e.g. a game's `assets/audio/` and prints the new paths.

## How search works

**Two similarity channels.** The query is compared with (1) the audio embedding of each file
and (2) an embedding of a description built from the file's path and tags. Sound libraries put
most of what they know in file names, e.g. `WEAPMisc_Steampunk Weapon Single Shots 26_JDOE_NONE.wav`
becomes "weapons: Steampunk Weapon Single Shots". Unnamed packs (`Fx 12.wav`) get by on audio alone. Each
channel's scores are z-scored across the library before mixing, because the channels sit on
different scales. `audio_weight` / `text_weight` set the mix (0.65 / 0.35).

**Metadata as filters, not as embedding.** BPM, duration and kind are hard filters, and they can
go inside the query string so a single argument carries everything:

| token | meaning |
|---|---|
| `bpm:120` / `bpm:110-130` | tempo (±4% for a single value). A BPM in the file/folder name wins over the estimate; estimates also match at half/double time and only count for music with a clear pulse. |
| `dur:<2` / `dur:>30` / `dur:1-3` / `dur:500ms` | duration in seconds |
| `kind:sfx` / `kind:music,ambience` | music, sfx, voice, ambience. Folder names decide when they say (`Music Pack`, `VO`, `Ambience`, UCS `AMB`/`VOX`); otherwise zero-shot classification of the audio embedding against text prompts. |
| `in:foley` | the path contains this |

Filtering happens before ranking, so `dur:<1` returns the five best short clips rather than the
short clips that happen to be among the best five.

**Results are sounds, not files.** Identical content, the same name in `wav/` and `ogg/` folders,
and duplicated packs (`Pack (1)/`) collapse into one result that lists its copies. Takes and
versions collapse into one result with variants: `Laser 004` / `Laser 017`,
`Wood Chop Break C` / `E`, `Epic Chase Main` / `Cut 30` / `Intensity 2`, `Torch` / `Torch Loop`.
Names are only a hint, though. Files merge only when their audio embeddings also agree
(`variant_similarity`, 0.75, single-linkage), so a pack of fifty different `Fx N.wav` files
stays fifty results. On this library, random pairs of files reach 0.75 only 0.4% of the time,
while the median same-name group sits at 0.83.

## Indexing

```
scan -> hash (threads) -> decode, analyse, cut segments (process pool) -> embed audio -> SQLite
                                                                        -> describe -> embed text (batched)
```

- **Cached by content.** Vectors are keyed by content hash plus model settings. Renamed, moved or
  duplicated files are never re-embedded. Switching models re-embeds, but the old vectors stay
  cached, so switching back is free.
- **Incremental.** An unchanged file (same size and mtime) costs one `stat`. `watch`/`serve`
  rescan every `scan_interval` seconds; the web UI's *rescan* link starts a rescan straight away.
  Files modified within `settle_seconds` are probably still being copied, so they wait for the
  next scan.
- **Kinds** come from folder names where they say (`Music Pack`, `VO`, `Ambience`, UCS `AMB`/`VOX`),
  otherwise from zero-shot CLAP prompts. On the 8k files here whose folders name their kind, the
  zero-shot guess agrees 89% of the time (music 95%, sfx 91%, ambience 73%, voice 67%). Changing
  the rules reclassifies from stored vectors, with no re-embedding.
- **Long files** are embedded as up to 3 × 10 s windows spread over the file, then averaged.
- **Analysis** stores duration, loudness, tags (ID3/Vorbis/MP4), and tempo with a pulse-clarity
  score. The tempo comes from librosa and is estimated for clips ≥ 6 s.
- **Robust.** Unreadable files are recorded with their error and retried only when they change.
  If the endpoint goes down, the sync stops and files are retried next time. A lock keeps two
  indexers off the same index. Searches keep working while indexing, because SQLite runs in WAL
  mode.

The index is a single SQLite file (`~/.cache/find-sound/index.sqlite` by default). Search is
exact brute force in numpy. Ranking takes about 7 ms for this 8.8k-sound library and would take
roughly 40 ms at 100k, so no ANN index is needed. The first indexing run over 8,843 files (35 GB)
took 5.5 minutes on this machine with the bundled CLAP server on the RTX 5090.

## Embedding servers

The OpenAI embeddings API has no audio input, so `audio_format` selects the wire format:

| `audio_format` | request | servers |
|---|---|---|
| `infinity` | `{"input": ["data:audio/wav;base64,…"], "modality": "audio"}` | [Infinity](https://github.com/michaelfeil/infinity) with a CLAP model; the bundled `find-sound-embed-server` |
| `messages` | `{"messages": [{"role": "user", "content": [{"type": "input_audio", …}]}]}` | vLLM-style chat embeddings of omni models (one clip per request) |
| `messages-url` | same, with `{"type": "audio_url", "audio_url": {"url": "data:…"}}` | servers that want audio as a data URL |

Text and queries always go out as a standard `{"input": [...]}` request. A separate text-only
model for the name/tag channel can be configured under `[text_embedding]`; see
`find-sound.example.toml`.

## JSON API (`serve`)

- `GET /api/search?q=…&k=5&kind=&bpm=&dur=&group=true`: results with `url` (stream) and `download_url`
- `GET /api/audio/{id}[?download=1]`: the file (supports range requests, so seeking works)
- `GET /api/status`, `POST /api/rescan`

## Development

```bash
uv run pytest        # uses a fake embedding server and synthetic tones; no GPU or model needed
```

## License

MIT; see [LICENSE](LICENSE). This repository contains no audio. The index (`index.sqlite`)
holds embeddings and metadata derived from your own library, so keep it out of version control
when the sounds are licensed. The default location, `~/.cache/find-sound/`, is outside any repo.
