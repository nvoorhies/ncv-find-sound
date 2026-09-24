# find-sound

Find the right sound effect or music track for a description ("heavy wooden door slam",
"tense synth loop bpm:120-140", "forest at night kind:ambience") in a local sound library.

Every file is embedded once by a multimodal audio+text model (CLAP by default), and its name and
folders once by a text embedding model (Qwen3-Embedding by default). Both sit behind
OpenAI-compatible `/embeddings` endpoints. The vectors are cached on disk. A search embeds only
the query and ranks the library against both. On the command line you get paths; in the
browser you get players and download links. A [Claude skill](skills/find-sound/SKILL.md)
drives it for picking game audio.

```
$ find-sound search "creaky wooden door opening" -k 3
~/sounds/Foley Props/Containers and Keys/Door Hinge Creaking Door.wav
~/sounds/Foley Props/Containers and Keys/Creaky Door.wav
~/sounds/Wood Sound FX/Spooky/Old Wood Creak C.wav
```

## Quick start

```bash
mkdir -p ~/.config/find-sound
cp find-sound.example.toml ~/.config/find-sound/config.toml    # edit `library`
bin/find-sound doctor --fix     # starts the bundled embedding server (CLAP + Qwen3-Embedding-4B, ~9 GB VRAM;
                                # downloads ~8.5 GB of weights once) and indexes the library
bin/find-sound search "laser zap" -l
bin/find-sound serve            # http://127.0.0.1:8765, keeps indexing in the background
```

`bin/find-sound` runs the CLI from this checkout's uv environment from any directory, including
the `server` extra (torch, transformers). `uv run find-sound ...` works inside the checkout.
Already running an audio-capable embedding server, such as Infinity? Point `[embedding]` at it
and skip the extra.

## Commands

| command | what it does |
|---|---|
| `index [--limit N]` | scan once; embed new/changed files; drop deleted ones |
| `watch [--interval S]` | `index` every `scan_interval` seconds (default 300) |
| `search QUERY [-k 5] [--json \| -l] [--kind] [--bpm] [--dur] [--in] [--copy-to DIR]` | best matches, one path per line by default |
| `serve [--port 8765] [--no-index]` | web UI + JSON API + background indexing |
| `doctor [--fix]` | checks config, library, both endpoints (text and a real audio clip) and the index; `--fix` starts the bundled server for local endpoints and updates the index |
| `eval CASES.toml [--weights ...] [-v]` | precision@5 / MRR@10 on queries with known answers, across audio/text weightings |
| `stats` / `config` | index size, kinds, errors, last sync / effective settings |

`--json` output is meant for scripts and skills: path, score, both similarities, kind, duration,
BPM, description, variants and duplicate copies for each result. `--copy-to DIR` copies the
results into e.g. a game's `assets/audio/` (`--with-variants` copies their takes too) and reports
each copy's `source`.

## How search works

**Two similarity channels.** The query is compared with (1) the audio embedding of each file
and (2) an embedding of a description built from the file's path and tags. Sound libraries put
most of what they know in file names, e.g. `WEAPMisc_Steampunk Weapon Single Shots 26_JDOE_NONE.wav`
becomes "weapons: Steampunk Weapon Single Shots". Unnamed packs (`Fx 12.wav`) get by on audio alone. Each
channel's scores are z-scored across the library before mixing, because the channels sit on
different scales. `audio_weight` sets the mix; see *Choosing models* for how 0.4 was picked.

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

## Choosing models

`find-sound eval evals/game-audio.toml` runs 36 game-audio queries phrased the way you'd ask,
mostly in words the files don't use ("blade clash", "jingling pocket change", "rapping on a
wooden door"). Relevance is judged by path regexes. Results on the 8.8k-file library this was
built on, with CLAP (`laion/larger_clap_general`) always doing the audio:

| name/tag channel | P@5 at audio 0.65 | P@5 at audio 0.4 | best P@5 (at weight) | name channel only |
|---|---|---|---|---|
| CLAP's text encoder | 0.51 | 0.55 | 0.56 (0.2) | 0.53 |
| Qwen3-Embedding-4B, 2560-d | 0.63 | 0.71 | 0.74 (0.2) | 0.67 |
| **Qwen3-Embedding-4B, 1024-d** (default) | 0.63 | **0.68** | 0.73 (0.2) | 0.69 |
| Qwen3-Embedding-8B, 1024-d | 0.63 | 0.73 | 0.74 (0.2) | 0.65 |

Audio alone scores 0.40. The dedicated text model is the biggest single gain. At the default
weighting, 8B beats 4B by 0.05 (MRR 0.88 vs 0.85) but needs about 16 GB of VRAM instead of 8.
If the GPU has room, switching is one line in the config plus `--text-model`. Truncating 4B to
1024 dims moves P@5 by at most ±0.02 depending on the weighting, with MRR unchanged, and makes
the index 2.5× smaller. The query instruction matters little: three phrasings landed within
0.01 of each other, and no instruction scored 0.05 lower. Judging by path rewards descriptive
names, so the eval leans toward the name channel. The default of 0.4, rather than the eval's
0.2-0.3, keeps well-sounding but badly named files findable. Swapping models is cheap: vectors
are cached per model, and only descriptions are re-embedded (4k of them in ~9 s).

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
  score, for clips ≥ 6 s. librosa picks the tempo on its default 512-sample hop. Near 140 BPM
  that grid only allows 136, 143.6 and 152, so the beat period is then refined on a 128-sample
  onset envelope with parabolic peak interpolation. Synthetic beats at 97, 128.5, 141 and 173
  BPM come back within 0.5. When the analysis improves, `ANALYSIS_REVISION` is bumped, and
  older results are recomputed on the next sync without re-embedding.
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

Text and queries always go out as a standard `{"input": [...]}` request, with `dimensions` when
configured. The bundled `find-sound-embed-server` hosts CLAP plus any number of decoder
embedding models (`--text-model`, last-token pooled, as Qwen3-Embedding expects) on one port,
routed by the request's `model`.

## Claude skill

[`skills/find-sound/SKILL.md`](skills/find-sound/SKILL.md) teaches an agent to pick game audio
with this tool: check the environment, phrase queries (acoustic words plus what the thing is),
read the JSON, let the user listen when it's a matter of taste, copy files and variants into
the project, and record sources, since commercial sound licences usually forbid redistributing
raw files, for example in a public repo. Install by symlinking from the checkout:

```bash
ln -s "$PWD/skills/find-sound" ~/.claude/skills/find-sound
```

The skill is deliberately generic. Facts about one machine (where the checkout is, which packs
the library holds and what it lacks, what their licences allow) belong in a private overlay: a
small SKILL.md of your own, installed as `find-sound`, that states those facts and tells the
agent to read this one (for example via an `upstream.md` symlink next to it). The instructions
then stay in step with the tool, and nothing personal lands in this repo.

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
