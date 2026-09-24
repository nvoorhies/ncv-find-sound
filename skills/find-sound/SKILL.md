---
name: find-sound
description: >-
  Pick sound effects, music, ambiences and voice clips for a game from a local sound library
  by describing them, via the `find-sound` CLI. CLAP audio embeddings plus Qwen3
  embeddings of file names, with duration/BPM/kind filters. Copies the chosen files and their
  variants into the project and records where they came from. Use whenever a game or prototype
  needs audio: "find a jump sound", "I need a coin pickup sfx", "pick menu music", "add footstep
  sounds", "what explosions do we have", "music around 120 bpm", "sfx for this enemy", "add
  audio to the game", "replace this placeholder beep". Not for generating new audio: it only
  finds what is in the library.
---

# find-sound: pick game audio from the local library

`find-sound` searches an index of every audio file under the configured `library`
directories (`$FS config` shows them). Each query is compared with the audio itself (CLAP) and
with the file's name and folders (Qwen3-Embedding). Output is paths, or JSON with metadata.

`<checkout>` below is the clone of
[nvoorhies/ncv-find-sound](https://github.com/nvoorhies/ncv-find-sound) that this skill lives
in, as `<checkout>/skills/find-sound`. When the skill directory is a symlink, resolve it:

```bash
FS="$(readlink -f <this skill's directory>/../..)/bin/find-sound"    # i.e. <checkout>/bin/find-sound
```

If the skill was installed through a local overlay (a SKILL.md that points here), the overlay
names the checkout, the library, what it holds and its licences. Its facts win over the
generic advice below.

## Commands at a glance

| Command | Does | Cost |
|---|---|---|
| `doctor [--fix]` | checks config, library, both embedding models (text + a real audio clip), index; `--fix` starts the embedding server and updates the index | seconds; ~20 s to load the models the first time |
| `search "query" [-k N] [--json\|-l] [--kind K] [--dur R] [--bpm R] [--in S]` | best matches, one path per line by default | ~1 s |
| `search ... --copy-to DIR [--with-variants] --json` | same, and copies the results (and their takes) into DIR | ~1 s |
| `index` | picks up files added to the library since the last scan (the service does this every 5 min) | seconds; ~15-25 new files/s |
| `serve` | web page with players at http://127.0.0.1:8765 (for the user to listen) | runs until stopped |
| `service status` | whether the background service (periodic rescans + web UI) is installed and running | instant |

Filters also work inside the query string: `dur:<1`, `dur:1-3`, `bpm:120`, `bpm:110-130`,
`kind:sfx`, `kind:music,ambience`, `in:foley` (the path contains this).

## What it can and cannot do

- **You cannot listen.** Ranking comes from embeddings. Judge a hit by its `description`
  (cleaned-up name + folders), `kind`, `duration` and `bpm`. When the choice is a matter of
  taste (music, the player's own sounds, anything heard constantly), let the user listen
  before committing (see step 4).
- **It only finds what is in the library.** `$FS stats` gives counts by kind, and the
  top-level folders under the library root usually name the packs. When nothing fits, the
  results give it away: the descriptions don't match the request, and `text_similarity` stays
  under ~0.6 on every result. On one 9k-file game-audio library with Qwen3-Embedding-4B, real
  matches scored 0.65-0.8 and misses at most 0.59. `audio_similarity` doesn't help here
  (misses reach 0.4 too). Say the library has nothing suitable; don't pass off the nearest
  miss as a match.
- **Kinds are partly guessed.** Folder names decide when they are explicit (`Music Pack`,
  `VO`, `Ambience`); otherwise CLAP guesses. That guess matches folder labels 89% of the time,
  but short grunts and yells often count as `sfx` rather than `voice`. If `kind:voice` comes up
  empty, retry without it.
- **BPM** comes from the file name when stated there, otherwise it is estimated (music only,
  shown as `"bpm_source": "estimated"`). Estimates can be off by half or double time, and the
  filter accepts both.
- **Results are sounds, not files.** `variants` are other takes or versions of the same sound
  (`Hit 001..006`, a track's `Main` / `Cut 30` / `Intensity 2`). `duplicates` are the same
  sound in another format or a duplicated pack.

## Workflow

1. **Check the environment once per session**

   ```bash
   $FS doctor --fix
   ```

   Every line must be `[ok]`. `--fix` starts the embedding server (CLAP + Qwen3-Embedding-4B,
   about 9 GB of VRAM, log at `~/.cache/find-sound/embed-server.log`) and rescans the library
   if the index is stale. If it fails with CUDA out of memory, another GPU job holds the card;
   tell the user rather than killing it.

   Without `--fix`, a server reported as `[..] asleep, starts on demand` is fine. It exits
   after 15 idle minutes to free the GPU, and the next search starts it again, which makes
   that search take ~20 s longer. When `$FS service status` shows the service running, new
   sounds are indexed within 5 minutes of landing in the library, and the web UI is always
   up.

2. **Write the query: what it sounds like, plus what it is**

   The audio model responds to acoustic words (bright, metallic, deep, short, reverberant,
   distant, crunchy, 8-bit, whoosh). The name model responds to what the thing is (coin
   pickup, door, goblin, menu confirm). Use both in one query, and add filters:

   | Need | Query |
   |---|---|
   | pickup / reward blip | `"short bright coin pickup chime dur:<1.5"` |
   | UI | `"soft menu confirm click dur:<0.5 kind:sfx"` |
   | footsteps | `"footsteps on wooden floor kind:sfx"` (keep all variants) |
   | enemy | `"small goblin laughing"`, `"large monster roar dur:<4"` |
   | weapon | `"heavy sword hitting metal armor"`, `"sci-fi blaster shot dur:<1"` |
   | ambience loop | `"cave ambience with dripping water kind:ambience dur:>20"` |
   | music | `"calm background music for a main menu kind:music"`, `"intense boss battle music bpm:140-180"` |

   For anything important, run two or three phrasings with `-k 8 --json` and compare.

3. **Read the JSON**

   ```bash
   $FS search -k 8 --json "short bright coin pickup chime dur:<1.5"
   ```

   Per result: `path`, `description`, `kind`, `duration`, `bpm` / `bpm_source`, `score`
   (only meaningful within one query), `audio_similarity`, `text_similarity`, `variants`,
   `duplicates`. Prefer results with variants for anything that repeats (footsteps, hits,
   shots, impacts), so the game can randomize them. For music, a track's variants are its
   intensity layers and short cuts, which suit adaptive music.

4. **Let the user choose when it's a matter of taste**

   Present 3-5 candidates, one line each: the name, why it fits, and its duration or BPM. For
   listening:

   ```bash
   curl -sf http://127.0.0.1:8765/api/status >/dev/null || (nohup $FS serve >/dev/null 2>&1 &)
   ```

   (With the service installed, it's already running.)

   Then give them `http://127.0.0.1:8765/?q=<url-encoded query>`. The page plays every result
   and its variants.

5. **Copy into the project and record the source**

   ```bash
   $FS search -k 1 --json --copy-to <project>/assets/audio/sfx --with-variants "short bright coin pickup chime dur:<1.5"
   ```

   Or `cp` specific paths from an earlier search. The JSON gives `source` (library path) and
   `path` (the copy). Rename copies to the project's convention (`coin_pickup_01.wav`, ...).
   Then append a line per sound to `<project>/assets/audio/SOURCES.md`: the file in the game,
   where it came from, and the pack (usually the top-level folder under the library root).

   **Licensing:** these are commercial, royalty-free packs. Such licences normally allow
   shipping the sounds inside a game, but not redistributing the raw files, and a public repo
   does exactly that. Before committing audio to a public repository, ask the user (the usual
   answer is to gitignore `assets/audio/` or use a private repo or LFS).

6. **Engine notes (Godot 4)**

   Short sound effects work well as WAV. Music and long ambience loops should be Ogg Vorbis (a
   fraction of the size, streamed); convert with
   `ffmpeg -i in.wav -c:a libvorbis -q:a 5 out.ogg`. Turn looping on in the Import dock for
   loops. Adaptive-music variants (`Intensity 1/2/3`) of one track share tempo and length, so
   they can be crossfaded in sync.

## Troubleshooting

- `doctor` shows `[!!]` for a model even after `--fix`: read
  `~/.cache/find-sound/embed-server.log`. `address already in use` means another server owns
  port 7997. Check that `curl -s localhost:7997/health` lists both models.
- Search prints `the index ... is empty`: run `$FS doctor --fix` (or `$FS index`).
- A pack was just added to the library: the service picks it up within 5 minutes; `$FS index`
  does it now. Files still being copied are left for the next run.
- `another indexer (pid N) is updating`: a `serve` or `watch` process is indexing. Searches
  still work; results fill in as it goes.
- Tuning: `$FS eval <checkout>/evals/game-audio.toml` measures precision on
  known queries; `--audio-weight` on `search` shifts the audio/name balance for one query
  (default 0.4).
