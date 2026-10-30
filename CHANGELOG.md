# Changelog

## 1.7.0 – 2026-10-30
### Added
- **"On this page" list (batch mode).** Every batch pass, and the session summary at stop, refreshes a `## 🧭 On this page` section just above `## 💡 Notes`: links to each `###` theme with its point count, the inbox ("n waiting") and each session summary. It appears once a note has two or more themes and is rebuilt (never duplicated). It is built directly from the headings, so it costs no model time and always matches the note. Setting *Batch: "On this page" list* (`toc_enabled`, default on).
- **Freeing GPU/RAM automatically.** `ollama_keep_alive_min` (default 10; 0 = unload after each answer, -1 = never) replaces the fixed 30-minute keep-alive. `unload_on_stop` (default on) unloads the live and batch models as soon as a recording or import ends. `whisper_idle_unload_min` (default 15, 0 = never) releases Whisper after that long without speech during a session (e.g. while paused); it reloads on the next thought.
- **Model check before processing.** Before every LLM call the backend asks Ollama (`/api/ps`) whether the model is loaded. If it isn't, it loads it first and reports `llm_loading` / `llm_ready`, so the dashboard shows "loading …" instead of a silent wait. A model that isn't installed gives a clear notice (`ollama pull …`), and the thought is kept unorganised (batch: stays in the inbox). In live mode the model is warmed up while Whisper loads.
- Plugin: checks that the configured models are installed before a session starts; *Models in memory* in settings (size and GPU share) with *Unload now*; command *Free memory: unload the Ollama models now*; a new guide section, *Freeing GPU and memory*.

## 1.6.1 – 2026-10-30
### Changed
- Hidden "thinking" is switched off for all LLM calls (`think: false`; `"low"` for gpt-oss, which can't disable it). This keeps reasoning models such as Qwen 3/3.5 fast and their JSON clean. If a server or model rejects the option, the call retries without it and remembers that. Inline `<think>…</think>` blocks are ignored when reading answers.
- In batch mode the session summary uses the batch model (with the batch context and output limits).

## 1.6.0 – 2026-10-30
### Added
- **Batch organising** as a second mode next to live organising (`organise_mode`: `live` | `batch`). Thoughts go straight into a `📥 Inbox (to organise)` section with their sources. The inbox is organised in one pass every `batch_interval_min` minutes, on stop, or via *Organise now*, optionally with a different model (`batch_ollama_model`, `batch_num_ctx`, `batch_num_predict`, `batch_max_thoughts`). Merged bullets keep all their sources and are grounding-checked. Thoughts the model leaves out are kept under "Other thoughts". If the model fails, the inbox is untouched and retried later. Organising passes can be undone.
- Mode choice in the start dialog, settings, quick options, and the command *Switch organising*. The dashboard shows the mode, an "in inbox" counter and an *Organise now* button.

## 1.5.1 – 2026-10-30
### Added
- *Install GPU support…* asks where to put the CUDA libraries: a local, non-synced folder (recommended), the Python environment, any folder, or an existing CUDA 12 install (no download). It shows free space and warns about synced folders (OneDrive / MEGA / Dropbox / the vault). New setting *GPU libraries folder* (`cuda_libs_dir`); the backend searches it (pip `--target` layout or a CUDA `bin` folder) and CUDA_PATH.

## 1.5.0 – 2026-10-30
### Added
- **GPU transcription.** `whisper_device` can be auto / cpu / cuda. The GPU is tested once in a separate process (a missing CUDA library cannot crash a session), the result is cached in `gpu_probe.json`, and pip-installed NVIDIA DLL folders are picked up automatically. Settings gain *Install GPU support* and *Check GPU*; diagnostics show the GPU status.
- **Undo** ("scratch that", "undo that", "remove the last point", Alt+Z, dashboard button). Restores the note, or removes only the added points when the note was edited since. Undoing a focused correction also restores the focus.
- **Session summary** on stop (and *Summary now*): summary, `- [ ]` action items and open questions, each grounded in the session's thoughts and linked to their sources. Re-running replaces the section.
- **Import recordings** (`--import-audio`; command and dashboard button): from the computer or the vault. Whisper segments are grouped into thoughts at pauses, clips are archived, and the router, voice commands, undo and summary all apply. The loudness envelope and progress show in the dashboard.
- **Automatic links** to existing notes, and **note properties** (`created`, `updated`, `voice_sessions`, `voice_minutes`, `tags`) that leave other properties untouched.
- **Pause / resume** listening (Alt+P, dashboard button). The current thought is kept, and paused spans are shaded on the timeline.

## 1.4.0 – 2026-10-30
### Added
- In-app **guide** (command "Open the guide", **Guide** button in settings, **?** on the dashboard; opens automatically on first install). It covers quick start, voice commands, focus, the dashboard, sources, every command with its current hotkey and a Run button, and troubleshooting.
- `Install_Plugin.bat`, `config.example.json`; `Start_Scribe.bat` creates `config.json` on first run.

## 1.3.0 – 2026-10-30
### Added
- **Choose the note when recording starts**: a new note (name template), a recent note, the current note, or any note. Setting: *When recording starts*.
- **Voice note commands**: "create a new note named X", "new note called X", "switch to a new note called X", "switch to Welcome.md", "go back to X", "open the note called X". Commands can be the first or last sentence of a thought.
- **Compact source links**: link chip with a count that expands on hover (time → transcript, Play, transcript quote). Works in Live Preview and Reading view.
- Setting *Minimum real sound* to ignore noise blips.
### Fixed
- Topic switches said as "switch to a new note called …" were not recognised, so no timeline marker appeared.
- Duplicate bullets: model output must now come from what was just said, `[n]` numbering is stripped, and repeated points add a source instead of a copy.
- Whisper hallucinations from noise ("see you next time"). Noise blips are discarded, the no-VAD retry only runs on sustained sound, filters are stricter for short clips, and you get a warning suggesting calibration.

## 1.2.0 – 2026-10-30
### Added
- **Session dashboard**: status, timer, live input meter, session audio timeline (click to play, note-switch markers), counters, filterable console. The note being written opens automatically and follows switches.
- **Focus**: Alt+F (add details) / Alt+Shift+F (correct) on selected bullets or sections. Editor highlights for focus and the latest change.
- Backend ↔ plugin command channel over stdin (`--ipc`); session timelines saved to `VoiceScribe_Archives/sessions/`.

## 1.1.0 – 2026-10-29
### Fixed
- Every speech chunk was discarded: chunk loudness was taken from its final silent block.
- Stopping killed the backend after 3 s and lost in-flight work. Stopping now finishes pending thoughts.
- The LLM rewrote the whole notes section with a small output cap, which would truncate notes. It now only proposes placement; edits are applied in code.
- Audio saved as `.ogg` (Opus) so Obsidian shows a player; transcript block ids are linkable; "Note that…" no longer switches notes; the filler remover no longer deletes meaningful words.
### Added
- Every backend setting in Obsidian, microphone calibration, diagnostics, and offline-by-default model loading.
