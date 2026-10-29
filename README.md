# VoiceScribe 🎙️

**Continuous, local voice-to-notes transcription, grounded synthesis, and routing system for Obsidian on Windows 11.**

VoiceScribe listens continuously to your microphone, dynamically segments your speech by natural pauses (~1.8–2.0 seconds), transcribes with high-accuracy `small.en` on CPU, saves full session transcripts to a dedicated `Transcripts/` file, and synthesizes structured, thematic notes in your active note—linking back to exact transcript lines and playable audio.

---

## ⚡ Obsidian-Native Control (No Bash / BAT Needed!)

VoiceScribe includes a native **Obsidian Plugin** installed directly into your `Note Rambling` vault:

1. **Ribbon Microphone Icon**: Look at the left sidebar ribbon in Obsidian. Click the **microphone** icon:
   - Starts recording silently in the background (turns bright red while active).
   - Click again to stop.
2. **Hotkey**: Press **`Alt + V`** anywhere in Obsidian to toggle recording on and off.
3. **Options & Configuration**:
   - **Settings Tab**: Open Obsidian Settings (`Ctrl + ,`) ➔ **VoiceScribe** (under Community Plugins).
   - **Quick Options Modal**: Click the **cog / settings icon** on the left ribbon, or press **`Alt + Shift + V`**, or run `VoiceScribe: Options: Quick Change Microphone & Models` from the Command Palette (`Ctrl + P`).
   - Change your input microphone, Whisper model, Ollama model, and speech pause duration with instant visual feedback!
4. **Status Bar Indicator**: In the bottom-right corner of Obsidian:
   - `🔴 VoiceScribe: Listening` when active.
   - `⏹️ VoiceScribe: Stopped` when idle.
5. **Command Palette**: Press `Ctrl + P` to trigger `VoiceScribe: Toggle Continuous Voice Recording` or `VoiceScribe: Open Options & Settings`.

*(To configure or launch outside Obsidian, double-click **`Configure_VoiceScribe.bat`** or **`Start_Scribe.bat`**).*

---

## 🎯 Architecture & New Hardened Capabilities

### 1. Dynamic Audio Segmentation (Context-Fitting Audio Files)
- Slices audio when you pause naturally for **~2.2 seconds** (configurable), instead of arbitrary fixed intervals.
- Rolling pre-speech buffer (~0.5s) prevents clipping initial syllables.
- Result: Each audio chunk contains a complete, coherent sentence or paragraph point.

### 2. State-of-the-Art Transcription (`large-v3-turbo` + Contextual Prompting)
- Uses `large-v3-turbo` (809M parameters) on CPU with `int8` quantization and `beam_size=3`.
- **Dynamic Context Prompting**: Whisper's decoder is primed with the active note topic and rolling recent speech history, resolving homophones and domain jargon.
- Decodes in ~0.8–1.2s on modern 8-core CPUs while preserving 100% of your RTX 3060 Laptop 6 GB VRAM for the local LLM (`llama3.2:3b`).

### 3. Separate Time-Coded Transcript Files
- Session transcripts are stored separately in:
  `Note Rambling/Transcripts/YYYY-MM-DD_Transcript.md`
- Contains the verbatim quote, timestamp, block ID `^t-HHMMSS`, and playable embedded audio player widget `![[...opus]]`.
- Keeps your note file clean and uncluttered.

### 4. Thematic Note Synthesis & Topic Preservation
- The active note (e.g. `Voice Notes.md` or `[Topic].md`) maintains only **`## 💡 Notes`** organized under `### ` thematic headings.
- **Elaboration & Correction**: When you mention new information or correct an earlier statement, VoiceScribe updates that bullet point without deleting other topics or existing sections.
- **Clickable Citations**: Every bullet point cites the separate transcript line and the audio recording:
  `- Deploy migration on Sunday morning to test replicas first. ([[Transcripts/2026-10-25_Transcript#^t-101500|🔗 10:15:00]] · [[VoiceScribe_Archives/2026-10-25/audio.opus|▶ Audio]])`
  - Clicking `🔗 10:15:00` opens the transcript and jumps straight to that quote.
  - Clicking `▶ Audio` plays the audio snippet.

---

## 🗣️ Voice Commands & Topic Switching

Say any of the following commands while speaking to switch active notes:

| Voice Command | Action | Active Obsidian Note |
|---|---|---|
| *"Topic Algorand"* | Switches active note | `Algorand.md` |
| *"Switch to Flutter"* | Switches active note | `Flutter.md` |
| *"Note French toast"* | Switches active note | `French toast.md` |
| *"Switch topic to Ideas"* | Switches active note | `Ideas.md` |

---

## ⚙️ Options & Parameter Customization

You can adjust all of VoiceScribe's core parameters at any time:

| Parameter | Recommended Setting | Description |
|---|---|---|
| **Input Microphone** | *System Default* or *Yeti / Headset* | Selects the active physical or virtual audio capture device. |
| **Whisper Model** | `large-v3-turbo` (CPU int8) | Speech-to-text model running on CPU. `large-v3-turbo` provides state-of-the-art accuracy with fast 4-layer decoding. |
| **Whisper Beam Size** | `3` (range 1–5) | Beam search width for multi-hypothesis sentence decoding. |
| **Context Prompting** | `true` | Dynamically injects topic and recent spoken sentences into Whisper's initial prompt. |
| **Ollama Model** | `llama3.2:3b` (GPU) | Local synthesis LLM running on your RTX 3060. Supports `llama3.1:8b`, `qwen2.5-coder:7b`, `mistral`, etc. |
| **Speech Pause (VAD)** | `2.2s` (range 1.0–3.5s) | Amount of silence needed before finalizing a coherent sentence/thought chunk. |
| **Default Note** | `Voice Notes` | Name of the default synthesis note in your vault (omitting `.md`). |
| **Transcripts Folder** | `Transcripts` | Folder where daily time-coded transcripts (`YYYY-MM-DD_Transcript.md`) are logged. |
| **Ollama Host** | `http://127.0.0.1:11434` | Explicit IPv4 host preventing Windows IPv6 resolution latency. |

