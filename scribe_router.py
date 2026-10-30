#!/usr/bin/env python3
"""
================================================================================
VoiceScribe: Continuous Voice-to-Notes Transcription & Routing for Obsidian
================================================================================
Fully local pipeline (no internet unless model downloads are explicitly allowed):

  Thread 1  AudioIngestionThread   microphone -> energy VAD -> speech chunks
  Thread 2  TranscriptionThread    chunk -> Ogg/Opus archive + faster-whisper text
  Thread 3  RouterSynthesisThread  text -> voice commands, daily transcript,
                                   structured note (local Ollama decides WHERE a
                                   point goes; the file edit itself is done in
                                   Python so existing notes are never lost)

The Obsidian plugin launches this script and reads machine-readable status
lines from stdout that start with "@@VS ".
================================================================================
"""

import argparse
import collections
import dataclasses
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Keep model hubs quiet / offline-friendly.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# UTF-8 + line-buffered output so the Obsidian plugin receives lines immediately.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("VoiceScribe")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _console = logging.StreamHandler(sys.stdout)
    _console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(_console)
    try:
        _file = logging.FileHandler(SCRIPT_DIR / "voicescribe.log", encoding="utf-8", mode="a")
        _file.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(_file)
    except OSError:
        pass

_EMIT_LOCK = threading.Lock()


def emit(event: str, **data) -> None:
    """Machine-readable status line for the Obsidian plugin."""
    payload = json.dumps({"event": event, **data}, ensure_ascii=False)
    with _EMIT_LOCK:
        try:
            sys.stdout.write(f"@@VS {payload}\n")
            sys.stdout.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ScribeConfig:
    # Vault layout
    vault_dir: str = "../Note Rambling"
    notes_subfolder: str = ""
    archive_dir: str = "VoiceScribe_Archives"
    transcripts_folder: str = "Transcripts"
    default_note: str = "Voice Notes"
    start_note: str = ""  # vault-relative note chosen in Obsidian for this session (overrides default_note)
    # Microphone / segmentation
    audio_device: Optional[object] = None
    sample_rate: int = 16000
    energy_threshold: float = 0.0012
    pause_duration_sec: float = 1.7
    min_speech_duration_sec: float = 0.8
    min_voiced_sec: float = 0.3  # chunks with less actual loud audio than this are noise blips
    max_speech_duration_sec: float = 45.0
    normalize_audio: bool = True
    # Transcription
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "int8"
    whisper_threads: int = 4
    whisper_beam_size: int = 3
    whisper_language: str = "en"
    whisper_vad_filter: bool = True
    context_prompt_enabled: bool = True
    allow_model_downloads: bool = False
    whisper_device: str = "auto"  # auto | cpu | cuda  (auto = GPU when it works, else CPU)
    whisper_gpu_compute_type: str = "int8_float16"
    cuda_libs_dir: str = ""  # folder with CUDA cuBLAS/cuDNN DLLs ("" = VoiceScribe's Python environment)
    # Organisation (local LLM)
    llm_enabled: bool = True
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_num_predict: int = 512
    ollama_num_ctx: int = 4096
    ollama_timeout_sec: float = 120.0
    ollama_keep_alive_min: float = 10.0  # Ollama unloads a model after this many idle minutes (0 = right away, -1 = never)
    unload_on_stop: bool = True          # free the Ollama models' GPU/RAM as soon as recording stops
    whisper_idle_unload_min: float = 15.0  # release Whisper after this many minutes without speech (0 = never); reloads on the next thought
    organise_mode: str = "live"      # live: place every thought immediately | batch: collect, then organise together
    batch_interval_min: float = 5.0  # batch mode: organise every N minutes (0 = only when recording stops)
    batch_ollama_model: str = ""     # batch mode: model for the batch pass ("" = same as ollama_model)
    batch_num_ctx: int = 8192
    batch_num_predict: int = 2048
    batch_max_thoughts: int = 25     # thoughts per LLM call (bigger inboxes are organised in parts)
    toc_enabled: bool = True         # batch passes keep an "On this page" list of the note's sections
    summary_enabled: bool = True     # write a session summary into each note when recording stops
    summary_min_thoughts: int = 3
    autolink_enabled: bool = True    # turn mentions of existing notes into [[links]]
    properties_enabled: bool = True  # keep created/updated/voice_sessions/voice_minutes/tags properties
    properties_tag: str = "voicescribe"
    # Storage
    opus_bitrate: str = "12k"
    # Filtering
    min_substantive_words: int = 1

    @classmethod
    def load(cls, config_path: str = "config.json") -> "ScribeConfig":
        path = Path(config_path)
        if not path.is_absolute() and not path.exists():
            path = SCRIPT_DIR / config_path
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                names = {f.name for f in dataclasses.fields(cls)}
                logger.info(f"Loaded configuration from {path}")
                return cls(**{k: v for k, v in data.items() if k in names})
            except Exception as e:
                logger.warning(f"Could not parse {path}: {e}. Using defaults.")
        return cls()


# ---------------------------------------------------------------------------
# Shared context (Whisper prompt priming)
# ---------------------------------------------------------------------------
class SharedContext:
    def __init__(self, default_topic: str = "Voice Notes", max_history: int = 3):
        self.active_topic = default_topic
        self.recent_transcripts: collections.deque = collections.deque(maxlen=max_history)
        self._lock = threading.Lock()

    def update_topic(self, topic: str):
        with self._lock:
            self.active_topic = topic.strip()

    def add_transcript(self, text: str):
        with self._lock:
            if text.strip():
                self.recent_transcripts.append(text.strip())

    def get_initial_prompt(self) -> Optional[str]:
        # NOTE: deliberately not "Topic: X" - Whisper sometimes echoes its prompt,
        # and an echoed "Topic ..." would be mistaken for a voice command.
        with self._lock:
            topic = self.active_topic
            history = list(self.recent_transcripts)
        parts = []
        if topic:
            parts.append(f"Notes about {topic}.")
        if history:
            parts.append(" ".join(history[-2:]))
        if not parts:
            return None
        words = " ".join(parts).split()
        return " ".join(words[-150:])


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class AudioChunk:
    timestamp: datetime
    data: np.ndarray
    duration: float
    rms_energy: float
    seg_id: int = 0
    t0: float = 0.0  # seconds since session start (for the dashboard timeline)
    voiced: float = 0.0  # seconds of audio above the speech threshold


@dataclasses.dataclass
class TranscribedChunk:
    timestamp: datetime
    text: str
    archive_path: Optional[Path]
    duration: float
    seg_id: int = 0


@dataclasses.dataclass
class Focus:
    """Part of a note the user selected to guide where new speech goes."""
    note: Path
    heading: Optional[str]
    bullets: List[str]
    mode: str = "add"  # "add" | "correct"

    def describe(self, vault_root: Path) -> dict:
        try:
            rel = self.note.resolve().relative_to(vault_root).as_posix()
        except ValueError:
            rel = self.note.name
        return {"note": rel, "heading": self.heading, "bullets": list(self.bullets), "mode": self.mode}


@dataclasses.dataclass
class NoteChange:
    ok: bool
    added: List[str] = dataclasses.field(default_factory=list)
    updated: List[str] = dataclasses.field(default_factory=list)
    headings: List[str] = dataclasses.field(default_factory=list)
    focus_bullets: Optional[List[str]] = None  # new focus texts after a correction
    path: Optional[Path] = None
    before: str = ""   # note text before / after this change (for undo)
    after: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
_SEP = r"(?:\s*[:,\-]\s*|\s+)"
_NOTE_WORD = r"(?:note|page|file|document)"
_NAMED = r"(?:(?:named|called|titled|name|for|about)" + _SEP + r")?"
# Leading filler that often precedes a spoken command ("Okay, so switch to ...").
_LEAD_FILLER = re.compile(r"^(?:(?:ok(?:ay)?|alright|all right|right|so|now|and|then|um+|uh+|hey|please|let's|lets)[\s,]+)+", re.I)

# (kind, pattern, strict) - strict patterns reject targets that start with ordinary words
# ("Note that the server is down" is not a command).
COMMAND_PATTERNS = [
    ("create", re.compile(r"^(?:create|make|start|open|begin|add)\s+(?:a\s+|another\s+)?(?:brand\s+)?new\s+" + _NOTE_WORD + _SEP + _NAMED + r"(.+)$", re.I), False),
    ("create", re.compile(r"^(?:switch|change|go|move|jump)\s+(?:over\s+)?to\s+(?:a\s+)?(?:brand\s+)?new\s+" + _NOTE_WORD + _SEP + _NAMED + r"(.+)$", re.I), False),
    ("create", re.compile(r"^new\s+" + _NOTE_WORD + _SEP + _NAMED + r"(.+)$", re.I), False),
    ("switch", re.compile(r"^(?:switch|change)\s+(?:the\s+)?(?:topic|note|notes)\s+to" + _SEP + _NAMED + r"(.+)$", re.I), False),
    ("switch", re.compile(r"^(?:switch|change|go|move|jump)\s+(?:back\s+|over\s+)?to\s+(?:the\s+|my\s+)?(?:(?:" + _NOTE_WORD + r"|topic(?:\s+of)?)" + _SEP + r")?" + _NAMED + r"(.+)$", re.I), False),
    ("switch", re.compile(r"^(?:open|use|go\s+to)\s+(?:the\s+|my\s+)?" + _NOTE_WORD + _SEP + _NAMED + r"(.+)$", re.I), False),
    ("switch", re.compile(r"^(?:new\s+)?topic" + _SEP + r"(.+)$", re.I), True),
    ("switch", re.compile(r"^note" + _SEP + r"(.+)$", re.I), True),
]
# A command target never starts with these (e.g. "Note that ...", "Topic of today ...").
_COMMAND_STOPWORDS = {
    "that", "to", "this", "these", "those", "is", "was", "of", "for", "about", "it", "i",
    "we", "you", "on", "in", "and", "so", "which", "what", "how", "if", "my", "a", "an",
    "here", "there", "down", "taken", "self", "yourself", "be", "are", "the",
}
_SWITCH_REJECT_FIRST = {"talking", "discussing", "doing", "using", "something", "another", "it", "that", "this"}
MAX_COMMAND_WORDS = 6


@dataclasses.dataclass
class NoteCommand:
    kind: str      # "switch" (existing note if there is one) | "create" (always a new note)
    name: str      # note name as spoken, without .md


def _clean_command_target(raw: str) -> str:
    t = raw.strip()
    t = re.sub(r"[\.\?!,:;]+$", "", t).strip()
    t = t.strip("\"'“”‘’`").strip()
    t = re.sub(r"[\.\?!,:;]+$", "", t).strip()
    t = re.sub(r"\s*(?:\.|\s+dot\s+)\s*md$", "", t, flags=re.I).strip()   # "Welcome.md", "welcome dot md"
    return t.strip("\"'“”‘’`").strip()


def parse_note_command(text: str) -> Optional[NoteCommand]:
    """
    Recognise spoken note commands:
      'Topic Algorand' / 'Note French toast'            -> switch
      'Switch to Flutter' / 'Switch to "Welcome.md"'     -> switch (existing note if found)
      'Create a new note named X' / 'New note called X'
      'Switch to a new note called X'                    -> create
    """
    stripped = _LEAD_FILLER.sub("", text.strip().strip("\"'“”")).strip()
    for kind, pattern, strict in COMMAND_PATTERNS:
        m = pattern.match(stripped)
        if not m:
            continue
        target = _clean_command_target(m.group(1))
        words = target.split()
        if not words or len(words) > MAX_COMMAND_WORDS:
            return None
        first = words[0].lower().strip(",.")
        if strict and first in _COMMAND_STOPWORDS:
            return None
        if kind == "switch" and first in _SWITCH_REJECT_FIRST:
            return None
        return NoteCommand(kind, target)
    return None


def parse_voice_command(text: str) -> Optional[str]:
    """Backwards-compatible: just the target name of any note command."""
    cmd = parse_note_command(text)
    return cmd.name if cmd else None


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_command(text: str) -> Tuple[str, Optional[NoteCommand], str]:
    """
    Find a note command at the start or end of an utterance.
    Returns (speech_before, command, speech_after); command is None when there is none.
    """
    whole = parse_note_command(text)
    if whole:
        return "", whole, ""
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    if len(sentences) < 2:
        return text, None, ""
    first = parse_note_command(sentences[0])
    if first:
        return "", first, " ".join(sentences[1:])
    last = parse_note_command(sentences[-1])
    if last:
        return " ".join(sentences[:-1]), last, ""
    return text, None, ""


WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)}, *{f"LPT{i}" for i in range(1, 10)},
}


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", name)
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(". ")
    if not cleaned:
        cleaned = "Voice Notes"
    if cleaned.split(".")[0].upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_note"
    if not cleaned.lower().endswith(".md"):
        cleaned += ".md"
    if len(cleaned) > 100:
        cleaned = cleaned[:96].rstrip(". ") + ".md"
    return cleaned


# Only true hesitation sounds - words like "like"/"actually" carry meaning.
VERBAL_FILLERS_REGEX = re.compile(r"\b(?:u+m+|u+h+|e+r+m+|e+r+|a+h+|h+m+|m+h*m+)\b[,.]?", re.I)


def strip_verbal_fillers(text: str) -> str:
    cleaned = VERBAL_FILLERS_REGEX.sub(" ", text)
    cleaned = re.sub(r"\s+([,.!?])", r"\1", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"^[,\.\s]+", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned


def count_substantive_words(text: str) -> int:
    words = re.findall(r"\b[a-zA-Z0-9_\-']+\b", text)
    return len([w for w in words if len(w) > 1 or w.lower() in ("a", "i")])


_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching!",
    "thank you for watching", "you", "bye", "bye.", "so", "okay", ".", "...",
    "please subscribe", "thank you very much",
}


# Phrases Whisper invents from noise/silence (it was trained on video subtitles).
_OUTRO_RE = re.compile(
    r"see you (?:next time|in the next|later|soon)|thanks? (?:you )?(?:so much )?for watching|"
    r"(?:like and )?subscribe|go ahead and see you|bye[- ]bye|next video|that's all for (?:today|now)",
    re.I)


def is_short_clip_hallucination(text: str, duration: float) -> bool:
    """Outro-style phrases from short clips are almost always invented by Whisper."""
    return duration < 6.0 and bool(_OUTRO_RE.search(text))


def is_whisper_hallucination(text: str) -> bool:
    t = re.sub(r"[^\w\s']", "", text.lower()).strip()
    if not t:
        return True
    if "amara.org" in text.lower() or "subtitles by" in text.lower():
        return True
    return t in {re.sub(r"[^\w\s']", "", h).strip() for h in _HALLUCINATIONS}


def normalize_audio(data: np.ndarray, target_peak: float = 0.9, max_gain: float = 30.0) -> np.ndarray:
    """Peak-normalise quiet microphones so Whisper (and playback) hear clear speech."""
    data = np.asarray(data, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= 1e-6:
        return data
    gain = min(target_peak / peak, max_gain)
    return np.clip(data * gain, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Audio archive: Ogg container + Opus codec (~10-12 kbps speech, ~5 MB/hour).
# .ogg (not .opus) so Obsidian renders an inline audio player.
# ---------------------------------------------------------------------------
def _encode_with_ffmpeg(audio: np.ndarray, sample_rate: int, out: Path, bitrate: str) -> bool:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        "-c:a", "libopus", "-b:a", bitrate, "-vbr", "on", "-application", "voip",
        "-f", "ogg", str(out),
    ]
    try:
        proc = subprocess.run(cmd, input=audio.astype(np.float32).tobytes(), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW, timeout=60)
        if proc.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return True
        logger.debug(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[:300]}")
    except FileNotFoundError:
        logger.debug("ffmpeg not found on PATH")
    except Exception as e:
        logger.debug(f"ffmpeg error: {e}")
    return False


def _encode_with_pyav(audio: np.ndarray, sample_rate: int, out: Path, bitrate: str) -> bool:
    """Fallback: PyAV ships with faster-whisper and bundles libopus."""
    try:
        import av
        br = str(bitrate).lower()
        bit_rate = int(float(br[:-1]) * 1000) if br.endswith("k") else int(float(br))
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        rate = 48000 if sample_rate not in (8000, 12000, 16000, 24000, 48000) else sample_rate
        with av.open(str(out), mode="w", format="ogg") as container:
            stream = container.add_stream("libopus", rate=rate)
            stream.bit_rate = bit_rate
            try:
                stream.layout = "mono"
            except Exception:
                pass
            frame = av.AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = sample_rate
            resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
            for rf in resampler.resample(frame) + resampler.resample(None):
                for packet in stream.encode(rf):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        return out.exists() and out.stat().st_size > 0
    except Exception as e:
        logger.debug(f"PyAV encode failed: {e}")
        return False


def archive_audio_opus(audio_data: np.ndarray, sample_rate: int, output_path: Path, bitrate: str = "12k") -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _encode_with_ffmpeg(audio_data, sample_rate, output_path, bitrate):
        return True
    if _encode_with_pyav(audio_data, sample_rate, output_path, bitrate):
        return True
    logger.error("Could not archive audio: neither ffmpeg nor PyAV could encode Opus.")
    return False


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
    return path


# ---------------------------------------------------------------------------
# File writing helpers
# ---------------------------------------------------------------------------
FILE_WRITE_LOCK = threading.RLock()


def _write_text_safely(path: Path, content: str, retries: int = 5) -> bool:
    """Write via temp file + replace; retry when Obsidian/OneDrive/MEGA hold a lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            fd, tmp = tempfile.mkstemp(prefix=".vs_", suffix=".tmp", dir=str(path.parent))
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.replace(tmp, path)
            except OSError:
                # Some sync clients block replace; fall back to an in-place write.
                os.remove(tmp)
                with open(path, "w", encoding="utf-8", newline="\n") as f:
                    f.write(content)
            return True
        except OSError as e:
            if attempt == retries - 1:
                logger.error(f"Failed to write {path}: {e}")
                return False
            time.sleep(0.15 * (2 ** attempt))
    return False


def _append_text_safely(path: Path, text: str, header: str = "", retries: int = 5) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            is_new = not path.exists() or path.stat().st_size == 0
            with open(path, "a", encoding="utf-8", newline="\n") as f:
                if is_new and header:
                    f.write(header)
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            return True
        except OSError as e:
            if attempt == retries - 1:
                logger.error(f"Failed to append to {path}: {e}")
                return False
            time.sleep(0.15 * (2 ** attempt))
    return False


def append_session_transcript(
    vault_dir: Path,
    transcripts_folder: str,
    topic_title: str,
    chunk_id: str,
    timestamp_str: str,
    text: str,
    audio_rel_path: str,
    date_str: Optional[str] = None,
) -> bool:
    """
    Daily verbatim transcript: <vault>/<Transcripts>/YYYY-MM-DD_Transcript.md
    The block ID sits on its own line after the quote so Obsidian can link to it.
    """
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    trans_file = Path(vault_dir) / transcripts_folder / f"{date_str}_Transcript.md"
    lines = [f"#### {timestamp_str} · {topic_title}", f"> {text}"]
    if audio_rel_path:
        lines.append(">")
        lines.append(f"> ![[{audio_rel_path}]]")
    entry = "\n".join(lines) + f"\n\n^{chunk_id}\n\n"
    with FILE_WRITE_LOCK:
        return _append_text_safely(trans_file, entry, header=f"# Session Transcript: {date_str}\n\n")


# ---------------------------------------------------------------------------
# Structured note model
# ---------------------------------------------------------------------------
NOTES_HEADER = "## 💡 Notes"
_PLACEHOLDER_RE = re.compile(r"^\*\((?:ready for|listening for)[^)]*\)\*$", re.I)
_CITATION_SPLIT_RE = re.compile(r"\s+(?=\(\[\[)")


@dataclasses.dataclass
class NoteBullet:
    text: str
    cites: List[str]
    extra: List[str] = dataclasses.field(default_factory=list)  # indented child lines

    def render(self) -> str:
        line = f"- {self.text}"
        if self.cites:
            line += " " + " ".join(self.cites)
        return "\n".join([line] + self.extra)


@dataclasses.dataclass
class NoteSection:
    heading: str
    items: List[object]  # NoteBullet or raw str lines


@dataclasses.dataclass
class StructuredNote:
    prefix: str
    preamble: List[str]
    sections: List[NoteSection]
    suffix: str

    # -- parsing ---------------------------------------------------------
    @classmethod
    def parse(cls, content: str, topic_title: str) -> "StructuredNote":
        header_re = re.compile(r"^##\s+💡\s*Notes\s*$", re.M)
        m = header_re.search(content)
        if m:
            prefix = content[:m.end()]
            rest = content[m.end():]
            nxt = re.search(r"^##\s+(?!#)", rest, re.M)
            body, suffix = (rest[:nxt.start()], rest[nxt.start():]) if nxt else (rest, "")
        else:
            existing = content.strip()
            if existing:
                prefix = f"{existing}\n\n{NOTES_HEADER}"
            else:
                prefix = f"# {topic_title}\n\n{NOTES_HEADER}"
            body, suffix = "", ""

        preamble: List[str] = []
        sections: List[NoteSection] = []
        current: Optional[NoteSection] = None
        for raw in body.splitlines():
            line = raw.rstrip()
            if line.startswith("### "):
                current = NoteSection(line[4:].strip(), [])
                sections.append(current)
                continue
            if not line.strip():
                continue
            if _PLACEHOLDER_RE.match(line.strip()):
                continue
            if line.startswith("- ") or line.startswith("* "):
                text = line[2:].strip()
                parts = _CITATION_SPLIT_RE.split(text, maxsplit=1)
                main = parts[0].strip()
                cites = re.findall(r"\([^()]*\[\[[^()]*\)", parts[1]) if len(parts) > 1 else []
                item: object = NoteBullet(main, cites)
            elif raw[:1] in (" ", "\t") and current and current.items and isinstance(current.items[-1], NoteBullet):
                current.items[-1].extra.append(line)
                continue
            else:
                item = line
            if current is None:
                preamble.append(item if isinstance(item, str) else item.render())
            else:
                current.items.append(item)
        return cls(prefix, preamble, sections, suffix)

    # -- helpers ---------------------------------------------------------
    def bullets(self) -> List[Tuple[NoteSection, NoteBullet]]:
        return [(s, it) for s in self.sections for it in s.items if isinstance(it, NoteBullet)]

    def find_section(self, heading: str) -> Optional[NoteSection]:
        key = heading.strip().lower()
        for s in self.sections:
            if s.heading.strip().lower() == key:
                return s
        return None

    def add_bullet(self, heading: str, text: str, cite: str):
        section = self.find_section(heading)
        if section is None:
            section = NoteSection(heading, [])
            self.sections.append(section)
        section.items.append(NoteBullet(text, [cite]))

    def locate(self, texts: List[str]) -> List[Tuple[NoteSection, int]]:
        """Positions of the bullets whose text is in `texts` (document order)."""
        wanted = {t.strip() for t in texts}
        found = []
        for s in self.sections:
            for i, it in enumerate(s.items):
                if isinstance(it, NoteBullet) and it.text.strip() in wanted:
                    found.append((s, i))
        return found

    def insert_after(self, section: NoteSection, index: int, text: str, cite: str) -> None:
        section.items.insert(index + 1, NoteBullet(text, [cite]))

    def replace_bullets(self, positions: List[Tuple[NoteSection, int]], new_texts: List[str], cite: str) -> List[str]:
        """Correct-mode edit: rewrite the focused bullets, keeping every old citation."""
        old = [p[0].items[p[1]] for p in positions]
        section, first = positions[0]
        old_cites: List[str] = []
        for b in old:
            for c in b.cites:
                if c not in old_cites:
                    old_cites.append(c)
        extras = [e for b in old for e in b.extra]
        new_bullets = []
        for i, text in enumerate(new_texts):
            if i < len(old):
                cites = list(old[i].cites)
            else:
                cites = []
            if cite not in cites:
                cites.append(cite)
            new_bullets.append(NoteBullet(text, cites))
        # Citations of bullets that were merged away stay on the last new bullet.
        kept = {c for b in new_bullets for c in b.cites}
        new_bullets[-1].cites = [c for c in old_cites if c not in kept] + new_bullets[-1].cites
        new_bullets[-1].extra = extras
        # Remove old bullets (from the end so indices stay valid), then insert.
        for s, i in sorted(positions, key=lambda p: (id(p[0]), p[1]), reverse=True):
            del s.items[i]
        for j, b in enumerate(new_bullets):
            section.items.insert(first + j, b)
        return [b.text for b in new_bullets]

    def update_bullet(self, old_text: str, new_text: str, cite: str) -> bool:
        for _, b in self.bullets():
            if b.text == old_text:
                b.text = new_text
                if cite not in b.cites:
                    b.cites.append(cite)
                return True
        return False

    def outline(self, max_bullets: int = 60, max_chars: int = 160) -> Tuple[str, List[str]]:
        """Numbered outline for the LLM (without citations). Returns (text, bullet texts)."""
        all_b = self.bullets()
        texts = [b.text for _, b in all_b]
        first_shown = max(0, len(all_b) - max_bullets)
        lines, idx = [], 0
        for s in self.sections:
            lines.append(f"### {s.heading}")
            for it in s.items:
                if isinstance(it, NoteBullet):
                    idx += 1
                    if idx - 1 >= first_shown:
                        t = it.text if len(it.text) <= max_chars else it.text[:max_chars] + "..."
                        lines.append(f"[{idx}] {t}")
        return ("\n".join(lines) if lines else "(empty - no headings yet)"), texts

    def render(self) -> str:
        out = [self.prefix.rstrip(), ""]
        if self.preamble:
            out.extend(self.preamble)
            out.append("")
        for s in self.sections:
            out.append(f"### {s.heading}")
            for it in s.items:
                out.append(it.render() if isinstance(it, NoteBullet) else it)
            out.append("")
        body = "\n".join(out).rstrip() + "\n"
        if self.suffix.strip():
            body += "\n" + self.suffix.lstrip("\n")
        return body


# ---------------------------------------------------------------------------
# LLM organisation
# ---------------------------------------------------------------------------
LLM_SYSTEM_PROMPT = (
    "You organise a person's spoken thoughts into a tidy Markdown note. "
    "You never invent facts; you only restate what was said, clearly and concisely. "
    "You reply with JSON only."
)


def build_llm_prompt(outline: str, statement: str, topic_title: str, focus_hint: str = "") -> str:
    return (
        f"NOTE TITLE: {topic_title}\n\n"
        f"CURRENT NOTE OUTLINE (### headings with numbered bullet points):\n{outline}\n\n"
        f"{focus_hint}"
        f"NEW SPOKEN STATEMENT (speech-to-text, may contain small recognition errors):\n\"{statement}\"\n\n"
        "Turn the new statement into note bullet(s) and decide where each belongs.\n"
        "Reply with JSON exactly in this shape:\n"
        '{"items": [{"action": "add", "heading": "Short Theme", "bullet": "Concise point", "target": null}]}\n\n'
        "Rules:\n"
        "- \"bullet\": rewrite the statement as a clear, concise note in the speaker's perspective. Keep every fact, name, number and idea. Drop filler and repetition. Do NOT add information.\n"
        "- If the statement covers several distinct ideas, return one item per idea (max 5).\n"
        "- \"heading\": reuse an existing heading exactly when the idea fits it; otherwise create a new 1-4 word theme heading.\n"
        "- Use \"action\": \"update\" with \"target\": <bullet number> ONLY when the statement corrects, changes or directly continues that specific bullet; the bullet must then contain the merged, corrected point.\n"
        "- Otherwise use \"action\": \"add\" and \"target\": null.\n"
    )


def _extract_json(raw: str) -> Optional[dict]:
    # Some model templates put their reasoning inline as <think>…</think>; never parse that.
    raw = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S | re.I).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def validate_llm_items(data: Optional[dict], bullet_texts: List[str]) -> List[dict]:
    """Normalise model output into safe edit operations."""
    if not isinstance(data, dict):
        return []
    items = data.get("items")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        if "bullet" in data:
            items = [data]
        else:
            return []
    ops: List[dict] = []
    for it in items[:5]:
        if not isinstance(it, dict):
            continue
        bullet = _clean_bullet_text(it.get("bullet"))
        if not bullet:
            continue
        heading = str(it.get("heading") or "General").strip().lstrip("#").strip()
        heading = re.sub(r"\s+", " ", heading)[:60] or "General"
        action = str(it.get("action") or "add").lower().strip()
        target = it.get("target")
        old_text = None
        if action == "update":
            try:
                t = int(target)
                if 1 <= t <= len(bullet_texts):
                    old_text = bullet_texts[t - 1]
            except (TypeError, ValueError):
                old_text = None
        ops.append({"heading": heading, "bullet": bullet, "old_text": old_text})
    return ops


def build_focus_hint(focus: Optional["Focus"], outline_texts: List[str]) -> str:
    if not focus:
        return ""
    nums = [str(i + 1) for i, t in enumerate(outline_texts) if t in focus.bullets]
    where = f'heading "{focus.heading}"' if focus.heading else "the selected part of the note"
    if nums:
        where += f" (bullet{'s' if len(nums) > 1 else ''} [{', '.join(nums)}])"
    return (
        f"FOCUS: the speaker selected {where} and is now adding details to it. "
        f"Add the statement there (use that exact heading) unless it is clearly about a different subject. "
        f"If it extends one focused bullet, you may use \"update\" with that bullet's number.\n\n"
    )


def build_correction_prompt(heading: Optional[str], bullets: List[str], statement: str, topic_title: str) -> str:
    listed = "\n".join(f"- {b}" for b in bullets)
    return (
        f"NOTE TITLE: {topic_title}\n"
        f"SECTION: {heading or '(none)'}\n\n"
        f"The speaker selected these bullet point(s) to correct:\n{listed}\n\n"
        f"What they just said (speech-to-text, may contain small recognition errors):\n\"{statement}\"\n\n"
        "Rewrite the selected bullet point(s) so they reflect what was just said. "
        "Keep every fact that was not corrected; change only what the speaker corrected or added; do not invent anything. "
        "Return the same number of bullets unless the speaker clearly merged or split points.\n"
        'Reply with JSON exactly in this shape: {"bullets": ["first rewritten point", "..."]}\n'
    )


def thinking_setting(model: str):
    """Hidden 'thinking' makes reasoning models (Qwen 3/3.5, DeepSeek-R1, gpt-oss…) far slower and can
    break JSON-only answers. Turn it off; gpt-oss cannot switch it off, so ask for the lowest level."""
    return "low" if (model or "").lower().startswith("gpt-oss") else False


_NO_THINK_SUPPORT: set = set()   # models/servers that rejected the `think` option


# ---- Ollama model memory: keep-alive, "is it loaded?" check, unloading -------------
def keep_alive_value(config: ScribeConfig):
    """Ollama keep_alive: how long a model stays in GPU/RAM after its last answer."""
    try:
        minutes = float(config.ollama_keep_alive_min)
    except (TypeError, ValueError):
        minutes = 10.0
    if minutes < 0:
        return -1  # never unload
    return f"{int(round(minutes * 60))}s"


def _ollama_json(host: str, path: str, payload: Optional[dict] = None, timeout: float = 3.0) -> Optional[dict]:
    """Small local-only HTTP helper for the Ollama API (None when the server is unreachable)."""
    import urllib.request
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(f"{host.rstrip('/')}{path}", data=data,
                                 headers={"User-Agent": "VoiceScribe", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            body = res.read().decode("utf-8", "replace").strip()
    except Exception:
        return None
    if not body:
        return {}
    try:
        return json.loads(body.splitlines()[-1])
    except ValueError:
        return {}


def _model_key(name: str) -> str:
    name = (name or "").strip().lower()
    return name if ":" in name else name + ":latest"


def ollama_model_names(host: str, path: str) -> Optional[List[str]]:
    """Models listed by /api/ps (loaded) or /api/tags (installed); None when Ollama is unreachable."""
    data = _ollama_json(host, path)
    if data is None:
        return None
    out = []
    for m in data.get("models") or []:
        if isinstance(m, dict):
            out.append(m.get("name") or m.get("model") or "")
    return [n for n in out if n]


MODEL_CHECK_ENABLED = True  # tests switch the HTTP check off


def ensure_model_loaded(config: ScribeConfig, model: str, http=None) -> bool:
    """
    Make sure `model` is in memory before it is asked to organise anything. If Ollama unloaded it
    (keep-alive ran out), load it now and tell the plugin, so a slow first answer is explained.
    Returns False only when Ollama answers but the model is not installed.
    """
    if not MODEL_CHECK_ENABLED and http is None:
        return True
    names = http or ollama_model_names
    host = config.ollama_host
    loaded = names(host, "/api/ps")
    if loaded is None:
        return True  # server not answering: the chat call reports the real error
    key = _model_key(model)
    if any(_model_key(n) == key for n in loaded):
        return True
    installed = names(host, "/api/tags")
    if installed is not None and not any(_model_key(n) == key for n in installed):
        logger.warning(f"Ollama model '{model}' is not installed.")
        emit("llm_missing", model=model)
        return False
    logger.info(f"Ollama model '{model}' is not in memory - loading it...")
    emit("llm_loading", model=model)
    started = time.time()
    if http is None:
        _ollama_json(host, "/api/generate", {"model": model, "keep_alive": keep_alive_value(config)},
                     timeout=max(float(config.ollama_timeout_sec), 300.0))
    took = round(time.time() - started, 1)
    logger.info(f"Ollama model '{model}' loaded in {took}s.")
    emit("llm_ready", model=model, seconds=took)
    return True


def session_models(config: ScribeConfig) -> List[str]:
    models = [config.ollama_model]
    batch = (config.batch_ollama_model or "").strip()
    if batch and batch not in models:
        models.append(batch)
    return [m for m in models if m]


def unload_models(config: ScribeConfig, models: Optional[List[str]] = None) -> List[str]:
    """Ask Ollama to drop the models from GPU/RAM now (keep_alive 0). Returns the ones unloaded."""
    loaded = ollama_model_names(config.ollama_host, "/api/ps")
    if not loaded:
        return []
    want = {_model_key(m) for m in (models or session_models(config))}
    done = []
    for name in loaded:
        if _model_key(name) in want:
            if _ollama_json(config.ollama_host, "/api/generate", {"model": name, "keep_alive": 0}, timeout=30.0) is not None:
                done.append(name)
    if done:
        logger.info(f"Unloaded Ollama model(s): {', '.join(done)}")
        emit("llm_unloaded", models=done)
    return done


def _chat_json(config: ScribeConfig, user_prompt: str, model: Optional[str] = None,
               num_ctx: Optional[int] = None, num_predict: Optional[int] = None,
               timeout: Optional[float] = None) -> Optional[dict]:
    import ollama
    model = model or config.ollama_model
    client = ollama.Client(host=config.ollama_host, timeout=timeout or config.ollama_timeout_sec)
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={
            "num_predict": num_predict or config.ollama_num_predict,
            "num_ctx": num_ctx or config.ollama_num_ctx,
            "temperature": 0.1,
            "top_p": 0.9,
        },
        keep_alive=keep_alive_value(config),
    )
    if not ensure_model_loaded(config, model):
        raise RuntimeError(f"Ollama model '{model}' is not installed (run: ollama pull {model})")
    if model not in _NO_THINK_SUPPORT:
        kwargs["think"] = thinking_setting(model)
    try:
        response = client.chat(**kwargs)
    except TypeError as e:
        # older ollama-python without the `think` argument
        if "think" not in kwargs or "think" not in str(e):
            raise
        _NO_THINK_SUPPORT.add(model)
        kwargs.pop("think")
        response = client.chat(**kwargs)
    except Exception as e:
        # older Ollama servers / models that reject the option: retry without it
        if "think" not in kwargs or "think" not in str(e).lower():
            raise
        logger.info(f"Model {model} does not accept the thinking option; continuing without it.")
        _NO_THINK_SUPPORT.add(model)
        kwargs.pop("think")
        response = client.chat(**kwargs)
    try:
        content = response["message"]["content"]
    except (KeyError, TypeError):
        content = getattr(getattr(response, "message", None), "content", "") or ""
    return _extract_json(content)


def ask_llm_for_placement(config: ScribeConfig, outline: str, statement: str, topic_title: str,
                          focus_hint: str = "") -> Optional[dict]:
    return _chat_json(config, build_llm_prompt(outline, statement, topic_title, focus_hint))


def ask_llm_for_correction(config: ScribeConfig, heading: Optional[str], bullets: List[str],
                           statement: str, topic_title: str) -> Optional[dict]:
    return _chat_json(config, build_correction_prompt(heading, bullets, statement, topic_title))


def _clean_bullet_text(value) -> str:
    text = str(value or "").replace("\n", " ").strip()
    for _ in range(3):  # "- [3] text", "[3] - text", "3. text" ...
        text = re.sub(r"^(?:[\-\*•]\s*|\[\d+\]\s*|\d+[.)]\s+)", "", text).strip()
    text = re.sub(r"\s*\(\[\[.*$", "", text)  # never let the model write citations
    return re.sub(r"\s+", " ", text).strip()


_STOPWORDS = set("""a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each else ever few for from further get got had has
have having he her here hers him his how i if in into is it its itself just like me more most my no nor not now of off
on once only or other our out over own really same she should so some such than that the their them then there these
they this those through to too under until up very was we were what when where which while who whom why will with would
you your yours yeah okay ok um uh oh well thing things actually basically gonna wanna going want""".split())


def _content_stems(text: Optional[str]) -> set:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w[:5] for w in words if len(w) > 2 and w not in _STOPWORDS}


def is_grounded(bullet: str, statement: str, old: Optional[str] = None) -> bool:
    """A bullet must restate the new statement (plus, for updates, the old bullet) - not echo other notes."""
    b = _content_stems(bullet)
    st = _content_stems(statement)
    if not b or not st:
        return False
    if not (b & st):
        return False
    novel = b - st - (_content_stems(old) if old else set())
    return len(novel) <= max(2, len(b) // 2)


def _similar(a: str, b: str) -> float:
    import difflib
    na = re.sub(r"[^a-z0-9 ]", "", a.lower())
    nb = re.sub(r"[^a-z0-9 ]", "", b.lower())
    return difflib.SequenceMatcher(None, na, nb).ratio()


def is_same_point(a: str, b: str) -> bool:
    """Same point = same content words and same numbers (30 vs 60 seconds are different facts)."""
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        return False
    sa, sb = _content_stems(a), _content_stems(b)
    return (bool(sa) and sa == sb) or _similar(a, b) >= 0.97


def filter_llm_ops(ops: List[dict], statement: str, existing: List[str]) -> List[dict]:
    """Drop echoed/ungrounded bullets; turn near-duplicates into 'add a citation to the existing bullet'."""
    out: List[dict] = []
    seen: List[str] = []
    for op in ops:
        text = op["bullet"]
        if any(_similar(text, t) >= 0.9 for t in seen):
            continue  # the model listed the same point twice
        if not is_grounded(text, statement, op.get("old_text")):
            logger.info(f"Ignored model bullet not based on what was said: '{text[:80]}'")
            continue
        if not op.get("old_text"):
            dup = next((t for t in existing if is_same_point(text, t)), None)
            if dup is not None:
                op = {"heading": op["heading"], "bullet": dup, "old_text": dup}  # repeat: cite existing point
        seen.append(text)
        out.append(op)
    return out


def ensure_note_exists(note_filepath: Path, topic_title: str) -> bool:
    """Create an empty structured note so Obsidian can open it straight away."""
    if note_filepath.exists():
        return False
    with FILE_WRITE_LOCK:
        if note_filepath.exists():
            return False
        return _write_text_safely(note_filepath, StructuredNote.parse("", topic_title).render())


def integrate_note_thematically(
    note_filepath: Path,
    topic_title: str,
    new_statement: str,
    citation_tag: str,
    config: ScribeConfig,
    llm_fn=None,
    focus: Optional[Focus] = None,
    correct_fn=None,
    linker=None,
    postprocess=None,
) -> NoteChange:
    """
    Adds the statement to the note's "## 💡 Notes" section under ### theme headings.
    The LLM only decides placement/wording; edits are applied in Python, so no existing
    content can be truncated or deleted by the model.

    With a focus (the user selected part of the note):
      mode "correct": only the focused bullets are rewritten;
      mode "add":     the statement is steered into the focused heading/bullet.
    """
    def read() -> str:
        try:
            return note_filepath.read_text(encoding="utf-8") if note_filepath.exists() else ""
        except Exception:
            return ""

    original = read()
    note = StructuredNote.parse(original, topic_title)
    outline, texts = note.outline()
    link = linker or (lambda t: t)

    def commit(note_obj: StructuredNote, base: str, change: NoteChange) -> NoteChange:
        out = note_obj.render()
        if postprocess:
            try:
                out = postprocess(out)
            except Exception as e:
                logger.warning(f"Could not update note properties: {e}")
        change.ok = _write_text_safely(note_filepath, out)
        change.path, change.before, change.after = note_filepath, base, out
        return change

    # ---- correct mode -------------------------------------------------------
    if focus and focus.mode == "correct":
        targets = [b for b in focus.bullets if b in texts]
        if not targets and focus.heading:
            section = note.find_section(focus.heading)
            if section:
                targets = [it.text for it in section.items if isinstance(it, NoteBullet)]
        if targets:
            new_texts: List[str] = []
            if config.llm_enabled:
                try:
                    fn = correct_fn or ask_llm_for_correction
                    data = fn(config, focus.heading, targets, new_statement, topic_title)
                    raw = data.get("bullets") if isinstance(data, dict) else None
                    if isinstance(raw, str):
                        raw = [raw]
                    if isinstance(raw, list):
                        new_texts = [t for t in (_clean_bullet_text(x) for x in raw[:10]) if t]
                        old_joined = " ".join(targets)
                        if new_texts and not all(is_grounded(t, new_statement + " " + old_joined) for t in new_texts):
                            logger.info("Ignored a correction that was not based on the note and what was said.")
                            new_texts = []
                except Exception as e:
                    logger.warning(f"Ollama correction failed ({e}).")
                    emit("warning", message=f"Ollama unavailable: {e}")
            with FILE_WRITE_LOCK:
                current = read()
                if current != original:
                    note = StructuredNote.parse(current, topic_title)
                positions = note.locate(targets)
                if positions and new_texts:
                    result = note.replace_bullets(positions, [link(t) for t in new_texts], citation_tag)
                    return commit(note, current, NoteChange(False, updated=result, headings=[positions[0][0].heading],
                                                            focus_bullets=result))
                if positions:
                    # Model unavailable: keep the correction next to what it corrects.
                    section, idx = positions[-1]
                    text = link(f"Correction: {new_statement}")
                    note.insert_after(section, idx, text, citation_tag)
                    return commit(note, current, NoteChange(False, added=[text], headings=[section.heading]))
            # focused bullets vanished (user edited them) -> fall through to add mode
        emit("warning", message="Focused text is no longer in the note; adding the thought normally.")

    # ---- add mode / no focus ------------------------------------------------
    hint = build_focus_hint(focus, texts) if (focus and focus.mode == "add") else ""
    ops: List[dict] = []
    if config.llm_enabled:
        try:
            fn = llm_fn or ask_llm_for_placement
            data = fn(config, outline, new_statement, topic_title, hint) if hint else fn(config, outline, new_statement, topic_title)
            ops = filter_llm_ops(validate_llm_items(data, texts), new_statement, texts)
            if not ops:
                logger.warning("LLM returned no usable placement; adding the statement as-is.")
        except Exception as e:
            logger.warning(f"Ollama organisation failed ({e}); adding the statement as-is.")
            emit("warning", message=f"Ollama unavailable: {e}")
    if not ops:
        if focus and focus.mode == "add" and focus.heading:
            heading = focus.heading
        else:
            heading = note.sections[-1].heading if note.sections else "General"
        ops = [{"heading": heading, "bullet": new_statement, "old_text": None}]

    change = NoteChange(False)
    inserted_here: List[str] = []
    with FILE_WRITE_LOCK:
        current = read()
        if current != original:  # user edited the note while the LLM was thinking
            note = StructuredNote.parse(current, topic_title)
        for op in ops:
            if op["bullet"] != op.get("old_text"):
                op = dict(op, bullet=link(op["bullet"]))
            if op["old_text"] and note.update_bullet(op["old_text"], op["bullet"], citation_tag):
                change.updated.append(op["bullet"])
                change.headings.append(op["heading"])
                continue
            anchor = None
            if focus and focus.mode == "add" and focus.heading and op["heading"].lower() == focus.heading.lower():
                positions = note.locate(focus.bullets + inserted_here)
                if positions:
                    anchor = positions[-1]
            if anchor:
                note.insert_after(anchor[0], anchor[1], op["bullet"], citation_tag)
                inserted_here.append(op["bullet"])
            else:
                note.add_bullet(op["heading"], op["bullet"], citation_tag)
            change.added.append(op["bullet"])
            change.headings.append(op["heading"])
        commit(note, current, change)
    return change


# ---------------------------------------------------------------------------
# Batch organisation: thoughts wait in an inbox section, then are organised together
# ---------------------------------------------------------------------------
INBOX_HEADING = "📥 Inbox (to organise)"


def append_to_inbox(note_filepath: Path, topic_title: str, statement: str, citation_tag: str,
                    linker=None, postprocess=None) -> NoteChange:
    """Batch mode, step 1: keep the thought (cleaned, cited) in the note straight away."""
    text = (linker or (lambda t: t))(statement)
    with FILE_WRITE_LOCK:
        try:
            current = note_filepath.read_text(encoding="utf-8") if note_filepath.exists() else ""
        except OSError:
            current = ""
        note = StructuredNote.parse(current, topic_title)
        note.add_bullet(INBOX_HEADING, text, citation_tag)
        out = note.render()
        if postprocess:
            try:
                out = postprocess(out)
            except Exception as e:
                logger.warning(f"Could not update note properties: {e}")
        ok = _write_text_safely(note_filepath, out)
    return NoteChange(ok, added=[text], headings=[INBOX_HEADING], path=note_filepath, before=current, after=out)


def inbox_bullets(note: StructuredNote) -> List[NoteBullet]:
    section = note.find_section(INBOX_HEADING)
    return [it for it in section.items if isinstance(it, NoteBullet)] if section else []


def build_batch_prompt(topic_title: str, outline: str, thoughts: List[str]) -> str:
    listed = "\n".join(f"(T{i + 1}) {t}" for i, t in enumerate(thoughts))
    return (
        f"NOTE TITLE: {topic_title}\n\n"
        f"CURRENT NOTE OUTLINE (### headings with numbered bullet points):\n{outline}\n\n"
        f"NEW THOUGHTS, spoken in this order (speech-to-text, may contain small recognition errors):\n{listed}\n\n"
        "Organise the new thoughts into the note. Reply with JSON exactly in this shape:\n"
        '{"items": [{"action": "add", "heading": "Short Theme", "bullet": "Concise point", "sources": ["T1", "T3"], "target": null}]}\n\n'
        "Rules:\n"
        "- Merge thoughts that say the same thing or build on each other into ONE bullet and list all their sources.\n"
        "- When a later thought corrects an earlier one (\"actually…\", \"no, …\"), keep only the corrected version.\n"
        "- Use \"action\": \"update\" with \"target\": <bullet number> when thoughts extend or correct an existing bullet; "
        "the bullet must then be the merged, corrected point.\n"
        "- Reuse an existing heading exactly when the theme fits; otherwise create a short 1-4 word theme heading. Group related points.\n"
        "- Every thought with real content must be used by at least one item. Skip only pure filler.\n"
        "- Keep every fact, name and number. Never add information that was not said.\n"
    )


def ask_llm_for_batch(config: ScribeConfig, topic_title: str, outline: str, thoughts: List[str]) -> Optional[dict]:
    timeout = max(config.ollama_timeout_sec, 60.0 + 20.0 * len(thoughts))
    return _chat_json(config, build_batch_prompt(topic_title, outline, thoughts),
                      model=(config.batch_ollama_model or "").strip() or None,
                      num_ctx=config.batch_num_ctx, num_predict=config.batch_num_predict, timeout=timeout)


def plan_batch(data: Optional[dict], thoughts: List[str], existing: List[str]) -> List[dict]:
    """Validate the model's plan: ops with their source thought indices (0-based)."""
    if not isinstance(data, dict):
        return []
    items = data.get("items")
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    ops: List[dict] = []
    for it in items[:60]:
        if not isinstance(it, dict):
            continue
        bullet = _clean_bullet_text(it.get("bullet"))
        if not bullet:
            continue
        srcs: List[int] = []
        for sref in it.get("sources") or []:
            m = re.search(r"\d+", str(sref))
            if m:
                k = int(m.group(0)) - 1
                if 0 <= k < len(thoughts) and k not in srcs:
                    srcs.append(k)
        if not srcs:
            continue
        heading = re.sub(r"\s+", " ", str(it.get("heading") or "General").strip().lstrip("#").strip())[:60] or "General"
        if heading == INBOX_HEADING:
            heading = "General"
        old_text = None
        if str(it.get("action") or "").lower() == "update":
            try:
                t = int(it.get("target"))
                if 1 <= t <= len(existing):
                    old_text = existing[t - 1]
            except (TypeError, ValueError):
                old_text = None
        basis = " ".join(thoughts[k] for k in srcs)
        if not is_grounded(bullet, basis, old_text):
            logger.info(f"Ignored batch bullet not based on its sources: '{bullet[:80]}'")
            continue
        if not old_text:
            dup = next((t for t in existing if is_same_point(bullet, t)), None)
            if dup is not None:
                bullet, old_text = dup, dup
        ops.append({"heading": heading, "bullet": bullet, "old_text": old_text, "sources": srcs})
    return ops


def organise_inbox(note_filepath: Path, topic_title: str, config: ScribeConfig, batch_fn=None,
                   linker=None, postprocess=None) -> NoteChange:
    """
    Batch mode, step 2: organise every inbox thought of a note in one go (in parts of
    batch_max_thoughts). Each resulting bullet keeps the citations of all thoughts it came from;
    thoughts the model left out are kept as they are, so nothing is lost.
    """
    link = linker or (lambda t: t)
    try:
        original = note_filepath.read_text(encoding="utf-8") if note_filepath.exists() else ""
    except OSError:
        return NoteChange(False)
    pending = inbox_bullets(StructuredNote.parse(original, topic_title))
    if not pending:
        return NoteChange(True, path=note_filepath, before=original, after=original)
    size = max(1, int(config.batch_max_thoughts))
    plans: List[Tuple[List[NoteBullet], List[dict]]] = []
    for start in range(0, len(pending), size):
        part = pending[start:start + size]
        note = StructuredNote.parse(original, topic_title)
        # outline without the inbox section
        inbox = note.find_section(INBOX_HEADING)
        if inbox:
            note.sections.remove(inbox)
        outline, existing = note.outline()
        ops: List[dict] = []
        if config.llm_enabled:
            try:
                data = (batch_fn or ask_llm_for_batch)(config, topic_title, outline, [b.text for b in part])
                ops = plan_batch(data, [b.text for b in part], existing)
            except Exception as e:
                logger.warning(f"Batch organisation failed ({e}).")
                emit("warning", message=f"Ollama could not organise the inbox: {e}")
                return NoteChange(False)
        plans.append((part, ops))

    change = NoteChange(False)
    with FILE_WRITE_LOCK:
        try:
            current = note_filepath.read_text(encoding="utf-8") if note_filepath.exists() else ""
        except OSError:
            return NoteChange(False)
        note = StructuredNote.parse(current, topic_title)
        inbox = note.find_section(INBOX_HEADING)
        for part, ops in plans:
            used = set()
            for op in ops:
                cites: List[str] = []
                for k in op["sources"]:
                    for c in part[k].cites:
                        if c not in cites:
                            cites.append(c)
                used.update(op["sources"])
                text = op["bullet"] if op["bullet"] == op.get("old_text") else link(op["bullet"])
                done = False
                if op["old_text"]:
                    for _, b in note.bullets():
                        if b.text == op["old_text"]:
                            b.text = text
                            b.cites += [c for c in cites if c not in b.cites]
                            change.updated.append(text)
                            done = True
                            break
                if not done:
                    section = note.find_section(op["heading"])
                    if section is None:
                        section = NoteSection(op["heading"], [])
                        idx = note.sections.index(inbox) if inbox in note.sections else len(note.sections)
                        note.sections.insert(idx, section)
                    section.items.append(NoteBullet(text, cites))
                    change.added.append(text)
                change.headings.append(op["heading"])
            leftovers = [b for k, b in enumerate(part) if k not in used]
            if leftovers and ops:
                logger.info(f"{len(leftovers)} inbox thought(s) were not used by the model; keeping them as they are.")
            for b in leftovers:
                if count_substantive_words(b.text) < 2 or is_whisper_hallucination(b.text):
                    continue  # pure filler: it stays in the transcript
                heading = "Other thoughts" if ops else (note.sections[0].heading if note.sections and note.sections[0] is not inbox else "General")
                section = note.find_section(heading)
                if section is None:
                    section = NoteSection(heading, [])
                    idx = note.sections.index(inbox) if inbox in note.sections else len(note.sections)
                    note.sections.insert(idx, section)
                section.items.append(NoteBullet(b.text, list(b.cites), list(b.extra)))
                change.added.append(b.text)
            # remove exactly the processed thoughts from the inbox (match text + sources)
            if inbox:
                keys = {(b.text, tuple(b.cites)) for b in part}
                inbox.items = [it for it in inbox.items
                               if not (isinstance(it, NoteBullet) and (it.text, tuple(it.cites)) in keys)]
        if inbox and not any(isinstance(it, NoteBullet) for it in inbox.items):
            note.sections.remove(inbox)
        out = note.render()
        if postprocess:
            try:
                out = postprocess(out)
            except Exception as e:
                logger.warning(f"Could not update note properties: {e}")
        change.ok = _write_text_safely(note_filepath, out)
        change.path, change.before, change.after = note_filepath, current, out
    return change


# ---------------------------------------------------------------------------
# "On this page" list (kept up to date by batch passes)
# ---------------------------------------------------------------------------
TOC_HEADING = "## 🧭 On this page"
_TOC_HEADING_RE = re.compile(r"^##\s+🧭\s*On this page\s*$")
_NOTES_HEADER_RE = re.compile(r"^##\s+💡\s*Notes\s*$")


def _heading_link(heading: str, label: Optional[str] = None) -> str:
    """Obsidian link to a heading in the same note ([[#Heading|label]])."""
    target = re.sub(r"\s+", " ", re.sub(r"[#|^\[\]]", " ", heading)).strip()
    label = re.sub(r"\s+", " ", re.sub(r"[|\[\]]", " ", label if label is not None else heading)).strip()
    return f"[[#{target}]]" if label == target else f"[[#{target}|{label}]]"


def build_toc(text: str, min_themes: int = 2) -> Optional[str]:
    """
    "On this page" section for a voice note: the note's ## sections, with the ### themes under
    💡 Notes (and how many points each holds). None when the note has fewer than `min_themes` themes.
    """
    body = text
    fm = _FM_RE.match(body)
    if fm:
        body = body[fm.end():]
    entries: List[Tuple[str, str, List[Tuple[str, int]]]] = []  # (heading, label, themes)
    current = None
    in_notes = False
    theme = None
    fence = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if line.startswith("## ") and not line.startswith("### "):
            heading = line[3:].strip()
            theme = None
            if _TOC_HEADING_RE.match(line):
                current, in_notes = None, False
                continue
            in_notes = bool(_NOTES_HEADER_RE.match(line))
            label = re.sub(r"^[^\w(]+", "", heading).strip() or heading  # drop the leading emoji
            current = (heading, label, [])
            entries.append(current)
            continue
        if line.startswith("### ") and current is not None:
            if in_notes:
                theme = [line[4:].strip(), 0]
                current[2].append(theme)
            continue
        if theme is not None and (line.startswith("- ") or line.startswith("* ")):
            theme[1] += 1
    themes = sum(len(e[2]) for e in entries if _NOTES_HEADER_RE.match("## " + e[0]))
    if themes < min_themes:
        return None
    lines = [TOC_HEADING, ""]
    for heading, label, sub in entries:
        lines.append(f"- {_heading_link(heading, label)}")
        for name, count in sub:
            if name == INBOX_HEADING:
                extra = f" · {count} waiting" if count else ""
            else:
                extra = f" · {count} point{'s' if count != 1 else ''}" if count else ""
            lines.append(f"\t- {_heading_link(name)}{extra}")
    return "\n".join(lines) + "\n"


def upsert_toc(text: str, min_themes: int = 2) -> str:
    """Insert or refresh the "On this page" section right above ## 💡 Notes (removed when not useful)."""
    toc = build_toc(text, min_themes)
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if _TOC_HEADING_RE.match(l.rstrip())), None)
    if start is not None:
        end = start + 1
        while end < len(lines) and not re.match(r"^#{1,2}\s", lines[end]):
            end += 1
        del lines[start:end]
    if toc:
        at = next((i for i, l in enumerate(lines) if _NOTES_HEADER_RE.match(l.rstrip())), None)
        block = toc.rstrip("\n").split("\n") + [""]
        if at is None:
            at = len(lines)
            block = [""] + block
        lines[at:at] = block
    out = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", out)


# ---------------------------------------------------------------------------
# Note properties (YAML frontmatter) and automatic links
# ---------------------------------------------------------------------------
_FM_RE = re.compile(r"^---\r?\n(.*?)\r?\n---\r?\n?", re.S)


def update_properties(text: str, *, date_str: str, add_session: bool, add_seconds: float, tag: str = "") -> str:
    """Maintain created / updated / voice_sessions / voice_minutes / tags without touching other properties."""
    m = _FM_RE.match(text)
    lines = m.group(1).split("\n") if m else []
    body = text[m.end():] if m else text

    def find(key: str) -> int:
        for i, line in enumerate(lines):
            if re.match(rf"^{re.escape(key)}\s*:", line):
                return i
        return -1

    def put(key: str, value, only_if_missing: bool = False):
        i = find(key)
        if i < 0:
            lines.append(f"{key}: {value}")
        elif not only_if_missing:
            lines[i] = f"{key}: {value}"

    def num(key: str) -> float:
        i = find(key)
        if i < 0:
            return 0.0
        try:
            return float(lines[i].split(":", 1)[1].strip().strip("\"'"))
        except ValueError:
            return 0.0

    put("created", date_str, only_if_missing=True)
    put("updated", date_str)
    if add_session or find("voice_sessions") < 0:
        put("voice_sessions", int(num("voice_sessions")) + 1)
    put("voice_minutes", round(num("voice_minutes") + max(0.0, add_seconds) / 60.0, 1))
    if tag:
        i = find("tags")
        if i < 0:
            lines.extend(["tags:", f"  - {tag}"])
        else:
            val = lines[i].split(":", 1)[1].strip()
            if val.startswith("["):
                items = [x.strip().strip("\"'") for x in val.strip("[]").split(",") if x.strip()]
                if tag not in items:
                    lines[i] = f"tags: [{', '.join(items + [tag])}]"
            elif val == "":
                j, items = i + 1, []
                while j < len(lines) and re.match(r"^\s*-\s*", lines[j]):
                    items.append(lines[j].split("-", 1)[1].strip().strip("\"'"))
                    j += 1
                if tag not in items:
                    lines.insert(j, f"  - {tag}")
            else:
                items = [x.strip().lstrip("#") for x in re.split(r"[,\s]+", val) if x.strip()]
                if tag not in items:
                    lines[i] = f"tags: [{', '.join(items + [tag])}]"
    return "---\n" + "\n".join(lines) + "\n---\n" + body.lstrip("\n")


class NoteIndex:
    """Names of the notes in the vault (refreshed at most every 30 s) for automatic [[links]]."""

    def __init__(self, vault_root: Path, skip_dirs: set):
        self.vault_root = vault_root
        self.skip = {d.lower() for d in skip_dirs}
        self._titles: List[str] = []
        self._at = 0.0

    def invalidate(self):
        self._at = 0.0

    def titles(self) -> List[str]:
        if time.time() - self._at < 30 and self._titles:
            return self._titles
        found = set()
        for root, dirs, files in os.walk(self.vault_root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in self.skip]
            for fn in files:
                if fn.lower().endswith(".md"):
                    found.add(fn[:-3])
        self._titles = sorted(found, key=len, reverse=True)
        self._at = time.time()
        return self._titles


_PROTECTED_RE = re.compile(r"\[\[.*?\]\]|`[^`]*`|\[[^\]]*\]\([^)]*\)|https?://\S+")


def link_mentions(text: str, titles: List[str], exclude: set) -> str:
    """Link the first mention of each existing note: multi-word names case-insensitively,
    single words only when written exactly like the note name (avoids linking common words)."""
    excl = {e.lower() for e in exclude}
    for title in titles:
        if title.lower() in excl or len(title) < 3 or re.search(r"\d{4}-\d{2}-\d{2}", title):
            continue
        multi = " " in title.strip()
        if not multi and (len(title) < 4 or not title[0].isupper() or title.lower() in _STOPWORDS):
            continue
        pat = re.compile(r"(?<![\w\[(|])" + re.escape(title) + r"(?![\w\]])", re.I if multi else 0)
        protected = [(m.start(), m.end()) for m in _PROTECTED_RE.finditer(text)]
        for m in pat.finditer(text):
            if any(a <= m.start() < b for a, b in protected):
                continue
            found = m.group(0)
            repl = f"[[{title}]]" if found == title else f"[[{title}|{found}]]"
            text = text[:m.start()] + repl + text[m.end():]
            break
    return text


# ---------------------------------------------------------------------------
# Undo ("scratch that") and session summaries
# ---------------------------------------------------------------------------
_UNDO_RE = re.compile(
    r"^(?:scratch that|strike that|cancel that|undo(?: that)?(?: last (?:one|point|change|bit|thing))?|"
    r"undo the last (?:one|point|change|bit|thing)|take (?:that|it) (?:back|out)|"
    r"(?:delete|remove) (?:that|the last (?:one|point|bullet|thing|sentence|note)))(?: please)?[.!]?$",
    re.I)


def is_undo_command(text: str) -> bool:
    t = _LEAD_FILLER.sub("", text.strip().strip("\"'“”")).strip()
    return bool(_UNDO_RE.match(t))


def build_summary_prompt(topic_title: str, thoughts: List[str]) -> str:
    listed = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(thoughts))
    return (
        f"NOTE: {topic_title}\n\nThese are the thoughts a person spoke in one session, in order:\n{listed}\n\n"
        "Write a short session summary. Reply with JSON exactly in this shape:\n"
        '{"summary": "2-4 sentences", "action_items": [{"text": "...", "sources": [1]}], '
        '"open_questions": [{"text": "...", "sources": [2]}]}\n'
        "Rules: use ONLY what was said; never invent facts. Action items are things the speaker intends or needs to do. "
        "Open questions are things they were unsure about, wondered or asked. Use empty lists when there are none. "
        "At most 6 of each. \"sources\" are the numbers of the thoughts each item comes from.\n"
    )


def summary_model(config: ScribeConfig) -> Optional[str]:
    """In batch mode the session summary uses the (usually bigger) batch model."""
    if str(config.organise_mode).lower() == "batch" and (config.batch_ollama_model or "").strip():
        return config.batch_ollama_model.strip()
    return None


def ask_llm_for_summary(config: ScribeConfig, topic_title: str, thoughts: List[str]) -> Optional[dict]:
    big = summary_model(config)
    if big:
        timeout = max(config.ollama_timeout_sec, 60.0 + 10.0 * len(thoughts))
        return _chat_json(config, build_summary_prompt(topic_title, thoughts), model=big,
                          num_ctx=config.batch_num_ctx, num_predict=config.batch_num_predict, timeout=timeout)
    return _chat_json(config, build_summary_prompt(topic_title, thoughts))


def render_summary_section(heading: str, data: dict, thoughts: List[dict]) -> Optional[str]:
    """Validated summary dict -> markdown section; items keep the citations of their source thoughts."""
    if not isinstance(data, dict):
        return None
    all_text = " ".join(t["text"] for t in thoughts)
    summary = _clean_bullet_text(data.get("summary"))
    if summary and not (_content_stems(summary) & _content_stems(all_text)):
        summary = ""

    def items(key: str) -> List[str]:
        out = []
        for it in (data.get(key) or [])[:6]:
            if isinstance(it, str):
                it = {"text": it, "sources": []}
            if not isinstance(it, dict):
                continue
            text = _clean_bullet_text(it.get("text"))
            srcs = []
            for n in it.get("sources") or []:
                try:
                    n = int(n)
                except (TypeError, ValueError):
                    continue
                if 1 <= n <= len(thoughts) and n not in srcs:
                    srcs.append(n)
            basis = " ".join(thoughts[n - 1]["text"] for n in srcs) if srcs else all_text
            if not text or not is_grounded(text, basis):
                continue
            cites = " ".join(dict.fromkeys(thoughts[n - 1]["cite"] for n in srcs))
            out.append(text + (" " + cites if cites else ""))
        return out

    actions, questions = items("action_items"), items("open_questions")
    if not (summary or actions or questions):
        return None
    lines = [heading, ""]
    if summary:
        lines += [summary, ""]
    if actions:
        lines += ["**Action items**"] + [f"- [ ] {a}" for a in actions] + [""]
    if questions:
        lines += ["**Open questions**"] + [f"- {q}" for q in questions] + [""]
    return "\n".join(lines).rstrip() + "\n"


def upsert_section(text: str, heading: str, section: str) -> str:
    """Replace the `## heading` section (up to the next `## `) or append it at the end."""
    start = text.find(heading + "\n")
    if start < 0 and text.rstrip().endswith(heading):
        start = text.rstrip().rfind(heading)
    if start >= 0:
        nxt = re.search(r"^##\s+(?!#)", text[start + len(heading):], re.M)
        end = start + len(heading) + nxt.start() if nxt else len(text)
        return text[:start] + section + ("\n" + text[end:] if nxt else "")
    return text.rstrip("\n") + "\n\n" + section


# ---------------------------------------------------------------------------
# Ollama availability
# ---------------------------------------------------------------------------
def ensure_ollama_running(host: str = "http://127.0.0.1:11434") -> bool:
    import urllib.request

    def ping(timeout: float) -> bool:
        try:
            req = urllib.request.Request(f"{host}/api/tags", headers={"User-Agent": "VoiceScribe"})
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return res.status == 200
        except Exception:
            return False

    if ping(1.5):
        logger.info(f"Ollama server is responsive at {host}.")
        return True

    logger.warning(f"Ollama at {host} is not running. Trying to start it...")
    import shutil
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            ollama_bin = str(candidate)
    if not ollama_bin:
        logger.error("Could not find Ollama. Notes will be saved without AI organisation.")
        emit("warning", message="Ollama not found - notes will be added without AI organisation.")
        return False
    try:
        subprocess.Popen([ollama_bin, "serve"], creationflags=CREATE_NO_WINDOW,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.error(f"Error starting Ollama: {e}")
        return False
    for _ in range(20):
        time.sleep(0.5)
        if ping(1.0):
            logger.info("Ollama server started.")
            return True
    logger.warning("Ollama was launched but is not answering yet.")
    return False


# ---------------------------------------------------------------------------
# Thread 1: audio ingestion with energy-based segmentation
# ---------------------------------------------------------------------------
class AudioIngestionThread(threading.Thread):
    LEVEL_INTERVAL_SEC = 0.1  # ~10 level updates per second for the dashboard meter

    def __init__(self, audio_queue: queue.Queue, stop_event: threading.Event, config: ScribeConfig,
                 session_start: Optional[float] = None):
        super().__init__(name="AudioIngestionThread", daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.config = config
        self.session_start = session_start or time.time()
        self.raw_blocks: queue.Queue = queue.Queue()
        self.total_chunks_captured = 0
        self._next_seg = 1
        self.paused = threading.Event()

    def set_paused(self, paused: bool):
        if paused == self.paused.is_set():
            return
        t = round(time.time() - self.session_start, 2)
        if paused:
            self.paused.set()
            logger.info("Listening paused.")
            emit("paused", t=t)
        else:
            self.paused.clear()
            logger.info("Listening resumed.")
            emit("resumed", t=t)

    def _audio_callback(self, indata, frames, time_info, status):
        self.raw_blocks.put(indata[:, 0].copy())

    def _emit_chunk(self, blocks: List[np.ndarray], start: datetime, voiced_rms: List[float], block_duration: float,
                    label: str, t0: float = 0.0):
        duration = len(blocks) * block_duration
        voiced = len(voiced_rms) * block_duration
        if duration < self.config.min_speech_duration_sec or voiced < self.config.min_voiced_sec:
            logger.info(f"[VAD] Ignored noise blip ({voiced:.2f}s of sound)")
            emit("discarded", t0=round(t0, 2), dur=round(duration, 2), voiced=round(voiced, 2))
            return
        chunk = AudioChunk(
            timestamp=start,
            data=np.concatenate(blocks),
            duration=float(duration),
            rms_energy=float(np.mean(voiced_rms)) if voiced_rms else 0.0,
            seg_id=self._next_seg,
            t0=t0,
            voiced=voiced,
        )
        self._next_seg += 1
        try:
            self.audio_queue.put(chunk, timeout=2.0)
            self.total_chunks_captured += 1
            logger.info(f"[VAD] {label} ({duration:.1f}s)")
            emit("chunk_captured", seg=chunk.seg_id, t0=round(t0, 2), dur=round(duration, 2),
                 seconds=round(duration, 1), pending=self.audio_queue.qsize())
        except queue.Full:
            logger.warning("Transcription is falling behind - audio queue full, chunk dropped.")
            emit("warning", message="Transcription is falling behind; a speech chunk was dropped.")

    def run(self):
        import sounddevice as sd

        sr = self.config.sample_rate
        block_size = 1024
        block_duration = block_size / sr
        pre_speech = collections.deque(maxlen=max(1, int(0.5 / block_duration)))

        is_speaking = False
        speech_blocks: List[np.ndarray] = []
        voiced_rms: List[float] = []
        silence = 0.0
        start_time = datetime.now()
        seg_t0 = 0.0
        level_peak = 0.0
        last_level = time.time()

        logger.info(f"Audio thread started ({sr} Hz, pause={self.config.pause_duration_sec}s, "
                    f"threshold={self.config.energy_threshold}).")

        while not self.stop_event.is_set():
            try:
                device = self.config.audio_device
                if isinstance(device, str) and device.strip().lstrip("-").isdigit():
                    device = int(device)
                device_name = "System Default"
                if device is not None:
                    try:
                        device_name = sd.query_devices(device, "input").get("name", str(device))
                    except Exception:
                        device_name = str(device)
                logger.info(f"Connecting to microphone: {device_name}")

                with sd.InputStream(samplerate=sr, channels=1, dtype="float32", device=device,
                                    callback=self._audio_callback, blocksize=block_size):
                    logger.info("Microphone active. Listening continuously...")
                    emit("listening", device=device_name)

                    while not self.stop_event.is_set():
                        try:
                            block = self.raw_blocks.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        if self.paused.is_set():
                            if is_speaking and speech_blocks:  # keep what was said before pausing
                                self._emit_chunk(speech_blocks, start_time, voiced_rms, block_duration,
                                                 "Captured speech chunk (paused)", t0=seg_t0)
                            is_speaking, speech_blocks, voiced_rms, silence = False, [], [], 0.0
                            pre_speech.clear()
                            now = time.time()
                            if now - last_level >= 1.0:
                                emit("level", t=round(now - self.session_start, 2), r=0.0, s=0, p=1)
                                last_level = now
                            continue

                        rms = float(np.sqrt(np.mean(block ** 2)))
                        loud = rms >= self.config.energy_threshold

                        now = time.time()
                        level_peak = max(level_peak, rms)
                        if now - last_level >= self.LEVEL_INTERVAL_SEC:
                            emit("level", t=round(now - self.session_start, 2), r=round(level_peak, 5),
                                 s=1 if is_speaking else 0)
                            level_peak = 0.0
                            last_level = now

                        if not is_speaking:
                            if loud:
                                is_speaking = True
                                start_time = datetime.now()
                                seg_t0 = max(0.0, now - self.session_start - len(pre_speech) * block_duration)
                                speech_blocks = list(pre_speech) + [block]
                                pre_speech.clear()
                                voiced_rms = [rms]
                                silence = 0.0
                                logger.info(f"[VAD] Speech detected (RMS={rms:.4f})")
                                emit("speech_start")
                            else:
                                pre_speech.append(block)
                            continue

                        speech_blocks.append(block)
                        if loud:
                            voiced_rms.append(rms)
                            silence = 0.0
                        else:
                            silence += block_duration

                        too_long = len(speech_blocks) * block_duration >= self.config.max_speech_duration_sec
                        if silence >= self.config.pause_duration_sec or too_long:
                            self._emit_chunk(speech_blocks, start_time, voiced_rms, block_duration,
                                             "Captured speech chunk" if not too_long else "Captured max-length chunk",
                                             t0=seg_t0)
                            speech_blocks, voiced_rms = [], []
                            is_speaking = False
                            silence = 0.0

                if is_speaking and speech_blocks:
                    self._emit_chunk(speech_blocks, start_time, voiced_rms, block_duration,
                                     "Flushed final speech chunk", t0=seg_t0)
                    speech_blocks, is_speaking = [], False

            except Exception as e:
                if self.stop_event.is_set():
                    break
                logger.warning(f"Microphone error: {e}. Retrying in 2.5s...")
                emit("warning", message=f"Microphone error: {e}")
                time.sleep(2.5)

        logger.info("Audio thread stopped.")


# ---------------------------------------------------------------------------
# GPU support for Whisper (CTranslate2 + CUDA 12 / cuDNN 9)
# ---------------------------------------------------------------------------
GPU_PROBE_FILE = SCRIPT_DIR / "gpu_probe.json"
_DLL_DIRS_ADDED = False


def cuda_dll_search_dirs(extra: str = "") -> List[Path]:
    """Folders that may hold CUDA DLLs: pip `nvidia-*` packages in the venv, a chosen folder
    (`pip install --target` layout or a CUDA Toolkit), and CUDA_PATH."""
    import site
    roots: List[Path] = []
    try:
        roots += [Path(r) for r in site.getsitepackages()]
    except Exception:
        pass
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")
    extra = extra or os.environ.get("VOICESCRIBE_CUDA_DIR", "")
    if extra:
        roots.insert(0, Path(extra))
    out: List[Path] = []
    for root in roots:
        for d in (root, root / "bin"):
            if d.is_dir() and (any(d.glob("cublas*.dll")) or any(d.glob("cudnn*.dll"))):
                out.append(d)
        nv = root / "nvidia"
        if nv.is_dir():
            for sub in nv.iterdir():
                for d in (sub / "bin", sub / "lib"):
                    if d.is_dir():
                        out.append(d)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and (Path(cuda_path) / "bin").is_dir():
        out.append(Path(cuda_path) / "bin")
    seen, uniq = set(), []
    for d in out:
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def add_nvidia_dll_dirs() -> List[str]:
    """Make CUDA libraries loadable on Windows (see cuda_dll_search_dirs)."""
    global _DLL_DIRS_ADDED
    added: List[str] = []
    if _DLL_DIRS_ADDED:
        return added
    _DLL_DIRS_ADDED = True
    for d in cuda_dll_search_dirs():
        try:
            if hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
            added.append(str(d))
        except OSError:
            pass
    return added


def cuda_device_count() -> int:
    try:
        add_nvidia_dll_dirs()
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def _probe_key(config: ScribeConfig) -> str:
    try:
        import ctranslate2
        ver = ctranslate2.__version__
    except Exception:
        ver = "?"
    return f"{config.whisper_model}|{config.whisper_gpu_compute_type}|ct2-{ver}|{config.cuda_libs_dir or 'venv'}"


def gpu_probe(config: ScribeConfig, force: bool = False) -> Tuple[bool, str]:
    """
    Check in a separate process that Whisper really runs on the GPU. A missing CUDA/cuDNN
    library can crash the whole process, so it is never tried in the main backend first.
    The result is cached per model / compute type / CTranslate2 version.
    """
    key = _probe_key(config)
    cache = {}
    if GPU_PROBE_FILE.exists():
        try:
            cache = json.loads(GPU_PROBE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}
    if not force and key in cache:
        r = cache[key]
        return bool(r.get("ok")), r.get("detail", "")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--gpu-probe",
           "--whisper-model", config.whisper_model, "--gpu-compute", config.whisper_gpu_compute_type]
    if config.allow_model_downloads:
        cmd.append("--allow-downloads")
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
                              creationflags=CREATE_NO_WINDOW, cwd=str(SCRIPT_DIR))
        out = proc.stdout.decode("utf-8", errors="replace")
        ok = proc.returncode == 0 and "GPU_PROBE_OK" in out
        lines = [l for l in out.strip().splitlines() if l.strip() and "GPU_PROBE_OK" not in l]
        detail = "" if ok else (lines[-1] if lines else f"exit code {proc.returncode}")[:300]
    except subprocess.TimeoutExpired:
        ok, detail = False, "GPU test timed out"
    except Exception as e:
        ok, detail = False, str(e)
    cache[key] = {"ok": ok, "detail": detail, "at": datetime.now().isoformat(timespec="seconds")}
    try:
        GPU_PROBE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except OSError:
        pass
    return ok, detail


def run_gpu_probe_main(model: str, compute: str, allow_downloads: bool) -> int:
    """Child process for gpu_probe(): load the model on CUDA and transcribe one second of audio."""
    try:
        add_nvidia_dll_dirs()
        from faster_whisper import WhisperModel
        m = WhisperModel(model, device="cuda", compute_type=compute, local_files_only=not allow_downloads)
        audio = (np.random.default_rng(0).standard_normal(16000) * 0.01).astype(np.float32)
        list(m.transcribe(audio, beam_size=1, language="en", vad_filter=False)[0])
        print("GPU_PROBE_OK", flush=True)
        return 0
    except Exception as e:
        print(f"GPU test failed: {e}", flush=True)
        return 1


def gpu_status_main() -> int:
    """`--gpu-status`: one JSON line for the plugin's diagnostics."""
    dirs = add_nvidia_dll_dirs()
    info = {"cuda_devices": 0, "nvidia_libs": bool(dirs), "lib_dirs": dirs,
            "cuda_dir": os.environ.get("VOICESCRIBE_CUDA_DIR", ""), "ctranslate2": None}
    try:
        import ctranslate2
        info["ctranslate2"] = ctranslate2.__version__
        info["cuda_devices"] = int(ctranslate2.get_cuda_device_count())
    except Exception as e:
        info["error"] = str(e)
    if GPU_PROBE_FILE.exists():
        try:
            info["probes"] = json.loads(GPU_PROBE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    print(json.dumps(info))
    return 0


# ---------------------------------------------------------------------------
# Thread 2: transcription + archival
# ---------------------------------------------------------------------------
class TranscriptionThread(threading.Thread):
    def __init__(self, audio_queue, transcription_queue, stop_event, config: ScribeConfig,
                 shared_context: Optional[SharedContext] = None, fatal_event: Optional[threading.Event] = None):
        super().__init__(name="TranscriptionThread", daemon=True)
        self.audio_queue = audio_queue
        self.transcription_queue = transcription_queue
        self.stop_event = stop_event
        self.config = config
        self.shared_context = shared_context
        self.fatal_event = fatal_event or threading.Event()
        self.whisper_model = None
        self.device = "cpu"
        self.total_transcribed = 0
        self.ready = threading.Event()
        self._recent_short: collections.deque = collections.deque(maxlen=6)
        self._noise_chunks = 0
        self._noise_warned = False
        self.last_activity = time.time()

    def release_whisper(self, reason: str = "idle") -> bool:
        """Free Whisper's GPU/RAM; it is loaded again when the next thought arrives."""
        if self.whisper_model is None:
            return False
        self.whisper_model = None
        import gc
        gc.collect()
        idle = round((time.time() - self.last_activity) / 60.0, 1)
        logger.info(f"Released Whisper from memory ({reason}, idle {idle} min).")
        emit("whisper_unloaded", reason=reason, idle_min=idle, device=self.device)
        return True

    def idle_check(self) -> None:
        try:
            limit = float(self.config.whisper_idle_unload_min or 0) * 60.0
        except (TypeError, ValueError):
            limit = 0.0
        if limit > 0 and self.whisper_model is not None and time.time() - self.last_activity >= limit:
            self.release_whisper("idle")

    def _choose_device(self) -> str:
        want = (self.config.whisper_device or "auto").lower()
        if want == "cpu":
            return "cpu"
        n = cuda_device_count()
        if n == 0:
            if want == "cuda":
                emit("warning", message="GPU transcription was requested but no CUDA GPU is usable; using the CPU.")
            return "cpu"
        emit("gpu_check", model=self.config.whisper_model)
        ok, detail = gpu_probe(self.config)
        if ok:
            return "cuda"
        logger.warning(f"GPU transcription unavailable ({detail}); using the CPU.")
        if want == "cuda" or "cublas" in detail.lower() or "cudnn" in detail.lower():
            emit("warning", message="GPU transcription isn't working yet (" + detail[:140] + "). Using the CPU. "
                                    "Settings → VoiceScribe → Transcription → Install GPU support can fix this.")
        return "cpu"

    def _init_whisper(self):
        from faster_whisper import WhisperModel
        name = self.config.whisper_model
        device = self._choose_device()
        self.device = device
        if device == "cuda":
            compute = self.config.whisper_gpu_compute_type
            logger.info(f"Loading Whisper '{name}' on the GPU ({compute})...")
            kwargs = dict(device="cuda", compute_type=compute)
        else:
            logger.info(f"Loading Whisper '{name}' on CPU ({self.config.whisper_compute_type}, "
                        f"{self.config.whisper_threads} threads)...")
            kwargs = dict(device="cpu", compute_type=self.config.whisper_compute_type,
                          cpu_threads=int(self.config.whisper_threads))
        emit("loading_model", model=name, device=device)
        try:
            self.whisper_model = WhisperModel(name, local_files_only=True, **kwargs)
        except Exception as local_err:
            if not self.config.allow_model_downloads and not Path(name).exists():
                raise RuntimeError(
                    f"Whisper model '{name}' is not downloaded yet and model downloads are disabled. "
                    f"Enable 'Allow model downloads' in VoiceScribe settings once, or pick a model you already have."
                ) from local_err
            logger.info(f"Model not cached locally - downloading '{name}' (one-time)...")
            emit("downloading_model", model=name)
            self.whisper_model = WhisperModel(name, **kwargs)
        logger.info(f"Whisper loaded on {device.upper()}.")
        emit("model_loaded", model=name, device=device)

    def _transcribe(self, audio: np.ndarray, duration: float = 99.0, voiced: float = 99.0) -> str:
        prompt = (self.shared_context.get_initial_prompt()
                  if (self.shared_context and self.config.context_prompt_enabled) else None)
        lang = (self.config.whisper_language or "").strip().lower()
        lang = None if lang in ("", "auto") else lang

        def run(vad: bool) -> str:
            segments, _ = self.whisper_model.transcribe(
                audio,
                beam_size=int(self.config.whisper_beam_size),
                language=lang,
                vad_filter=vad,
                vad_parameters=dict(min_silence_duration_ms=500) if vad else None,
                condition_on_previous_text=False,
                initial_prompt=prompt,
            )
            kept = []
            short = duration < 4.0
            for seg in segments:
                if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                    continue
                if getattr(seg, "compression_ratio", 0) > 2.4:  # repetitive gibberish
                    continue
                if short and (seg.no_speech_prob > 0.45 or seg.avg_logprob < -0.9):
                    continue
                kept.append(seg.text.strip())
            return " ".join(kept).strip()

        text = run(self.config.whisper_vad_filter)
        if not text and self.config.whisper_vad_filter and voiced >= 1.0:
            # Silero VAD can reject very quiet speech; retry once - but only when there was real
            # sustained sound, otherwise Whisper invents text from noise.
            text = run(False)
        if text and (is_whisper_hallucination(text) or is_short_clip_hallucination(text, duration)):
            logger.info(f"Ignored likely non-speech transcription: '{text}'")
            return ""
        if text and duration < 5.0:
            norm = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
            if norm and norm in self._recent_short:
                logger.info(f"Ignored repeated short transcription (likely noise): '{text}'")
                return ""
            self._recent_short.append(norm)
        return text

    def run(self):
        try:
            self._init_whisper()
        except Exception as e:
            logger.critical(f"Whisper failed to load: {e}")
            emit("error", message=f"Whisper failed to load: {e}")
            self.fatal_event.set()
            self.stop_event.set()
            return
        self.ready.set()

        vault_root = Path(self.config.vault_dir).resolve()
        archive_root = vault_root / self.config.archive_dir

        def rel(p: Path) -> str:
            try:
                return p.resolve().relative_to(vault_root).as_posix()
            except ValueError:
                return p.name

        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                chunk: AudioChunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                self.idle_check()
                continue
            self.last_activity = time.time()
            try:
                if self.whisper_model is None:
                    emit("whisper_reload", model=self.config.whisper_model)
                    self._init_whisper()
                audio = normalize_audio(chunk.data) if self.config.normalize_audio else chunk.data.astype(np.float32)

                day_dir = archive_root / chunk.timestamp.strftime("%Y-%m-%d")
                ogg_path = unique_path(day_dir / f"audio_{chunk.timestamp.strftime('%Y%m%d_%H%M%S')}.ogg")
                archived = archive_audio_opus(audio, self.config.sample_rate, ogg_path, self.config.opus_bitrate)

                emit("transcribing", seg=chunk.seg_id, seconds=round(chunk.duration, 1))
                started = time.time()
                text = self._transcribe(audio, chunk.duration, chunk.voiced)
                took = time.time() - started
                self.last_activity = time.time()
                logger.info(f"Transcribed {chunk.duration:.1f}s of audio in {took:.1f}s")

                if text:
                    if self.shared_context:
                        self.shared_context.add_transcript(text)
                    self.total_transcribed += 1
                    emit("segment", seg=chunk.seg_id, t0=round(chunk.t0, 2), dur=round(chunk.duration, 2),
                         clip=rel(ogg_path) if archived else "", speech=True, took=round(took, 1))
                    self.transcription_queue.put(TranscribedChunk(
                        timestamp=chunk.timestamp, text=text,
                        archive_path=ogg_path if archived else None, duration=chunk.duration,
                        seg_id=chunk.seg_id))
                else:
                    logger.info("No words recognised in chunk.")
                    emit("no_speech", seg=chunk.seg_id)
                    self._noise_chunks += 1
                    if self._noise_chunks >= 3 and not self._noise_warned:
                        self._noise_warned = True
                        emit("warning", message="Background noise keeps triggering recordings. Run Settings → VoiceScribe → "
                                                "Speech volume threshold → Calibrate to ignore it.")
                    emit("segment", seg=chunk.seg_id, t0=round(chunk.t0, 2), dur=round(chunk.duration, 2),
                         clip="", speech=False, took=round(took, 1))
                    if archived:
                        try:
                            ogg_path.unlink()  # don't keep audio nobody can reference
                        except OSError:
                            pass
            except Exception as e:
                logger.error(f"Error during transcription/archival: {e}", exc_info=True)
                emit("error", message=f"Transcription error: {e}")
            finally:
                self.audio_queue.task_done()

        logger.info("Transcription thread stopped.")


# ---------------------------------------------------------------------------
# Thread 3: router + note synthesis
# ---------------------------------------------------------------------------
class RouterSynthesisThread(threading.Thread):
    def __init__(self, transcription_queue, stop_event, config: ScribeConfig,
                 shared_context: Optional[SharedContext] = None, upstream_done: Optional[threading.Event] = None):
        super().__init__(name="RouterSynthesisThread", daemon=True)
        self.transcription_queue = transcription_queue
        self.stop_event = stop_event
        self.config = config
        self.shared_context = shared_context
        self.upstream_done = upstream_done or stop_event

        self.vault_root = Path(self.config.vault_dir).resolve()
        self.notes_dir = self.vault_root / self.config.notes_subfolder if self.config.notes_subfolder else self.vault_root
        self.notes_dir.mkdir(parents=True, exist_ok=True)

        self.state_lock = threading.Lock()
        self.active_path: Path = self.notes_dir / sanitize_filename(self.config.default_note)
        if self.config.start_note:
            chosen = self.resolve_vault_note(self.config.start_note)
            if chosen:
                self.active_path = chosen
        self.focus: Optional[Focus] = None
        self.total_notes_written = 0
        self._used_ids: Dict[str, int] = {}
        self.undo_stack: collections.deque = collections.deque(maxlen=30)
        self.inbox_pending: Dict[Path, dict] = {}   # batch mode: note -> {"count", "since"}
        self.session_thoughts: Dict[Path, List[dict]] = {}
        self.session_start = datetime.now()
        self.skip_summary = False
        self.note_index = NoteIndex(self.vault_root, self._SKIP_DIRS | {self.config.archive_dir, self.config.transcripts_folder})

        if self.shared_context:
            self.shared_context.update_topic(self.active_path.stem)
        logger.info(f"Notes folder: {self.notes_dir}")
        logger.info(f"Active note:  {self.active_path.name}")

    # Backwards-compatible name used by older callers/tests
    @property
    def active_filename(self) -> str:
        with self.state_lock:
            return self.active_path.name

    def vault_rel(self, p: Path) -> str:
        try:
            return p.resolve().relative_to(self.vault_root).as_posix()
        except ValueError:
            return p.name

    def get_active_filepath(self) -> Path:
        with self.state_lock:
            return self.active_path

    def _switch_to(self, path: Path, label: str, reason: str, created: bool = False):
        with self.state_lock:
            old = self.active_path
            self.active_path = path
            focus = self.focus
        if self.shared_context:
            self.shared_context.update_topic(path.stem)
        if ensure_note_exists(path, path.stem):
            self.note_index.invalidate()
        if old != path:
            logger.info(f"[{reason}] Switched active note: '{old.name}' -> '{path.name}'")
            emit("topic", note=self.vault_rel(path), name=path.stem, previous=self.vault_rel(old), reason=reason,
                 created=created)
            if focus and focus.note != path:
                self.clear_focus("note switched")

    def set_active_topic(self, topic_name: str):
        self.apply_note_command(NoteCommand("switch", topic_name))

    _SKIP_DIRS = {".obsidian", ".trash", ".git"}

    def find_note(self, name: str) -> Optional[Path]:
        """Existing note with this name (case-insensitive), preferring the notes folder."""
        wanted = sanitize_filename(name).lower()
        skip = self._SKIP_DIRS | {self.config.archive_dir.lower(), self.config.transcripts_folder.lower()}
        direct = self.notes_dir / sanitize_filename(name)
        if direct.exists():
            return direct
        try:
            for f in self.notes_dir.iterdir():
                if f.is_file() and f.name.lower() == wanted:
                    return f
        except OSError:
            pass
        best = None
        for root, dirs, files in os.walk(self.vault_root):
            dirs[:] = [d for d in dirs if d.lower() not in skip and not d.startswith(".")]
            for fn in files:
                if fn.lower() == wanted:
                    cand = Path(root) / fn
                    if best is None or len(cand.parts) < len(best.parts):
                        best = cand
        return best

    def apply_note_command(self, cmd: NoteCommand):
        name = cmd.name.strip()
        if cmd.kind == "switch":
            found = self.find_note(name)
            if not found and re.search(r"\s+(?:note|notes|file|page)$", name, re.I):
                found = self.find_note(re.sub(r"\s+(?:note|notes|file|page)$", "", name, flags=re.I))
            if found:
                self._switch_to(found, found.stem, "command")
                return
            target = self.notes_dir / sanitize_filename(name)
            created = not target.exists()
            self._switch_to(target, name, "command", created=created)
            return
        # create: always a new note; never reuse an existing file
        target = self.notes_dir / sanitize_filename(name)
        if target.exists():
            stem = target.stem
            for i in range(2, 100):
                cand = self.notes_dir / sanitize_filename(f"{stem} {i}")
                if not cand.exists():
                    target = cand
                    break
        self._switch_to(target, target.stem, "command", created=True)

    def resolve_vault_note(self, rel_path: str) -> Optional[Path]:
        rel_path = (rel_path or "").replace("\\", "/").strip("/")
        if not rel_path.lower().endswith(".md"):
            return None
        p = (self.vault_root / rel_path).resolve()
        try:
            p.relative_to(self.vault_root)
        except ValueError:
            return None
        return p

    # ---- focus (set from the Obsidian plugin) --------------------------------
    def set_focus(self, note_rel: str, heading: Optional[str], bullets: List[str], mode: str):
        path = self.resolve_vault_note(note_rel)
        if path is None:
            emit("warning", message=f"Cannot focus on '{note_rel}' (not a note inside the vault).")
            return
        mode = "correct" if str(mode).lower() == "correct" else "add"
        bullets = [str(b).strip() for b in (bullets or []) if str(b).strip()][:20]
        heading = (str(heading).strip() or None) if heading else None
        if not bullets and not heading:
            emit("warning", message="Nothing to focus on: put the cursor on a bullet or ### heading.")
            return
        if path != self.get_active_filepath():
            self._switch_to(path, path.stem, "focus")
        with self.state_lock:
            self.focus = Focus(path, heading, bullets, mode)
            info = self.focus.describe(self.vault_root)
        logger.info(f"[FOCUS] {mode} -> {heading or ''} {bullets[:1]}")
        emit("focus", **info)

    def set_focus_mode(self, mode: str):
        with self.state_lock:
            if not self.focus:
                return
            self.focus.mode = "correct" if str(mode).lower() == "correct" else "add"
            info = self.focus.describe(self.vault_root)
        emit("focus", **info)

    def clear_focus(self, reason: str = "cleared"):
        with self.state_lock:
            had = self.focus is not None
            self.focus = None
        if had:
            logger.info(f"[FOCUS] cleared ({reason})")
            emit("focus_cleared", reason=reason)

    def handle_command(self, msg: dict):
        cmd = str(msg.get("cmd", "")).lower()
        if cmd == "focus":
            self.set_focus(msg.get("note", ""), msg.get("heading"), msg.get("bullets") or [], msg.get("mode", "add"))
        elif cmd == "focus_mode":
            self.set_focus_mode(msg.get("mode", "add"))
        elif cmd == "clear_focus":
            self.clear_focus("cleared")
        elif cmd == "undo":
            self.undo()
        elif cmd == "organise":
            self._organise_requested = True
        elif cmd == "summarize":
            threading.Thread(target=self.summarize_all, kwargs={"reason": "requested"}, daemon=True).start()
        elif cmd == "switch_note":
            path = self.resolve_vault_note(msg.get("note", ""))
            if path:
                self._switch_to(path, path.stem, "manual")

    # ---- processing ------------------------------------------------------------
    def _chunk_id(self, ts: datetime) -> str:
        base = f"t-{ts.strftime('%H%M%S')}"
        key = f"{ts.strftime('%Y%m%d')}{base}"
        n = self._used_ids.get(key, 0)
        self._used_ids[key] = n + 1
        return base if n == 0 else f"{base}-{n + 1}"

    def process(self, item: TranscribedChunk) -> bool:
        raw_text = item.text.strip()
        logger.info(f'Heard: "{raw_text}"')
        emit("heard", text=raw_text, seg=item.seg_id)

        if is_undo_command(raw_text):
            logger.info("[COMMAND] undo")
            self.undo()
            return False

        before, command, after = split_command(raw_text)
        if command:
            logger.info(f"[COMMAND] {command.kind} note '{command.name}'")
            wrote = False
            if before.strip():
                wrote = self._write_thought(item, before.strip()) or wrote
            self.apply_note_command(command)
            if after.strip():
                wrote = self._write_thought(item, after.strip()) or wrote
            return wrote
        return self._write_thought(item, raw_text)

    def _write_thought(self, item: TranscribedChunk, raw_text: str) -> bool:
        cleaned = strip_verbal_fillers(raw_text)
        if count_substantive_words(cleaned) < max(1, int(self.config.min_substantive_words)):
            logger.info(f"Skipped fragment: '{cleaned}'")
            emit("skipped", seg=item.seg_id, text=cleaned)
            return False

        chunk_id = self._chunk_id(item.timestamp)
        time_str = item.timestamp.strftime("%H:%M:%S")
        date_str = item.timestamp.strftime("%Y-%m-%d")
        audio_rel = self.vault_rel(item.archive_path) if item.archive_path else ""

        with self.state_lock:
            target = self.active_path
            focus = dataclasses.replace(self.focus, bullets=list(self.focus.bullets)) if self.focus else None
        if focus and focus.note != target:
            focus = None
        topic_title = target.stem

        append_session_transcript(
            vault_dir=self.vault_root,
            transcripts_folder=self.config.transcripts_folder,
            topic_title=topic_title,
            chunk_id=chunk_id,
            timestamp_str=time_str,
            text=raw_text,
            audio_rel_path=audio_rel,
            date_str=date_str,
        )

        trans_link = f"{self.config.transcripts_folder.strip('/')}/{date_str}_Transcript"
        citation = f"([[{trans_link}#^{chunk_id}|🔗 {time_str}]]"
        if audio_rel:
            citation += f" · [[{audio_rel}|▶ Audio]]"
        citation += ")"

        emit("organising", note=self.vault_rel(target), seg=item.seg_id,
             focus=focus.mode if focus else None)
        first_time = target not in self.session_thoughts
        linker = None
        if self.config.autolink_enabled:
            linker = lambda t, _ex={target.stem}: link_mentions(t, self.note_index.titles(), _ex)
        postprocess = None
        if self.config.properties_enabled:
            postprocess = lambda out: update_properties(out, date_str=date_str, add_session=first_time,
                                                        add_seconds=item.duration, tag=self.config.properties_tag.strip())
        prev_focus = list(focus.bullets) if focus else None
        batch = self.batch_mode and not focus  # focus is an explicit instruction: always applied at once
        if batch:
            change = append_to_inbox(target, topic_title, cleaned, citation, linker=linker, postprocess=postprocess)
            if change:
                pend = self.inbox_pending.setdefault(target, {"count": 0, "since": time.time()})
                pend["count"] += 1
                emit("inbox", note=self.vault_rel(target), count=pend["count"])
        else:
            change = integrate_note_thematically(target, topic_title, cleaned, citation, self.config, focus=focus,
                                                 linker=linker, postprocess=postprocess)
        if change:
            self.session_thoughts.setdefault(target, []).append({"text": cleaned, "cite": citation})
            self.undo_stack.append({"path": target, "before": change.before, "after": change.after,
                                    "added": list(change.added), "updated": list(change.updated),
                                    "prev_focus": prev_focus, "focus_after": change.focus_bullets,
                                    "thought": cleaned, "inbox": batch})
            self.total_notes_written += 1
            if change.focus_bullets is not None:
                with self.state_lock:
                    if self.focus and self.focus.note == target:
                        self.focus.bullets = list(change.focus_bullets)
                        info = self.focus.describe(self.vault_root)
                    else:
                        info = None
                if info:
                    emit("focus", **info)
            logger.info(f"[{target.name}] Updated structured note ({trans_link}#^{chunk_id}).")
            emit("note_updated", note=self.vault_rel(target), transcript=f"{trans_link}.md", seg=item.seg_id,
                 chunk=chunk_id, added=change.added, updated=change.updated, headings=change.headings,
                 focus=focus.mode if focus else None, inbox=batch)
        else:
            emit("error", message=f"Could not write {target.name}")
        return bool(change)

    # ---- undo ------------------------------------------------------------------
    def undo(self) -> bool:
        if not self.undo_stack:
            emit("warning", message="Nothing to undo in this session.")
            return False
        entry = self.undo_stack.pop()
        path: Path = entry["path"]
        with FILE_WRITE_LOCK:
            try:
                current = path.read_text(encoding="utf-8") if path.exists() else ""
            except OSError:
                current = ""
            if current == entry["after"]:
                ok = _write_text_safely(path, entry["before"])
                how = "restored"
            else:
                # The note was edited since; remove only the bullets this thought added.
                note = StructuredNote.parse(current, path.stem)
                removed = 0
                for s_, i in sorted(note.locate(entry["added"]), key=lambda p: p[1], reverse=True):
                    del s_.items[i]
                    removed += 1
                ok = _write_text_safely(path, note.render()) if removed else True
                how = "removed" if removed else "nothing"
        thoughts = self.session_thoughts.get(path)
        if thoughts and thoughts[-1]["text"] == entry["thought"]:
            thoughts.pop()
        if entry.get("inbox") and path in self.inbox_pending:
            self.inbox_pending[path]["count"] = max(0, self.inbox_pending[path]["count"] - 1)
            emit("inbox", note=self.vault_rel(path), count=self.inbox_pending[path]["count"])
        if entry.get("batch"):
            self.inbox_pending[path] = {"count": entry["batch"], "since": time.time()}
            emit("inbox", note=self.vault_rel(path), count=entry["batch"])
        with self.state_lock:
            if self.focus and self.focus.note == path and entry["focus_after"] is not None \
                    and self.focus.bullets == entry["focus_after"] and entry["prev_focus"] is not None:
                self.focus.bullets = list(entry["prev_focus"])
                info = self.focus.describe(self.vault_root)
            else:
                info = None
        if info:
            emit("focus", **info)
        self.total_notes_written = max(0, self.total_notes_written - 1)
        logger.info(f"[UNDO] {path.name}: {how}")
        if how == "nothing":
            emit("warning", message="The last change was already edited away; nothing to undo.")
        emit("undone", note=self.vault_rel(path), mode=how, thought=entry["thought"], ok=bool(ok),
             removed=entry["added"], reverted=entry["updated"])
        return bool(ok)

    # ---- batch organisation ----------------------------------------------------
    @property
    def batch_mode(self) -> bool:
        return str(self.config.organise_mode).lower() == "batch"

    @property
    def toc_wanted(self) -> bool:
        return bool(self.config.toc_enabled) and self.batch_mode

    def organise_note(self, path: Path, reason: str = "interval", batch_fn=None) -> bool:
        """Organise the inbox of one note (batch mode, or on request in any mode)."""
        try:
            text = path.read_text(encoding="utf-8") if path.exists() else ""
        except OSError:
            return False
        pending = inbox_bullets(StructuredNote.parse(text, path.stem))
        if not pending:
            self.inbox_pending.pop(path, None)
            return False
        emit("organising_batch", note=self.vault_rel(path), count=len(pending), reason=reason)
        logger.info(f"[BATCH] Organising {len(pending)} thought(s) in {path.name} ({reason})")
        linker = None
        if self.config.autolink_enabled:
            linker = lambda t, _ex={path.stem}: link_mentions(t, self.note_index.titles(), _ex)
        steps = []
        if self.config.properties_enabled:
            today = datetime.now().strftime("%Y-%m-%d")
            steps.append(lambda out: update_properties(out, date_str=today, add_session=False, add_seconds=0,
                                                       tag=self.config.properties_tag.strip()))
        if self.toc_wanted:
            steps.append(upsert_toc)
        postprocess = None
        if steps:
            def postprocess(out, _steps=steps):
                for f in _steps:
                    out = f(out)
                return out
        started = time.time()
        change = organise_inbox(path, path.stem, self.config, batch_fn=batch_fn, linker=linker, postprocess=postprocess)
        if not change:
            emit("batch_failed", note=self.vault_rel(path), count=len(pending))
            # try again later instead of hammering a busy/missing model
            self.inbox_pending[path] = {"count": len(pending), "since": time.time()}
            return False
        self.inbox_pending.pop(path, None)
        if change.before != change.after:
            self.undo_stack.append({"path": path, "before": change.before, "after": change.after,
                                    "added": list(change.added), "updated": list(change.updated),
                                    "prev_focus": None, "focus_after": None,
                                    "thought": f"organising {len(pending)} thoughts", "batch": len(pending)})
        emit("batch_done", note=self.vault_rel(path), organised=len(pending), added=change.added,
             updated=change.updated, headings=sorted(set(change.headings)), seconds=round(time.time() - started, 1))
        emit("inbox", note=self.vault_rel(path), count=0)
        return True

    def organise_due(self, force: bool = False, reason: str = "interval") -> int:
        if not (self.batch_mode or force):
            return 0
        interval = float(self.config.batch_interval_min or 0) * 60.0
        n = 0
        for path, pend in list(self.inbox_pending.items()):
            if pend["count"] <= 0:
                continue
            if force or (interval > 0 and time.time() - pend["since"] >= interval):
                if self.organise_note(path, reason):
                    n += 1
        return n

    # ---- session summary -------------------------------------------------------
    def summary_heading(self) -> str:
        return f"## 🧾 Session summary · {self.session_start.strftime('%Y-%m-%d %H:%M')}"

    def summarize_note(self, path: Path, summary_fn=None) -> bool:
        thoughts = list(self.session_thoughts.get(path) or [])
        if len(thoughts) < max(1, int(self.config.summary_min_thoughts)) or not self.config.llm_enabled:
            return False
        emit("summarizing", note=self.vault_rel(path), thoughts=len(thoughts))
        try:
            data = (summary_fn or ask_llm_for_summary)(self.config, path.stem, [t["text"] for t in thoughts])
        except Exception as e:
            emit("warning", message=f"Could not write the session summary (Ollama: {e}).")
            return False
        heading = self.summary_heading()
        section = render_summary_section(heading, data, thoughts)
        if not section:
            logger.info(f"No usable summary for {path.name}")
            return False
        with FILE_WRITE_LOCK:
            try:
                current = path.read_text(encoding="utf-8") if path.exists() else ""
            except OSError:
                return False
            updated = upsert_section(current, heading, section)
            if self.toc_wanted:
                updated = upsert_toc(updated)
            ok = _write_text_safely(path, updated)
        if ok:
            logger.info(f"[SUMMARY] {path.name}")
            emit("summary_written", note=self.vault_rel(path), heading=heading.lstrip("# ").strip())
        return ok

    def summarize_all(self, reason: str = "stop", summary_fn=None) -> int:
        if not self.config.summary_enabled and reason == "stop":
            return 0
        n = 0
        for path in list(self.session_thoughts):
            if self.skip_summary:
                break
            if self.summarize_note(path, summary_fn):
                n += 1
        if reason == "requested" and n == 0:
            emit("warning", message=f"Nothing to summarise yet (needs at least {self.config.summary_min_thoughts} thoughts in a note, and Ollama).")
        return n

    def run(self):
        logger.info("Router thread started.")
        ensure_note_exists(self.get_active_filepath(), self.get_active_filepath().stem)
        self._organise_requested = False
        while True:
            if self._organise_requested:
                self._organise_requested = False
                active = self.get_active_filepath()
                self.inbox_pending.setdefault(active, {"count": 1, "since": 0})
                if not self.organise_due(force=True, reason="requested"):
                    emit("warning", message="Nothing in the inbox to organise.")
            try:
                item = self.transcription_queue.get(timeout=0.5)
            except queue.Empty:
                if self.upstream_done.is_set():
                    break
                self.organise_due()
                continue
            try:
                self.process(item)
            except Exception as e:
                logger.error(f"Error in router: {e}", exc_info=True)
                emit("error", message=f"Note writing error: {e}")
            finally:
                self.transcription_queue.task_done()
            try:
                self.organise_due()
            except Exception as e:
                logger.error(f"Batch organisation failed: {e}", exc_info=True)
        if self.inbox_pending and not self.skip_summary:
            try:
                self.organise_due(force=True, reason="stop")
            except Exception as e:
                logger.error(f"Batch organisation failed: {e}", exc_info=True)
        if not self.skip_summary:
            try:
                self.summarize_all("stop")
            except Exception as e:
                logger.error(f"Summary failed: {e}", exc_info=True)
        logger.info("Router thread stopped.")


class CommandReaderThread(threading.Thread):
    """JSON-lines commands from the Obsidian plugin on stdin (focus, stop, ...)."""

    def __init__(self, service: "VoiceScribeService", stream=None):
        super().__init__(name="CommandReaderThread", daemon=True)
        self.service = service
        self.stream = stream or sys.stdin

    def run(self):
        try:
            for line in self.stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    logger.warning(f"Ignoring malformed command: {line[:120]}")
                    continue
                if not isinstance(msg, dict):
                    continue
                cmd = str(msg.get("cmd", "")).lower()
                if cmd == "stop":
                    self.service.request_stop()
                    continue
                if cmd in ("pause", "resume"):
                    self.service.thread_audio.set_paused(cmd == "pause")
                    continue
                try:
                    self.service.thread_router.handle_command(msg)
                except Exception as e:
                    logger.error(f"Command failed: {e}", exc_info=True)
        except Exception as e:
            logger.debug(f"Command reader ended: {e}")
        # stdin closed: Obsidian went away -> finish and exit instead of lingering
        if not self.service.stop_event.is_set():
            logger.info("Obsidian connection closed; finishing session.")
            self.service.request_stop()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
STOP_SIGNAL_FILE = SCRIPT_DIR / "stop.signal"
PID_FILE = SCRIPT_DIR / "voicescribe.pid"


class VoiceScribeService:
    def __init__(self, config: ScribeConfig, ipc: bool = False):
        self.config = config
        self.ipc = ipc
        self.stop_event = threading.Event()
        self.fatal_event = threading.Event()
        self.transcriber_done = threading.Event()
        self._stop_requests = 0
        self._stop_lock = threading.Lock()
        self.audio_queue: queue.Queue = queue.Queue(maxsize=200)
        self.transcription_queue: queue.Queue = queue.Queue(maxsize=500)
        self.start_time = time.time()
        self.shared_context = SharedContext(default_topic=self.config.default_note)

        self.thread_audio = AudioIngestionThread(self.audio_queue, self.stop_event, self.config,
                                                 session_start=self.start_time)
        self.thread_transcribe = TranscriptionThread(self.audio_queue, self.transcription_queue, self.stop_event,
                                                     self.config, self.shared_context, self.fatal_event)
        self.thread_router = RouterSynthesisThread(self.transcription_queue, self.stop_event, self.config,
                                                   self.shared_context, upstream_done=self.transcriber_done)
        self.thread_commands = CommandReaderThread(self) if ipc else None

    # A first request finishes pending work; a second one exits right away.
    def request_stop(self):
        with self._stop_lock:
            self._stop_requests += 1
        self.stop_event.set()

    def _skip_waiting(self) -> bool:
        return self._stop_requests >= 2 or STOP_SIGNAL_FILE.exists()

    def stats(self) -> dict:
        return dict(
            chunks=self.thread_audio.total_chunks_captured,
            transcribed=self.thread_transcribe.total_transcribed,
            notes_written=self.thread_router.total_notes_written,
            pending=self.audio_queue.qsize() + self.transcription_queue.qsize(),
        )

    def start(self) -> int:
        logger.info("=" * 60)
        logger.info("VOICESCRIBE: continuous voice-to-notes for Obsidian")
        logger.info(f"Vault        : {Path(self.config.vault_dir).resolve()}")
        logger.info(f"Whisper      : {self.config.whisper_model} (beam={self.config.whisper_beam_size})")
        logger.info(f"Organiser    : {self.config.ollama_model if self.config.llm_enabled else 'disabled'}")
        logger.info(f"Pause / gate : {self.config.pause_duration_sec}s / RMS {self.config.energy_threshold}")
        logger.info("=" * 60)

        try:
            if STOP_SIGNAL_FILE.exists():
                STOP_SIGNAL_FILE.unlink()
            PID_FILE.write_text(str(os.getpid()))
        except OSError:
            pass

        emit("started",
             note=self.thread_router.vault_rel(self.thread_router.get_active_filepath()),
             session_start=self.start_time,
             threshold=self.config.energy_threshold,
             whisper=self.config.whisper_model,
             llm=self.config.ollama_model if self.config.llm_enabled else None)
        if self.thread_commands:
            self.thread_commands.start()

        if self.config.llm_enabled:
            if ensure_ollama_running(self.config.ollama_host) and str(self.config.organise_mode).lower() != "batch":
                # warm the live model while Whisper loads, so the first thought isn't slow
                threading.Thread(target=self._preload_llm, daemon=True, name="LLMPreload").start()

        self.thread_transcribe.start()
        self.thread_router.start()
        self.thread_audio.start()

        def _handle_exit(sig, frame):
            logger.info("Exit requested.")
            self.request_stop()

        signal.signal(signal.SIGINT, _handle_exit)
        try:
            signal.signal(signal.SIGTERM, _handle_exit)
        except Exception:
            pass

        last_stats = 0.0
        try:
            while not self.stop_event.is_set():
                time.sleep(0.3)
                if STOP_SIGNAL_FILE.exists():
                    logger.info("Stop requested from Obsidian.")
                    try:
                        STOP_SIGNAL_FILE.unlink()
                    except OSError:
                        pass
                    self.request_stop()
                    break
                if time.time() - last_stats >= 1.0:
                    last_stats = time.time()
                    emit("stats", **self.stats())
        except KeyboardInterrupt:
            pass
        self.stop()
        return 1 if self.fatal_event.is_set() else 0

    def _preload_llm(self):
        try:
            ensure_model_loaded(self.config, self.config.ollama_model)
        except Exception as e:
            logger.info(f"Could not preload the Ollama model: {e}")

    def stop(self, max_wait_sec: float = 600.0):
        self.stop_event.set()
        deadline = time.time() + max_wait_sec

        self.thread_audio.join(timeout=5.0)
        pending = self.audio_queue.qsize() + self.transcription_queue.qsize()
        if pending:
            logger.info(f"Finishing {pending} pending thought(s) before exit...")
        emit("finishing", pending=pending)

        last_stats = 0.0
        while self.thread_transcribe.is_alive() and time.time() < deadline:
            self.thread_transcribe.join(timeout=0.5)
            if time.time() - last_stats >= 1.0:
                last_stats = time.time()
                emit("stats", **self.stats())
            if self._skip_waiting():
                logger.warning("Second stop request - exiting without finishing.")
                break
        self.transcriber_done.set()
        while self.thread_router.is_alive() and time.time() < deadline:
            self.thread_router.join(timeout=0.5)
            if self._skip_waiting():
                self.thread_router.skip_summary = True
                break

        for f in (STOP_SIGNAL_FILE, PID_FILE):
            try:
                f.unlink()
            except OSError:
                pass

        if self.config.llm_enabled and self.config.unload_on_stop:
            try:
                unload_models(self.config)
            except Exception as e:
                logger.info(f"Could not unload Ollama models: {e}")

        summary = dict(duration_sec=int(time.time() - self.start_time), **self.stats())
        summary.pop("pending", None)
        logger.info(f"Session summary: {summary}")
        emit("stopped", **summary)


# ---------------------------------------------------------------------------
# Importing a recording (voice memo, meeting audio, ...)
# ---------------------------------------------------------------------------
IMPORT_EXTENSIONS = (".mp3", ".m4a", ".wav", ".ogg", ".opus", ".webm", ".flac", ".aac", ".mp4", ".mov", ".wma", ".amr", ".3gp")


def group_segments(segments: List[Tuple[float, float, str]], pause: float, max_len: float) -> List[Tuple[float, float, str]]:
    """Merge Whisper segments into 'thoughts': split at pauses >= `pause` or when longer than `max_len`."""
    out: List[Tuple[float, float, str]] = []
    cur = None
    for start, end, text in segments:
        text = text.strip()
        if not text:
            continue
        if cur and (start - cur[1] < pause) and (end - cur[0] <= max_len):
            cur = (cur[0], end, (cur[2] + " " + text).strip())
        else:
            if cur:
                out.append(cur)
            cur = (start, end, text)
    if cur:
        out.append(cur)
    return out


class ImportService:
    """Runs one audio file through the normal pipeline: clips, transcript, organised note, summary."""

    def __init__(self, config: ScribeConfig, audio_path: str, ipc: bool = False):
        self.config = config
        self.audio_path = Path(audio_path)
        self.stop_event = threading.Event()
        self.fatal_event = threading.Event()
        self.transcriber_done = threading.Event()
        self._stop_requests = 0
        self.start_time = time.time()
        self.audio_queue: queue.Queue = queue.Queue()
        self.transcription_queue: queue.Queue = queue.Queue()
        self.shared_context = SharedContext(default_topic=self.config.default_note)
        self.thread_transcribe = TranscriptionThread(self.audio_queue, self.transcription_queue, self.stop_event,
                                                     self.config, self.shared_context, self.fatal_event)
        self.thread_router = RouterSynthesisThread(self.transcription_queue, self.stop_event, self.config,
                                                   self.shared_context, upstream_done=self.transcriber_done)
        self.thread_audio = types_simple(total_chunks_captured=0, set_paused=lambda p: None)
        self.thread_commands = CommandReaderThread(self) if ipc else None

    def request_stop(self):
        self._stop_requests += 1
        self.stop_event.set()
        if self._stop_requests >= 2:
            self.thread_router.skip_summary = True

    def run(self) -> int:
        p = self.audio_path
        if not p.exists():
            emit("error", message=f"Recording not found: {p}")
            return 1
        logger.info(f"Importing recording: {p}")
        if self.thread_commands:
            self.thread_commands.start()
        try:
            from faster_whisper import decode_audio
            audio = decode_audio(str(p), sampling_rate=self.config.sample_rate).astype(np.float32)
        except Exception as e:
            emit("error", message=f"Could not read {p.name}: {e}")
            return 1
        sr = self.config.sample_rate
        total = len(audio) / sr
        # Recording start: file time minus its length (phones save memos when recording ends).
        try:
            base = datetime.fromtimestamp(p.stat().st_mtime - total)
        except OSError:
            base = datetime.now()
        self.thread_router.session_start = base
        step = 0.5
        hop = int(sr * step)
        env = [round(float(np.max(np.abs(audio[i:i + hop]))) * 0.35, 5) for i in range(0, len(audio), hop)]
        emit("started", note=self.thread_router.vault_rel(self.thread_router.get_active_filepath()),
             session_start=self.start_time, threshold=self.config.energy_threshold,
             whisper=self.config.whisper_model, llm=self.config.ollama_model if self.config.llm_enabled else None,
             mode="import", file=p.name, duration=round(total, 1))
        emit("envelope", step=step, values=env, duration=round(total, 1))
        if self.config.llm_enabled:
            ensure_ollama_running(self.config.ollama_host)

        tt = self.thread_transcribe
        try:
            tt._init_whisper()
        except Exception as e:
            emit("error", message=f"Whisper failed to load: {e}")
            return 1
        self.thread_router.start()

        lang = (self.config.whisper_language or "").strip().lower()
        lang = None if lang in ("", "auto") else lang
        emit("import_progress", done=0.0, total=round(total, 1))
        raw: List[Tuple[float, float, str]] = []
        segments, _ = tt.whisper_model.transcribe(
            audio, beam_size=int(self.config.whisper_beam_size), language=lang, vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500), condition_on_previous_text=False)
        pending: List[Tuple[float, float, str]] = []
        seg_id = 0
        archive_root = Path(self.config.vault_dir).resolve() / self.config.archive_dir

        def flush(groups):
            nonlocal seg_id
            for start, end, text in groups:
                if is_whisper_hallucination(text):
                    continue
                seg_id += 1
                a = max(0, int((start - 0.2) * sr))
                b = min(len(audio), int((end + 0.3) * sr))
                clip = normalize_audio(audio[a:b]) if self.config.normalize_audio else audio[a:b]
                ts = base + timedelta(seconds=start)
                out = unique_path(archive_root / ts.strftime("%Y-%m-%d") / f"audio_{ts.strftime('%Y%m%d_%H%M%S')}.ogg")
                ok = archive_audio_opus(clip, sr, out, self.config.opus_bitrate)
                rel = self.thread_router.vault_rel(out) if ok else ""
                emit("segment", seg=seg_id, t0=round(start, 2), dur=round(end - start, 2), clip=rel, speech=True)
                tt.total_transcribed += 1
                self.thread_audio.total_chunks_captured += 1
                self.transcription_queue.put(TranscribedChunk(timestamp=ts, text=text, archive_path=out if ok else None,
                                                              duration=end - start, seg_id=seg_id))

        for seg in segments:
            if self.stop_event.is_set():
                break
            if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:
                continue
            raw.append((seg.start, seg.end, seg.text))
            emit("import_progress", done=round(seg.end, 1), total=round(total, 1))
            # hand complete thoughts to the router as soon as a long pause is seen
            groups = group_segments(raw, self.config.pause_duration_sec, self.config.max_speech_duration_sec)
            if len(groups) > 1:
                flush(groups[:-1])
                last = groups[-1]
                raw = [(last[0], last[1], last[2])]
            emit("stats", chunks=self.thread_audio.total_chunks_captured, transcribed=tt.total_transcribed,
                 notes_written=self.thread_router.total_notes_written, pending=self.transcription_queue.qsize())
        if raw and not self.stop_event.is_set():
            flush(group_segments(raw, self.config.pause_duration_sec, self.config.max_speech_duration_sec))
        emit("import_progress", done=round(total, 1), total=round(total, 1))
        emit("finishing", pending=self.transcription_queue.qsize())
        self.transcriber_done.set()
        while self.thread_router.is_alive():
            self.thread_router.join(timeout=0.5)
            emit("stats", chunks=self.thread_audio.total_chunks_captured, transcribed=tt.total_transcribed,
                 notes_written=self.thread_router.total_notes_written, pending=self.transcription_queue.qsize())
        summary = dict(duration_sec=int(time.time() - self.start_time), chunks=self.thread_audio.total_chunks_captured,
                       transcribed=tt.total_transcribed, notes_written=self.thread_router.total_notes_written)
        if self.config.llm_enabled and self.config.unload_on_stop:
            try:
                unload_models(self.config)
            except Exception as e:
                logger.info(f"Could not unload Ollama models: {e}")
        logger.info(f"Import finished: {summary}")
        emit("stopped", **summary)
        return 0


def types_simple(**kw):
    import types
    return types.SimpleNamespace(**kw)


# ---------------------------------------------------------------------------
# Microphone calibration (used by the Obsidian settings tab)
# ---------------------------------------------------------------------------
def calibrate(config: ScribeConfig, seconds: float = 3.0) -> dict:
    import sounddevice as sd
    sr = config.sample_rate
    device = config.audio_device
    if isinstance(device, str) and device.strip().lstrip("-").isdigit():
        device = int(device)
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32", device=device)
    sd.wait()
    data = rec[:, 0]
    blocks = data[: len(data) // 1024 * 1024].reshape(-1, 1024)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    noise_p90 = float(np.percentile(rms, 90))
    suggested = round(max(noise_p90 * 2.5, 0.0008), 4)
    return {"noise_median": round(float(np.median(rms)), 5), "noise_p90": round(noise_p90, 5),
            "peak": round(float(np.max(np.abs(data))), 4), "suggested_threshold": suggested}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VoiceScribe: local voice-to-notes for Obsidian.")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.json"))
    parser.add_argument("--vault-dir")
    parser.add_argument("--notes-subfolder")
    parser.add_argument("--default-note")
    parser.add_argument("--note", help="Vault-relative note to write to this session (created if missing)")
    parser.add_argument("--whisper-model")
    parser.add_argument("--ollama-model")
    parser.add_argument("--device", help="Audio input device index or name")
    parser.add_argument("--calibrate", action="store_true", help="Measure background noise and print JSON")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--menu", "-m", action="store_true", help="Interactive configuration menu")
    parser.add_argument("--gpu-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-compute", default="int8_float16", help=argparse.SUPPRESS)
    parser.add_argument("--allow-downloads", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-status", action="store_true", help="Print GPU/CUDA status as JSON")
    parser.add_argument("--import-audio", help="Transcribe and organise an audio file instead of the microphone")
    parser.add_argument("--ipc", action="store_true", help="Read JSON commands from stdin (used by the Obsidian plugin)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.gpu_probe:  # child process: the folder comes from VOICESCRIBE_CUDA_DIR set by the parent
        return run_gpu_probe_main(args.whisper_model or "large-v3-turbo", args.gpu_compute, args.allow_downloads)
    if args.gpu_status:
        st_cfg = ScribeConfig.load(args.config)
        if st_cfg.cuda_libs_dir:
            os.environ["VOICESCRIBE_CUDA_DIR"] = st_cfg.cuda_libs_dir
        return gpu_status_main()

    if args.menu:
        from config_menu import interactive_menu
        interactive_menu()
        return 0

    cfg = ScribeConfig.load(args.config)
    if cfg.cuda_libs_dir:
        os.environ["VOICESCRIBE_CUDA_DIR"] = cfg.cuda_libs_dir
    if args.vault_dir:
        cfg.vault_dir = args.vault_dir
    if args.notes_subfolder is not None:
        cfg.notes_subfolder = args.notes_subfolder
    if args.default_note:
        cfg.default_note = args.default_note
    if args.note:
        cfg.start_note = args.note
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.ollama_model:
        cfg.ollama_model = args.ollama_model
    if args.device:
        cfg.audio_device = int(args.device) if args.device.lstrip("-").isdigit() else args.device
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.calibrate:
        logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)
                           or isinstance(h, logging.FileHandler)]
        try:
            print(json.dumps(calibrate(cfg, args.seconds)))
            return 0
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            return 1

    if args.import_audio:
        return ImportService(cfg, args.import_audio, ipc=args.ipc).run()
    return VoiceScribeService(cfg, ipc=args.ipc).start()


if __name__ == "__main__":
    sys.exit(main() or 0)
