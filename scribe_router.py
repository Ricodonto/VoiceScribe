#!/usr/bin/env python3
"""
================================================================================
VoiceScribe: Continuous Voice-to-Notes Transcription & Routing for Obsidian
================================================================================
Hardware Target: Windows 11, RTX 3060 Laptop (6 GB VRAM), 32 GB RAM.
Allocation:
  - Whisper: Runs on CPU (int8 quantization via faster-whisper) to conserve VRAM.
  - LLM: Runs on GPU via local Ollama (llama3.2:3b).
Vault Target: Direct atomic updates to Markdown (.md) notes in Obsidian Vault.

Architecture:
  - Thread 1: Continuous Audio Ingestion (sounddevice, 16 kHz mono, resilient driver loop)
  - Thread 2: Transcription (faster-whisper CPU int8) & Storage Compression (FFmpeg Opus @ 10 kbps)
  - Thread 3: Intelligent Router & Synthesis (Filler stripping, keyword routing, Ollama distillation, Obsidian atomic writer)
================================================================================
"""

import argparse
import collections
import dataclasses
from datetime import datetime
import json
import logging
import math
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
from typing import List, Optional, Tuple

import numpy as np

# Windows UTF-8 console output hardening
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Windows subprocess creation flag to avoid flashing console windows
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
class ColoredFormatter(logging.Formatter):
    """Clean, colored console output with timestamps."""
    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: GREY + "%(asctime)s [DEBUG] %(message)s" + RESET,
        logging.INFO: CYAN + "%(asctime)s" + RESET + " %(message)s",
        logging.WARNING: YELLOW + "%(asctime)s [WARN] %(message)s" + RESET,
        logging.ERROR: RED + "%(asctime)s [ERROR] %(message)s" + RESET,
        logging.CRITICAL: BOLD_RED + "%(asctime)s [CRITICAL] %(message)s" + RESET,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, "%(asctime)s %(message)s")
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
logger = logging.getLogger("VoiceScribe")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False


# ---------------------------------------------------------------------------
# Configuration Model
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ScribeConfig:
    vault_dir: str = "../Note Rambling"
    notes_subfolder: str = ""
    archive_dir: str = "VoiceScribe_Archives"
    transcripts_folder: str = "Transcripts"
    default_note: str = "Voice Notes"
    sample_rate: int = 16000
    pause_duration_sec: float = 2.2
    min_speech_duration_sec: float = 1.5
    max_speech_duration_sec: float = 45.0
    whisper_model: str = "large-v3-turbo"
    whisper_compute_type: str = "int8"
    whisper_threads: int = 4
    whisper_beam_size: int = 3
    context_prompt_enabled: bool = True
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_num_predict: int = 384
    ollama_num_ctx: int = 2048
    opus_bitrate: str = "10k"
    min_substantive_words: int = 3
    energy_threshold: float = 0.003
    audio_device: Optional[int | str] = None

    @classmethod
    def load(cls, config_path: str = "config.json") -> "ScribeConfig":
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
            except Exception as e:
                logger.warning(f"Could not parse {config_path}: {e}. Using defaults.")
        return cls()


# ---------------------------------------------------------------------------
# Shared Context Provider for Whisper & Synthesis
# ---------------------------------------------------------------------------
class SharedContext:
    """
    Thread-safe context provider linking the active topic and recent transcription
    history to Whisper's initial_prompt and Ollama's thematic reasoning.
    """
    def __init__(self, default_topic: str = "Voice Notes", max_history: int = 3):
        self.active_topic: str = default_topic
        self.recent_transcripts: collections.deque = collections.deque(maxlen=max_history)
        self._lock = threading.Lock()

    def update_topic(self, topic: str):
        with self._lock:
            self.active_topic = topic.strip()

    def add_transcript(self, text: str):
        with self._lock:
            cleaned = text.strip()
            if cleaned:
                self.recent_transcripts.append(cleaned)

    def get_initial_prompt(self) -> Optional[str]:
        with self._lock:
            topic = self.active_topic
            history = list(self.recent_transcripts)

        parts = []
        if topic:
            parts.append(f"Topic: {topic}.")
        if history:
            parts.append(" ".join(history[-2:]))

        if not parts:
            return None
        prompt = " ".join(parts).strip()
        tokens = prompt.split()
        if len(tokens) > 150:
            prompt = " ".join(tokens[-150:])
        return prompt


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class AudioChunk:
    timestamp: datetime
    data: np.ndarray  # 1D float32 array, 16000 Hz
    duration: float
    rms_energy: float


@dataclasses.dataclass
class TranscribedChunk:
    timestamp: datetime
    text: str
    archive_path: Optional[Path]
    duration: float


# ---------------------------------------------------------------------------
# Text Pre-Processing, Commands & Filename Sanitization
# ---------------------------------------------------------------------------
COMMAND_PATTERNS = [
    re.compile(
        r"^(?:topic|switch\s+(?:topic\s+)?to|note)(?:\s*[:\-]\s*|\s+)(.+)$",
        re.IGNORECASE
    ),
]

VERBAL_FILLERS_REGEX = re.compile(
    r"\b(um+|uh+|you\s+know|like|sort\s+of|kind\s+of|i\s+mean|basically|actually|er+|ah+)\b",
    re.IGNORECASE,
)

WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
}


def parse_voice_command(text: str) -> Optional[str]:
    """
    Detect voice commands like:
      - 'Topic Algorand'
      - 'Switch to Flutter'
      - 'Note French toast'
      - 'Switch topic to Note Rambling.'
    Returns the target topic name if matched, else None.
    """
    stripped = text.strip()
    for pattern in COMMAND_PATTERNS:
        match = pattern.match(stripped)
        if match:
            target = match.group(1).strip()
            # Strip trailing punctuation (Whisper frequently attaches periods/commas)
            target = re.sub(r"[\.\?!,:;]+$", "", target).strip()
            if target:
                return target
    return None


def sanitize_filename(name: str) -> str:
    """
    Sanitize a topic string into a valid, safe Windows filename for Obsidian:
      - Replaces illegal Windows characters (\\ / : * ? " < > |) with space
      - Strips non-printable control characters
      - Collapses multiple spaces and strips trailing dots/spaces
      - Protects against reserved Windows device names (CON, NUL, COM1, etc.)
      - Ensures .md extension
    """
    # Replace Windows forbidden characters with a space
    cleaned = re.sub(r'[\\/:*?"<>|]', " ", name)
    # Remove control characters
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", cleaned)
    # Collapse multiple spaces and strip outer dots/spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip(". ")

    if not cleaned:
        cleaned = "Voice Notes"

    # Avoid Windows reserved device names
    base_upper = cleaned.split(".")[0].upper()
    if base_upper in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_note"

    # Add .md extension if missing
    if not cleaned.lower().endswith(".md"):
        cleaned += ".md"

    # Windows MAX_PATH safety: keep filename within reasonable length
    if len(cleaned) > 100:
        cleaned = cleaned[:96] + ".md"

    return cleaned


def strip_verbal_fillers(text: str) -> str:
    """
    Remove verbal fillers ('um', 'uh', 'you know', 'like', etc.)
    and clean up dangling punctuation or extra whitespace.
    """
    cleaned = VERBAL_FILLERS_REGEX.sub(" ", text)
    # Clean dangling commas/punctuation created by filler removal (e.g., ", , " or leading ", ")
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"^[,\.\s]+", "", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def count_substantive_words(text: str) -> int:
    """
    Count substantive words (ignoring purely numeric tokens or single-letter noise).
    """
    words = re.findall(r"\b[a-zA-Z0-9_\-']+\b", text)
    # Filter out single-letter fragments (except meaningful ones like 'a' or 'i')
    substantive = [w for w in words if len(w) > 1 or w.lower() in ("a", "i")]
    return len(substantive)


# ---------------------------------------------------------------------------
# Storage Compression: Opus via FFmpeg
# ---------------------------------------------------------------------------
def archive_audio_opus(
    audio_data: np.ndarray,
    sample_rate: int,
    output_path: Path,
    bitrate: str = "10k"
) -> bool:
    """
    Archive raw 16kHz float32 audio to Opus format at ~10 kbps using FFmpeg.
    Ensures 10+ hours of speech consumes < 50 MB total.
    Uses stdin pipe directly to prevent uncompressed WAV writes on disk.
    If pipe fails, falls back to a temp WAV with immediate deletion.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_wav_path: Optional[Path] = None

    try:
        # Preferred method: Stream raw PCM float32 directly via stdin
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "f32le",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-b:a", bitrate,
            "-vbr", "on",
            "-application", "voip",
            str(output_path),
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
        _, err = proc.communicate(input=audio_data.tobytes())
        if proc.returncode == 0 and output_path.exists():
            return True

        logger.debug(f"Direct pipe FFmpeg returned {proc.returncode}. Trying temp file fallback.")
    except Exception as e:
        logger.debug(f"Direct pipe failed ({e}). Using temporary WAV fallback.")

    # Fallback method: Write temporary WAV and immediately remove it
    try:
        import scipy.io.wavfile as wavfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            temp_wav_path = Path(tmp_wav.name)

        wavfile.write(str(temp_wav_path), sample_rate, audio_data)

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(temp_wav_path),
            "-c:a", "libopus",
            "-b:a", bitrate,
            "-vbr", "on",
            "-application", "voip",
            str(output_path),
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
        return res.returncode == 0 and output_path.exists()
    except Exception as e:
        logger.error(f"Failed to encode Opus audio via FFmpeg: {e}")
        return False
    finally:
        # Guarantee removal of uncompressed temporary WAV immediately
        if temp_wav_path and temp_wav_path.exists():
            try:
                os.remove(temp_wav_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Obsidian Atomic Writing
# ---------------------------------------------------------------------------
FILE_WRITE_LOCK = threading.Lock()


def atomic_append_note(
    filepath: Path,
    bullet_text: str,
    timestamp_str: Optional[str] = None,
    max_retries: int = 3
) -> bool:
    """
    Atomically appends a distilled bullet to the target markdown note.
    Appends with format: - [HH:MM:SS] <text>
    Flushes and fsyncs immediately, avoiding file lock contention with
    Obsidian file watchers, OneDrive, or MEGA sync engines.
    """
    if timestamp_str is None:
        timestamp_str = datetime.now().strftime("%H:%M:%S")

    filepath.parent.mkdir(parents=True, exist_ok=True)
    clean_text = bullet_text.strip()
    # Strip any accidental bullet symbol from LLM output
    clean_text = re.sub(r"^[\-\*\•]\s*", "", clean_text)
    entry_line = f"- [{timestamp_str}] {clean_text}\n"

    for attempt in range(max_retries):
        try:
            with FILE_WRITE_LOCK:
                is_new_or_empty = not filepath.exists() or filepath.stat().st_size == 0
                with open(filepath, "a", encoding="utf-8") as f:
                    if is_new_or_empty:
                        # Write clean Obsidian header
                        topic_title = filepath.stem.replace("_", " ")
                        date_str = datetime.now().strftime("%Y-%m-%d")
                        header = (
                            f"# {topic_title}\n\n"
                            f"*VoiceScribe Session: {date_str}*\n\n"
                            f"---\n\n"
                        )
                        f.write(header)
                    f.write(entry_line)
                    f.flush()
                    os.fsync(f.fileno())
            return True
        except (PermissionError, OSError) as e:
            # File might be temporarily locked by Obsidian watcher or OneDrive/MEGA sync
            if attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))
            else:
                logger.error(f"Failed to write to {filepath} after {max_retries} attempts: {e}")
                return False
    return False


def append_session_transcript(
    vault_dir: Path,
    transcripts_folder: str,
    topic_title: str,
    chunk_id: str,
    timestamp_str: str,
    text: str,
    audio_rel_path: str
) -> bool:
    """
    Appends verbatim spoken thought to the dedicated daily session transcript file:
    vault_dir / transcripts_folder / YYYY-MM-DD_Transcript.md
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    trans_dir = vault_dir / transcripts_folder
    trans_dir.mkdir(parents=True, exist_ok=True)
    trans_file = trans_dir / f"{date_str}_Transcript.md"

    audio_embed = f"\n> ![[{audio_rel_path}]]" if audio_rel_path else ""
    block_entry = (
        f"### Topic: {topic_title}\n"
        f"> **[{timestamp_str}]** {text} ^{chunk_id}{audio_embed}\n\n"
    )

    with FILE_WRITE_LOCK:
        try:
            is_new = not trans_file.exists() or trans_file.stat().st_size == 0
            with open(trans_file, "a", encoding="utf-8") as f:
                if is_new:
                    f.write(f"# Session Transcript: {date_str}\n\n---\n\n")
                f.write(block_entry)
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            logger.error(f"Failed to append to transcript {trans_file}: {e}")
            return False


def integrate_note_thematically(
    note_filepath: Path,
    topic_title: str,
    new_statement: str,
    citation_tag: str,
    config: ScribeConfig,
    timeout_sec: float = 12.0
) -> bool:
    """
    Integrates the new spoken statement into the structured thematic notes:
    - Alters/updates existing lines if the statement corrects or elaborates.
    - Groups related points under ### thematic headers.
    - Strictly preserves all other sections and topics without deleting them.
    - Grounded exclusively in the spoken speech.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    existing_content = ""
    if note_filepath.exists():
        try:
            existing_content = note_filepath.read_text(encoding="utf-8")
        except Exception:
            existing_content = ""

    # Parse existing notes section under "## 💡 Notes"
    notes_header_pattern = re.compile(r"## 💡 Notes\s*\n", re.IGNORECASE)
    match = notes_header_pattern.search(existing_content)

    prefix = ""
    current_notes = ""
    suffix = ""

    if match:
        prefix = existing_content[:match.end()]
        rest = existing_content[match.end():]
        # Check if there is another ## section after notes
        next_sec = re.search(r"\n## [^#]", rest)
        if next_sec:
            current_notes = rest[:next_sec.start()].strip()
            suffix = rest[next_sec.start():]
        else:
            current_notes = rest.strip()
    else:
        prefix = f"# {topic_title}\n\n*Session Date: {date_str}*\n\n## 💡 Notes\n\n"
        current_notes = existing_content.strip()

    # Synthesize updated notes section
    if not current_notes or current_notes == "*(Listening for thoughts...)*":
        updated_notes = f"### General\n- {new_statement} {citation_tag}"
    else:
        prompt = (
            "You are a faithful note synthesizer. Integrate the new spoken statement into the current notes.\n\n"
            f"CURRENT NOTES:\n{current_notes}\n\n"
            f"NEW SPOKEN STATEMENT:\n- \"{new_statement}\" {citation_tag}\n\n"
            "RULES:\n"
            "1. If the new statement updates, refines, or corrects an existing bullet point, revise that point and append the new citation tag.\n"
            "2. If it is a new point under an existing theme, add it under that ### heading.\n"
            "3. If it introduces a new theme, create a new ### heading.\n"
            "4. CRITICAL: Preserve all existing points and headers about other topics. Do NOT delete unrelated notes.\n"
            "5. Rely ONLY on the spoken statements. Do NOT add outside facts, trivia, or explanations.\n\n"
            "OUTPUT UPDATED NOTES:"
        )

        try:
            import ollama
            client = ollama.Client(host=config.ollama_host, timeout=timeout_sec)
            response = client.generate(
                model=config.ollama_model,
                prompt=prompt,
                options={
                    "num_predict": config.ollama_num_predict,
                    "num_ctx": config.ollama_num_ctx,
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            )
            raw_res = response.get("response", "").strip()
            if raw_res and not any(phrase in raw_res.lower() for phrase in ("i don't see", "i need more context", "please provide", "as an ai")):
                updated_notes = raw_res
            else:
                updated_notes = f"{current_notes}\n- {new_statement} {citation_tag}"
        except Exception as e:
            logger.warning(f"Ollama integration call failed ({e}). Appending bullet.")
            updated_notes = f"{current_notes}\n- {new_statement} {citation_tag}"

    full_doc = f"{prefix.rstrip()}\n\n{updated_notes.strip()}\n{suffix}"

    with FILE_WRITE_LOCK:
        try:
            with open(note_filepath, "w", encoding="utf-8") as f:
                f.write(full_doc)
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            logger.error(f"Failed to update note {note_filepath}: {e}")
            return False


def ensure_ollama_running(host: str = "http://127.0.0.1:11434") -> bool:
    """Verify Ollama is reachable; if down, launch 'ollama serve' in background."""
    import urllib.request
    try:
        req = urllib.request.Request(f"{host}/api/tags", headers={"User-Agent": "VoiceScribe"})
        with urllib.request.urlopen(req, timeout=1.5) as res:
            if res.status == 200:
                logger.info(f"✅ Ollama server is responsive at {host}.")
                return True
    except Exception:
        pass

    logger.warning(f"⚠️ Ollama server at {host} is not running. Starting background service...")
    try:
        import shutil
        ollama_bin = shutil.which("ollama")
        if not ollama_bin:
            local_app = os.environ.get("LOCALAPPDATA", "")
            candidate = Path(local_app) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.exists():
                ollama_bin = str(candidate)

        if ollama_bin:
            flags = CREATE_NO_WINDOW
            subprocess.Popen([ollama_bin, "serve"], creationflags=flags)
            # Wait up to 5s for Ollama to become responsive
            for _ in range(10):
                time.sleep(0.5)
                try:
                    req = urllib.request.Request(f"{host}/api/tags", headers={"User-Agent": "VoiceScribe"})
                    with urllib.request.urlopen(req, timeout=1.0) as res:
                        if res.status == 200:
                            logger.info("✅ Ollama server successfully started on GPU.")
                            return True
                except Exception:
                    continue
            logger.info("Ollama server process spawned.")
            return True
        else:
            logger.error("Could not find 'ollama.exe'. Ensure Ollama is installed.")
            return False
    except Exception as e:
        logger.error(f"Error auto-launching Ollama: {e}")
        return False


# ---------------------------------------------------------------------------
# Thread 1: Continuous Audio Ingestion
# ---------------------------------------------------------------------------
class AudioIngestionThread(threading.Thread):
    """
    Captures continuous microphone audio using sounddevice at 16 kHz mono.
    Slices audio into 6-8s intervals without dropping frames.
    Hardened against Windows audio driver hiccups and device disconnects.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        stop_event: threading.Event,
        config: ScribeConfig
    ):
        super().__init__(name="AudioIngestionThread", daemon=True)
        self.audio_queue = audio_queue
        self.stop_event = stop_event
        self.config = config
        self.raw_blocks: queue.Queue = queue.Queue()
        self.total_chunks_captured = 0

    def _audio_callback(self, indata, frames, time_info, status):
        """Non-blocking audio callback."""
        if status:
            logger.debug(f"Audio stream status: {status}")
        # Fast copy to avoid any memory sharing or blocking in PortAudio callback
        self.raw_blocks.put(indata[:, 0].copy())

    def run(self):
        import sounddevice as sd
        import collections
        logger.info(
            f"🎙️ Audio Ingestion Thread started (16 kHz, pause_threshold={self.config.pause_duration_sec}s)."
        )

        block_size = 1024
        block_duration = block_size / self.config.sample_rate  # ~0.064s
        pre_speech_blocks = collections.deque(maxlen=int(0.5 / block_duration))  # 0.5s pre-speech buffer

        is_speaking = False
        speech_blocks: List[np.ndarray] = []
        silence_seconds = 0.0
        thought_start_time = datetime.now()

        while not self.stop_event.is_set():
            try:
                # Resolve audio device
                device = self.config.audio_device
                device_name = "System Default"
                if device is not None:
                    try:
                        dev_info = sd.query_devices(device, "input")
                        device_name = dev_info.get("name", str(device))
                    except Exception:
                        device_name = str(device)

                logger.info(f"Connecting to microphone: {device_name}")

                with sd.InputStream(
                    samplerate=self.config.sample_rate,
                    channels=1,
                    dtype="float32",
                    device=device,
                    callback=self._audio_callback,
                    blocksize=block_size,
                ):
                    logger.info("🟢 Microphone active. Listening continuously...")

                    while not self.stop_event.is_set():
                        try:
                            block = self.raw_blocks.get(timeout=0.2)
                            rms = float(np.sqrt(np.mean(block ** 2)))

                            if rms >= self.config.energy_threshold:
                                # Active speech detected
                                if not is_speaking:
                                    is_speaking = True
                                    thought_start_time = datetime.now()
                                    speech_blocks = list(pre_speech_blocks)
                                speech_blocks.append(block)
                                silence_seconds = 0.0
                            else:
                                # Silence block
                                if is_speaking:
                                    speech_blocks.append(block)
                                    silence_seconds += block_duration
                                    speech_duration = len(speech_blocks) * block_duration

                                    is_pause_complete = silence_seconds >= self.config.pause_duration_sec
                                    is_max_reached = speech_duration >= self.config.max_speech_duration_sec

                                    if is_pause_complete or is_max_reached:
                                        if speech_duration >= self.config.min_speech_duration_sec:
                                            chunk_data = np.concatenate(speech_blocks)
                                            chunk = AudioChunk(
                                                timestamp=thought_start_time,
                                                data=chunk_data,
                                                duration=float(speech_duration),
                                                rms_energy=rms,
                                            )
                                            try:
                                                self.audio_queue.put(chunk, block=False)
                                                self.total_chunks_captured += 1
                                                logger.debug(
                                                    f"Captured coherent speech chunk: {speech_duration:.1f}s"
                                                )
                                            except queue.Full:
                                                logger.warning("Audio queue full! Dropping chunk.")

                                        # Reset state
                                        speech_blocks = []
                                        is_speaking = False
                                        silence_seconds = 0.0
                                else:
                                    pre_speech_blocks.append(block)

                        except queue.Empty:
                            continue

            except (sd.PortAudioError, OSError, Exception) as e:
                if self.stop_event.is_set():
                    break
                logger.warning(
                    f"⚠️ Audio device disconnected or driver error: {e}. "
                    "Re-initializing in 2.5s..."
                )
                time.sleep(2.5)

        logger.info("Audio Ingestion Thread terminated gracefully.")


# ---------------------------------------------------------------------------
# Thread 2: Transcription & Storage Compression
# ---------------------------------------------------------------------------
class TranscriptionThread(threading.Thread):
    """
    Ingests audio chunks from Thread 1:
      1. Archives raw audio to Opus format via FFmpeg CLI at ~10 kbps (< 50MB / 10hrs).
      2. Transcribes speech using faster-whisper on CPU with int8 quantization.
      3. Dispatches transcribed text and archive metadata to Thread 3 queue.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        transcription_queue: queue.Queue,
        stop_event: threading.Event,
        config: ScribeConfig,
        shared_context: Optional[SharedContext] = None,
    ):
        super().__init__(name="TranscriptionThread", daemon=True)
        self.audio_queue = audio_queue
        self.transcription_queue = transcription_queue
        self.stop_event = stop_event
        self.config = config
        self.shared_context = shared_context
        self.whisper_model = None
        self.total_transcribed = 0

    def _init_whisper(self):
        from faster_whisper import WhisperModel
        logger.info(
            f"🧠 Loading Whisper model '{self.config.whisper_model}' on CPU "
            f"(compute_type='{self.config.whisper_compute_type}', threads={self.config.whisper_threads})..."
        )
        self.whisper_model = WhisperModel(
            self.config.whisper_model,
            device="cpu",
            compute_type=self.config.whisper_compute_type,
            cpu_threads=self.config.whisper_threads,
        )
        logger.info("✅ Whisper loaded on CPU. (GPU preserved for Ollama).")

    def run(self):
        self._init_whisper()
        logger.info("Transcription & Archival Thread started.")

        # Resolve archives root directory
        archive_root = Path(self.config.vault_dir) / self.config.archive_dir

        while not self.stop_event.is_set() or not self.audio_queue.empty():
            try:
                chunk: AudioChunk = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # 1. Check for pure silence
                if chunk.rms_energy < self.config.energy_threshold:
                    # Skip silent chunks
                    continue

                # 2. Archive audio to Opus via FFmpeg
                date_folder = chunk.timestamp.strftime("%Y-%m-%d")
                archive_dir = archive_root / date_folder
                opus_filename = f"audio_{chunk.timestamp.strftime('%Y%m%d_%H%M%S')}.opus"
                opus_path = archive_dir / opus_filename

                archive_success = archive_audio_opus(
                    chunk.data,
                    self.config.sample_rate,
                    opus_path,
                    bitrate=self.config.opus_bitrate,
                )

                # 3. Transcribe speech using faster-whisper with context prompt and beam search
                initial_prompt = (
                    self.shared_context.get_initial_prompt()
                    if (self.shared_context and self.config.context_prompt_enabled)
                    else None
                )
                if initial_prompt:
                    logger.debug(f"Whisper initial_prompt: '{initial_prompt}'")

                segments, _ = self.whisper_model.transcribe(
                    chunk.data,
                    beam_size=self.config.whisper_beam_size,
                    language="en",
                    vad_filter=True,
                    condition_on_previous_text=False,
                    initial_prompt=initial_prompt,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()

                if text:
                    if self.shared_context:
                        self.shared_context.add_transcript(text)
                    self.total_transcribed += 1
                    t_chunk = TranscribedChunk(
                        timestamp=chunk.timestamp,
                        text=text,
                        archive_path=opus_path if archive_success else None,
                        duration=chunk.duration,
                    )
                    self.transcription_queue.put(t_chunk)

            except Exception as e:
                logger.error(f"Error during transcription/archival: {e}", exc_info=True)
            finally:
                # Explicitly discard audio numpy array to prevent memory retention
                del chunk.data
                self.audio_queue.task_done()

        logger.info("Transcription Thread terminated gracefully.")


# ---------------------------------------------------------------------------
# Thread 3: Intelligent Router & Synthesis
# ---------------------------------------------------------------------------
class RouterSynthesisThread(threading.Thread):
    """
    Consumes transcribed speech:
      1. Detects Voice Keyword Commands ("Topic [Name]", "Switch to [Name]", "Note [Name]")
         and dynamically updates the active target file (Windows-sanitized).
      2. Strips verbal fillers ("um", "uh", "you know", "like") & filters fragments < min_substantive_words.
      3. Appends verbatim transcript, block ID, and playable audio widget to dedicated daily transcript file.
      4. Integrates and reorganizes thematic bullet points in the active note, preserving all unrelated content.
    """

    def __init__(
        self,
        transcription_queue: queue.Queue,
        stop_event: threading.Event,
        config: ScribeConfig,
        shared_context: Optional[SharedContext] = None,
    ):
        super().__init__(name="RouterSynthesisThread", daemon=True)
        self.transcription_queue = transcription_queue
        self.stop_event = stop_event
        self.config = config
        self.shared_context = shared_context

        # Target notes directory setup
        base_vault = Path(self.config.vault_dir).resolve()
        if self.config.notes_subfolder:
            self.notes_dir = base_vault / self.config.notes_subfolder
        else:
            self.notes_dir = base_vault
        self.notes_dir.mkdir(parents=True, exist_ok=True)

        # Active note state (protected by lock)
        self.state_lock = threading.Lock()
        self.active_filename = sanitize_filename(self.config.default_note)
        self.total_notes_written = 0

        if self.shared_context:
            self.shared_context.update_topic(self.config.default_note)

        logger.info(f"📁 Target Notes Directory: {self.notes_dir}")
        logger.info(f"🎯 Default Active Note:    {self.active_filename}")

    def get_active_filepath(self) -> Path:
        with self.state_lock:
            return self.notes_dir / self.active_filename

    def set_active_topic(self, topic_name: str):
        sanitized = sanitize_filename(topic_name)
        with self.state_lock:
            old_name = self.active_filename
            self.active_filename = sanitized
        if self.shared_context:
            self.shared_context.update_topic(topic_name)
        logger.info(f"🎯 [COMMAND] Switched active note: '{old_name}' ➔ '{sanitized}'")

    def run(self):
        logger.info("Router & Synthesis Thread started. Ready for speech thoughts.")

        while not self.stop_event.is_set() or not self.transcription_queue.empty():
            try:
                item: TranscribedChunk = self.transcription_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                raw_text = item.text.strip()
                logger.info(f'🗣️ Heard: "{raw_text}"')

                # Step 1: Voice Keyword Detection
                voice_command = parse_voice_command(raw_text)
                if voice_command:
                    self.set_active_topic(voice_command)
                    continue

                # Step 2: Verbal Filler Stripping & Word Filtering
                cleaned_text = strip_verbal_fillers(raw_text)
                substantive_count = count_substantive_words(cleaned_text)

                if substantive_count < self.config.min_substantive_words:
                    logger.debug(
                        f"Dropped low-substance fragment ({substantive_count} words < "
                        f"{self.config.min_substantive_words}): '{cleaned_text}'"
                    )
                    continue

                # Step 3: IDs and Relative Paths
                chunk_id = f"t-{item.timestamp.strftime('%H%M%S')}"
                time_str = item.timestamp.strftime("%H:%M:%S")
                date_str = item.timestamp.strftime("%Y-%m-%d")

                audio_rel = ""
                if item.archive_path:
                    try:
                        audio_rel = str(item.archive_path.relative_to(self.notes_dir)).replace("\\", "/")
                    except ValueError:
                        audio_rel = item.archive_path.name

                with self.state_lock:
                    active_name = self.active_filename
                    target_note_path = self.notes_dir / active_name
                    topic_title = target_note_path.stem.replace("_", " ")

                # Step 4: Write to Dedicated Session Transcript File
                append_session_transcript(
                    vault_dir=Path(self.config.vault_dir).resolve(),
                    transcripts_folder=self.config.transcripts_folder,
                    topic_title=topic_title,
                    chunk_id=chunk_id,
                    timestamp_str=time_str,
                    text=cleaned_text,
                    audio_rel_path=audio_rel,
                )

                # Step 5: Citation Tag linking to separate transcript and audio
                trans_doc_name = f"{date_str}_Transcript"
                citation_tag = (
                    f"([[Transcripts/{trans_doc_name}#^{chunk_id}|🔗 {time_str}]]"
                    f"{f' · [[{audio_rel}|▶ Audio]]' if audio_rel else ''})"
                )

                # Step 6: Thematic Note Synthesis & Preservation
                success = integrate_note_thematically(
                    note_filepath=target_note_path,
                    topic_title=topic_title,
                    new_statement=cleaned_text,
                    citation_tag=citation_tag,
                    config=self.config,
                )

                if success:
                    self.total_notes_written += 1
                    logger.info(
                        f"📝 [{target_note_path.name}] Updated structured note (linked to {trans_doc_name}#^{chunk_id})."
                    )
                else:
                    logger.error(f"Failed to record note into {target_note_path}")

            except Exception as e:
                logger.error(f"Error in router synthesis: {e}", exc_info=True)
            finally:
                self.transcription_queue.task_done()

        logger.info("Router Synthesis Thread terminated gracefully.")


# ---------------------------------------------------------------------------
# Main Orchestrator & Signal Handling
# ---------------------------------------------------------------------------
class VoiceScribeService:
    """Orchestrates audio capture, transcription, routing, and lifecycle."""

    def __init__(self, config: ScribeConfig):
        self.config = config
        self.stop_event = threading.Event()
        self.audio_queue = queue.Queue(maxsize=40)
        self.transcription_queue = queue.Queue(maxsize=100)
        self.start_time = time.time()

        self.shared_context = SharedContext(default_topic=self.config.default_note)

        self.thread_audio = AudioIngestionThread(
            audio_queue=self.audio_queue,
            stop_event=self.stop_event,
            config=self.config,
        )
        self.thread_transcribe = TranscriptionThread(
            audio_queue=self.audio_queue,
            transcription_queue=self.transcription_queue,
            stop_event=self.stop_event,
            config=self.config,
            shared_context=self.shared_context,
        )
        self.thread_router = RouterSynthesisThread(
            transcription_queue=self.transcription_queue,
            stop_event=self.stop_event,
            config=self.config,
            shared_context=self.shared_context,
        )

    def start(self):
        logger.info("=" * 65)
        logger.info("   🎙️  VOICESCRIBE: CONTINUOUS OBSIDIAN VOICE ROUTER  🎙️   ")
        logger.info("=" * 65)
        logger.info(f"Target Vault Directory : {Path(self.config.vault_dir).resolve()}")
        logger.info(f"Whisper Model (CPU int8): {self.config.whisper_model} (beam_size={self.config.whisper_beam_size})")
        logger.info(f"Contextual Prompting   : {'Enabled' if self.config.context_prompt_enabled else 'Disabled'}")
        logger.info(f"Ollama Model (GPU)     : {self.config.ollama_model} ({self.config.ollama_host})")
        logger.info(f"Pause VAD Threshold    : {self.config.pause_duration_sec}s")
        logger.info(f"Commands Supported     : 'Topic [Name]', 'Switch to [Name]', 'Note [Name]'")
        logger.info("=" * 65)

        # Ensure Ollama server is running in the background for note synthesis
        ensure_ollama_running(self.config.ollama_host)

        self.thread_audio.start()
        self.thread_transcribe.start()
        self.thread_router.start()

        # Handle Ctrl+C / SIGINT / SIGTERM gracefully
        def _handle_exit(sig, frame):
            logger.info("\n🛑 Exit requested (Ctrl+C). Flashing shutdown sequence...")
            self.stop()

        signal.signal(signal.SIGINT, _handle_exit)
        signal.signal(signal.SIGTERM, _handle_exit)

        stop_signal_file = Path(__file__).resolve().parent / "stop.signal"

        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
                # Check for stop signal from Obsidian plugin
                if stop_signal_file.exists():
                    logger.info("🛑 stop.signal detected from Obsidian. Stopping service...")
                    try:
                        stop_signal_file.unlink()
                    except OSError:
                        pass
                    self.stop()
                    break
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        if self.stop_event.is_set():
            return
        self.stop_event.set()

        logger.info("Draining audio and transcription queues...")
        self.thread_audio.join(timeout=3.0)
        self.thread_transcribe.join(timeout=5.0)
        self.thread_router.join(timeout=5.0)

        elapsed = time.time() - self.start_time
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Calculate archive directory size
        archive_root = Path(self.config.vault_dir) / self.config.archive_dir
        archive_size_mb = 0.0
        if archive_root.exists():
            total_bytes = sum(f.stat().st_size for f in archive_root.rglob("*.opus"))
            archive_size_mb = total_bytes / (1024 * 1024)

        logger.info("=" * 65)
        logger.info("                SESSION SUMMARY                         ")
        logger.info("=" * 65)
        logger.info(f"Total Session Duration : {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s")
        logger.info(f"Audio Chunks Processed : {self.thread_audio.total_chunks_captured}")
        logger.info(f"Speech Fragments Found : {self.thread_transcribe.total_transcribed}")
        logger.info(f"Notes Written to Vault : {self.thread_router.total_notes_written}")
        logger.info(f"Opus Archive Size      : {archive_size_mb:.2f} MB")
        logger.info("=" * 65)
        logger.info("Goodbye! ✨")


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VoiceScribe: Continuous voice-to-notes transcription and routing for Obsidian."
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--vault-dir", help="Obsidian vault directory (default: ..)")
    parser.add_argument("--notes-subfolder", help="Subfolder within vault for notes (default: root)")
    parser.add_argument("--default-note", help="Default note topic (default: Voice Notes)")
    parser.add_argument("--whisper-model", choices=["tiny.en", "base.en", "small.en", "medium.en"], help="Whisper model on CPU")
    parser.add_argument("--ollama-model", help="Ollama model for distillation (default: llama3.2:3b)")
    parser.add_argument("--chunk-duration", type=float, help="Audio slice interval in seconds (default: 7.0)")
    parser.add_argument("--menu", "-m", action="store_true", help="Launch interactive configuration options menu and exit")
    parser.add_argument("--device", help="Audio input device index or substring")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    if args.menu:
        from config_menu import interactive_menu
        interactive_menu()
        return

    # Load base config
    cfg = ScribeConfig.load(args.config)

    # CLI Overrides
    if args.vault_dir:
        cfg.vault_dir = args.vault_dir
    if args.notes_subfolder is not None:
        cfg.notes_subfolder = args.notes_subfolder
    if args.default_note:
        cfg.default_note = args.default_note
    if args.whisper_model:
        cfg.whisper_model = args.whisper_model
    if args.ollama_model:
        cfg.ollama_model = args.ollama_model
    if args.chunk_duration:
        cfg.chunk_duration_sec = args.chunk_duration
    if args.device:
        try:
            cfg.audio_device = int(args.device)
        except ValueError:
            cfg.audio_device = args.device
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    service = VoiceScribeService(cfg)
    service.start()


if __name__ == "__main__":
    main()
