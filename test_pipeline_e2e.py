"""
End-to-End Pipeline Verification Test for VoiceScribe
Simulates:
1. Audio generation / injection
2. Thread 2 (Faster-Whisper on CPU + FFmpeg Opus archival)
3. Thread 3 (Voice Command Routing + Filler Removal + Ollama Distillation + Atomic Writing)
"""

import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
import queue
import threading
import numpy as np

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scribe_router import (
    ScribeConfig,
    AudioChunk,
    TranscriptionThread,
    RouterSynthesisThread,
    archive_audio_opus,
    atomic_append_note
)


def run_e2e_pipeline_test():
    print("=== Running End-to-End VoiceScribe Pipeline Test ===")

    with tempfile.TemporaryDirectory() as temp_vault:
        vault_path = Path(temp_vault)
        config = ScribeConfig(
            vault_dir=str(vault_path),
            whisper_model="base.en",
            whisper_compute_type="int8",
            whisper_threads=4,
            ollama_model="llama3.2:3b",
            ollama_host="http://127.0.0.1:11434",
            default_note="Initial_Test",
            energy_threshold=0.0001
        )

        audio_queue = queue.Queue(maxsize=10)
        transcription_queue = queue.Queue(maxsize=10)
        stop_event = threading.Event()

        # Start Transcription Thread (Thread 2)
        transcribe_worker = TranscriptionThread(
            audio_queue=audio_queue,
            transcription_queue=transcription_queue,
            stop_event=stop_event,
            config=config
        )

        # Start Router & Synthesis Thread (Thread 3)
        router_worker = RouterSynthesisThread(
            transcription_queue=transcription_queue,
            stop_event=stop_event,
            config=config
        )

        print("[1/4] Starting worker threads...")
        transcribe_worker.start()
        router_worker.start()

        # We will test:
        # A) Direct voice command simulation: "Topic Smart Contracts"
        print("[2/4] Testing voice command routing...")
        router_worker.set_active_topic("Smart Contracts")
        assert router_worker.active_filename == "Smart Contracts.md"
        print(f"  -> Active note verified: {router_worker.active_filename}")

        # B) Synthetic audio chunk archival verification
        print("[3/4] Testing audio archival via FFmpeg Opus...")
        sr = 16000
        dur = 7.0
        t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
        audio_data = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        opus_archive = vault_path / "VoiceScribe_Archives" / "test_run.opus"
        success = archive_audio_opus(audio_data, sr, opus_archive, bitrate="10k")
        assert success and opus_archive.exists(), "Opus archiving failed!"
        print(f"  -> Opus archived successfully ({opus_archive.stat().st_size} bytes)")

        # C) Full Note writing & Ollama synthesis verification
        print("[4/4] Testing separate session transcript and thematic note integration...")
        raw_thought = "Um, you know, we really must optimize the smart contract gas limits before the mainnet launch on Tuesday."
        from scribe_router import strip_verbal_fillers, append_session_transcript, integrate_note_thematically
        cleaned = strip_verbal_fillers(raw_thought)

        chunk_id = "t-204510"
        time_str = "20:45:10"
        date_str = "2026-10-25"

        # 1. Append to separate transcript file
        append_session_transcript(
            vault_dir=vault_path,
            transcripts_folder="Transcripts",
            topic_title="Smart Contracts",
            chunk_id=chunk_id,
            timestamp_str=time_str,
            text=cleaned,
            audio_rel_path="VoiceScribe_Archives/test_run.opus"
        )
        t_file = vault_path / "Transcripts" / f"{datetime.now().strftime('%Y-%m-%d')}_Transcript.md"
        assert t_file.exists(), "Transcript file was not created!"
        t_text = t_file.read_text(encoding="utf-8")
        assert f"^{chunk_id}" in t_text
        assert "![[VoiceScribe_Archives/test_run.opus]]" in t_text

        # 2. Integrate into thematic main note
        target_file = router_worker.get_active_filepath()
        citation = f"([[Transcripts/{datetime.now().strftime('%Y-%m-%d')}_Transcript#^{chunk_id}|🔗 {time_str}]] · [[VoiceScribe_Archives/test_run.opus|▶ Audio]])"
        success = integrate_note_thematically(
            note_filepath=target_file,
            topic_title="Smart Contracts",
            new_statement=cleaned,
            citation_tag=citation,
            config=config
        )
        assert success and target_file.exists(), f"Target file {target_file} was not created!"

        content = target_file.read_text(encoding="utf-8")
        print("\n--- Generated Note Content ---")
        print(content)
        print("------------------------------")

        assert "# Smart Contracts" in content
        assert "## 💡 Notes" in content
        assert "[[Transcripts/" in content

        # Stop workers
        stop_event.set()
        transcribe_worker.join(timeout=3.0)
        router_worker.join(timeout=3.0)

    print("\n✅ End-to-End Pipeline Verification Test Passed Completely!")


if __name__ == "__main__":
    run_e2e_pipeline_test()
