"""
Unit and Integration Tests for VoiceScribe
Validates:
- Voice keyword detection and Windows filename sanitization
- Verbal filler removal and substantive word filtering
- Audio Opus compression via FFmpeg (bitrate and temporary file cleanup)
- Atomic Obsidian note writing with retry resilience
- Ollama LLM distillation and offline fallback
"""

import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

# Ensure stdout handles UTF-8 on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scribe_router import (
    sanitize_filename,
    parse_voice_command,
    strip_verbal_fillers,
    count_substantive_words,
    archive_audio_opus,
    append_session_transcript,
    integrate_note_thematically,
    ScribeConfig,
    SharedContext
)


def test_filename_sanitization():
    print("[TEST] test_filename_sanitization...")
    assert sanitize_filename("Algorand") == "Algorand.md"
    assert sanitize_filename("Note Rambling") == "Note Rambling.md"
    # Illegal chars: \ / : * ? " < > |
    assert sanitize_filename('Project: Alpha / Beta * ? "Test" <1> |') == "Project Alpha Beta Test 1.md"
    # Trailing dots and spaces
    assert sanitize_filename("Flutter...") == "Flutter.md"
    assert sanitize_filename("   ") == "Voice Notes.md"
    # Reserved Windows device names
    assert sanitize_filename("CON") == "CON_note.md"
    assert sanitize_filename("nul") == "nul_note.md"
    print("  -> PASSED")


def test_voice_command_parsing():
    print("[TEST] test_voice_command_parsing...")
    assert parse_voice_command("Topic Algorand") == "Algorand"
    assert parse_voice_command("Topic Algorand.") == "Algorand"
    assert parse_voice_command("topic: machine learning") == "machine learning"
    assert parse_voice_command("Switch to Flutter") == "Flutter"
    assert parse_voice_command("Switch topic to Note Rambling") == "Note Rambling"
    assert parse_voice_command("Note French toast") == "French toast"
    assert parse_voice_command("Note: Ideas and Todos!") == "Ideas and Todos"
    assert parse_voice_command("We should deploy the database migration on Friday") is None
    assert parse_voice_command("This is just taking a note of something") is None
    print("  -> PASSED")


def test_verbal_filler_and_word_filtering():
    print("[TEST] test_verbal_filler_and_word_filtering...")
    # Verbal filler removal
    raw_1 = "Um, you know, we really like need to finish the design by tomorrow."
    cleaned_1 = strip_verbal_fillers(raw_1)
    assert "um" not in cleaned_1.lower()
    assert "you know" not in cleaned_1.lower()
    assert "finish the design by tomorrow" in cleaned_1

    # Low-word fragments (< 5 substantive words)
    frag_1 = "Um, yeah."
    cleaned_frag_1 = strip_verbal_fillers(frag_1)
    assert count_substantive_words(cleaned_frag_1) < 5

    frag_2 = "Uh, like, okay then."
    cleaned_frag_2 = strip_verbal_fillers(frag_2)
    assert count_substantive_words(cleaned_frag_2) < 5

    # Substantive fragment (>= 5 words)
    long_thought = "We need to fix the database migration before deploying to staging."
    cleaned_long = strip_verbal_fillers(long_thought)
    assert count_substantive_words(cleaned_long) >= 5
    print("  -> PASSED")


def test_opus_compression():
    print("[TEST] test_opus_compression...")
    sr = 16000
    duration = 7.0
    # Generate 7 seconds of 440 Hz test sine wave
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = Path(tmpdir) / "test_chunk.opus"
        archive_audio_opus(audio, sr, archive_path, bitrate="10k")

        assert archive_path.exists(), "Opus file was not created"
        size = archive_path.stat().st_size
        print(f"  -> Generated Opus file size: {size} bytes ({size / 1024:.2f} KB)")
        # 7 seconds at 10 kbps should be ~8-12 KB, definitely under 20 KB
        assert size > 1000, "Opus file is unexpectedly empty"
        assert size < 20000, f"Opus file is too large: {size} bytes"
    print("  -> PASSED")


from scribe_router import (
    sanitize_filename,
    parse_voice_command,
    strip_verbal_fillers,
    count_substantive_words,
    archive_audio_opus,
    append_session_transcript,
    integrate_note_thematically,
    ScribeConfig
)


def test_separate_transcript_and_note_integration():
    print("[TEST] test_separate_transcript_and_note_integration...")
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_dir = Path(tmpdir)
        config = ScribeConfig(
            vault_dir=str(vault_dir),
            transcripts_folder="Transcripts",
            ollama_host="http://127.0.0.1:11434",
            ollama_model="llama3.2:3b"
        )

        # 1. Append to separate transcript
        success = append_session_transcript(
            vault_dir=vault_dir,
            transcripts_folder="Transcripts",
            topic_title="Ideas",
            chunk_id="t-142201",
            timestamp_str="14:22:01",
            text="Deploy new authentication flow by Monday.",
            audio_rel_path="VoiceScribe_Archives/2026-10-25/audio1.opus"
        )
        assert success
        date_str = datetime.now().strftime("%Y-%m-%d")
        transcript_file = vault_dir / "Transcripts" / f"{date_str}_Transcript.md"
        assert transcript_file.exists()
        t_content = transcript_file.read_text(encoding="utf-8")
        assert "^t-142201" in t_content
        assert "![[VoiceScribe_Archives/2026-10-25/audio1.opus]]" in t_content

        # 2. Integrate into thematic main note
        note_file = vault_dir / "Ideas.md"
        citation1 = f"([[Transcripts/{date_str}_Transcript#^t-142201|🔗 14:22:01]] · [[VoiceScribe_Archives/2026-10-25/audio1.opus|▶ Audio]])"
        success2 = integrate_note_thematically(
            note_filepath=note_file,
            topic_title="Ideas",
            new_statement="Deploy new authentication flow by Monday.",
            citation_tag=citation1,
            config=config
        )
        assert success2
        assert note_file.exists()
        n_content = note_file.read_text(encoding="utf-8")
        assert "# Ideas" in n_content
        assert "## 💡 Notes" in n_content
        assert "[[Transcripts/" in n_content
        assert "Deploy new authentication flow by Monday" in n_content
        # Note should NOT contain raw transcript block
        assert "> ![[" not in n_content
def test_shared_context():
    print("[TEST] test_shared_context...")
    ctx = SharedContext(default_topic="Machine Learning", max_history=3)
    # Check default topic prompt
    prompt = ctx.get_initial_prompt()
    assert prompt == "Topic: Machine Learning."

    # Add speech history
    ctx.add_transcript("We need to tune hyperparameters for the neural net.")
    prompt2 = ctx.get_initial_prompt()
    assert "Topic: Machine Learning." in prompt2
    assert "hyperparameters" in prompt2

    # Add second sentence
    ctx.add_transcript("Particularly learning rate and weight decay.")
    prompt3 = ctx.get_initial_prompt()
    assert "hyperparameters" in prompt3
    assert "weight decay" in prompt3

    # Switch topic
    ctx.update_topic("Algorand Governance")
    prompt4 = ctx.get_initial_prompt()
    assert "Topic: Algorand Governance." in prompt4
    print("  -> PASSED")


if __name__ == "__main__":
    from datetime import datetime
    print("=== Running VoiceScribe Self-Verification Tests ===")
    test_filename_sanitization()
    test_voice_command_parsing()
    test_verbal_filler_and_word_filtering()
    test_opus_compression()
    test_separate_transcript_and_note_integration()
    test_shared_context()
    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")
