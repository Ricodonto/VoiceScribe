"""
End-to-end pipeline test with a simulated microphone, fake Whisper and fake Ollama.
Exercises the real threads, VAD segmentation, Ogg/Opus archiving, transcript + note
writing, voice commands and the graceful stop/drain sequence.
Run:  venv\\Scripts\\python.exe test_pipeline_e2e.py
"""
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SR = 16000
BLOCK = 1024
SCRIPT = ["Topic French toast", "Use brioche and soak it for thirty seconds.",
          "Um, add a pinch of cinnamon to the eggs.", "Thank you."]


# --- fake microphone: speech bursts (quiet, like the real log) separated by 2.5 s of silence
class FakeInputStream:
    def __init__(self, samplerate, channels, dtype, device, callback, blocksize):
        self.cb = callback
        self._stop = threading.Event()

    def __enter__(self):
        def feed():
            rng = np.random.default_rng(0)
            plan = []
            for _ in SCRIPT:
                plan += [0.003] * int(2.0 * SR / BLOCK) + [0.0002] * int(2.5 * SR / BLOCK)
            for amp in plan + [0.0002] * 2000:
                if self._stop.is_set():
                    return
                block = (rng.standard_normal((BLOCK, 1)) * amp).astype(np.float32)
                self.cb(block, BLOCK, None, None)
                time.sleep(0.002)  # ~30x faster than real time
        threading.Thread(target=feed, daemon=True).start()
        return self

    def __exit__(self, *a):
        self._stop.set()


sys.modules["sounddevice"] = types.SimpleNamespace(
    InputStream=FakeInputStream, query_devices=lambda *a, **k: {"name": "Fake mic"}, PortAudioError=Exception)


# --- fake Whisper: returns the next scripted sentence per chunk
class FakeSeg:
    def __init__(self, text):
        self.text, self.no_speech_prob, self.avg_logprob = text, 0.01, -0.2


class FakeWhisper:
    calls = 0

    def __init__(self, name, **kw):
        self.local_only = kw.get("local_files_only")

    def transcribe(self, audio, **kw):
        assert np.max(np.abs(audio)) > 0.2, "audio should be normalised before transcription"
        text = SCRIPT[FakeWhisper.calls % len(SCRIPT)]
        FakeWhisper.calls += 1
        time.sleep(0.3)
        return iter([FakeSeg(text)]), None


class FakeTimedSeg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text
        self.no_speech_prob, self.avg_logprob, self.compression_ratio = 0.01, -0.2, 1.2


IMPORT_SCRIPT = [(1.0, 4.0, "First idea about the garden hose."), (4.2, 6.0, "It should be at least twenty metres."),
                 (9.0, 12.0, "Create a new note named Holiday."), (14.0, 18.0, "We could visit Lisbon in May."),
                 (21.0, 24.0, "Scratch that."), (26.0, 30.0, "We could visit Porto in June.")]


def fake_decode_audio(path, sampling_rate=16000):
    rng = np.random.default_rng(1)
    return (rng.standard_normal(sampling_rate * 32) * 0.05).astype(np.float32)


class FakeImportWhisper(FakeWhisper):
    def transcribe(self, audio, **kw):
        if len(audio) > 16000 * 20:  # whole recording
            return iter([FakeTimedSeg(*s) for s in IMPORT_SCRIPT]), None
        return super().transcribe(audio, **kw)


sys.modules["faster_whisper"] = types.SimpleNamespace(WhisperModel=FakeImportWhisper, decode_audio=fake_decode_audio)

import scribe_router as sr  # noqa: E402


def fake_llm(config, outline, statement, topic):
    return {"items": [{"action": "add", "heading": "Recipe", "bullet": statement.rstrip("."), "target": None}]}


sr.ask_llm_for_placement = fake_llm
sr.ensure_ollama_running = lambda host: True


def main():
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        cfg = sr.ScribeConfig(vault_dir=str(vault), energy_threshold=0.0012, pause_duration_sec=1.7)
        svc = sr.VoiceScribeService(cfg)
        sr.STOP_SIGNAL_FILE = vault / "stop.signal"
        sr.PID_FILE = vault / "pid"

        def stopper():
            # wait until all speech has been captured, then request stop like Obsidian does
            while svc.thread_audio.total_chunks_captured < len(SCRIPT):
                time.sleep(0.05)
            (vault / "stop.signal").write_text("stop")
        threading.Thread(target=stopper, daemon=True).start()

        code = svc.start()
        assert code == 0
        assert svc.thread_audio.total_chunks_captured == 4, svc.thread_audio.total_chunks_captured
        assert svc.thread_transcribe.total_transcribed == 3  # "Thank you." filtered as hallucination
        assert svc.thread_router.total_notes_written == 2

        note = (vault / "French toast.md").read_text(encoding="utf-8")
        print(note)
        assert "### Recipe" in note
        assert "Use brioche and soak it for thirty seconds" in note
        assert "Add a pinch of cinnamon to the eggs" in note
        assert "▶ Audio]]" in note and ".ogg|" in note

        transcripts = list((vault / "Transcripts").glob("*_Transcript.md"))
        assert len(transcripts) == 1
        t = transcripts[0].read_text(encoding="utf-8")
        assert "Um, add a pinch" in t  # transcript stays verbatim
        oggs = list((vault / "VoiceScribe_Archives").rglob("*.ogg"))
        assert len(oggs) == 3, oggs  # no-speech chunk's audio removed
        default_note = (vault / "Voice Notes.md").read_text(encoding="utf-8")
        assert "- " not in default_note  # created empty at start; the command chunk added nothing
        print("\nE2E PASSED")


def import_test():
    """An audio file goes through the same pipeline: clips, transcript, notes, voice commands, undo."""
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        rec = vault / "memo.m4a"
        rec.write_bytes(b"fake")
        cfg = sr.ScribeConfig(vault_dir=str(vault), start_note="Garden.md", properties_enabled=False,
                              autolink_enabled=False, summary_enabled=False)
        code = sr.ImportService(cfg, str(rec)).run()
        assert code == 0
        garden = (vault / "Garden.md").read_text(encoding="utf-8")
        holiday = (vault / "Holiday.md").read_text(encoding="utf-8")
        print(garden, holiday, sep="\n")
        assert "First idea about the garden hose It should be at least twenty metres" in garden.replace(".", "")
        assert "Porto in June" in holiday and "Lisbon" not in holiday  # "scratch that" undid Lisbon
        assert len(list((vault / "VoiceScribe_Archives").rglob("*.ogg"))) == 5
        print("IMPORT PASSED")


def batch_import_test():
    """Batch mode: thoughts collect in the inbox and are organised together when the run ends."""
    seen = {}

    def fake_batch(config, title, outline, thoughts):
        seen.setdefault(title, []).append(list(thoughts))
        return {"items": [{"action": "add", "heading": "Plans", "bullet": t.rstrip("."), "sources": [f"T{i + 1}"], "target": None}
                          for i, t in enumerate(thoughts) if "Scratch" not in t]}
    orig = sr.ask_llm_for_batch
    sr.ask_llm_for_batch = fake_batch
    try:
        with tempfile.TemporaryDirectory() as d:
            vault = Path(d)
            rec = vault / "memo.m4a"
            rec.write_bytes(b"fake")
            cfg = sr.ScribeConfig(vault_dir=str(vault), start_note="Garden.md", organise_mode="batch", batch_interval_min=0,
                                  properties_enabled=False, autolink_enabled=False, summary_enabled=False)
            assert sr.ImportService(cfg, str(rec)).run() == 0
            garden = (vault / "Garden.md").read_text(encoding="utf-8")
            holiday = (vault / "Holiday.md").read_text(encoding="utf-8")
            print(garden, holiday, sep="\n")
            assert "Inbox" not in garden and "Inbox" not in holiday
            assert "### Plans" in garden and "### Plans" in holiday
            assert "Porto in June" in holiday and "Lisbon" not in holiday     # "scratch that" removed it from the inbox
            assert seen["Garden"] and seen["Holiday"] == [["We could visit Porto in June."]]
            print("BATCH PASSED")
    finally:
        sr.ask_llm_for_batch = orig


def pause_test():
    """While paused, speech is ignored; after resuming it is captured again."""
    import queue as q
    cfg = sr.ScribeConfig(energy_threshold=0.0012, pause_duration_sec=1.7)
    aq = q.Queue()
    stop = threading.Event()
    th = sr.AudioIngestionThread(aq, stop, cfg)
    th.set_paused(True)
    th.start()
    time.sleep(1.5)                        # fake mic plays its speech while paused
    assert aq.qsize() == 0
    th.set_paused(False)
    rng = np.random.default_rng(3)
    for amp in [0.004] * 30 + [0.0002] * 40:  # one spoken thought after resuming
        th.raw_blocks.put((rng.standard_normal(BLOCK) * amp).astype(np.float32))
    deadline = time.time() + 20
    while aq.qsize() == 0 and time.time() < deadline:
        time.sleep(0.1)
    stop.set()
    th.join(5)
    assert aq.qsize() >= 1
    print("PAUSE PASSED")


if __name__ == "__main__":
    main()
    import_test()
    batch_import_test()
    pause_test()
