"""
Unit tests for VoiceScribe (no microphone, Whisper or Ollama required).
Run:  venv\\Scripts\\python.exe test_voice_scribe.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scribe_router import (  # noqa: E402
    ScribeConfig, SharedContext, StructuredNote, append_session_transcript, archive_audio_opus,
    count_substantive_words, integrate_note_thematically, is_whisper_hallucination, normalize_audio,
    parse_voice_command, sanitize_filename, strip_verbal_fillers, validate_llm_items,
)


def test_filename_sanitization():
    assert sanitize_filename("Algorand") == "Algorand.md"
    assert sanitize_filename('Project: Alpha / Beta * ? "Test" <1> |') == "Project Alpha Beta Test 1.md"
    assert sanitize_filename("Flutter...") == "Flutter.md"
    assert sanitize_filename("   ") == "Voice Notes.md"
    assert sanitize_filename("CON") == "CON_note.md"


def test_voice_command_parsing():
    assert parse_voice_command("Topic Algorand") == "Algorand"
    assert parse_voice_command("Topic, Algorand.") == "Algorand"
    assert parse_voice_command("topic: machine learning") == "machine learning"
    assert parse_voice_command("Switch to Flutter") == "Flutter"
    assert parse_voice_command("Switch topic to Note Rambling") == "Note Rambling"
    assert parse_voice_command("Note French toast") == "French toast"
    assert parse_voice_command("Note: Ideas and Todos!") == "Ideas and Todos"
    # Ordinary speech must not switch notes
    assert parse_voice_command("Note that the server is down on Fridays") is None
    assert parse_voice_command("Note to self, buy milk") is None
    assert parse_voice_command("Topic of today is how we deploy the database to production") is None
    assert parse_voice_command("We should deploy the database migration on Friday") is None
    assert parse_voice_command("Notes about Voice Notes. hello") is None


def test_fillers_keep_meaning():
    cleaned = strip_verbal_fillers("Um, I really like the idea, uh, of actually shipping it.")
    assert "um" not in cleaned.lower().split()
    assert "like the idea" in cleaned and "actually" in cleaned
    assert count_substantive_words("Ok.") == 1
    assert count_substantive_words("We need to fix the migration") == 6


def test_hallucination_filter_and_normalize():
    assert is_whisper_hallucination("Thank you.")
    assert is_whisper_hallucination("Subtitles by the Amara.org community")
    assert not is_whisper_hallucination("Thank you for the coffee recipe, it worked.")
    quiet = (0.003 * np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
    louder = normalize_audio(quiet)
    assert 0.08 < np.max(np.abs(louder)) <= 0.9 + 1e-6


def test_opus_archive_is_small_ogg():
    sr = 16000
    t = np.linspace(0, 7.0, sr * 7, dtype=np.float32)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "chunk.ogg"
        assert archive_audio_opus(audio, sr, out, bitrate="12k")
        data = out.read_bytes()
        assert data[:4] == b"OggS" and b"OpusHead" in data[:200]
        assert 1000 < len(data) < 25000, len(data)


def test_transcript_block_id_on_own_line():
    with tempfile.TemporaryDirectory() as d:
        ok = append_session_transcript(Path(d), "Transcripts", "Ideas", "t-142201", "14:22:01",
                                       "Deploy the new auth flow by Monday.",
                                       "VoiceScribe_Archives/2026-10-29/a.ogg", date_str="2026-10-29")
        assert ok
        text = (Path(d) / "Transcripts" / "2026-10-29_Transcript.md").read_text(encoding="utf-8")
        assert "\n\n^t-142201\n\n" in text
        assert "> ![[VoiceScribe_Archives/2026-10-29/a.ogg]]" in text


def test_note_parse_render_roundtrip_preserves_user_content():
    content = (
        "# Voice Notes\n\n*Session Date: 2026-10-25*\n\n## 💡 Notes\n\n"
        "*(Ready for your thoughts. Speak naturally into your microphone.)*\n"
        "[Here's the vid](youtube.com)\n\n"
        "### Plans\n- Ship v2 ([[Transcripts/x#^t-1|🔗 1]] · [[a.ogg|▶ Audio]])\n"
        "    - a manual sub point\n\n"
        "## My own section\nkeep me\n"
    )
    note = StructuredNote.parse(content, "Voice Notes")
    assert note.preamble == ["[Here's the vid](youtube.com)"]
    assert note.sections[0].items[0].text == "Ship v2"
    assert len(note.sections[0].items[0].cites) == 1
    out = note.render()
    assert "*(Ready for" not in out
    assert "[Here's the vid](youtube.com)" in out
    assert "    - a manual sub point" in out
    assert "## My own section\nkeep me" in out
    assert StructuredNote.parse(out, "Voice Notes").render() == out


def test_llm_validation():
    texts = ["Ship v2"]
    ops = validate_llm_items({"items": [{"action": "update", "heading": "Plans", "bullet": "Ship v2 on Friday", "target": 1},
                                        {"action": "add", "heading": "### Food", "bullet": "- Try French toast", "target": None},
                                        {"action": "update", "heading": "X", "bullet": "bad target", "target": 99},
                                        {"bullet": ""}]}, texts)
    assert ops[0] == {"heading": "Plans", "bullet": "Ship v2 on Friday", "old_text": "Ship v2"}
    assert ops[1] == {"heading": "Food", "bullet": "Try French toast", "old_text": None}
    assert ops[2]["old_text"] is None and len(ops) == 3
    assert validate_llm_items("garbage", texts) == []


def test_integration_never_loses_notes():
    cfg = ScribeConfig()
    with tempfile.TemporaryDirectory() as d:
        note = Path(d) / "Ideas.md"
        # 30 thoughts; the fake LLM alternates headings and sometimes updates
        def fake_llm(config, outline, statement, topic):
            n = int(statement.split()[1])
            if n % 7 == 0:
                return {"items": [{"action": "update", "heading": "A", "bullet": f"Thought {n} about gardening, merged with thought 1", "target": 1}]}
            return {"items": [{"action": "add", "heading": "A" if n % 2 else "B", "bullet": f"Thought {n} about gardening", "target": None}]}
        for n in range(1, 31):
            assert integrate_note_thematically(note, "Ideas", f"thought {n} about gardening", f"([[T#^t-{n}|🔗 {n}]])", cfg, llm_fn=fake_llm)
        text = note.read_text(encoding="utf-8")
        bullets = [ln for ln in text.splitlines() if ln.startswith("- ")]
        assert len(bullets) == 30 - 4  # 7,14,21,28 were updates
        assert "### A" in text and "### B" in text
        assert "[[T#^t-7|🔗 7]]" in text  # update kept new citation

        # LLM failure -> statement still saved
        def broken(*a):
            raise ConnectionError("ollama down")
        assert integrate_note_thematically(note, "Ideas", "Offline thought", "(c)", cfg, llm_fn=broken)
        assert "- Offline thought (c)" in note.read_text(encoding="utf-8")


FOCUS_NOTE = (
    "# French toast\n\n## 💡 Notes\n\n"
    "### Ingredients\n- Use brioche (c1)\n\n"
    "### Method\n- Soak each slice for a minute ([[T#^t-1|🔗 1]])\n- Fry in butter ([[T#^t-2|🔗 2]])\n\n"
    "### Serving\n- Maple syrup ([[T#^t-3|🔗 3]])\n"
)


def test_focus_correct_rewrites_only_focused_bullet():
    from scribe_router import Focus
    cfg = ScribeConfig()
    with tempfile.TemporaryDirectory() as d:
        note = Path(d) / "French toast.md"
        note.write_text(FOCUS_NOTE, encoding="utf-8")
        seen = {}

        def fake_correct(config, heading, bullets, statement, topic):
            seen["args"] = (heading, bullets)
            return {"bullets": ["Soak each slice for about 30 seconds"]}

        focus = Focus(note, "Method", ["Soak each slice for a minute"], "correct")
        ch = integrate_note_thematically(note, "French toast", "Actually thirty seconds", "([[T#^t-9|🔗 9]])",
                                         cfg, focus=focus, correct_fn=fake_correct)
        assert ch and ch.updated == ["Soak each slice for about 30 seconds"] and ch.focus_bullets == ch.updated
        assert seen["args"] == ("Method", ["Soak each slice for a minute"])
        text = note.read_text(encoding="utf-8")
        assert "- Soak each slice for about 30 seconds ([[T#^t-1|🔗 1]]) ([[T#^t-9|🔗 9]])" in text
        assert "for a minute" not in text
        assert "- Fry in butter" in text and "- Maple syrup" in text and "- Use brioche" in text
        # order kept: soak still before fry
        assert text.index("Soak each") < text.index("Fry in butter")

        # LLM down -> correction placed right after the focused bullet, nothing lost
        def down(*a):
            raise ConnectionError("x")
        focus2 = Focus(note, "Method", ["Fry in butter"], "correct")
        ch2 = integrate_note_thematically(note, "French toast", "use salted butter", "(c)", cfg,
                                          focus=focus2, correct_fn=down)
        text = note.read_text(encoding="utf-8")
        assert ch2 and "- Fry in butter ([[T#^t-2|🔗 2]])\n- Correction: use salted butter (c)" in text


def test_focus_correct_merges_two_bullets_keeps_citations():
    from scribe_router import Focus
    cfg = ScribeConfig()
    with tempfile.TemporaryDirectory() as d:
        note = Path(d) / "French toast.md"
        note.write_text(FOCUS_NOTE, encoding="utf-8")
        focus = Focus(note, "Method", ["Soak each slice for a minute", "Fry in butter"], "correct")
        integrate_note_thematically(note, "French toast", "soak briefly then fry", "(c9)", cfg, focus=focus,
                                    correct_fn=lambda *a: {"bullets": ["Soak briefly, then fry in butter"]})
        text = note.read_text(encoding="utf-8")
        line = [l for l in text.splitlines() if l.startswith("- Soak briefly")][0]
        assert "🔗 1" in line and "🔗 2" in line and "(c9)" in line


def test_focus_add_inserts_after_focused_bullet():
    from scribe_router import Focus
    cfg = ScribeConfig()
    with tempfile.TemporaryDirectory() as d:
        note = Path(d) / "French toast.md"
        note.write_text(FOCUS_NOTE, encoding="utf-8")
        hints = {}

        def fake_llm(config, outline, statement, topic, focus_hint=""):
            hints["h"] = focus_hint
            return {"items": [{"action": "add", "heading": "Method", "bullet": "Use thick slices", "target": None},
                              {"action": "add", "heading": "Method", "bullet": "Press gently", "target": None}]}

        focus = Focus(note, "Method", ["Soak each slice for a minute"], "add")
        ch = integrate_note_thematically(note, "French toast", "thick slices, press gently", "(c)", cfg,
                                         llm_fn=fake_llm, focus=focus)
        assert ch.added == ["Use thick slices", "Press gently"]
        assert "FOCUS" in hints["h"] and "[2]" in hints["h"]
        lines = [l for l in note.read_text(encoding="utf-8").splitlines() if l.startswith("- ")]
        order = [l.split(" (")[0] for l in lines]
        assert order == ["- Use brioche", "- Soak each slice for a minute", "- Use thick slices",
                         "- Press gently", "- Fry in butter", "- Maple syrup"], order


def test_echoed_bullets_are_rejected_and_repeats_merge():
    cfg = ScribeConfig()
    with tempfile.TemporaryDirectory() as d:
        note = Path(d) / "Voice Notes.md"
        note.write_text("# Voice Notes\n\n## 💡 Notes\n\n### Physical Characteristics\n"
                        "- Elephant skin is tough ([[T#^t-1|🔗 1]])\n- Soak for 30 seconds ([[T#^t-2|🔗 2]])\n", encoding="utf-8")
        statement = "I have been using Cloud Pro for the past day and it is amazing, I can control it from my phone"

        def echo_llm(config, outline, st, topic):  # what llama3.2:3b did: echo + numbered bullet
            return {"items": [
                {"action": "add", "heading": "Physical Characteristics", "bullet": "[1] Elephant skin is tough and serves as a protective barrier", "target": None},
                {"action": "add", "heading": "Cloud Pro", "bullet": "[2] Cloud Pro can be controlled from your phone", "target": None}]}
        ch = integrate_note_thematically(note, "Voice Notes", statement, "([[T#^t-3|🔗 3]])", cfg, llm_fn=echo_llm)
        text = note.read_text(encoding="utf-8")
        assert text.count("Elephant skin") == 1 and "[1]" not in text and "[2]" not in text
        assert "- Cloud Pro can be controlled from your phone ([[T#^t-3|🔗 3]])" in text
        assert ch.added == ["Cloud Pro can be controlled from your phone"]

        # Saying the same point again adds a source to the existing bullet instead of a copy
        integrate_note_thematically(note, "Voice Notes", "elephant skin is really tough", "([[T#^t-4|🔗 4]])", cfg,
                                    llm_fn=lambda *a: {"items": [{"action": "add", "heading": "Physical Characteristics", "bullet": "Elephant skin is tough", "target": None}]})
        text = note.read_text(encoding="utf-8")
        assert text.count("Elephant skin") == 1 and "- Elephant skin is tough ([[T#^t-1|🔗 1]]) ([[T#^t-4|🔗 4]])" in text

        # Different numbers = different facts, never merged
        integrate_note_thematically(note, "Voice Notes", "soak for 60 seconds", "([[T#^t-5|🔗 5]])", cfg,
                                    llm_fn=lambda *a: {"items": [{"action": "add", "heading": "Physical Characteristics", "bullet": "Soak for 60 seconds", "target": None}]})
        text = note.read_text(encoding="utf-8")
        assert "- Soak for 30 seconds ([[T#^t-2|🔗 2]])" in text and "- Soak for 60 seconds ([[T#^t-5|🔗 5]])" in text

        # Model "updates" an unrelated bullet with nothing from the statement -> statement kept as-is instead
        integrate_note_thematically(note, "Voice Notes", "Remember to water the garden", "([[T#^t-6|🔗 6]])", cfg,
                                    llm_fn=lambda *a: {"items": [{"action": "update", "heading": "Cloud Pro", "bullet": "Cloud Pro can be controlled from your phone", "target": 3}]})
        text = note.read_text(encoding="utf-8")
        assert "- Cloud Pro can be controlled from your phone ([[T#^t-3|🔗 3]])\n" in text
        assert "Remember to water the garden ([[T#^t-6|🔗 6]])" in text


def test_note_commands():
    from scribe_router import parse_note_command, split_command
    p = parse_note_command
    assert (p("Switch to a new note called Cloud Pro.").kind, p("Switch to a new note called Cloud Pro.").name) == ("create", "Cloud Pro")
    assert p("Switch to 'Welcome.md'.").name == "Welcome" and p("Switch to 'Welcome.md'.").kind == "switch"
    assert p("switch to welcome dot md").name == "welcome"
    assert p("Okay, so create a new note named grocery list.").name == "grocery list"
    assert p("New note: Project Phoenix").kind == "create"
    assert p("Switch topic to Note Rambling").name == "Note Rambling"
    assert p("Switch to talking about elephants") is None
    before, cmd, after = split_command("Switch to a new note called Cloud Pro. I've been using it all day.")
    assert cmd.name == "Cloud Pro" and after == "I've been using it all day." and before == ""
    before, cmd, after = split_command("That's all for elephants. Create a new note named Cloud Pro.")
    assert before == "That's all for elephants." and cmd.kind == "create"


def test_router_switch_existing_and_create():
    from scribe_router import RouterSynthesisThread, NoteCommand
    import queue, threading
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        (vault / "Projects").mkdir()
        (vault / "Projects" / "Welcome.md").write_text("# Welcome\nhello\n", encoding="utf-8")
        cfg = ScribeConfig(vault_dir=str(vault), notes_subfolder="Voice")
        r = RouterSynthesisThread(queue.Queue(), threading.Event(), cfg)
        r.apply_note_command(NoteCommand("switch", "welcome"))
        assert r.get_active_filepath() == (vault / "Projects" / "Welcome.md").resolve()
        r.apply_note_command(NoteCommand("create", "Welcome"))
        assert r.get_active_filepath() == (vault / "Voice" / "Welcome.md").resolve() or r.get_active_filepath().name == "Welcome.md"
        r.apply_note_command(NoteCommand("create", "Welcome"))
        assert r.get_active_filepath().name == "Welcome 2.md"
        assert (vault / "Projects" / "Welcome.md").read_text(encoding="utf-8") == "# Welcome\nhello\n"
        cfg2 = ScribeConfig(vault_dir=str(vault), start_note="Projects/Welcome.md")
        r2 = RouterSynthesisThread(queue.Queue(), threading.Event(), cfg2)
        assert r2.get_active_filepath().name == "Welcome.md"


def _router(vault, **kw):
    from scribe_router import RouterSynthesisThread
    import queue, threading
    cfg = ScribeConfig(vault_dir=str(vault), **kw)
    return RouterSynthesisThread(queue.Queue(), threading.Event(), cfg)


def _thought(text, n=1, dur=4.0):
    from scribe_router import TranscribedChunk
    from datetime import datetime
    return TranscribedChunk(timestamp=datetime(2026, 10, 30, 10, 0, n), text=text, archive_path=None, duration=dur, seg_id=n)


def test_undo_restores_previous_note():
    import scribe_router as sr
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        r = _router(vault, properties_enabled=False, autolink_enabled=False)
        orig = sr.ask_llm_for_placement
        sr.ask_llm_for_placement = lambda c, o, st, t, h="": {"items": [{"action": "add", "heading": "Ideas", "bullet": st.rstrip("."), "target": None}]}
        try:
            r.process(_thought("Buy a new garden hose.", 1))
            r.process(_thought("Paint the fence green.", 2))
            note = r.get_active_filepath()
            assert "Paint the fence green" in note.read_text(encoding="utf-8")
            r.process(_thought("Okay, scratch that.", 3))
            text = note.read_text(encoding="utf-8")
            assert "Paint the fence green" not in text and "Buy a new garden hose" in text
            # edited since -> only the added bullet is removed, user's edit kept
            r.process(_thought("Call the plumber tomorrow.", 4))
            note.write_text(note.read_text(encoding="utf-8") + "\nMy own line\n", encoding="utf-8")
            assert r.undo()
            text = note.read_text(encoding="utf-8")
            assert "Call the plumber" not in text and "My own line" in text and "garden hose" in text
        finally:
            sr.ask_llm_for_placement = orig


def test_properties_and_autolinks_on_write():
    import scribe_router as sr
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        (vault / "Cloud Pro.md").write_text("# Cloud Pro\n", encoding="utf-8")
        r = _router(vault)
        orig = sr.ask_llm_for_placement
        sr.ask_llm_for_placement = lambda c, o, st, t, h="": {"items": [{"action": "add", "heading": "Tools", "bullet": st.rstrip("."), "target": None}]}
        try:
            r.process(_thought("I control Cloud Pro from my phone.", 1, dur=30))
            r.process(_thought("Cloud Pro syncs my files too.", 2, dur=30))
            text = r.get_active_filepath().read_text(encoding="utf-8")
            assert text.startswith("---\ncreated: 2026-10-30\n")
            assert "voice_sessions: 1\n" in text and "voice_minutes: 1.0\n" in text and "  - voicescribe" in text
            assert "- I control [[Cloud Pro]] from my phone" in text
            # the line still parses as a bullet with its citation
            note = StructuredNote.parse(text, "Voice Notes")
            assert note.bullets()[0][1].text == "I control [[Cloud Pro]] from my phone" and note.bullets()[0][1].cites
        finally:
            sr.ask_llm_for_placement = orig


def test_session_summary_section():
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        r = _router(vault, properties_enabled=False)
        note = r.get_active_filepath()
        note.write_text("# Voice Notes\n\n## 💡 Notes\n\n### Garden\n- Buy a hose (c1)\n", encoding="utf-8")
        r.session_thoughts[note] = [
            {"text": "I need to buy a new garden hose this weekend", "cite": "([[T#^t-1|🔗 1]])"},
            {"text": "Not sure whether the fence needs a second coat of paint", "cite": "([[T#^t-2|🔗 2]])"},
            {"text": "The tomatoes are growing well this year", "cite": "([[T#^t-3|🔗 3]])"},
        ]

        def fake(config, title, thoughts):
            return {"summary": "Garden jobs: a new hose, the fence paint and the tomatoes growing well.",
                    "action_items": [{"text": "Buy a new garden hose this weekend", "sources": [1]},
                                     {"text": "Book a holiday to Spain", "sources": [2]}],
                    "open_questions": [{"text": "Does the fence need a second coat of paint?", "sources": [2, 9]}]}
        assert r.summarize_note(note, fake)
        text = note.read_text(encoding="utf-8")
        assert r.summary_heading() in text
        assert "- [ ] Buy a new garden hose this weekend ([[T#^t-1|🔗 1]])" in text
        assert "Spain" not in text  # not grounded -> dropped
        assert "- Does the fence need a second coat of paint? ([[T#^t-2|🔗 2]])" in text
        assert "- Buy a hose (c1)" in text
        # summarising again replaces the section instead of adding a second one
        assert r.summarize_note(note, fake)
        assert note.read_text(encoding="utf-8").count(r.summary_heading()) == 1


def test_group_segments_and_undo_phrases():
    from scribe_router import group_segments, is_undo_command
    g = group_segments([(0, 2, "a"), (2.3, 4, "b"), (7, 9, "c"), (9.1, 60, "d")], pause=1.7, max_len=45)
    assert [x[2] for x in g] == ["a b", "c", "d"]
    assert is_undo_command("Scratch that.") and is_undo_command("okay, undo that") and is_undo_command("Remove the last point")
    assert not is_undo_command("Remove one of the sentences.") and not is_undo_command("I want to undo my mistakes in life")


def test_cuda_library_folder_search():
    from scribe_router import cuda_dll_search_dirs
    with tempfile.TemporaryDirectory() as d:
        target = Path(d) / "pip-target"                      # pip install --target layout
        (target / "nvidia" / "cublas" / "bin").mkdir(parents=True)
        (target / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
        dirs = [str(x) for x in cuda_dll_search_dirs(str(target))]
        assert str(target / "nvidia" / "cublas" / "bin") in dirs and str(target / "nvidia" / "cudnn" / "bin") in dirs
        toolkit = Path(d) / "CUDA" / "v12.6"                # existing CUDA Toolkit
        (toolkit / "bin").mkdir(parents=True)
        (toolkit / "bin" / "cublas64_12.dll").write_bytes(b"")
        assert str(toolkit / "bin") in [str(x) for x in cuda_dll_search_dirs(str(toolkit))]
        assert str(toolkit / "bin") in [str(x) for x in cuda_dll_search_dirs(str(toolkit / "bin"))]


def test_batch_mode_inbox_then_organise():
    import scribe_router as sr
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        r = _router(vault, organise_mode="batch", batch_interval_min=0, properties_enabled=False, autolink_enabled=False)
        note = r.get_active_filepath()
        note.write_text("# Voice Notes\n\n## 💡 Notes\n\n### Recipe\n- Use brioche ([[T#^t-0|🔗 0]])\n", encoding="utf-8")
        called = {"n": 0}
        orig = sr.ask_llm_for_placement
        sr.ask_llm_for_placement = lambda *a, **k: called.__setitem__("n", called["n"] + 1)
        try:
            for n, text in enumerate(["Soak each slice for a minute.", "Actually, thirty seconds, longer and it falls apart.",
                                      "Fry in butter on medium heat.", "Serve with maple syrup and berries.",
                                      "Um, yeah okay."], start=1):
                r.process(_thought(text, n))
        finally:
            sr.ask_llm_for_placement = orig
        assert called["n"] == 0  # batch mode never calls the per-thought organiser
        text = note.read_text(encoding="utf-8")
        assert "### 📥 Inbox (to organise)" in text and text.count("🔗 10:00:0") == 5
        assert r.inbox_pending[note]["count"] == 5

        def fake_batch(config, title, outline, thoughts):
            assert "[1] Use brioche" in outline and "Inbox" not in outline
            assert thoughts[1].startswith("Actually, thirty seconds")
            return {"items": [
                {"action": "add", "heading": "Method", "bullet": "Soak each slice for about thirty seconds; longer and it falls apart", "sources": ["T1", "T2"], "target": None},
                {"action": "add", "heading": "Method", "bullet": "Fry in butter on medium heat", "sources": ["T3"], "target": None},
                {"action": "add", "heading": "Method", "bullet": "Book flights to Spain", "sources": ["T4"], "target": None},  # not grounded
                {"action": "update", "heading": "Recipe", "bullet": "Use brioche", "sources": ["T9"], "target": 1},        # bad source
            ]}
        assert r.organise_note(note, "test", batch_fn=fake_batch)
        text = note.read_text(encoding="utf-8")
        print(text)
        assert "Inbox" not in text
        merged = [l for l in text.splitlines() if l.startswith("- Soak each slice")][0]
        assert "🔗 10:00:01" in merged and "🔗 10:00:02" in merged  # both sources kept
        assert "- Fry in butter on medium heat ([[" in text
        assert "Spain" not in text
        assert "### Other thoughts\n- Serve with maple syrup and berries" in text   # left out by the model -> kept
        assert "Um, yeah" not in text                                              # filler stays only in the transcript
        assert "- Use brioche ([[T#^t-0|🔗 0]])" in text
        # undo restores the inbox
        assert r.undo()
        assert "### 📥 Inbox (to organise)" in note.read_text(encoding="utf-8")


def test_batch_model_down_keeps_inbox():
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        r = _router(vault, organise_mode="batch", properties_enabled=False, autolink_enabled=False)
        note = r.get_active_filepath()
        r.process(_thought("Remember to call the plumber about the leak.", 1))
        before = note.read_text(encoding="utf-8")

        def down(*a):
            raise ConnectionError("ollama down")
        assert not r.organise_note(note, "test", batch_fn=down)
        assert note.read_text(encoding="utf-8") == before   # nothing lost, still in the inbox


def test_focus_applies_immediately_in_batch_mode():
    import scribe_router as sr
    from scribe_router import Focus
    with tempfile.TemporaryDirectory() as d:
        vault = Path(d)
        r = _router(vault, organise_mode="batch", properties_enabled=False, autolink_enabled=False)
        note = r.get_active_filepath()
        note.write_text("# Voice Notes\n\n## 💡 Notes\n\n### Method\n- Soak for a minute ([[T#^t-0|🔗 0]])\n", encoding="utf-8")
        r.focus = Focus(note, "Method", ["Soak for a minute"], "correct")
        orig = sr.ask_llm_for_correction
        sr.ask_llm_for_correction = lambda *a: {"bullets": ["Soak for thirty seconds"]}
        try:
            r.process(_thought("Soak for thirty seconds instead.", 1))
        finally:
            sr.ask_llm_for_correction = orig
        text = note.read_text(encoding="utf-8")
        assert "- Soak for thirty seconds ([[T#^t-0" in text and "Inbox" not in text


def test_thinking_off_and_summary_model():
    import sys as _sys, types
    import scribe_router as sr
    calls = []

    class FakeClient:
        reject_think = False

        def __init__(self, host, timeout):
            pass

        def chat(self, **kw):
            calls.append(kw)
            if "think" in kw and FakeClient.reject_think:
                raise RuntimeError('"old-model" does not support thinking')
            return {"message": {"content": '<think>planning…</think>{"summary": "ok", "action_items": [], "open_questions": []}'}}

    saved = _sys.modules.get("ollama")
    _sys.modules["ollama"] = types.SimpleNamespace(Client=FakeClient)
    sr.MODEL_CHECK_ENABLED = False   # never touch a real Ollama server from the tests
    try:
        sr._NO_THINK_SUPPORT.clear()
        live = ScribeConfig(ollama_model="llama3.2:3b")
        assert sr.ask_llm_for_summary(live, "N", ["a thought"])["summary"] == "ok"   # <think> stripped
        assert calls[-1]["model"] == "llama3.2:3b" and calls[-1]["think"] is False
        batch = ScribeConfig(ollama_model="llama3.2:3b", organise_mode="batch", batch_ollama_model="qwen3:30b-a3b")
        sr.ask_llm_for_summary(batch, "N", ["a thought"])
        assert calls[-1]["model"] == "qwen3:30b-a3b" and calls[-1]["options"]["num_ctx"] == batch.batch_num_ctx
        sr.ask_llm_for_batch(batch, "N", "(empty)", ["a thought"])
        assert calls[-1]["model"] == "qwen3:30b-a3b" and calls[-1]["think"] is False
        gpt = ScribeConfig(ollama_model="gpt-oss:20b")
        sr.ask_llm_for_placement(gpt, "(empty)", "x", "N")
        assert calls[-1]["think"] == "low"
        # a server/model that rejects the option: retried without it, then remembered
        FakeClient.reject_think = True
        old = ScribeConfig(ollama_model="old-model")
        assert sr.ask_llm_for_summary(old, "N", ["a thought"]) is not None
        assert "think" not in calls[-1]
        n = len(calls)
        sr.ask_llm_for_summary(old, "N", ["a thought"])
        assert len(calls) == n + 1 and "think" not in calls[-1]
    finally:
        if saved is not None:
            _sys.modules["ollama"] = saved
        else:
            _sys.modules.pop("ollama", None)
        sr._NO_THINK_SUPPORT.clear()
        sr.MODEL_CHECK_ENABLED = True


def test_shared_context_prompt():
    ctx = SharedContext(default_topic="Machine Learning")
    assert ctx.get_initial_prompt() == "Notes about Machine Learning."
    ctx.add_transcript("We need to tune hyperparameters.")
    assert "hyperparameters" in ctx.get_initial_prompt()
    assert parse_voice_command(ctx.get_initial_prompt()) is None


def test_on_this_page_section():
    import scribe_router as sr
    note = ("---\ntags: [voicescribe]\n---\n# Trip\n\nMy own intro line.\n\n## 💡 Notes\n\n"
            "### Flights\n- Book early ([[T#^a|🔗 1]])\n- Window seat\n\n### Food | drink\n- Try paella\n\n"
            "### 📥 Inbox (to organise)\n- one more thing\n\n## 🧾 Session summary · 2026-10-30 14:00\n\nShort.\n")
    out = sr.upsert_toc(note)
    print(out)
    assert out.index("## 🧭 On this page") < out.index("## 💡 Notes") and out.index("My own intro line.") < out.index("## 🧭")
    assert "- [[#💡 Notes|Notes]]" in out
    assert "\t- [[#Flights]] · 2 points" in out
    assert "\t- [[#Food drink]] · 1 point" in out          # '|' cannot be in a link
    assert "\t- [[#📥 Inbox (to organise)]] · 1 waiting" in out
    assert "- [[#🧾 Session summary · 2026-10-30 14:00|Session summary · 2026-10-30 14:00]]" in out
    # refreshing replaces the old list instead of adding a second one
    again = sr.upsert_toc(out.replace("- Window seat\n", ""))
    assert again.count("## 🧭 On this page") == 1 and "[[#Flights]] · 1 point\n" in again
    # the notes themselves still parse the same
    a, b = StructuredNote.parse(note, "Trip"), StructuredNote.parse(out, "Trip")
    assert [x.text for _, x in a.bullets()] == [x.text for _, x in b.bullets()]
    # too few themes: no list (and an old one is removed)
    small = "# X\n\n## 💡 Notes\n\n### Only one\n- a\n"
    assert "On this page" not in sr.upsert_toc(small)
    assert "On this page" not in sr.upsert_toc(sr.upsert_toc(note).replace("### Flights", "- Flights").replace("### Food | drink", "- Food"))


def test_batch_pass_writes_on_this_page():
    with tempfile.TemporaryDirectory() as d:
        r = _router(Path(d), organise_mode="batch", properties_enabled=False, autolink_enabled=False)
        note = r.get_active_filepath()
        r.process(_thought("Buy flour and eggs for the cake.", 1))
        r.process(_thought("The garden fence needs painting.", 2))

        def fake_batch(config, title, outline, thoughts):
            return {"items": [
                {"action": "add", "heading": "Baking", "bullet": "Buy flour and eggs for the cake", "sources": ["T1"], "target": None},
                {"action": "add", "heading": "Garden", "bullet": "The garden fence needs painting", "sources": ["T2"], "target": None},
            ]}
        assert r.organise_note(note, "test", batch_fn=fake_batch)
        text = note.read_text(encoding="utf-8")
        assert "## 🧭 On this page" in text and "[[#Baking]] · 1 point" in text and "[[#Garden]]" in text
        r.config.toc_enabled = False
        r2 = _router(Path(d), organise_mode="live", properties_enabled=False, autolink_enabled=False)
        assert not r2.toc_wanted


def test_model_memory_helpers():
    import scribe_router as sr
    cfg = ScribeConfig(ollama_keep_alive_min=10)
    assert sr.keep_alive_value(cfg) == "600s"
    assert sr.keep_alive_value(ScribeConfig(ollama_keep_alive_min=0)) == "0s"
    assert sr.keep_alive_value(ScribeConfig(ollama_keep_alive_min=-1)) == -1
    lists = {"/api/ps": ["llama3.2:3b"], "/api/tags": ["llama3.2:3b", "qwen3:30b-a3b", "mistral:latest"]}
    http = lambda host, path: lists[path]
    assert sr.ensure_model_loaded(cfg, "llama3.2:3b", http=http)        # already loaded
    assert sr.ensure_model_loaded(cfg, "qwen3:30b-a3b", http=http)      # installed -> loaded on demand
    assert sr.ensure_model_loaded(cfg, "mistral", http=http)            # ':latest' is implied
    assert not sr.ensure_model_loaded(cfg, "not-pulled:7b", http=http)  # not installed
    assert sr.ensure_model_loaded(cfg, "anything", http=lambda h, p: None)  # server down: let the call report it
    assert sr.session_models(ScribeConfig(ollama_model="a", batch_ollama_model="b")) == ["a", "b"]


def test_whisper_released_when_idle():
    import queue as _q
    import threading as _th
    import scribe_router as sr
    t = sr.TranscriptionThread(_q.Queue(), _q.Queue(), _th.Event(), ScribeConfig(whisper_idle_unload_min=1))
    t.whisper_model = object()
    t.last_activity -= 30
    t.idle_check()
    assert t.whisper_model is not None
    t.last_activity -= 60
    t.idle_check()
    assert t.whisper_model is None
    t2 = sr.TranscriptionThread(_q.Queue(), _q.Queue(), _th.Event(), ScribeConfig(whisper_idle_unload_min=0))
    t2.whisper_model = object()
    t2.last_activity -= 10 ** 6
    t2.idle_check()
    assert t2.whisper_model is not None   # 0 = never


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL  {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
