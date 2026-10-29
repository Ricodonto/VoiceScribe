#!/usr/bin/env python3
"""
VoiceScribe Interactive Configuration Menu
Allows viewing and updating:
- Input microphone device
- Whisper speech-to-text model (CPU int8)
- Ollama synthesis model (GPU)
- Natural pause duration (VAD)
- Vault folder and default note topic
"""

import json
import os
import sys
from pathlib import Path
import sounddevice as sd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> dict:
    default_cfg = {
        "vault_dir": "../Note Rambling",
        "notes_subfolder": "",
        "archive_dir": "VoiceScribe_Archives",
        "transcripts_folder": "Transcripts",
        "default_note": "Voice Notes",
        "sample_rate": 16000,
        "pause_duration_sec": 2.2,
        "min_speech_duration_sec": 1.5,
        "max_speech_duration_sec": 45.0,
        "whisper_model": "large-v3-turbo",
        "whisper_compute_type": "int8",
        "whisper_threads": 4,
        "whisper_beam_size": 3,
        "context_prompt_enabled": True,
        "ollama_model": "llama3.2:3b",
        "ollama_host": "http://127.0.0.1:11434",
        "ollama_num_predict": 384,
        "ollama_num_ctx": 2048,
        "opus_bitrate": "10k",
        "min_substantive_words": 3,
        "energy_threshold": 0.003,
        "audio_device": None,
    }
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                default_cfg.update(data)
        except Exception as e:
            print(f"[WARN] Error reading {CONFIG_PATH}: {e}")
    return default_cfg


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n[OK] Configuration successfully saved to:\n     {CONFIG_PATH}")


def get_available_mics():
    from list_inputs import get_input_devices
    return get_input_devices()


def get_available_ollama_models(host="http://127.0.0.1:11434"):
    try:
        import urllib.request
        req = urllib.request.Request(f"{host}/api/tags", headers={"User-Agent": "VoiceScribe"})
        with urllib.request.urlopen(req, timeout=2.0) as res:
            data = json.loads(res.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return ["llama3.2:3b", "llama3.1:8b", "qwen2.5-coder:1.5b", "qwen2.5-coder:7b", "mistral:latest"]


def select_microphone(cfg: dict):
    mics = get_available_mics()
    print("\n--- AVAILABLE INPUT MICROPHONES ---")
    is_def_cur = " [CURRENT]" if cfg.get("audio_device") is None else ""
    print(f"  [0] System Default Microphone{is_def_cur}")
    for i, m in enumerate(mics, start=1):
        cur = " [CURRENT]" if cfg.get("audio_device") == m["id"] else ""
        print(f"  [{i}] {m['label']}{cur}")

    choice = input("\nSelect microphone number (Enter to cancel): ").strip()
    if not choice:
        return
    try:
        idx = int(choice)
        if idx == 0:
            cfg["audio_device"] = None
            print("[SET] Audio device set to System Default.")
        elif 1 <= idx <= len(mics):
            sel = mics[idx - 1]
            cfg["audio_device"] = sel["id"]
            print(f"[SET] Audio device set to: {sel['name']} (ID: {sel['id']})")
        else:
            print("[WARN] Invalid selection.")
    except ValueError:
        print("[WARN] Invalid input.")


def select_whisper_model(cfg: dict):
    models = [
        ("large-v3-turbo", "large-v3-turbo (Recommended - State-of-the-Art Accuracy & Fast Decoding)"),
        ("small.en", "small.en (High accuracy, robust punctuation)"),
        ("base.en", "base.en (Fastest, lower accuracy)"),
        ("medium.en", "medium.en (High accuracy, heavier CPU usage)"),
        ("tiny.en", "tiny.en (Fastest, lowest accuracy)"),
    ]
    print("\n--- WHISPER MODELS (CPU int8) ---")
    for i, (m, desc) in enumerate(models, start=1):
        cur = " [CURRENT]" if cfg.get("whisper_model") == m else ""
        print(f"  [{i}] {desc}{cur}")

    choice = input("\nSelect Whisper model number (Enter to cancel): ").strip()
    if not choice:
        return
    try:
        idx = int(choice)
        if 1 <= idx <= len(models):
            cfg["whisper_model"] = models[idx - 1][0]
            print(f"[SET] Whisper model set to: {cfg['whisper_model']}")
        else:
            print("[WARN] Invalid selection.")
    except ValueError:
        print("[WARN] Invalid input.")


def select_ollama_model(cfg: dict):
    models = get_available_ollama_models(cfg.get("ollama_host", "http://127.0.0.1:11434"))
    print("\n--- DETECTED OLLAMA MODELS (GPU) ---")
    for i, m in enumerate(models, start=1):
        cur = " [CURRENT]" if cfg.get("ollama_model") == m else ""
        print(f"  [{i}] {m}{cur}")
    print(f"  [{len(models) + 1}] Custom Model Name...")

    choice = input("\nSelect Ollama model number (Enter to cancel): ").strip()
    if not choice:
        return
    try:
        idx = int(choice)
        if 1 <= idx <= len(models):
            cfg["ollama_model"] = models[idx - 1]
            print(f"[SET] Ollama model set to: {cfg['ollama_model']}")
        elif idx == len(models) + 1:
            custom = input("Enter custom Ollama model name: ").strip()
            if custom:
                cfg["ollama_model"] = custom
                print(f"[SET] Ollama model set to: {cfg['ollama_model']}")
        else:
            print("[WARN] Invalid selection.")
    except ValueError:
        print("[WARN] Invalid input.")


def set_pause_duration(cfg: dict):
    current = cfg.get("pause_duration_sec", 1.9)
    print(f"\nCurrent pause duration: {current} seconds.")
    print("Recommended: 1.8 to 2.0s for natural sentence flow.")
    val = input("Enter new pause duration in seconds (e.g. 1.9, Enter to cancel): ").strip()
    if not val:
        return
    try:
        num = float(val)
        if 0.5 <= num <= 5.0:
            cfg["pause_duration_sec"] = round(num, 2)
            print(f"[SET] Pause duration set to: {cfg['pause_duration_sec']}s")
        else:
            print("[WARN] Duration must be between 0.5 and 5.0 seconds.")
    except ValueError:
        print("[WARN] Invalid number.")


def test_system_setup(cfg: dict):
    print("\n--- RUNNING SYSTEM DIAGNOSTICS & SETUP TEST ---")
    import shutil

    # 1. Test FFmpeg
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    if ffmpeg_ok:
        print("  [OK] FFmpeg found in PATH (Opus audio archival ready).")
    else:
        print("  [WARN] FFmpeg not found in PATH! Audio cannot be compressed to Opus.")

    # 2. Test Audio Device
    dev = cfg.get("audio_device")
    try:
        if dev is None:
            dev_info = sd.query_devices(kind="input")
            print(f"  [OK] Audio Input: System Default -> {dev_info['name']}")
        else:
            dev_info = sd.query_devices(dev, "input")
            print(f"  [OK] Audio Input: ID {dev} -> {dev_info['name']}")
    except Exception as e:
        print(f"  [FAIL] Audio Input Error: {e}")

    # 3. Test Ollama Connectivity & Model
    host = cfg.get("ollama_host", "http://127.0.0.1:11434")
    target_model = cfg.get("ollama_model", "llama3.2:3b")
    try:
        models = get_available_ollama_models(host)
        if target_model in models:
            print(f"  [OK] Ollama connected at {host}. Model '{target_model}' is installed and ready.")
        else:
            print(f"  [WARN] Ollama connected, but model '{target_model}' not found in installed models: {models}")
    except Exception as e:
        print(f"  [FAIL] Cannot reach Ollama at {host}: {e}")

    # 4. Test Whisper Cache
    wh_model = cfg.get("whisper_model", "small.en")
    print(f"  [INFO] Whisper configured for '{wh_model}' (CPU int8).")
    print("------------------------------------------------")
    input("Press Enter to return to main menu...")


def interactive_menu():
    cfg = load_config()
    while True:
        device_label = "System Default"
        if cfg.get("audio_device") is not None:
            try:
                d = sd.query_devices(cfg["audio_device"], "input")
                device_label = f"{d['name']} (ID {cfg['audio_device']})"
            except Exception:
                device_label = f"Device ID {cfg['audio_device']}"

        print("\n" + "=" * 65)
        print("          VOICESCRIBE CONFIGURATION OPTIONS MENU          ")
        print("=" * 65)
        print(f"  1. Input Microphone    : {device_label}")
        print(f"  2. Whisper Model (CPU) : {cfg.get('whisper_model', 'small.en')}")
        print(f"  3. Ollama Model (GPU)  : {cfg.get('ollama_model', 'llama3.2:3b')}")
        print(f"  4. Speech Pause (VAD)  : {cfg.get('pause_duration_sec', 1.9)}s")
        print(f"  5. Default Note Topic  : {cfg.get('default_note', 'Voice Notes')}")
        print(f"  6. Transcripts Folder  : {cfg.get('transcripts_folder', 'Transcripts')}")
        print(f"  7. Target Vault Folder : {cfg.get('vault_dir', '../Note Rambling')}")
        print(f"  8. Run Diagnostics & Test Setup")
        print("-" * 65)
        print("  [S] Save Configuration    [Q] Quit without Saving")
        print("=" * 65)

        choice = input("Select an option (1-8, S, Q): ").strip().upper()
        if choice == "1":
            select_microphone(cfg)
        elif choice == "2":
            select_whisper_model(cfg)
        elif choice == "3":
            select_ollama_model(cfg)
        elif choice == "4":
            set_pause_duration(cfg)
        elif choice == "5":
            val = input(f"Enter default note name (current: {cfg.get('default_note')}): ").strip()
            if val:
                cfg["default_note"] = val
        elif choice == "6":
            val = input(f"Enter transcripts folder (current: {cfg.get('transcripts_folder')}): ").strip()
            if val:
                cfg["transcripts_folder"] = val
        elif choice == "7":
            val = input(f"Enter vault path (current: {cfg.get('vault_dir')}): ").strip()
            if val:
                cfg["vault_dir"] = val
        elif choice == "8":
            test_system_setup(cfg)
        elif choice == "S":
            save_config(cfg)
            break
        elif choice == "Q":
            print("Exited without saving.")
            break
        else:
            print("[WARN] Invalid option. Enter 1-8, S, or Q.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnostics":
        test_system_setup(load_config())
    else:
        interactive_menu()

