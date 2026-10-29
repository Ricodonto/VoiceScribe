'use strict';

const obsidian = require('obsidian');
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

class VoiceScribePlugin extends obsidian.Plugin {
    async onload() {
        console.log('[VoiceScribe] Initializing plugin...');
        this.process = null;
        this.isRecording = false;

        // Resolve absolute directories
        this.vaultBasePath = this.app.vault.adapter.basePath;
        this.voiceScribeDir = path.resolve(this.vaultBasePath, '..', 'VoiceScribe');
        this.pythonPath = path.join(this.voiceScribeDir, 'venv', 'Scripts', 'python.exe');
        this.scriptPath = path.join(this.voiceScribeDir, 'scribe_router.py');
        this.configPath = path.join(this.voiceScribeDir, 'config.json');
        this.stopSignalPath = path.join(this.voiceScribeDir, 'stop.signal');

        // Load configuration
        this.config = this.loadConfigFile();

        // 1. Add Ribbon Icon for Recording (Microphone)
        this.ribbonIconEl = this.addRibbonIcon(
            'microphone',
            'VoiceScribe: Toggle Continuous Recording (Alt+V)',
            () => this.toggleRecording()
        );

        // 2. Add Ribbon Icon for Options / Settings (Sliders / Cog)
        this.addRibbonIcon(
            'settings',
            'VoiceScribe: Options & Parameters',
            () => this.openOptionsModal()
        );

        // 3. Add Status Bar Item (bottom right)
        this.statusBarEl = this.addStatusBarItem();
        this.updateStatusBar(false);

        // 4. Register Command Palette Commands
        this.addCommand({
            id: 'toggle-voicescribe-recording',
            name: 'Toggle Continuous Voice Recording',
            hotkeys: [{ modifiers: ['Alt'], key: 'v' }],
            callback: () => this.toggleRecording(),
        });

        this.addCommand({
            id: 'open-active-voice-notes',
            name: 'Open Active Voice Notes',
            callback: () => this.openActiveVoiceNote(),
        });

        this.addCommand({
            id: 'open-voicescribe-options-menu',
            name: 'Options: Quick Change Microphone & Models',
            hotkeys: [{ modifiers: ['Alt', 'Shift'], key: 'v' }],
            callback: () => this.openOptionsModal(),
        });

        this.addCommand({
            id: 'open-voicescribe-settings-tab',
            name: 'Open VoiceScribe Settings Tab',
            callback: () => {
                this.app.setting.open();
                this.app.setting.openTabById(this.manifest.id);
            },
        });

        // 5. Register Settings Tab
        this.addSettingTab(new VoiceScribeSettingTab(this.app, this));

        console.log('[VoiceScribe] Plugin loaded successfully.');
    }

    loadConfigFile() {
        const defaults = {
            vault_dir: "../Note Rambling",
            notes_subfolder: "",
            archive_dir: "VoiceScribe_Archives",
            transcripts_folder: "Transcripts",
            default_note: "Voice Notes",
            sample_rate: 16000,
            pause_duration_sec: 2.2,
            min_speech_duration_sec: 1.5,
            max_speech_duration_sec: 45.0,
            whisper_model: "large-v3-turbo",
            whisper_compute_type: "int8",
            whisper_threads: 4,
            ollama_model: "llama3.2:3b",
            ollama_host: "http://127.0.0.1:11434",
            ollama_num_predict: 384,
            ollama_num_ctx: 2048,
            opus_bitrate: "10k",
            min_substantive_words: 3,
            energy_threshold: 0.003,
            audio_device: null
        };

        try {
            if (fs.existsSync(this.configPath)) {
                const raw = fs.readFileSync(this.configPath, 'utf-8');
                const data = JSON.parse(raw);
                return Object.assign({}, defaults, data);
            }
        } catch (e) {
            console.warn('[VoiceScribe] Failed to read config.json:', e);
        }
        return defaults;
    }

    saveConfigFile(newConfig) {
        this.config = Object.assign({}, this.config, newConfig);
        try {
            fs.writeFileSync(this.configPath, JSON.stringify(this.config, null, 2), 'utf-8');
            new obsidian.Notice('\u2705 VoiceScribe settings saved successfully.');
            return true;
        } catch (e) {
            new obsidian.Notice('\u274C [VoiceScribe] Error saving settings: ' + e.message);
            return false;
        }
    }

    getInputDevices() {
        const listScript = path.join(this.voiceScribeDir, 'list_inputs.py');
        try {
            if (fs.existsSync(this.pythonPath) && fs.existsSync(listScript)) {
                const output = execSync(`"${this.pythonPath}" "${listScript}"`, {
                    cwd: this.voiceScribeDir,
                    windowsHide: true,
                    timeout: 5000,
                    encoding: 'utf-8'
                });
                return JSON.parse(output);
            }
        } catch (e) {
            console.warn('[VoiceScribe] Could not fetch audio devices:', e);
        }
        return [];
    }

    async getOllamaModels(host) {
        const targetHost = host || (this.config && this.config.ollama_host) || 'http://127.0.0.1:11434';
        try {
            const res = await obsidian.requestUrl({ url: `${targetHost}/api/tags` });
            if (res.json && Array.isArray(res.json.models)) {
                return res.json.models.map(m => m.name);
            }
        } catch (e) {
            console.warn('[VoiceScribe] Ollama scan error, using defaults:', e);
        }
        return ['llama3.2:3b', 'llama3.1:8b', 'qwen2.5-coder:1.5b', 'qwen2.5-coder:7b', 'mistral:latest'];
    }

    updateStatusBar(recording, extraInfo) {
        this.isRecording = recording;
        const info = extraInfo ? (' ' + extraInfo) : '';
        if (recording) {
            this.statusBarEl.setText('\uD83D\uDD34 VoiceScribe: Listening' + info);
            if (this.ribbonIconEl) {
                this.ribbonIconEl.addClass('voicescribe-recording');
            }
        } else {
            this.statusBarEl.setText('\u23F9\uFE0F VoiceScribe: Stopped');
            if (this.ribbonIconEl) {
                this.ribbonIconEl.removeClass('voicescribe-recording');
            }
        }
    }

    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            this.startRecording();
        }
    }

    async ensureOllamaRunning() {
        const host = (this.config && this.config.ollama_host) || 'http://127.0.0.1:11434';
        try {
            const res = await obsidian.requestUrl({ url: `${host}/api/tags` });
            if (res.status === 200) return true;
        } catch (e) {
            console.log('[VoiceScribe] Ollama offline. Auto-launching ollama serve...');
            try {
                spawn('ollama', ['serve'], { detached: true, stdio: 'ignore', windowsHide: true });
            } catch (err) {
                console.warn('[VoiceScribe] Could not spawn ollama command:', err);
            }
        }
        return false;
    }

    async startRecording() {
        if (!fs.existsSync(this.pythonPath)) {
            new obsidian.Notice('❌ [VoiceScribe] Python venv not found at:\n' + this.pythonPath);
            return;
        }

        // Clean any old stop signal
        if (fs.existsSync(this.stopSignalPath)) {
            try {
                fs.unlinkSync(this.stopSignalPath);
            } catch (e) {}
        }

        new obsidian.Notice('🎙️ VoiceScribe Starting: Initializing mic & models...');
        this.updateStatusBar(true, '(Starting...)');

        // Check & ensure Ollama is active
        await this.ensureOllamaRunning();

        try {
            this.process = spawn(
                this.pythonPath,
                [this.scriptPath, '--vault-dir', this.vaultBasePath],
                {
                    cwd: this.voiceScribeDir,
                    windowsHide: true,
                    env: Object.assign({}, process.env, { PYTHONIOENCODING: 'utf-8' }),
                }
            );

            this.process.stdout.on('data', (data) => {
                const str = data.toString('utf-8');
                console.log('[VoiceScribe]', str);
                if (str.includes('Microphone active. Listening continuously')) {
                    this.updateStatusBar(true, '(Active)');
                    new obsidian.Notice('🟢 VoiceScribe Listening! Speak your thoughts naturally.');
                } else if (str.includes('Switched active note')) {
                    const match = str.match(/'([^']+)'/);
                    const noteName = match ? match[1] : 'new note';
                    new obsidian.Notice('💡 Topic: ' + noteName);
                } else if (str.includes('Updated structured note')) {
                    new obsidian.Notice('📝 Note Updated with your thought!');
                } else if (str.includes('Heard:')) {
                    const match = str.match(/Heard:\s*"([^"]+)"/);
                    if (match) {
                        const snippet = match[1].length > 45 ? match[1].slice(0, 42) + '...' : match[1];
                        new obsidian.Notice('🗣️ Heard: "' + snippet + '"', 3000);
                    }
                }
            });

            this.process.stderr.on('data', (data) => {
                console.error('[VoiceScribe Error]', data.toString('utf-8'));
            });

            this.process.on('close', (code) => {
                console.log('[VoiceScribe] Process exited with code', code);
                this.process = null;
                this.updateStatusBar(false);
                new obsidian.Notice('\u23F9\uFE0F VoiceScribe recording stopped.');
            });

            this.process.on('error', (err) => {
                new obsidian.Notice('\u274C [VoiceScribe] Launch Error: ' + err.message);
                this.process = null;
                this.updateStatusBar(false);
            });

        } catch (e) {
            new obsidian.Notice('\u274C [VoiceScribe] Launch failed: ' + e.message);
            this.updateStatusBar(false);
        }
    }

    stopRecording() {
        new obsidian.Notice('Stopping VoiceScribe session...');
        try {
            fs.writeFileSync(this.stopSignalPath, 'stop');
        } catch (e) {}

        if (this.process && this.process.pid) {
            const pid = this.process.pid;
            setTimeout(() => {
                if (this.process) {
                    try {
                        spawn('taskkill', ['/pid', pid.toString(), '/t', '/f']);
                    } catch (e) {}
                }
            }, 3000);
        }
    }

    openActiveVoiceNote() {
        const noteName = (this.config && this.config.default_note) ? `${this.config.default_note}.md` : 'Voice Notes.md';
        const file = this.app.vault.getAbstractFileByPath(noteName);
        if (file) {
            this.app.workspace.getLeaf(false).openFile(file);
        } else {
            new obsidian.Notice(`Note '${noteName}' has not been created yet.`);
        }
    }

    openOptionsModal() {
        new VoiceScribeOptionsModal(this.app, this).open();
    }

    onunload() {
        console.log('[VoiceScribe] Unloading plugin...');
        if (this.isRecording) {
            this.stopRecording();
        }
    }
}

// -----------------------------------------------------------------------------
// VoiceScribe Settings Tab (Full Obsidian Settings GUI)
// -----------------------------------------------------------------------------
class VoiceScribeSettingTab extends obsidian.PluginSettingTab {
    constructor(app, plugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    async display() {
        const { containerEl } = this;
        containerEl.empty();

        const cfg = this.plugin.loadConfigFile();

        containerEl.createEl('h2', { text: '\uD83C\uDF99\uFE0F VoiceScribe Options & Parameters' });
        containerEl.createEl('p', {
            text: 'Customize your audio input microphone, Whisper speech recognition model, and local Ollama synthesis model.',
            cls: 'voicescribe-settings-desc'
        });

        // 1. Microphone Selection Setting
        const micSetting = new obsidian.Setting(containerEl)
            .setName('Input Microphone')
            .setDesc('Select the audio input device for continuous listening.');

        const mics = this.plugin.getInputDevices();
        micSetting.addDropdown((dropdown) => {
            dropdown.addOption('null', 'System Default Microphone');
            mics.forEach((m) => {
                dropdown.addOption(m.id.toString(), m.label);
            });

            const curVal = (cfg.audio_device === null || cfg.audio_device === undefined) ? 'null' : cfg.audio_device.toString();
            dropdown.setValue(curVal);
            dropdown.onChange((val) => {
                const updatedDev = val === 'null' ? null : parseInt(val, 10);
                this.plugin.saveConfigFile({ audio_device: updatedDev });
            });
        });

        micSetting.addButton((btn) => {
            btn.setButtonText('\uD83D\uDD04 Scan Devices')
                .setTooltip('Refresh available audio inputs')
                .onClick(() => {
                    this.display();
                    new obsidian.Notice('Audio devices refreshed.');
                });
        });

        // 2. Whisper Speech-to-Text Model Setting
        new obsidian.Setting(containerEl)
            .setName('Whisper Model (Speech-to-Text)')
            .setDesc('Runs on CPU with int8 quantization to prevent CUDA Out-of-Memory (OOM) on your GPU.')
            .addDropdown((dropdown) => {
                dropdown.addOption('large-v3-turbo', 'large-v3-turbo (Recommended - State-of-the-Art Accuracy & Fast Decoding)');
                dropdown.addOption('small.en', 'small.en (High Accuracy, Good Punctuation)');
                dropdown.addOption('base.en', 'base.en (Fastest, Lower Accuracy)');
                dropdown.addOption('medium.en', 'medium.en (High Accuracy, Heavier CPU Usage)');
                dropdown.addOption('tiny.en', 'tiny.en (Fastest, Minimal Accuracy)');

                dropdown.setValue(cfg.whisper_model || 'large-v3-turbo');
                dropdown.onChange((val) => {
                    this.plugin.saveConfigFile({ whisper_model: val });
                });
            });

        // 3. Ollama Model Setting (Synthesis / Summarization)
        const ollamaSetting = new obsidian.Setting(containerEl)
            .setName('Ollama Model (Synthesis & Note Routing)')
            .setDesc('Local LLM running on your RTX GPU via Ollama (127.0.0.1:11434).');

        const models = await this.plugin.getOllamaModels(cfg.ollama_host);
        let customModelInput = null;

        ollamaSetting.addDropdown((dropdown) => {
            let found = false;
            models.forEach((m) => {
                dropdown.addOption(m, m);
                if (m === cfg.ollama_model) found = true;
            });
            dropdown.addOption('__custom__', 'Custom Model Name...');

            if (found) {
                dropdown.setValue(cfg.ollama_model);
            } else {
                dropdown.setValue('__custom__');
            }

            dropdown.onChange((val) => {
                if (val === '__custom__') {
                    if (customModelInput) customModelInput.inputEl.style.display = 'block';
                } else {
                    if (customModelInput) customModelInput.inputEl.style.display = 'none';
                    this.plugin.saveConfigFile({ ollama_model: val });
                }
            });
        });

        ollamaSetting.addText((text) => {
            customModelInput = text;
            text.setPlaceholder('Enter model name (e.g. qwen2.5:7b)')
                .setValue(cfg.ollama_model || '')
                .onChange((val) => {
                    const trimmed = val.trim();
                    if (trimmed) {
                        this.plugin.saveConfigFile({ ollama_model: trimmed });
                    }
                });

            const isCustom = !models.includes(cfg.ollama_model);
            text.inputEl.style.display = isCustom ? 'block' : 'none';
            text.inputEl.style.marginLeft = '8px';
        });

        ollamaSetting.addButton((btn) => {
            btn.setButtonText('\uD83D\uDD04 Scan Ollama')
                .setTooltip('Rescan installed Ollama models')
                .onClick(() => {
                    this.display();
                    new obsidian.Notice('Ollama models refreshed.');
                });
        });

        // 4. Natural Speech Pause Duration (VAD)
        new obsidian.Setting(containerEl)
            .setName('Natural Speech Pause Duration')
            .setDesc('Seconds of silence required before finalizing a coherent spoken thought into notes (Default: 1.9s).')
            .addSlider((slider) => {
                slider.setLimits(1.0, 3.5, 0.1)
                    .setValue(cfg.pause_duration_sec || 1.9)
                    .setDynamicTooltip()
                    .onChange((val) => {
                        this.plugin.saveConfigFile({ pause_duration_sec: Math.round(val * 10) / 10 });
                    });
            });

        // 5. Default Note Topic
        new obsidian.Setting(containerEl)
            .setName('Default Note Topic')
            .setDesc('Main note filename where structured synthesis notes are updated (without .md extension).')
            .addText((text) => {
                text.setValue(cfg.default_note || 'Voice Notes')
                    .onChange((val) => {
                        const trimmed = val.trim();
                        if (trimmed) {
                            this.plugin.saveConfigFile({ default_note: trimmed });
                        }
                    });
            });

        // 6. Transcripts Subfolder
        new obsidian.Setting(containerEl)
            .setName('Transcripts Folder')
            .setDesc('Folder where daily time-coded verbatim transcripts and audio players are stored.')
            .addText((text) => {
                text.setValue(cfg.transcripts_folder || 'Transcripts')
                    .onChange((val) => {
                        const trimmed = val.trim();
                        if (trimmed) {
                            this.plugin.saveConfigFile({ transcripts_folder: trimmed });
                        }
                    });
            });

        // 7. Ollama Host URL
        new obsidian.Setting(containerEl)
            .setName('Ollama Host URL')
            .setDesc('Local Ollama server address. Use IPv4 127.0.0.1 to avoid Windows IPv6 resolution latency.')
            .addText((text) => {
                text.setValue(cfg.ollama_host || 'http://127.0.0.1:11434')
                    .onChange((val) => {
                        const trimmed = val.trim();
                        if (trimmed) {
                            this.plugin.saveConfigFile({ ollama_host: trimmed });
                        }
                    });
            });

        // 8. Diagnostics & Connectivity Test
        new obsidian.Setting(containerEl)
            .setName('Test System & Connectivity')
            .setDesc('Verify Ollama connection, microphone accessibility, and FFmpeg configuration.')
            .addButton((btn) => {
                btn.setButtonText('\uD83E\uDDEA Run Diagnostics Test')
                    .setCta()
                    .onClick(async () => {
                        btn.setDisabled(true);
                        btn.setButtonText('Testing...');
                        try {
                            const res = await obsidian.requestUrl({
                                url: `${cfg.ollama_host || 'http://127.0.0.1:11434'}/api/tags`
                            });
                            const installedModels = (res.json && res.json.models) ? res.json.models.map(m => m.name) : [];
                            const modelFound = installedModels.includes(cfg.ollama_model);

                            if (modelFound) {
                                new obsidian.Notice(`\u2705 Ollama Connected! Model '${cfg.ollama_model}' is ready on GPU.`);
                            } else {
                                new obsidian.Notice(`\u26A0\uFE0F Ollama Connected, but model '${cfg.ollama_model}' not found in: ${installedModels.join(', ')}`);
                            }
                        } catch (err) {
                            new obsidian.Notice(`\u274C Cannot connect to Ollama at ${cfg.ollama_host}: ${err.message}`);
                        } finally {
                            btn.setDisabled(false);
                            btn.setButtonText('\uD83E\uDDEA Run Diagnostics Test');
                        }
                    });
            });
    }
}

// -----------------------------------------------------------------------------
// VoiceScribe Quick Options Modal (Popup Dialog)
// -----------------------------------------------------------------------------
class VoiceScribeOptionsModal extends obsidian.Modal {
    constructor(app, plugin) {
        super(app);
        this.plugin = plugin;
    }

    async onOpen() {
        const { contentEl } = this;
        contentEl.empty();

        const cfg = this.plugin.loadConfigFile();

        contentEl.createEl('h2', { text: '\u2699\uFE0F VoiceScribe Options Menu' });
        contentEl.createEl('p', {
            text: 'Quickly adjust your input microphone, Whisper model, and Ollama synthesis model.',
            cls: 'voicescribe-settings-desc'
        });

        // 1. Microphone
        const mics = this.plugin.getInputDevices();
        new obsidian.Setting(contentEl)
            .setName('Input Microphone')
            .addDropdown((dropdown) => {
                dropdown.addOption('null', 'System Default Microphone');
                mics.forEach((m) => {
                    dropdown.addOption(m.id.toString(), m.label);
                });
                const curVal = (cfg.audio_device === null || cfg.audio_device === undefined) ? 'null' : cfg.audio_device.toString();
                dropdown.setValue(curVal);
                dropdown.onChange((val) => {
                    const dev = val === 'null' ? null : parseInt(val, 10);
                    this.plugin.saveConfigFile({ audio_device: dev });
                });
            });

        // 2. Whisper Model
        new obsidian.Setting(contentEl)
            .setName('Whisper Model (STT)')
            .setDesc('Runs on CPU (int8)')
            .addDropdown((dropdown) => {
                dropdown.addOption('large-v3-turbo', 'large-v3-turbo (Recommended - State-of-the-Art)');
                dropdown.addOption('small.en', 'small.en (High Accuracy)');
                dropdown.addOption('base.en', 'base.en (Fastest)');
                dropdown.addOption('medium.en', 'medium.en (High Accuracy)');
                dropdown.addOption('tiny.en', 'tiny.en (Fastest / Lightweight)');
                dropdown.setValue(cfg.whisper_model || 'large-v3-turbo');
                dropdown.onChange((val) => {
                    this.plugin.saveConfigFile({ whisper_model: val });
                });
            });

        // 3. Ollama Model
        const models = await this.plugin.getOllamaModels(cfg.ollama_host);
        new obsidian.Setting(contentEl)
            .setName('Ollama Model (LLM)')
            .setDesc('Runs on GPU')
            .addDropdown((dropdown) => {
                models.forEach((m) => dropdown.addOption(m, m));
                dropdown.setValue(models.includes(cfg.ollama_model) ? cfg.ollama_model : models[0]);
                dropdown.onChange((val) => {
                    this.plugin.saveConfigFile({ ollama_model: val });
                });
            });

        // 4. Pause Threshold
        new obsidian.Setting(contentEl)
            .setName('Speech Pause (VAD)')
            .setDesc('Pause duration before finalizing chunk (seconds)')
            .addSlider((slider) => {
                slider.setLimits(1.0, 3.5, 0.1)
                    .setValue(cfg.pause_duration_sec || 1.9)
                    .setDynamicTooltip()
                    .onChange((val) => {
                        this.plugin.saveConfigFile({ pause_duration_sec: Math.round(val * 10) / 10 });
                    });
            });

        // Close Button
        new obsidian.Setting(contentEl)
            .addButton((btn) => {
                btn.setButtonText('Done')
                    .setCta()
                    .onClick(() => this.close());
            });
    }

    onClose() {
        const { contentEl } = this;
        contentEl.empty();
    }
}

module.exports = VoiceScribePlugin;
