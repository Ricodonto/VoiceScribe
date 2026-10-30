'use strict';

/*
 * VoiceScribe Obsidian plugin
 * Launches the local Python backend (VoiceScribe/scribe_router.py), shows a live session
 * dashboard (status, audio timeline, console), follows the note being written, lets you
 * focus the model on a selected part of a note, and exposes every backend parameter
 * (stored in VoiceScribe/config.json).
 */

const obsidian = require('obsidian');
const { spawn, execFile } = require('child_process');
const path = require('path');
const fs = require('fs');

const IS_WIN = process.platform === 'win32';
const VIEW_TYPE_DASHBOARD = 'voicescribe-dashboard';
const ENV_STEP = 0.5;          // seconds per stored loudness value in the session timeline
const MAX_LOG_LINES = 1500;
const RECENT_HIGHLIGHT_MS = 30000;

const CONFIG_DEFAULTS = {
    vault_dir: '../Note Rambling',
    notes_subfolder: '',
    archive_dir: 'VoiceScribe_Archives',
    transcripts_folder: 'Transcripts',
    default_note: 'Voice Notes',
    audio_device: null,
    sample_rate: 16000,
    energy_threshold: 0.0012,
    pause_duration_sec: 1.7,
    min_speech_duration_sec: 0.8,
    min_voiced_sec: 0.3,
    max_speech_duration_sec: 45.0,
    normalize_audio: true,
    whisper_model: 'large-v3-turbo',
    whisper_compute_type: 'int8',
    whisper_threads: 4,
    whisper_beam_size: 3,
    whisper_language: 'en',
    whisper_vad_filter: true,
    context_prompt_enabled: true,
    allow_model_downloads: false,
    whisper_device: 'auto',
    whisper_gpu_compute_type: 'int8_float16',
    cuda_libs_dir: '',
    organise_mode: 'live',
    batch_interval_min: 5,
    batch_ollama_model: '',
    batch_num_ctx: 8192,
    batch_num_predict: 2048,
    batch_max_thoughts: 25,
    toc_enabled: true,
    summary_enabled: true,
    summary_min_thoughts: 3,
    autolink_enabled: true,
    properties_enabled: true,
    properties_tag: 'voicescribe',
    llm_enabled: true,
    ollama_model: 'llama3.2:3b',
    ollama_host: 'http://127.0.0.1:11434',
    ollama_num_predict: 512,
    ollama_num_ctx: 4096,
    ollama_timeout_sec: 120,
    ollama_keep_alive_min: 10,
    unload_on_stop: true,
    whisper_idle_unload_min: 15,
    opus_bitrate: '12k',
    min_substantive_words: 1,
};

const PLUGIN_DEFAULTS = {
    backendDir: '',      // empty = <vault>/../VoiceScribe
    pythonPath: '',      // empty = <backendDir>/venv/Scripts/python.exe
    showSpeechNotices: false,
    openDashboardOnStart: true,
    followActiveNote: true,
    saveSessionTimeline: true,
    startNoteMode: 'ask',          // ask | new | default | last
    newNoteTemplate: 'Voice note {date} {time}',
    recentNotes: [],
    compactSources: true,
    guideSeen: false,
};

const WHISPER_MODELS = [
    ['large-v3-turbo', 'large-v3-turbo — best accuracy/speed balance (recommended)'],
    ['distil-large-v3', 'distil-large-v3 — fast, English'],
    ['large-v3', 'large-v3 — most accurate, slowest'],
    ['medium.en', 'medium.en — accurate, heavy'],
    ['small.en', 'small.en — good, light'],
    ['base.en', 'base.en — fast, lower accuracy'],
    ['tiny.en', 'tiny.en — fastest, lowest accuracy'],
];

function runFile(file, args, opts) {
    return new Promise((resolve, reject) => {
        execFile(file, args, Object.assign({ windowsHide: true, encoding: 'utf-8', maxBuffer: 4 * 1024 * 1024 }, opts || {}),
            (err, stdout, stderr) => {
                if (err) { err.stdout = stdout; err.stderr = stderr; reject(err); }
                else resolve(stdout);
            });
    });
}

const pad2 = (n) => String(n).padStart(2, '0');
function clock(d) { d = d || new Date(); return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`; }
function duration(sec) {
    sec = Math.max(0, Math.floor(sec));
    return `${pad2(Math.floor(sec / 3600))}:${pad2(Math.floor(sec / 60) % 60)}:${pad2(sec % 60)}`;
}
function shorten(s, n) { s = String(s || ''); return s.length > n ? s.slice(0, n - 1) + '…' : s; }

/** Same rule as the Python note parser: top-level "- " line, text before the first "([[" citation. */
function bulletText(line) {
    if (!(line.startsWith('- ') || line.startsWith('* '))) return null;
    const text = line.slice(2).trim();
    const m = text.search(/\s+\(\[\[/);
    return (m >= 0 ? text.slice(0, m) : text).trim();
}

// ---- source links ("([[Transcripts/x#^t-1|🔗 10:00:00]] · [[clip.ogg|▶ Audio]])") ----
const CITE_GROUP_RE = /\(\[\[[^\]\n]+?\]\](?:\s*·\s*\[\[[^\]\n]+?\]\])*\)/g;
const CITE_RUN_RE = /(?:[ \t]*\(\[\[[^\]\n]+?\]\](?:\s*·\s*\[\[[^\]\n]+?\]\])*\))+/g;
const AUDIO_EXT_RE = /\.(ogg|opus|mp3|wav|m4a|webm|flac)$/i;

function parseCitationRun(text) {
    const groups = text.match(CITE_GROUP_RE) || [];
    const out = [];
    for (const g of groups) {
        const links = Array.from(g.matchAll(/\[\[([^\]|]+)(?:\|([^\]]*))?\]\]/g)).map((m) => ({ path: m[1].trim(), label: (m[2] || '').trim() }));
        const tr = links.find((l) => /#\^t-/.test(l.path));
        if (!tr) return null;
        const au = links.find((l) => AUDIO_EXT_RE.test(l.path.split('#')[0]));
        out.push({ transcript: tr, audio: au || null });
    }
    return out.length ? out : null;
}

function citationTime(src) {
    const m = (src.transcript.label || '').match(/\d{1,2}:\d{2}(?::\d{2})?/);
    return m ? m[0] : (src.transcript.path.split('#^')[1] || 'source');
}

function newSession(fromPython) {
    return {
        startedAt: Date.now(),
        pyStart: fromPython || null,
        env: [],
        segs: new Map(),
        markers: [],
        pauses: [],
        inbox: {},
        stats: { chunks: 0, transcribed: 0, notes_written: 0, pending: 0 },
        level: 0,
        speaking: false,
        lastT: 0,
        threshold: null,
        note: null,
        device: null,
        live: true,
        file: null,
    };
}

class VoiceScribePlugin extends obsidian.Plugin {
    async onload() {
        this.settings = Object.assign({}, PLUGIN_DEFAULTS, await this.loadData());
        this.process = null;
        this.state = 'idle'; // idle | starting | listening | stopping
        this.activeNotePath = null;
        this.lastError = '';
        this.session = null;
        this.lastSession = null;
        this.logLines = [];
        this.focus = null;     // {note, heading, bullets, mode, pending}
        this.recent = null;    // {note, texts, until}
        this.pinned = false;
        this.paused = false;
        this.mode = 'record';  // record | import
        this.decoVersion = 0;
        this.views = new Set();
        this.resolvePaths();
        this.config = this.loadConfigFile();

        this.registerView(VIEW_TYPE_DASHBOARD, (leaf) => new VoiceScribeDashboardView(leaf, this));
        this.registerView(VIEW_TYPE_GUIDE, (leaf) => new VoiceScribeGuideView(leaf, this));
        this.registerEditorExtension([buildEditorHighlighter(this), buildCitationCompactor(this)]);
        this.registerMarkdownPostProcessor((el, ctx) => this.compactRenderedCitations(el, ctx.sourcePath));

        this.ribbonIconEl = this.addRibbonIcon('microphone', 'VoiceScribe: start / stop recording', () => this.toggleRecording());
        this.addRibbonIcon('layout-dashboard', 'VoiceScribe: open session dashboard', () => this.activateDashboard());
        this.statusBarEl = this.addStatusBarItem();
        this.statusBarEl.addClass('mod-clickable');
        this.statusBarEl.addEventListener('click', () => this.activateDashboard());
        this.setStatus('idle');

        this.addCommand({ id: 'toggle-voicescribe-recording', name: 'Start / stop recording', hotkeys: [{ modifiers: ['Alt'], key: 'v' }], callback: () => this.toggleRecording() });
        this.addCommand({ id: 'force-stop', name: 'Force stop (discard unfinished processing)', callback: () => this.forceStop() });
        this.addCommand({ id: 'open-dashboard', name: 'Open session dashboard', callback: () => this.activateDashboard() });
        this.addCommand({ id: 'open-active-voice-notes', name: 'Open active voice note', callback: () => this.openActiveVoiceNote() });
        this.addCommand({ id: 'open-today-transcript', name: "Open today's transcript", callback: () => this.openTodayTranscript() });
        this.addCommand({
            id: 'focus-add', name: 'Focus here: add details to the selection', hotkeys: [{ modifiers: ['Alt'], key: 'f' }],
            editorCallback: (editor, ctx) => this.focusFromEditor(editor, ctx.file, 'add'),
        });
        this.addCommand({
            id: 'focus-correct', name: 'Focus here: correct the selection', hotkeys: [{ modifiers: ['Alt', 'Shift'], key: 'f' }],
            editorCallback: (editor, ctx) => this.focusFromEditor(editor, ctx.file, 'correct'),
        });
        this.addCommand({ id: 'clear-focus', name: 'Clear focus', callback: () => this.clearFocus() });
        this.addCommand({ id: 'pause-resume', name: 'Pause / resume listening', hotkeys: [{ modifiers: ['Alt'], key: 'p' }], callback: () => this.togglePause() });
        this.addCommand({ id: 'undo-last-change', name: 'Undo the last note change (scratch that)', hotkeys: [{ modifiers: ['Alt'], key: 'z' }], callback: () => this.undoLast() });
        this.addCommand({ id: 'summarize-now', name: 'Write the session summary now', callback: () => this.summarizeNow() });
        this.addCommand({ id: 'organise-now', name: 'Organise the inbox now (batch mode)', callback: () => this.organiseNow() });
        this.addCommand({ id: 'unload-models', name: 'Free memory: unload the Ollama models now', callback: () => this.unloadModelsNow() });
        this.addCommand({ id: 'toggle-organise-mode', name: 'Switch organising: live ⇄ in batches (next session)', callback: () => this.toggleOrganiseMode() });
        this.addCommand({ id: 'import-recording', name: 'Import an audio recording…', callback: () => this.importRecording() });
        this.addCommand({ id: 'open-guide', name: 'Open the guide (how to use VoiceScribe)', callback: () => this.openGuide() });
        this.addCommand({ id: 'open-voicescribe-options-menu', name: 'Quick options (microphone & models)', hotkeys: [{ modifiers: ['Alt', 'Shift'], key: 'v' }], callback: () => new VoiceScribeOptionsModal(this.app, this).open() });
        this.addCommand({ id: 'open-voicescribe-settings-tab', name: 'Open settings', callback: () => { this.app.setting.open(); this.app.setting.openTabById(this.manifest.id); } });

        this.registerEvent(this.app.workspace.on('editor-menu', (menu, editor, info) => {
            if (!info || !info.file) return;
            menu.addItem((item) => item.setTitle('VoiceScribe: add details here').setIcon('message-square-plus')
                .onClick(() => this.focusFromEditor(editor, info.file, 'add')));
            menu.addItem((item) => item.setTitle('VoiceScribe: correct this').setIcon('pencil')
                .onClick(() => this.focusFromEditor(editor, info.file, 'correct')));
            if (this.focus) menu.addItem((item) => item.setTitle('VoiceScribe: clear focus').setIcon('x').onClick(() => this.clearFocus()));
        }));

        this.addSettingTab(new VoiceScribeSettingTab(this.app, this));
        this.app.workspace.onLayoutReady(() => {
            this.loadLastSession();
            if (!this.settings.guideSeen) {
                this.settings.guideSeen = true;
                this.saveData(this.settings);
                this.openGuide();
            }
        });
        this.registerInterval(window.setInterval(() => this.tick(), 1000));
    }

    onunload() {
        // Ask the backend to finish and exit on its own (stdin closing does the same).
        if (this.process) { this.send({ cmd: 'stop' }); this.writeStopSignal(); }
        this.saveSession();
    }

    // ------------------------------------------------------------------ paths / config
    resolvePaths() {
        this.vaultBasePath = this.app.vault.adapter.basePath;
        this.backendDir = this.settings.backendDir && this.settings.backendDir.trim()
            ? this.settings.backendDir.trim()
            : path.resolve(this.vaultBasePath, '..', 'VoiceScribe');
        const defaultPy = IS_WIN ? path.join(this.backendDir, 'venv', 'Scripts', 'python.exe')
                                 : path.join(this.backendDir, 'venv', 'bin', 'python');
        this.pythonPath = this.settings.pythonPath && this.settings.pythonPath.trim() ? this.settings.pythonPath.trim() : defaultPy;
        this.scriptPath = path.join(this.backendDir, 'scribe_router.py');
        this.configPath = path.join(this.backendDir, 'config.json');
        this.stopSignalPath = path.join(this.backendDir, 'stop.signal');
    }

    async savePluginSettings() {
        await this.saveData(this.settings);
        this.resolvePaths();
        this.config = this.loadConfigFile();
    }

    loadConfigFile() {
        try {
            if (fs.existsSync(this.configPath)) {
                return Object.assign({}, CONFIG_DEFAULTS, JSON.parse(fs.readFileSync(this.configPath, 'utf-8')));
            }
        } catch (e) {
            console.warn('[VoiceScribe] Failed to read config.json:', e);
            new obsidian.Notice('VoiceScribe: config.json is unreadable — using defaults.');
        }
        return Object.assign({}, CONFIG_DEFAULTS);
    }

    saveConfigFile(patch) {
        this.config = Object.assign({}, this.loadConfigFile(), patch);
        try {
            fs.writeFileSync(this.configPath, JSON.stringify(this.config, null, 2), 'utf-8');
            if (this.state !== 'idle' && !this._restartHintShown) {
                this._restartHintShown = true;
                new obsidian.Notice('VoiceScribe: settings saved. They apply the next time you start recording.');
            }
            return true;
        } catch (e) {
            new obsidian.Notice('VoiceScribe: could not save settings — ' + e.message);
            return false;
        }
    }

    // ------------------------------------------------------------------ backend helpers
    backendReady() {
        if (!fs.existsSync(this.scriptPath)) {
            new obsidian.Notice(`VoiceScribe: backend not found at\n${this.backendDir}\nSet the folder in VoiceScribe settings.`, 10000);
            return false;
        }
        if (!fs.existsSync(this.pythonPath)) {
            new obsidian.Notice(`VoiceScribe: Python not found at\n${this.pythonPath}\nRun Start_Scribe.bat once to create the venv, or set the Python path in settings.`, 10000);
            return false;
        }
        return true;
    }

    async getInputDevices() {
        const listScript = path.join(this.backendDir, 'list_inputs.py');
        if (!fs.existsSync(this.pythonPath) || !fs.existsSync(listScript)) return [];
        try {
            const out = await runFile(this.pythonPath, [listScript], { cwd: this.backendDir, timeout: 15000 });
            return JSON.parse(out);
        } catch (e) {
            console.warn('[VoiceScribe] Could not list audio devices:', e);
            return [];
        }
    }

    async getOllamaModels() {
        try {
            const res = await obsidian.requestUrl({ url: `${this.config.ollama_host}/api/tags`, throw: false });
            if (res.status === 200 && res.json && Array.isArray(res.json.models)) return res.json.models.map(m => m.name);
        } catch (e) { /* offline */ }
        return [];
    }

    /** Models Ollama currently holds in memory: [{name, vram, size, until}] or null when Ollama is not running. */
    async getLoadedModels() {
        try {
            const res = await obsidian.requestUrl({ url: `${this.config.ollama_host}/api/ps`, throw: false });
            if (res.status === 200 && res.json && Array.isArray(res.json.models)) {
                return res.json.models.map((m) => ({ name: m.name || m.model, vram: m.size_vram || 0, size: m.size || 0, until: m.expires_at || '' }));
            }
        } catch (e) { /* not running */ }
        return null;
    }

    sessionModels() {
        const cfg = this.config || CONFIG_DEFAULTS;
        const out = [cfg.ollama_model];
        const b = (cfg.batch_ollama_model || '').trim();
        if (b && !out.includes(b)) out.push(b);
        return out.filter(Boolean);
    }

    /** Ask Ollama to drop models from GPU/RAM right away (keep_alive 0). */
    async unloadModels(names) {
        const loaded = await this.getLoadedModels();
        if (!loaded) return null;
        const key = (n) => { n = String(n || '').toLowerCase(); return n.includes(':') ? n : n + ':latest'; };
        const want = names ? new Set(names.map(key)) : null;
        const done = [];
        for (const m of loaded) {
            if (want && !want.has(key(m.name))) continue;
            try {
                const res = await obsidian.requestUrl({
                    url: `${this.config.ollama_host}/api/generate`, method: 'POST', contentType: 'application/json',
                    body: JSON.stringify({ model: m.name, keep_alive: 0 }), throw: false,
                });
                if (res.status === 200) done.push(m.name);
            } catch (e) { /* ignore */ }
        }
        return done;
    }

    async unloadModelsNow() {
        this.config = this.loadConfigFile();
        if (this.process && this.state !== 'idle') {
            new obsidian.Notice('VoiceScribe: a session is running — the model would be loaded again for the next thought. Unloading anyway.');
        }
        const done = await this.unloadModels(null);
        if (done === null) new obsidian.Notice('VoiceScribe: Ollama is not running, so no model is using memory.');
        else if (!done.length) new obsidian.Notice('VoiceScribe: no Ollama model is loaded.');
        else { new obsidian.Notice(`VoiceScribe: unloaded ${done.join(', ')} — GPU/RAM freed.`); this.log('info', `Unloaded ${done.join(', ')}`); }
        return done;
    }

    /**
     * Before a session: make sure the models it needs are installed. Returns false when the user
     * should not start (never at the moment: thoughts are kept unorganised instead), true otherwise.
     */
    async checkModelsBeforeStart() {
        const cfg = this.config || CONFIG_DEFAULTS;
        if (!cfg.llm_enabled) return true;
        const installed = await this.getOllamaModels();
        if (!installed.length) return true; // Ollama not running yet: the backend starts it
        const key = (n) => { n = String(n || '').toLowerCase(); return n.includes(':') ? n : n + ':latest'; };
        const have = new Set(installed.map(key));
        const wanted = String(cfg.organise_mode) === 'batch' ? this.sessionModels() : [cfg.ollama_model];
        const missing = wanted.filter((m) => !have.has(key(m)));
        if (missing.length) {
            new obsidian.Notice(`VoiceScribe: Ollama model ${missing.map((m) => `'${m}'`).join(' and ')} is not installed. ` +
                `Run “ollama pull ${missing[0]}” or pick an installed model in settings. Until then thoughts are kept without organising.`, 12000);
            this.log('warn', `Ollama model not installed: ${missing.join(', ')}`);
        }
        return true;
    }

    async calibrateMic() {
        if (!this.backendReady()) return null;
        if (this.state !== 'idle') {
            new obsidian.Notice('VoiceScribe: stop recording before calibrating.');
            return null;
        }
        const notice = new obsidian.Notice('🤫 Calibrating — stay quiet for 3 seconds…', 0);
        try {
            const out = await runFile(this.pythonPath, [this.scriptPath, '--calibrate', '--seconds', '3'], { cwd: this.backendDir, timeout: 30000 });
            const line = out.trim().split(/\r?\n/).filter(l => l.startsWith('{')).pop();
            const res = JSON.parse(line);
            if (res.error) throw new Error(res.error);
            return res;
        } catch (e) {
            new obsidian.Notice('VoiceScribe: calibration failed — ' + (e.message || e));
            return null;
        } finally {
            notice.hide();
        }
    }

    send(msg) {
        const p = this.process;
        if (!p || !p.stdin || !p.stdin.writable) return false;
        try { p.stdin.write(JSON.stringify(msg) + '\n'); return true; } catch (e) { return false; }
    }

    // ------------------------------------------------------------------ status & dashboard plumbing
    setStatus(state, detail) {
        this.state = state;
        this.statusDetail = detail || '';
        const text = {
            idle: '⏹ VoiceScribe',
            starting: '⏳ VoiceScribe: starting…',
            listening: this.paused ? '⏸ VoiceScribe: paused' : '🔴 VoiceScribe: listening',
            importing: '📥 VoiceScribe: importing',
            stopping: '⏳ VoiceScribe: finishing…',
        }[state];
        this.statusBarEl.setText(text + (detail ? ` · ${detail}` : ''));
        this.statusBarEl.setAttr('aria-label', 'Open the VoiceScribe dashboard');
        if (this.ribbonIconEl) this.ribbonIconEl.toggleClass('voicescribe-recording', (state === 'listening' && !this.paused) || state === 'starting' || state === 'importing');
        this.updateViews('header');
    }

    updateViews(part) { for (const v of this.views) v.refresh(part); }

    log(tag, text) {
        const line = { t: clock(), tag, text: String(text) };
        this.logLines.push(line);
        if (this.logLines.length > MAX_LOG_LINES) this.logLines.splice(0, this.logLines.length - MAX_LOG_LINES);
        for (const v of this.views) v.appendLog(line);
    }

    refreshEditors() {
        this.decoVersion++;
        this.app.workspace.updateOptions();
    }

    async openGuide() {
        let leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_GUIDE)[0];
        if (!leaf) {
            leaf = this.app.workspace.getLeaf('tab');
            await leaf.setViewState({ type: VIEW_TYPE_GUIDE, active: true });
        }
        this.app.workspace.revealLeaf(leaf);
    }

    async activateDashboard() {
        let leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE_DASHBOARD)[0];
        if (!leaf) {
            leaf = this.app.workspace.getRightLeaf(false);
            await leaf.setViewState({ type: VIEW_TYPE_DASHBOARD, active: true });
        }
        this.app.workspace.revealLeaf(leaf);
    }

    tick() {
        if (this.session && this.session.live) this.updateViews('timer');
        if (this.recent && Date.now() > this.recent.until) { this.recent = null; this.refreshEditors(); }
        if (this.session && this.session.live && this.settings.saveSessionTimeline) {
            this._saveCountdown = (this._saveCountdown || 60) - 1;
            if (this._saveCountdown <= 0) { this._saveCountdown = 60; this.saveSession(); }
        }
    }

    // ------------------------------------------------------------------ session timeline persistence
    sessionsFolder() {
        const cfg = this.config || CONFIG_DEFAULTS;
        return obsidian.normalizePath(`${cfg.archive_dir}/sessions`);
    }

    async saveSession() {
        const s = this.session;
        if (!s || !this.settings.saveSessionTimeline || s.env.length < 2) return;
        try {
            const adapter = this.app.vault.adapter;
            const folder = this.sessionsFolder();
            if (!(await adapter.exists(folder))) await adapter.mkdir(folder);
            if (!s.file) {
                const d = new Date(s.startedAt);
                s.file = `${folder}/${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}_${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}.json`;
            }
            const data = {
                v: 1, started: new Date(s.startedAt).toISOString(), step: ENV_STEP,
                env: s.env.map((v) => +(v || 0).toFixed(5)),
                segments: Array.from(s.segs.values()),
                markers: s.markers, pauses: s.pauses, stats: s.stats, threshold: s.threshold, note: s.note, importFile: s.importFile || null,
            };
            await adapter.write(s.file, JSON.stringify(data));
        } catch (e) {
            console.warn('[VoiceScribe] could not save session timeline', e);
        }
    }

    async loadLastSession() {
        try {
            const adapter = this.app.vault.adapter;
            const folder = this.sessionsFolder();
            if (!(await adapter.exists(folder))) return;
            const listing = await adapter.list(folder);
            const files = listing.files.filter((f) => f.endsWith('.json')).sort();
            if (!files.length) return;
            const data = JSON.parse(await adapter.read(files[files.length - 1]));
            const s = newSession();
            s.live = false;
            s.startedAt = Date.parse(data.started) || Date.now();
            s.env = data.env || [];
            (data.segments || []).forEach((g) => s.segs.set(g.seg, g));
            s.markers = data.markers || [];
            s.pauses = data.pauses || [];
            s.importFile = data.importFile || null;
            s.stats = Object.assign(s.stats, data.stats || {});
            s.threshold = data.threshold;
            s.note = data.note;
            s.lastT = s.env.length * (data.step || ENV_STEP);
            s.file = files[files.length - 1];
            this.lastSession = s;
            this.updateViews('all');
        } catch (e) {
            console.warn('[VoiceScribe] could not load last session', e);
        }
    }

    // ------------------------------------------------------------------ events from the backend
    segment(id) {
        const s = this.session;
        if (!s || !id) return null;
        if (!s.segs.has(id)) s.segs.set(id, { seg: id, t0: 0, dur: 0, state: 'captured', clip: '', text: '' });
        return s.segs.get(id);
    }

    handleEvent(ev) {
        const s = this.session;
        const stopping = this.state === 'stopping';
        const keep = (st, d) => this.setStatus(stopping ? 'stopping' : (this.mode === 'import' && st === 'listening' ? 'importing' : st), d);
        switch (ev.event) {
            case 'started':
                if (s) { s.pyStart = ev.session_start; s.threshold = ev.threshold; s.note = ev.note; s.markers.push({ t: 0, name: path.basename(ev.note || '', '.md'), reason: 'start' }); if (ev.mode === 'import') { s.importFile = ev.file; s.live = true; } }
                if (ev.note) { this.activeNotePath = ev.note; this.rememberNote(ev.note); }
                this.log('info', `Session started → ${ev.note}  (whisper ${ev.whisper}${ev.llm ? ', ' + ev.llm : ''})`);
                this.followNote(ev.note, { force: true, retries: 10 });
                this.updateViews('all');
                break;
            case 'level':
                if (s) {
                    s.level = ev.r; s.speaking = !!ev.s; s.lastT = Math.max(s.lastT, ev.t);
                    const i = Math.floor(ev.t / ENV_STEP);
                    s.env[i] = Math.max(s.env[i] || 0, ev.r);
                }
                this.updateViews('level');
                break;
            case 'stats':
                if (s) Object.assign(s.stats, ev);
                this.updateViews('stats');
                break;
            case 'envelope':
                if (s) { s.env = ev.values || []; s.lastT = ev.duration || s.env.length * (ev.step || ENV_STEP); s.importFile = s.importFile || ''; }
                this.updateViews('timeline');
                break;
            case 'import_progress': {
                const pct = ev.total ? Math.round((ev.done / ev.total) * 100) : 0;
                if (s) s.progress = pct;
                keep('importing', `${pct}%`);
                break;
            }
            case 'gpu_check':
                keep('starting', 'testing GPU…');
                this.log('info', 'Checking whether Whisper can run on the GPU (first time only)…');
                break;
            case 'paused':
                this.paused = true;
                if (s) s.pauses.push([ev.t, null]);
                this.log('info', 'Listening paused');
                this.setStatus(this.state);
                this.updateViews('all');
                break;
            case 'resumed':
                this.paused = false;
                if (s && s.pauses.length && s.pauses[s.pauses.length - 1][1] === null) s.pauses[s.pauses.length - 1][1] = ev.t;
                this.log('info', 'Listening resumed');
                this.setStatus(this.state);
                this.updateViews('all');
                break;
            case 'undone': {
                const name = path.basename(ev.note || '', '.md');
                this.log('note', ev.mode === 'restored' ? `${name} · undid “${shorten(ev.thought, 60)}”`
                    : ev.mode === 'removed' ? `${name} · removed the points from “${shorten(ev.thought, 60)}” (note was edited since)`
                        : `${name} · nothing left to undo`);
                if (ev.mode !== 'nothing') new obsidian.Notice('↩ Undone: ' + shorten(ev.thought, 60), 4000);
                if (s) s.stats.notes_written = Math.max(0, (s.stats.notes_written || 0) - 1);
                this.recent = null;
                this.refreshEditors();
                this.updateViews('stats');
                break;
            }
            case 'summarizing':
                keep(this.state === 'stopping' ? 'stopping' : 'listening', `summarising ${path.basename(ev.note || '', '.md')}…`);
                this.log('note', `Writing session summary for ${path.basename(ev.note || '', '.md')} (${ev.thoughts} thoughts)…`);
                break;
            case 'summary_written':
                this.log('note', `${path.basename(ev.note || '', '.md')} · ${ev.heading} written`);
                break;
            case 'loading_model':
                if (this._whisperReload) { keep('listening', `reloading Whisper${ev.device === 'cuda' ? ' on GPU' : ''}…`); break; }
                keep('starting', `loading ${ev.model}${ev.device === 'cuda' ? ' on GPU' : ''}`);
                this.log('info', `Loading Whisper model ${ev.model} on the ${ev.device === 'cuda' ? 'GPU' : 'CPU'}…`);
                break;
            case 'downloading_model':
                new obsidian.Notice(`VoiceScribe: downloading Whisper model '${ev.model}' (one-time)…`, 8000);
                this.log('info', `Downloading Whisper model ${ev.model} (one-time)…`);
                break;
            case 'model_loaded':
                if (this._whisperReload) { this._whisperReload = false; keep('listening', ''); }
                if (s) s.whisperDevice = ev.device;
                this.log('info', `Whisper ready (${ev.model}, ${ev.device === 'cuda' ? 'GPU' : 'CPU'})`);
                this.updateViews('header');
                break;
            case 'whisper_unloaded':
                this.log('info', `Whisper released from ${ev.device === 'cuda' ? 'GPU' : 'memory'} after ${ev.idle_min} min without speech — it reloads when you speak.`);
                break;
            case 'whisper_reload':
                this._whisperReload = true;
                this.log('info', `Reloading Whisper ${ev.model}…`);
                break;
            case 'llm_loading':
                if (this.state !== 'starting') keep('listening', `loading ${ev.model}…`);
                this.log('info', `Ollama model ${ev.model} is not in memory — loading it first…`);
                break;
            case 'llm_ready':
                if (this.state !== 'starting') keep('listening', 'organising…');
                this.log('info', `Ollama model ${ev.model} ready (${ev.seconds}s)`);
                break;
            case 'llm_missing':
                new obsidian.Notice(`VoiceScribe: Ollama model '${ev.model}' is not installed — run “ollama pull ${ev.model}”. Thoughts are kept without organising.`, 12000);
                this.log('warn', `Ollama model ${ev.model} is not installed (ollama pull ${ev.model})`);
                break;
            case 'llm_unloaded':
                this.log('info', `Unloaded ${(ev.models || []).join(', ')} from memory`);
                break;
            case 'listening':
                if (s) s.device = ev.device;
                keep('listening');
                this.log('mic', `Microphone active: ${ev.device}`);
                this.updateViews('all');
                break;
            case 'speech_start':
                if (!stopping) this.setStatus('listening', 'hearing you…');
                break;
            case 'chunk_captured': {
                const g = this.segment(ev.seg);
                if (g) { g.t0 = ev.t0; g.dur = ev.dur; g.state = 'captured'; }
                keep('listening', 'transcribing…');
                this.log('mic', `speech ${ev.dur}s captured`);
                this.updateViews('timeline');
                break;
            }
            case 'transcribing': {
                const g = this.segment(ev.seg);
                if (g) g.state = 'transcribing';
                this.updateViews('timeline');
                break;
            }
            case 'segment': {
                const g = this.segment(ev.seg);
                if (g) { g.t0 = ev.t0; g.dur = ev.dur; g.clip = ev.clip; g.state = ev.speech ? 'speech' : 'nospeech'; }
                if (!ev.speech) this.log('mic', `no words recognised (${ev.dur}s)`);
                this.updateViews('timeline');
                break;
            }
            case 'heard': {
                const g = this.segment(ev.seg);
                if (g) g.text = ev.text;
                if (ev.text) {
                    this.log('heard', `“${ev.text}”`);
                    if (this.settings.showSpeechNotices) new obsidian.Notice('🗣 ' + shorten(ev.text, 80), 4000);
                }
                break;
            }
            case 'skipped':
                this.log('info', `fragment skipped: “${ev.text}”`);
                break;
            case 'organising':
                keep('listening', ev.focus ? `organising (${ev.focus})…` : 'organising…');
                break;
            case 'inbox':
                if (s) { s.inbox[ev.note] = ev.count; s.stats.inbox = Object.values(s.inbox).reduce((a, b) => a + b, 0); }
                this.updateViews('stats');
                break;
            case 'organising_batch':
                keep('listening', `organising ${ev.count} thought${ev.count === 1 ? '' : 's'}…`);
                this.log('note', `${path.basename(ev.note || '', '.md')} · organising ${ev.count} thought(s) from the inbox (${ev.reason})…`);
                break;
            case 'batch_done': {
                const name = path.basename(ev.note || '', '.md');
                const parts = [];
                if (ev.added && ev.added.length) parts.push(`${ev.added.length} point(s) added`);
                if (ev.updated && ev.updated.length) parts.push(`${ev.updated.length} updated`);
                this.log('note', `${name} · organised ${ev.organised} thought(s) in ${ev.seconds}s → ${parts.join(', ') || 'no changes'}${ev.headings && ev.headings.length ? ' under ' + ev.headings.join(', ') : ''}`);
                this.recent = { note: ev.note, texts: [].concat(ev.added || [], ev.updated || []), until: Date.now() + RECENT_HIGHLIGHT_MS };
                if (s) { s.stats.notes_written = (s.stats.notes_written || 0); }
                this.refreshEditors();
                keep('listening', `organised ${name}`);
                break;
            }
            case 'batch_failed':
                this.log('warn', `${path.basename(ev.note || '', '.md')} · could not organise the inbox (${ev.count} thought(s) kept). Will retry.`);
                break;
            case 'note_updated': {
                if (ev.note) this.activeNotePath = ev.note;
                const g = this.segment(ev.seg);
                if (g) g.note = ev.note;
                const name = path.basename(ev.note || '', '.md');
                if (ev.inbox) {
                    const waiting = s && s.inbox[ev.note] ? s.inbox[ev.note] : 1;
                    this.log('note', `${name} · added to inbox (${waiting} waiting)`);
                    this.recent = { note: ev.note, texts: [].concat(ev.added || []), until: Date.now() + RECENT_HIGHLIGHT_MS };
                    this.refreshEditors();
                    keep('listening', `inbox ${name}`);
                    break;
                }
                const heads = Array.from(new Set(ev.headings || [])).join(', ');
                if (ev.updated && ev.updated.length && !(ev.added && ev.added.length)) this.log('note', `${name} · corrected bullet in ${heads}`);
                else this.log('note', `${name} · added to ${heads || 'notes'}${ev.updated && ev.updated.length ? ' (and updated a point)' : ''}`);
                this.recent = { note: ev.note, texts: [].concat(ev.added || [], ev.updated || []), until: Date.now() + RECENT_HIGHLIGHT_MS };
                this.refreshEditors();
                keep('listening', `updated ${name}`);
                break;
            }
            case 'no_speech':
                keep('listening');
                break;
            case 'topic':
                this.activeNotePath = ev.note;
                if (s) { s.note = ev.note; s.markers.push({ t: s.lastT, name: ev.name, reason: ev.reason }); }
                this.rememberNote(ev.note);
                this.log('topic', `${ev.created ? 'Created new note' : 'Switched note'} → ${ev.name}${ev.reason === 'focus' ? ' (focus)' : ''}`);
                if (ev.reason !== 'focus') new obsidian.Notice((ev.created ? '🆕 New note: ' : '💡 Now writing to: ') + ev.name);
                this.followNote(ev.note, { retries: 10 });
                this.updateViews('all');
                break;
            case 'focus':
                this.focus = { note: ev.note, heading: ev.heading, bullets: ev.bullets || [], mode: ev.mode, pending: false };
                this.log('focus', `${ev.heading || path.basename(ev.note, '.md')} › ${ev.bullets && ev.bullets.length ? '“' + shorten(ev.bullets[0], 48) + '”' + (ev.bullets.length > 1 ? ` +${ev.bullets.length - 1}` : '') : 'whole section'} (${ev.mode})`);
                this.refreshEditors();
                this.updateViews('focus');
                break;
            case 'focus_cleared':
                this.focus = null;
                this.log('focus', `focus cleared (${ev.reason})`);
                this.refreshEditors();
                this.updateViews('focus');
                break;
            case 'finishing':
                this.setStatus('stopping', ev.pending ? `${ev.pending} pending` : '');
                this.log('info', ev.pending ? `Finishing ${ev.pending} pending thought(s)…` : 'Finishing…');
                break;
            case 'warning':
                this.log('warn', ev.message);
                new obsidian.Notice('⚠️ VoiceScribe: ' + ev.message, 6000);
                break;
            case 'error':
                this.lastError = ev.message || '';
                this.log('error', ev.message);
                new obsidian.Notice('❌ VoiceScribe: ' + ev.message, 10000);
                break;
            case 'stopped':
                this.lastSummary = ev;
                if (s) Object.assign(s.stats, ev, { pending: 0 });
                this.log('info', `Stopped — ${ev.transcribed} transcribed, ${ev.notes_written} note update(s) in ${duration(ev.duration_sec)}`);
                break;
        }
    }

    // ------------------------------------------------------------------ recording
    async toggleRecording() {
        if (this.state === 'idle') return this.chooseNoteAndStart();
        if (this.state === 'importing') return this.stopRecording();
        if (this.state === 'stopping') return this.forceStop();
        return this.stopRecording();
    }

    // ------------------------------------------------------------------ choosing the session's note
    formatNoteName(template) {
        const d = new Date();
        const date = `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
        const time = `${pad2(d.getHours())}.${pad2(d.getMinutes())}`;
        return (template || PLUGIN_DEFAULTS.newNoteTemplate).replace(/\{date\}/g, date).replace(/\{time\}/g, time)
            .replace(/[\\/:*?"<>|#^\[\]]/g, ' ').replace(/\s+/g, ' ').trim() || 'Voice note';
    }

    notePathFor(name) {
        const cfg = this.config || CONFIG_DEFAULTS;
        let clean = String(name || '').trim().replace(/\.md$/i, '').replace(/[\\/:*?"<>|#^\[\]]/g, ' ').replace(/\s+/g, ' ').trim();
        if (!clean) clean = this.formatNoteName();
        const folder = (cfg.notes_subfolder || '').replace(/^\/+|\/+$/g, '');
        let rel = obsidian.normalizePath(folder ? `${folder}/${clean}.md` : `${clean}.md`);
        for (let i = 2; this.app.vault.getAbstractFileByPath(rel) && i < 100; i++) {
            rel = obsidian.normalizePath(folder ? `${folder}/${clean} ${i}.md` : `${clean} ${i}.md`);
        }
        return rel;
    }

    recentNoteFiles() {
        const seen = new Set();
        const out = [];
        for (const rel of this.settings.recentNotes || []) {
            if (seen.has(rel)) continue;
            seen.add(rel);
            const f = this.app.vault.getAbstractFileByPath(rel);
            if (f instanceof obsidian.TFile) out.push(f);
        }
        return out.slice(0, 6);
    }

    rememberNote(rel) {
        if (!rel) return;
        const list = [rel].concat((this.settings.recentNotes || []).filter((r) => r !== rel)).slice(0, 12);
        this.settings.recentNotes = list;
        this.saveData(this.settings);
    }

    async chooseNoteAndStart() {
        if (this.process) return;
        this.config = this.loadConfigFile();
        await this.checkModelsBeforeStart();
        if (this.process) return;
        const mode = this.settings.startNoteMode || 'ask';
        if (mode === 'new') return this.startRecording(this.notePathFor(this.formatNoteName(this.settings.newNoteTemplate)));
        if (mode === 'default') return this.startRecording(null);
        if (mode === 'last') {
            const last = this.recentNoteFiles()[0];
            return this.startRecording(last ? last.path : null);
        }
        new VoiceScribeStartModal(this.app, this, (rel) => this.startRecording(rel)).open();
    }

    startRecording(noteRel, opts) {
        opts = opts || {};
        if (this.process) return;
        this.mode = opts.importPath ? 'import' : 'record';
        this.paused = false;
        this.resolvePaths();
        if (!this.backendReady()) return;
        this.config = this.loadConfigFile();
        try { if (fs.existsSync(this.stopSignalPath)) fs.unlinkSync(this.stopSignalPath); } catch (e) { /* ignore */ }

        this.lastError = '';
        this.lastSummary = null;
        this._restartHintShown = false;
        this._forced = false;
        this.pinned = false;
        this.session = newSession();
        this._saveCountdown = 60;
        this.setStatus('starting', opts.importPath ? 'import' : '');
        if (opts.importPath) this.log('info', `Importing ${path.basename(opts.importPath)}…`);
        if (this.settings.openDashboardOnStart || opts.importPath) this.activateDashboard();
        else new obsidian.Notice('🎙 VoiceScribe starting (loading models)…');

        let proc;
        try {
            const args = ['-u', this.scriptPath, '--ipc', '--vault-dir', this.vaultBasePath];
            if (noteRel) args.push('--note', noteRel);
            if (opts.importPath) args.push('--import-audio', opts.importPath);
            proc = spawn(this.pythonPath, args, {
                cwd: this.backendDir,
                windowsHide: true,
                env: Object.assign({}, process.env, { PYTHONIOENCODING: 'utf-8', PYTHONUNBUFFERED: '1' }),
            });
        } catch (e) {
            new obsidian.Notice('VoiceScribe: failed to launch — ' + e.message);
            this.setStatus('idle');
            return;
        }
        this.process = proc;
        if (this.focus) this.send({ cmd: 'focus', note: this.focus.note, heading: this.focus.heading, bullets: this.focus.bullets, mode: this.focus.mode });

        let buf = '';
        proc.stdout.on('data', (data) => {
            buf += data.toString('utf-8');
            const lines = buf.split(/\r?\n/);
            buf = lines.pop();
            for (const line of lines) {
                if (line.startsWith('@@VS ')) {
                    try { this.handleEvent(JSON.parse(line.slice(5))); } catch (e) { console.warn('[VoiceScribe] bad event', line, e); }
                } else if (line.trim()) {
                    this.log('log', line);
                }
            }
        });
        let stderrTail = '';
        proc.stderr.on('data', (data) => {
            const str = data.toString('utf-8');
            stderrTail = (stderrTail + str).slice(-2000);
            str.split(/\r?\n/).filter((l) => l.trim()).forEach((l) => this.log('err', l));
        });
        if (proc.stdin) proc.stdin.on('error', () => { /* backend gone */ });
        proc.on('error', (err) => {
            new obsidian.Notice('VoiceScribe: launch error — ' + err.message);
            this.log('error', 'Launch error: ' + err.message);
        });
        proc.on('close', (code) => {
            clearTimeout(this.killTimer);
            this.process = null;
            this.paused = false;
            if (this.session) {
                this.session.live = false;
                const last = this.session.pauses[this.session.pauses.length - 1];
                if (last && last[1] === null) last[1] = this.session.lastT;
                this.saveSession();
                this.lastSession = this.session;
            }
            if (this.focus) { this.focus = null; this.refreshEditors(); } // focus lasts one recording session
            this.setStatus('idle');
            const sum = this.lastSummary;
            if (code !== 0 && code !== null && !this._forced) {
                const tail = this.lastError || stderrTail.trim().split(/\r?\n/).pop() || '';
                new obsidian.Notice(`⚠️ VoiceScribe stopped unexpectedly (exit ${code})${tail ? ': ' + tail.slice(0, 200) : ''}\nSee the dashboard console.`, 12000);
                this.log('error', `Backend exited with code ${code}`);
            } else if (sum) {
                new obsidian.Notice(`⏹ VoiceScribe stopped — ${sum.transcribed} thought(s) transcribed, ${sum.notes_written} note update(s).`, 6000);
            } else {
                new obsidian.Notice('⏹ VoiceScribe stopped.');
            }
            this._forced = false;
            this.updateViews('all');
        });
    }

    writeStopSignal() {
        try { fs.writeFileSync(this.stopSignalPath, 'stop'); } catch (e) { console.warn('[VoiceScribe] could not write stop signal', e); }
    }

    stopRecording() {
        if (!this.process) { this.setStatus('idle'); return; }
        this.setStatus('stopping');
        this.log('info', 'Stop requested — finishing what you already said (click Stop again to force).');
        if (!this.send({ cmd: 'stop' })) this.writeStopSignal();
        clearTimeout(this.killTimer);
        this.killTimer = setTimeout(() => this.forceStop(), 10 * 60 * 1000); // safety net only
    }

    forceStop() {
        if (!this.process) { this.setStatus('idle'); return; }
        this._forced = true;
        this.log('warn', 'Force stop.');
        this.send({ cmd: 'stop' });
        this.send({ cmd: 'stop' });
        this.writeStopSignal();
        const pid = this.process.pid;
        setTimeout(() => {
            if (!this.process || this.process.pid !== pid) return;
            try {
                if (IS_WIN) spawn('taskkill', ['/pid', String(pid), '/t', '/f'], { windowsHide: true });
                else this.process.kill('SIGKILL');
            } catch (e) { /* ignore */ }
        }, 4000);
    }

    // ------------------------------------------------------------------ pause / undo / summary / import
    togglePause() {
        if (!this.process || this.mode !== 'record' || this.state === 'stopping') {
            new obsidian.Notice('VoiceScribe: pausing works while recording.');
            return;
        }
        this.send({ cmd: this.paused ? 'resume' : 'pause' });
    }

    undoLast() {
        if (!this.send({ cmd: 'undo' })) {
            new obsidian.Notice('VoiceScribe: undo works during a session. Use Ctrl+Z in the note otherwise.');
        }
    }

    organiseNow() {
        if (!this.send({ cmd: 'organise' })) {
            new obsidian.Notice('VoiceScribe: in batch mode the inbox is organised automatically every few minutes and when you stop. “Organise now” works during a session.');
        }
    }

    isBatchMode() { return String((this.config || CONFIG_DEFAULTS).organise_mode) === 'batch'; }

    toggleOrganiseMode() {
        const next = this.isBatchMode() ? 'live' : 'batch';
        this.saveConfigFile({ organise_mode: next });
        new obsidian.Notice(next === 'batch' ? 'VoiceScribe will organise in batches (from the next session).' : 'VoiceScribe will organise live (from the next session).');
        this.updateViews('all');
    }

    summarizeNow() {
        if (!this.send({ cmd: 'summarize' })) {
            new obsidian.Notice('VoiceScribe: a summary is written automatically when recording stops (or use this during a session).');
        }
    }

    importRecording() {
        if (this.process) {
            new obsidian.Notice('VoiceScribe: stop the current session before importing.');
            return;
        }
        this.resolvePaths();
        if (!this.backendReady()) return;
        new VoiceScribeImportModal(this.app, this, (absPath) => {
            this.config = this.loadConfigFile();
            new VoiceScribeStartModal(this.app, this, (rel) => this.startRecording(rel, { importPath: absPath }), {
                title: `Where should “${path.basename(absPath)}” go?`,
                newName: path.basename(absPath).replace(/\.[^.]+$/, ''),
                startLabel: 'Import into new note',
            }).open();
        }).open();
    }

    // ---- GPU libraries: where they live -------------------------------------
    venvSitePackages() {
        const venv = path.dirname(path.dirname(this.pythonPath));
        return IS_WIN ? path.join(venv, 'Lib', 'site-packages') : venv;
    }

    recommendedCudaDir() {
        const base = process.env.LOCALAPPDATA || path.join(require('os').homedir(), IS_WIN ? 'AppData/Local' : '.local/share');
        return path.join(base, 'VoiceScribe', 'cuda');
    }

    isSyncedPath(p) {
        const lower = String(p || '').toLowerCase();
        return /onedrive|[\\/]mega[\\/ ]|mega sync|dropbox|google drive|icloud/.test(lower)
            || lower.startsWith(String(this.vaultBasePath || '').toLowerCase() + path.sep);
    }

    freeSpaceGB(p) {
        try {
            let probe = p;
            while (probe && !fs.existsSync(probe)) {
                const up = path.dirname(probe);
                if (up === probe) break;
                probe = up;
            }
            const st = fs.statfsSync(probe);
            return (st.bavail * st.bsize) / 1e9;
        } catch (e) {
            return null;
        }
    }

    findCudaDlls(dir) {
        // A CUDA Toolkit / existing install: cublas64_12.dll in the folder or its bin\
        if (!dir) return false;
        for (const d of [dir, path.join(dir, 'bin')]) {
            try {
                if (fs.readdirSync(d).some((f) => /^cublas64_12\.dll$/i.test(f))) return true;
            } catch (e) { /* missing */ }
        }
        try { // pip --target layout
            return fs.existsSync(path.join(dir, 'nvidia', 'cublas'));
        } catch (e) { return false; }
    }

    async chooseFolder(defaultPath) {
        try {
            const electron = require('electron');
            const dialog = electron.remote && electron.remote.dialog;
            if (dialog) {
                const res = await dialog.showOpenDialog({ title: 'Folder for the GPU libraries', defaultPath, properties: ['openDirectory', 'createDirectory'] });
                return res && !res.canceled && res.filePaths && res.filePaths[0] ? res.filePaths[0] : null;
            }
        } catch (e) { /* not available in this Obsidian version */ }
        return undefined; // caller falls back to typing the path
    }

    /** dir: '' = VoiceScribe's Python environment, otherwise a folder (pip --target). */
    async installGpuSupport(dir) {
        if (!fs.existsSync(this.pythonPath)) { this.backendReady(); return false; }
        const where = dir || this.venvSitePackages();
        const notice = new obsidian.Notice(`VoiceScribe: downloading GPU libraries (about 1 GB) to\n${where}\nThis uses the internet once.`, 0);
        this.log('info', `Installing GPU support into ${where}`);
        const args = ['-m', 'pip', 'install', '--upgrade', '--disable-pip-version-check'];
        if (dir) {
            try { fs.mkdirSync(dir, { recursive: true }); } catch (e) { /* pip reports it */ }
            args.push('--target', dir);
        }
        args.push('nvidia-cublas-cu12', 'nvidia-cudnn-cu12>=9,<10');
        try {
            const out = await runFile(this.pythonPath, args, { cwd: this.backendDir, timeout: 45 * 60 * 1000 });
            out.split(/\r?\n/).filter((l) => /Successfully|already satisfied/i.test(l)).slice(-3).forEach((l) => this.log('info', l));
            this.useCudaDir(dir);
            new obsidian.Notice('✅ GPU libraries installed. VoiceScribe tests the GPU the next time you start recording.', 10000);
            return true;
        } catch (e) {
            const msg = ((e.stderr || e.message || '') + '').trim().split(/\r?\n/).pop();
            this.log('error', 'GPU support install failed: ' + msg);
            new obsidian.Notice('❌ Could not install GPU support: ' + msg, 12000);
            return false;
        } finally {
            notice.hide();
        }
    }

    useCudaDir(dir) {
        this.saveConfigFile({ cuda_libs_dir: dir || '' });
        try { fs.unlinkSync(path.join(this.backendDir, 'gpu_probe.json')); } catch (e) { /* not there */ }
        this.log('info', `GPU libraries folder: ${dir || 'VoiceScribe Python environment'}`);
    }

    async gpuStatus() {
        try {
            const out = await runFile(this.pythonPath, [this.scriptPath, '--gpu-status'], { cwd: this.backendDir, timeout: 60000 });
            return JSON.parse(out.trim().split(/\r?\n/).filter((l) => l.startsWith('{')).pop());
        } catch (e) {
            return null;
        }
    }

    // ------------------------------------------------------------------ following the active note
    findLeafWithFile(rel) {
        let found = null;
        this.app.workspace.iterateAllLeaves((leaf) => {
            if (!found && leaf.view instanceof obsidian.MarkdownView && leaf.view.file && leaf.view.file.path === rel) found = leaf;
        });
        return found;
    }

    isRootLeaf(leaf) {
        let ok = false;
        if (!leaf) return false;
        this.app.workspace.iterateRootLeaves((l) => { if (l === leaf) ok = true; });
        return ok;
    }

    async followNote(rel, opts) {
        opts = opts || {};
        if (!rel) return;
        if (!opts.force && (!this.settings.followActiveNote || this.pinned)) return;
        if (opts.force && !this.settings.followActiveNote) return;
        const file = this.app.vault.getAbstractFileByPath(obsidian.normalizePath(rel));
        if (!(file instanceof obsidian.TFile)) {
            // The backend has just created the file; wait for Obsidian to index it.
            if ((opts.retries || 0) > 0) setTimeout(() => this.followNote(rel, Object.assign({}, opts, { retries: opts.retries - 1 })), 400);
            return;
        }
        const existing = this.findLeafWithFile(file.path);
        if (existing) {
            if (opts.force) this.app.workspace.revealLeaf(existing);
            this.noteLeaf = existing;
            return;
        }
        const target = this.isRootLeaf(this.noteLeaf) ? this.noteLeaf : this.app.workspace.getLeaf(false);
        await target.openFile(file, { active: false });
        this.noteLeaf = target;
    }

    togglePinned() {
        this.pinned = !this.pinned;
        if (!this.pinned && this.activeNotePath) this.followNote(this.activeNotePath, { force: true });
        this.updateViews('header');
    }

    // ------------------------------------------------------------------ focus
    focusFromEditor(editor, file, mode) {
        if (!file) return;
        const from = editor.getCursor('from');
        const to = editor.getCursor('to');
        const bullets = [];
        const addBulletAt = (ln) => {
            for (let i = ln; i >= 0; i--) {
                const t = editor.getLine(i);
                const bt = bulletText(t);
                if (bt !== null) { if (bt && !bullets.includes(bt)) bullets.push(bt); return; }
                if (!/^\s+\S/.test(t)) return; // only indented child lines belong to a bullet above
            }
        };
        for (let ln = from.line; ln <= to.line; ln++) addBulletAt(ln);
        let heading = null;
        for (let i = from.line; i >= 0; i--) {
            const t = editor.getLine(i);
            if (t.startsWith('### ')) { heading = t.slice(4).trim(); break; }
            if (/^#{1,2}\s/.test(t)) break;
        }
        if (!bullets.length && !heading) {
            new obsidian.Notice('VoiceScribe: put the cursor on a bullet or a ### heading in the notes to focus there.');
            return;
        }
        this.setFocus({ note: file.path, heading, bullets, mode });
    }

    setFocus(f) {
        this.focus = Object.assign({}, f, { pending: true });
        const sent = this.send({ cmd: 'focus', note: f.note, heading: f.heading, bullets: f.bullets, mode: f.mode });
        if (!sent) {
            new obsidian.Notice(`VoiceScribe focus set (${f.mode === 'correct' ? 'correct' : 'add details'}). It applies when recording starts.`);
        }
        this.refreshEditors();
        this.updateViews('focus');
    }

    setFocusMode(mode) {
        if (!this.focus) return;
        this.focus.mode = mode;
        if (!this.send({ cmd: 'focus_mode', mode })) this.focus.pending = true;
        this.refreshEditors();
        this.updateViews('focus');
    }

    clearFocus() {
        if (!this.focus) return;
        this.focus = null;
        if (!this.send({ cmd: 'clear_focus' })) this.log('focus', 'focus cleared');
        this.refreshEditors();
        this.updateViews('focus');
    }

    focusFromActiveEditor(mode) {
        const view = this.app.workspace.getActiveViewOfType(obsidian.MarkdownView)
            || (this.noteLeaf && this.noteLeaf.view instanceof obsidian.MarkdownView ? this.noteLeaf.view : null);
        if (!view || !view.editor) {
            new obsidian.Notice('VoiceScribe: open a note, select text and press Alt+F (add) or Alt+Shift+F (correct).');
            return;
        }
        this.focusFromEditor(view.editor, view.file, mode);
    }

    // ------------------------------------------------------------------ navigation
    async openVaultFile(relPath, label) {
        const file = relPath && this.app.vault.getAbstractFileByPath(obsidian.normalizePath(relPath));
        if (file instanceof obsidian.TFile) {
            await this.app.workspace.getLeaf(false).openFile(file);
        } else {
            new obsidian.Notice(`${label} has not been created yet.`);
        }
    }

    openActiveVoiceNote() {
        const cfg = this.config || CONFIG_DEFAULTS;
        const fallback = [cfg.notes_subfolder, `${cfg.default_note}.md`].filter(Boolean).join('/');
        return this.openVaultFile(this.activeNotePath || fallback, 'The voice note');
    }

    openTodayTranscript() {
        const cfg = this.config || CONFIG_DEFAULTS;
        const d = new Date();
        const today = `${cfg.transcripts_folder}/${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}_Transcript.md`;
        return this.openVaultFile(today, "Today's transcript");
    }

    playClip(rel) {
        try {
            if (this.audioEl) { this.audioEl.pause(); }
            if (this._playing === rel) { this._playing = null; return; }
            const url = this.app.vault.adapter.getResourcePath(obsidian.normalizePath(rel));
            this.audioEl = new Audio(url);
            this._playing = rel;
            this.audioEl.addEventListener('ended', () => { this._playing = null; });
            this.audioEl.play().catch((e) => new obsidian.Notice('VoiceScribe: could not play clip — ' + e.message));
        } catch (e) {
            new obsidian.Notice('VoiceScribe: could not play clip — ' + e.message);
        }
    }

    async transcriptQuote(linkPath) {
        // "Transcripts/2026-10-30_Transcript#^t-074036" -> the quoted words for that block
        const [file, block] = linkPath.split('#^');
        if (!block) return '';
        this._quoteCache = this._quoteCache || new Map();
        const key = linkPath;
        if (this._quoteCache.has(key)) return this._quoteCache.get(key);
        try {
            const dest = this.app.metadataCache.getFirstLinkpathDest(file, '');
            const path = dest ? dest.path : obsidian.normalizePath(file.endsWith('.md') ? file : file + '.md');
            const text = await this.app.vault.adapter.read(path);
            const lines = text.split(/\r?\n/);
            const idx = lines.findIndex((l) => l.trim() === '^' + block || l.trim().endsWith(' ^' + block));
            let quote = '';
            for (let i = idx; i >= 0 && idx >= 0 && i > idx - 12; i--) {
                const l = lines[i];
                if (/^>\s*[^!\s>]/.test(l) && !/^>\s*!\[\[/.test(l)) { quote = l.replace(/^>\s*/, '').replace(/\s*\^t-[\w-]+$/, '').replace(/^\*\*\[[^\]]+\]\*\*\s*/, ''); break; }
            }
            this._quoteCache.set(key, quote);
            if (this._quoteCache.size > 300) this._quoteCache.delete(this._quoteCache.keys().next().value);
            return quote;
        } catch (e) {
            return '';
        }
    }

    /** Small link icon (count when several) that expands into sources on hover / keyboard focus. */
    buildCiteChip(sources, sourcePath) {
        const n = sources.length;
        const wrap = createSpan({ cls: 'vs-cite', attr: { tabindex: '0', 'aria-label': `${n} source${n > 1 ? 's' : ''} — hover to show` } });
        const chip = wrap.createSpan({ cls: 'vs-cite-chip' });
        obsidian.setIcon(chip.createSpan({ cls: 'vs-cite-icon' }), 'link');
        if (n > 1) chip.createSpan({ cls: 'vs-cite-n', text: String(n) });
        const pop = wrap.createSpan({ cls: 'vs-cite-pop' });
        let quotesLoaded = false;
        const rows = sources.map((src) => {
            const row = pop.createSpan({ cls: 'vs-cite-row' });
            const top = row.createSpan({ cls: 'vs-cite-top' });
            const a = top.createEl('a', { cls: 'vs-cite-time', text: citationTime(src), href: '#', attr: { 'aria-label': 'Open in transcript' } });
            a.addEventListener('click', (e) => {
                e.preventDefault(); e.stopPropagation();
                this.app.workspace.openLinkText(src.transcript.path, sourcePath || '', obsidian.Keymap.isModEvent(e));
            });
            if (src.audio) {
                const b = top.createEl('button', { cls: 'vs-cite-play', attr: { 'aria-label': 'Play voice clip' } });
                obsidian.setIcon(b, 'play');
                b.appendText('Play');
                b.addEventListener('click', (e) => { e.preventDefault(); e.stopPropagation(); this.playClip(src.audio.path.split('#')[0]); });
            }
            const q = row.createSpan({ cls: 'vs-cite-quote' });
            return { src, q };
        });
        const loadQuotes = async () => {
            if (quotesLoaded) return;
            quotesLoaded = true;
            for (const r of rows) {
                const quote = await this.transcriptQuote(r.src.transcript.path);
                if (quote) r.q.setText(`“${shorten(quote, 160)}”`);
            }
        };
        wrap.addEventListener('mouseenter', loadQuotes);
        wrap.addEventListener('focusin', loadQuotes);
        wrap.addEventListener('mousedown', (e) => e.stopPropagation());
        return wrap;
    }

    /** Reading view: replace rendered source-link groups with the compact chip. */
    compactRenderedCitations(el, sourcePath) {
        if (!this.settings.compactSources) return;
        const isTranscriptLink = (n) => n && n.nodeName === 'A' && /#\^t-/.test(n.getAttribute('data-href') || n.getAttribute('href') || '');
        const linkInfo = (a) => ({ path: (a.getAttribute('data-href') || a.getAttribute('href') || '').trim(), label: a.textContent.trim() });
        const takeGroup = (first) => {
            const links = [first];
            let n = first.nextSibling;
            while (n && n.nodeType === 3 && /^\s*·\s*$/.test(n.nodeValue) && n.nextSibling && n.nextSibling.nodeName === 'A') {
                links.push(n.nextSibling);
                n = n.nextSibling.nextSibling;
            }
            if (!n || n.nodeType !== 3 || !/^\s*\)/.test(n.nodeValue)) return null;
            return { links, closer: n };
        };
        const toSource = (links) => {
            const infos = links.map(linkInfo);
            return { transcript: infos.find((l) => /#\^t-/.test(l.path)), audio: infos.find((l) => AUDIO_EXT_RE.test(l.path.split('#')[0])) || null };
        };
        const anchors = Array.from(el.querySelectorAll('a.internal-link')).filter(isTranscriptLink);
        for (const a of anchors) {
            if (!a.isConnected) continue;
            const prev = a.previousSibling;
            if (!prev || prev.nodeType !== 3 || !/\(\s*$/.test(prev.nodeValue)) continue;
            let g = takeGroup(a);
            if (!g) continue;
            const sources = [toSource(g.links)];
            const toRemove = [];
            g.links.forEach((l) => { toRemove.push(l); if (l.previousSibling && l.previousSibling !== prev && l.previousSibling.nodeType === 3 && /^\s*·\s*$/.test(l.previousSibling.nodeValue)) toRemove.push(l.previousSibling); });
            prev.nodeValue = prev.nodeValue.replace(/\s*\(\s*$/, ' ');
            let closer = g.closer;
            closer.nodeValue = closer.nodeValue.replace(/^\s*\)/, '');
            // merge directly following groups: ") (" [[..]] ... ")"
            while (/^\s*\(\s*$/.test(closer.nodeValue) && isTranscriptLink(closer.nextSibling)) {
                const g2 = takeGroup(closer.nextSibling);
                if (!g2) break;
                sources.push(toSource(g2.links));
                g2.links.forEach((l) => { toRemove.push(l); if (l.previousSibling && l.previousSibling.nodeType === 3 && /^\s*·\s*$/.test(l.previousSibling.nodeValue)) toRemove.push(l.previousSibling); });
                toRemove.push(closer);
                closer = g2.closer;
                closer.nodeValue = closer.nodeValue.replace(/^\s*\)/, '');
            }
            const chip = this.buildCiteChip(sources, sourcePath);
            closer.parentNode.insertBefore(chip, closer);
            toRemove.forEach((n) => n.remove());
        }
    }

    openLogFile() {
        const p = path.join(this.backendDir, 'voicescribe.log');
        try { require('electron').shell.openPath(p); } catch (e) { new obsidian.Notice(p); }
    }
}

// =============================================================================
// Editor highlighting: focused bullets (purple) and the latest change (green)
// =============================================================================
function buildEditorHighlighter(plugin) {
    let cmView, cmState;
    try {
        cmView = require('@codemirror/view');
        cmState = require('@codemirror/state');
    } catch (e) {
        console.warn('[VoiceScribe] CodeMirror not available; editor highlights disabled.');
        return [];
    }
    const { ViewPlugin, Decoration } = cmView;
    const focusLine = Decoration.line({ class: 'vs-focus-line' });
    const focusHeading = Decoration.line({ class: 'vs-focus-heading' });
    const recentLine = Decoration.line({ class: 'vs-recent-line' });

    function build(view) {
        const builder = new cmState.RangeSetBuilder();
        let info = null;
        try { info = view.state.field(obsidian.editorInfoField, false); } catch (e) { info = null; }
        const filePath = info && info.file ? info.file.path : null;
        const f = plugin.focus;
        const r = plugin.recent;
        const focusOn = !!(f && filePath && f.note === filePath);
        const recentOn = !!(r && filePath && r.note === filePath && Date.now() < r.until);
        if (!focusOn && !recentOn) return builder.finish();

        const focusSet = new Set(focusOn ? f.bullets : []);
        const recentSet = new Set(recentOn ? r.texts : []);
        const sectionFocus = focusOn && !f.bullets.length && f.heading ? f.heading.toLowerCase() : null;
        const doc = view.state.doc;
        let heading = null;
        for (let i = 1; i <= doc.lines; i++) {
            const line = doc.line(i);
            const t = line.text;
            if (t.startsWith('### ')) {
                heading = t.slice(4).trim().toLowerCase();
                if (sectionFocus && heading === sectionFocus) builder.add(line.from, line.from, focusHeading);
                continue;
            }
            if (/^#{1,2}\s/.test(t)) { heading = null; continue; }
            const bt = bulletText(t);
            if (bt === null) continue;
            if (focusOn && (focusSet.has(bt) || (sectionFocus && heading === sectionFocus))) builder.add(line.from, line.from, focusLine);
            else if (recentSet.has(bt)) builder.add(line.from, line.from, recentLine);
        }
        return builder.finish();
    }

    return ViewPlugin.fromClass(class {
        constructor(view) { this.ver = plugin.decoVersion; this.decorations = build(view); }
        update(u) {
            if (u.docChanged || u.viewportChanged || this.ver !== plugin.decoVersion) {
                this.ver = plugin.decoVersion;
                this.decorations = build(u.view);
            }
        }
    }, { decorations: (v) => v.decorations });
}

// =============================================================================
// Live Preview: source-link groups -> compact chip (raw text returns when the cursor is on it)
// =============================================================================
function buildCitationCompactor(plugin) {
    let cmView, cmState;
    try {
        cmView = require('@codemirror/view');
        cmState = require('@codemirror/state');
    } catch (e) {
        return [];
    }
    const { ViewPlugin, Decoration, WidgetType } = cmView;

    class CiteWidget extends WidgetType {
        constructor(sources, sourcePath, key) { super(); this.sources = sources; this.sourcePath = sourcePath; this.key = key; }
        eq(other) { return other.key === this.key && other.sourcePath === this.sourcePath; }
        toDOM() { return plugin.buildCiteChip(this.sources, this.sourcePath); }
        ignoreEvent() { return true; }
    }

    function build(view) {
        const builder = new cmState.RangeSetBuilder();
        if (!plugin.settings.compactSources) return builder.finish();
        let live = false;
        try { live = view.state.field(obsidian.editorLivePreviewField); } catch (e) { live = false; }
        if (!live) return builder.finish();
        let info = null;
        try { info = view.state.field(obsidian.editorInfoField, false); } catch (e) { info = null; }
        const sourcePath = info && info.file ? info.file.path : '';
        const sel = view.state.selection.ranges;
        const doc = view.state.doc;
        for (const { from, to } of view.visibleRanges) {
            let pos = from;
            while (pos <= to) {
                const line = doc.lineAt(pos);
                if (line.text.includes('([[')) {
                    CITE_RUN_RE.lastIndex = 0;
                    let m;
                    while ((m = CITE_RUN_RE.exec(line.text))) {
                        const start = line.from + m.index, end = start + m[0].length;
                        if (sel.some((r) => r.from <= end && r.to >= start)) continue; // editing: show raw text
                        const sources = parseCitationRun(m[0]);
                        if (!sources) continue;
                        builder.add(start, end, Decoration.replace({ widget: new CiteWidget(sources, sourcePath, m[0]) }));
                    }
                }
                pos = line.to + 1;
            }
        }
        return builder.finish();
    }

    const vp = ViewPlugin.fromClass(class {
        constructor(view) { this.ver = plugin.decoVersion; this.decorations = build(view); }
        update(u) {
            if (u.docChanged || u.viewportChanged || u.selectionSet || u.focusChanged || this.ver !== plugin.decoVersion
                || u.transactions.some((t) => t.reconfigured)) {
                this.ver = plugin.decoVersion;
                this.decorations = build(u.view);
            }
        }
    }, { decorations: (v) => v.decorations });
    return cmState.Prec ? cmState.Prec.high(vp) : vp;
}

// =============================================================================
// Session dashboard (right sidebar view)
// =============================================================================
const LOG_FILTERS = [
    ['all', 'All', ['info', 'mic', 'heard', 'note', 'topic', 'focus', 'warn', 'error', 'err']],
    ['speech', 'Speech', ['mic', 'heard']],
    ['notes', 'Notes', ['note', 'topic', 'focus']],
    ['issues', 'Issues', ['warn', 'error', 'err']],
    ['raw', 'Raw', ['log', 'err']],
];

function levelToUnit(v) {
    const lo = Math.log10(0.0003), hi = Math.log10(0.1);
    if (!v || v <= 0) return 0;
    return Math.min(1, Math.max(0, (Math.log10(v) - lo) / (hi - lo)));
}

class VoiceScribeDashboardView extends obsidian.ItemView {
    constructor(leaf, plugin) {
        super(leaf);
        this.plugin = plugin;
        this.filter = 'all';
        this.timelineDirty = true;
    }

    getViewType() { return VIEW_TYPE_DASHBOARD; }
    getDisplayText() { return 'VoiceScribe'; }
    getIcon() { return 'microphone'; }

    async onOpen() {
        this.plugin.views.add(this);
        const root = this.contentEl;
        root.empty();
        root.addClass('vs-dash');

        // header
        const header = root.createDiv({ cls: 'vs-header' });
        const left = header.createDiv({ cls: 'vs-row' });
        left.createSpan({ cls: 'vs-title', text: 'VoiceScribe' });
        this.pillEl = left.createSpan({ cls: 'vs-pill' });
        const right = header.createDiv({ cls: 'vs-row' });
        this.timerEl = right.createSpan({ cls: 'vs-timer', text: '00:00:00' });
        const help = right.createEl('button', { cls: 'vs-btn vs-btn-ghost vs-help', text: '?', attr: { 'aria-label': 'How to use VoiceScribe' } });
        help.addEventListener('click', () => this.plugin.openGuide());
        this.pauseBtn = right.createEl('button', { cls: 'vs-btn vs-btn-ghost' });
        this.pauseBtn.addEventListener('click', () => this.plugin.togglePause());
        this.mainBtn = right.createEl('button', { cls: 'vs-btn' });
        this.mainBtn.addEventListener('click', () => this.plugin.toggleRecording());

        this.infoEl = root.createDiv({ cls: 'vs-info' });
        this.followEl = root.createDiv({ cls: 'vs-follow' });

        // focus
        this.focusEl = root.createDiv();

        // audio
        const audio = root.createEl('section', { cls: 'vs-card', attr: { 'aria-label': 'Session audio' } });
        const ah = audio.createDiv({ cls: 'vs-card-head' });
        this.audioTitleEl = ah.createSpan({ cls: 'vs-label', text: 'Session audio' });
        ah.createSpan({ cls: 'vs-muted', text: 'click a segment to play' });
        const meterRow = audio.createDiv({ cls: 'vs-meter-row' });
        meterRow.createSpan({ cls: 'vs-muted vs-meter-label', text: 'Live' });
        const meter = meterRow.createDiv({ cls: 'vs-meter' });
        this.meterFill = meter.createDiv({ cls: 'vs-meter-fill' });
        this.meterThreshold = meter.createDiv({ cls: 'vs-meter-threshold', attr: { title: 'Speech threshold' } });
        this.meterText = meterRow.createSpan({ cls: 'vs-meter-text', text: '—' });
        const tlWrap = audio.createDiv({ cls: 'vs-timeline-wrap' });
        this.canvas = tlWrap.createEl('canvas', { cls: 'vs-timeline', attr: { role: 'img', 'aria-label': 'Session loudness timeline' } });
        this.canvas.addEventListener('click', (e) => this.onTimelineClick(e));
        this.canvas.addEventListener('mousemove', (e) => this.onTimelineHover(e));
        this.axisEl = audio.createDiv({ cls: 'vs-axis' });
        const legend = audio.createDiv({ cls: 'vs-legend' });
        [['vs-sw-speech', 'speech (saved clip)'], ['vs-sw-pending', 'processing'], ['vs-sw-silence', 'silence'], ['vs-sw-topic', 'note switch'], ['vs-sw-pause', 'paused']]
            .forEach(([c, t]) => { const s = legend.createSpan(); s.createSpan({ cls: 'vs-swatch ' + c }); s.appendText(t); });

        // counters + actions
        this.statsEl = root.createDiv({ cls: 'vs-stats' });
        this.actionsEl = root.createDiv({ cls: 'vs-actions' });

        // console
        const con = root.createEl('section', { cls: 'vs-console', attr: { 'aria-label': 'Console' } });
        const ch = con.createDiv({ cls: 'vs-console-head' });
        ch.createSpan({ cls: 'vs-label', text: 'Console' });
        const chips = ch.createDiv({ cls: 'vs-chips', attr: { role: 'group', 'aria-label': 'Filter log' } });
        this.chipEls = {};
        LOG_FILTERS.forEach(([id, label]) => {
            const b = chips.createEl('button', { cls: 'vs-chip', text: label });
            b.addEventListener('click', () => { this.filter = id; this.renderLog(); });
            this.chipEls[id] = b;
        });
        this.logEl = con.createDiv({ cls: 'vs-log' });
        const foot = con.createDiv({ cls: 'vs-console-foot' });
        this.logCountEl = foot.createSpan({ cls: 'vs-muted' });
        const openLog = foot.createEl('a', { text: 'Open voicescribe.log', href: '#' });
        openLog.addEventListener('click', (e) => { e.preventDefault(); this.plugin.openLogFile(); });

        this.resizeObs = new ResizeObserver(() => { this.timelineDirty = true; this.scheduleDraw(); });
        this.resizeObs.observe(tlWrap);
        this.refresh('all');
        this.renderLog();
    }

    async onClose() {
        this.plugin.views.delete(this);
        if (this.resizeObs) this.resizeObs.disconnect();
        if (this.raf) cancelAnimationFrame(this.raf);
    }

    get session() { return this.plugin.session || this.plugin.lastSession; }

    refresh(part) {
        if (!this.pillEl) return;
        const all = part === 'all';
        if (all || part === 'header' || part === 'timer') this.renderHeader();
        if (all || part === 'header') { this.renderInfo(); this.renderFollow(); }
        if (all || part === 'focus') this.renderFocus();
        if (all || part === 'level') this.renderMeter();
        if (all || part === 'stats' || part === 'timer') this.renderStats();
        if (all || part === 'timeline' || part === 'level' || part === 'timer') { this.timelineDirty = true; this.scheduleDraw(); }
    }

    renderHeader() {
        const p = this.plugin;
        const st = p.state;
        const labels = { idle: 'Stopped', starting: 'Starting…', listening: p.paused ? 'Paused' : 'Listening', importing: 'Importing', stopping: 'Finishing…' };
        this.pillEl.setText(labels[st] + ((st === 'listening' || st === 'importing') && p.statusDetail && !p.paused ? ` · ${p.statusDetail}` : ''));
        this.pillEl.className = 'vs-pill vs-pill-' + (p.paused && st === 'listening' ? 'paused' : st);
        const canPause = st === 'listening' && p.mode === 'record';
        this.pauseBtn.toggle(canPause);
        this.pauseBtn.setText(p.paused ? '▶ Resume' : '❚❚ Pause');
        this.pauseBtn.setAttr('aria-label', p.paused ? 'Resume listening (Alt+P)' : 'Pause listening (Alt+P)');
        this.renderActions();
        const s = this.session;
        let secs = 0;
        if (s) secs = s.live ? (Date.now() - s.startedAt) / 1000 : s.lastT;
        this.timerEl.setText(duration(secs));
        this.mainBtn.empty();
        if (st === 'idle') { this.mainBtn.setText('● Start'); this.mainBtn.className = 'vs-btn vs-btn-start'; }
        else if (st === 'stopping') { this.mainBtn.setText('Force stop'); this.mainBtn.className = 'vs-btn vs-btn-stop'; }
        else if (st === 'importing') { this.mainBtn.setText('■ Stop import'); this.mainBtn.className = 'vs-btn vs-btn-stop'; }
        else { this.mainBtn.setText('■ Stop'); this.mainBtn.className = 'vs-btn vs-btn-stop'; }
        if (this.audioTitleEl) {
            this.audioTitleEl.setText(s && s.importFile ? `Recording · ${s.importFile}` : (s && !s.live ? `Last session · ${new Date(s.startedAt).toLocaleString()}` : 'Session audio'));
        }
    }

    renderInfo() {
        const p = this.plugin;
        const cfg = p.config || CONFIG_DEFAULTS;
        const s = this.session;
        this.infoEl.empty();
        const note = p.activeNotePath || (s && s.note);
        if (note) {
            this.infoEl.appendText('Writing to ');
            const a = this.infoEl.createEl('a', { text: path.basename(note), href: '#' });
            a.addEventListener('click', (e) => { e.preventDefault(); p.openVaultFile(note, 'The note'); });
        } else {
            this.infoEl.appendText(`Default note: ${cfg.default_note}`);
        }
        const bits = [];
        if (s && s.device) bits.push(`mic: ${s.device}`);
        bits.push(`whisper ${cfg.whisper_model}${s && s.whisperDevice ? (s.whisperDevice === 'cuda' ? ' (GPU)' : ' (CPU)') : ''}`);
        bits.push(cfg.llm_enabled ? (cfg.organise_mode === 'batch'
            ? `batches every ${cfg.batch_interval_min} min (${(cfg.batch_ollama_model || '').trim() || cfg.ollama_model})`
            : `live (${cfg.ollama_model})`) : 'no LLM');
        this.infoEl.appendText(' · ' + bits.join(' · '));
    }

    renderActions() {
        if (!this.actionsEl) return;
        const p = this.plugin;
        const running = !!p.process && p.state !== 'stopping';
        const key = [running, p.mode, p.state, p.isBatchMode()].join('|');
        if (this._actionsKey === key) return;
        this._actionsKey = key;
        this.actionsEl.empty();
        const btn = (label, title, fn) => {
            const b = this.actionsEl.createEl('button', { cls: 'vs-btn vs-btn-ghost', text: label, attr: { 'aria-label': title } });
            b.addEventListener('click', fn);
            return b;
        };
        if (running) {
            btn('↩ Undo last change', 'Undo the last note change (Alt+Z, or say “scratch that”)', () => p.undoLast());
            if (p.isBatchMode()) btn('🗂 Organise now', 'Organise the inbox now instead of waiting', () => p.organiseNow());
            btn('🧾 Summary now', 'Write the session summary into the note now', () => p.summarizeNow());
        } else {
            btn('📥 Import recording…', 'Transcribe and organise an audio file', () => p.importRecording());
        }
    }

    renderFollow() {
        const p = this.plugin;
        this.followEl.empty();
        if (!p.settings.followActiveNote) {
            this.followEl.createSpan({ cls: 'vs-muted', text: 'Auto-follow is off (settings).' });
            return;
        }
        this.followEl.createSpan({ text: p.pinned ? 'Pinned — the open note stays put.' : 'Following the active note.' });
        const b = this.followEl.createEl('button', { cls: 'vs-btn vs-btn-ghost', text: p.pinned ? 'Follow active note' : 'Pin this note' });
        b.addEventListener('click', () => p.togglePinned());
    }

    renderFocus() {
        const p = this.plugin;
        const f = p.focus;
        this.focusEl.empty();
        if (!f) {
            const b = this.focusEl.createEl('button', { cls: 'vs-focus-hint', text: 'Select text in a note, then press Alt+F (add) or Alt+Shift+F (correct) to focus VoiceScribe there' });
            b.addEventListener('click', () => p.focusFromActiveEditor('add'));
            return;
        }
        const card = this.focusEl.createEl('section', { cls: 'vs-card vs-focus-card', attr: { 'aria-label': 'Focus' } });
        const head = card.createDiv({ cls: 'vs-card-head' });
        head.createSpan({ cls: 'vs-label vs-focus-label', text: f.pending ? 'Focus (waiting for recording)' : 'Focus' });
        const clear = head.createEl('button', { cls: 'vs-btn vs-btn-ghost', text: 'Clear focus' });
        clear.addEventListener('click', () => p.clearFocus());
        const what = card.createDiv({ cls: 'vs-focus-what' });
        const noteName = path.basename(f.note, '.md');
        let text = `${noteName} › ${f.heading || '(no heading)'}`;
        if (f.bullets.length) text += ` › “${shorten(f.bullets[0], 60)}”` + (f.bullets.length > 1 ? ` +${f.bullets.length - 1} more` : '');
        else text += ' (whole section)';
        what.setText(text);
        const modes = card.createDiv({ cls: 'vs-row vs-wrap', attr: { role: 'group', 'aria-label': 'What your next thoughts do' } });
        [['add', 'Add details here'], ['correct', 'Correct this']].forEach(([id, label]) => {
            const b = modes.createEl('button', { cls: 'vs-mode' + (f.mode === id ? ' is-on' : ''), text: label, attr: { 'aria-pressed': String(f.mode === id) } });
            b.addEventListener('click', () => p.setFocusMode(id));
        });
        card.createDiv({
            cls: 'vs-muted',
            text: f.mode === 'correct'
                ? 'What you say next rewrites the focused point(s). Nothing else in the note changes.'
                : 'What you say next is added here. Thoughts about other topics still go elsewhere.',
        });
    }

    renderMeter() {
        const s = this.session;
        const cfg = this.plugin.config || CONFIG_DEFAULTS;
        const thr = (s && s.threshold) || cfg.energy_threshold;
        const live = s && s.live;
        const lvl = live ? s.level : 0;
        this.meterFill.style.width = `${(levelToUnit(lvl) * 100).toFixed(1)}%`;
        this.meterFill.toggleClass('is-speech', !!(live && lvl >= thr));
        this.meterThreshold.style.left = `${(levelToUnit(thr) * 100).toFixed(1)}%`;
        this.meterText.setText(!live ? '—' : (this.plugin.paused ? 'paused' : (s.speaking ? 'speech' : (lvl >= thr ? 'loud' : 'quiet'))));
    }

    renderStats() {
        const s = this.session;
        const st = s ? s.stats : { chunks: 0, transcribed: 0, notes_written: 0, pending: 0 };
        this.statsEl.empty();
        const tiles = [[st.chunks, 'thoughts'], [st.transcribed, 'transcribed'], [st.notes_written, 'note updates'], [st.pending || 0, 'in queue']];
        if (this.plugin.isBatchMode() || st.inbox) tiles.push([st.inbox || 0, 'in inbox']);
        tiles
            .forEach(([v, l]) => {
                const c = this.statsEl.createDiv({ cls: 'vs-stat' });
                c.createDiv({ cls: 'vs-stat-v', text: String(v || 0) });
                c.createDiv({ cls: 'vs-muted', text: l });
            });
    }

    // ---- timeline ----------------------------------------------------------
    scheduleDraw() {
        if (this.raf || !this.canvas) return;
        const since = Date.now() - (this.lastDraw || 0);
        const wait = Math.max(0, 200 - since); // at most ~5 redraws per second
        this.drawTimer = setTimeout(() => {
            this.raf = requestAnimationFrame(() => { this.raf = null; this.drawTimeline(); });
        }, wait);
        this.raf = -1;
    }

    timelineSpan() {
        const s = this.session;
        return Math.max(60, s ? s.lastT : 0);
    }

    drawTimeline() {
        this.lastDraw = Date.now();
        const c = this.canvas;
        if (!c || !c.isConnected || !this.timelineDirty) return;
        if (!c.offsetParent) return; // panel hidden: skip work
        this.timelineDirty = false;
        const w = c.clientWidth, h = 72;
        const dpr = window.devicePixelRatio || 1;
        if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
            c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
        }
        const ctx = c.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, w, h);
        const css = getComputedStyle(this.contentEl);
        const col = (name, fb) => (css.getPropertyValue(name) || '').trim() || fb;
        const cSpeech = col('--vs-speech', '#4cc2a4');
        const cPending = col('--vs-pending', '#6c8fd6');
        const cSilence = col('--vs-silence', '#5a5a60');
        const cTopic = col('--vs-topic', '#a594f9');
        const cText = col('--text-muted', '#999');

        const s = this.session;
        this.renderAxis();
        if (!s || !s.env.length) {
            ctx.fillStyle = cText;
            ctx.font = '12px var(--font-interface, sans-serif)';
            ctx.fillText('Start recording to see the session timeline.', 8, h / 2 + 4);
            return;
        }
        const span = this.timelineSpan();
        const n = Math.max(20, Math.floor(w / 3));
        const bw = w / n;
        const segs = Array.from(s.segs.values());
        const top = 14; // room for topic labels
        const ih = h - top - 2;
        for (let i = 0; i < n; i++) {
            const t0 = (i * span) / n, t1 = ((i + 1) * span) / n;
            if (t0 > s.lastT) break;
            let v = 0;
            for (let k = Math.floor(t0 / ENV_STEP); k <= Math.floor(t1 / ENV_STEP); k++) v = Math.max(v, s.env[k] || 0);
            const mid = (t0 + t1) / 2;
            const seg = segs.find((g) => mid >= g.t0 && mid <= g.t0 + g.dur);
            let color = cSilence;
            if (seg) color = seg.state === 'speech' ? cSpeech : (seg.state === 'nospeech' ? cSilence : cPending);
            const bh = Math.max(2, levelToUnit(v) * ih);
            ctx.fillStyle = color;
            ctx.fillRect(i * bw, top + (ih - bh) / 2, Math.max(1, bw - 1), bh);
        }
        const cPause = col('--vs-pause', 'rgba(245,165,36,0.18)');
        for (const [a, b] of s.pauses || []) {
            const x0 = (a / span) * w, x1 = ((b === null ? s.lastT : b) / span) * w;
            ctx.fillStyle = cPause;
            ctx.fillRect(x0, top, Math.max(2, x1 - x0), ih);
        }
        ctx.font = '10px var(--font-interface, sans-serif)';
        for (const m of s.markers) {
            const x = Math.min(w - 2, (m.t / span) * w);
            ctx.fillStyle = cTopic;
            ctx.fillRect(x, 0, 2, h);
            ctx.fillText(shorten(m.name, 22), x + 4, 10);
        }
    }

    renderAxis() {
        if (!this.axisEl) return;
        const s = this.session;
        this.axisEl.empty();
        if (!s) return;
        const span = this.timelineSpan();
        const start = s.startedAt;
        [0, 1 / 3, 2 / 3].forEach((f) => this.axisEl.createSpan({ text: clock(new Date(start + f * span * 1000)).slice(0, 5) }));
        this.axisEl.createSpan({ cls: s.live ? 'vs-now' : '', text: s.live ? 'now' : clock(new Date(start + s.lastT * 1000)).slice(0, 5) });
    }

    segmentAt(e) {
        const s = this.session;
        if (!s) return null;
        const rect = this.canvas.getBoundingClientRect();
        const t = ((e.clientX - rect.left) / rect.width) * this.timelineSpan();
        let best = null;
        for (const g of s.segs.values()) {
            if (t >= g.t0 - 0.4 && t <= g.t0 + g.dur + 0.4) { best = g; break; }
        }
        return best;
    }

    onTimelineHover(e) {
        const g = this.segmentAt(e);
        this.canvas.style.cursor = g && g.clip ? 'pointer' : 'default';
        this.canvas.title = g ? (g.text ? `“${shorten(g.text, 140)}”` : (g.state === 'nospeech' ? 'no words recognised' : 'processing…')) : '';
    }

    onTimelineClick(e) {
        const g = this.segmentAt(e);
        if (g && g.clip) this.plugin.playClip(g.clip);
    }

    // ---- console -----------------------------------------------------------
    lineVisible(line) {
        const f = LOG_FILTERS.find((x) => x[0] === this.filter);
        return f ? f[2].includes(line.tag) : true;
    }

    makeLine(line) {
        const row = createDiv({ cls: 'vs-log-line vs-tag-' + line.tag });
        row.createSpan({ cls: 'vs-log-t', text: line.t });
        row.createSpan({ cls: 'vs-log-tag', text: line.tag });
        row.createSpan({ cls: 'vs-log-text', text: line.text });
        return row;
    }

    renderLog() {
        if (!this.logEl) return;
        Object.entries(this.chipEls).forEach(([id, b]) => {
            b.toggleClass('is-on', id === this.filter);
            b.setAttr('aria-pressed', String(id === this.filter));
        });
        this.logEl.empty();
        const lines = this.plugin.logLines.filter((l) => this.lineVisible(l)).slice(-500);
        if (!lines.length) this.logEl.createDiv({ cls: 'vs-muted', text: 'Status messages appear here while VoiceScribe runs.' });
        lines.forEach((l) => this.logEl.appendChild(this.makeLine(l)));
        this.logEl.scrollTop = this.logEl.scrollHeight;
        this.logCountEl.setText(`${this.plugin.logLines.length} lines`);
    }

    appendLog(line) {
        if (!this.logEl) return;
        this.logCountEl.setText(`${this.plugin.logLines.length} lines`);
        if (!this.lineVisible(line)) return;
        const atBottom = this.logEl.scrollHeight - this.logEl.scrollTop - this.logEl.clientHeight < 24;
        const empty = this.logEl.querySelector('.vs-muted');
        if (empty) empty.remove();
        this.logEl.appendChild(this.makeLine(line));
        while (this.logEl.childElementCount > 500) this.logEl.firstElementChild.remove();
        if (atBottom) this.logEl.scrollTop = this.logEl.scrollHeight;
    }
}

// =============================================================================
// Settings tab — every backend parameter
// =============================================================================
class VoiceScribeSettingTab extends obsidian.PluginSettingTab {
    constructor(app, plugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    numberSetting(el, name, desc, key, opts) {
        const o = Object.assign({ min: -Infinity, max: Infinity, int: false }, opts || {});
        const cfg = this.cfg;
        return new obsidian.Setting(el).setName(name).setDesc(desc).addText((t) => {
            t.inputEl.type = 'number';
            if (o.step) t.inputEl.step = String(o.step);
            t.setValue(String(cfg[key])).onChange((v) => {
                let n = o.int ? parseInt(v, 10) : parseFloat(v);
                if (!isFinite(n)) return;
                n = Math.min(o.max, Math.max(o.min, n));
                this.plugin.saveConfigFile({ [key]: n });
            });
        });
    }

    textSetting(el, name, desc, key, placeholder) {
        const cfg = this.cfg;
        return new obsidian.Setting(el).setName(name).setDesc(desc).addText((t) => {
            t.setPlaceholder(placeholder || '').setValue(cfg[key] == null ? '' : String(cfg[key])).onChange((v) => {
                this.plugin.saveConfigFile({ [key]: v.trim() });
            });
        });
    }

    toggleSetting(el, name, desc, key) {
        const cfg = this.cfg;
        return new obsidian.Setting(el).setName(name).setDesc(desc).addToggle((t) => {
            t.setValue(!!cfg[key]).onChange((v) => this.plugin.saveConfigFile({ [key]: v }));
        });
    }

    async display() {
        const { containerEl } = this;
        containerEl.empty();
        this.plugin.resolvePaths();
        this.cfg = this.plugin.loadConfigFile();
        const cfg = this.cfg;

        new obsidian.Setting(containerEl)
            .setName('New to VoiceScribe?')
            .setDesc('The guide explains recording, voice commands, focus, the dashboard and troubleshooting, and lists every command and hotkey.')
            .addButton((b) => b.setButtonText('Open guide').setCta().onClick(() => { this.app.setting.close(); this.plugin.openGuide(); }));
        containerEl.createEl('p', {
            text: 'Everything runs locally: Whisper on your CPU for transcription, Ollama for organising notes. Changes apply the next time recording starts.',
            cls: 'voicescribe-settings-desc',
        });

        // ---------------------------------------------------------- Notes
        new obsidian.Setting(containerEl).setName('Notes').setHeading();
        this.textSetting(containerEl, 'Default note', 'Note that thoughts go into until you say "Topic …" / "Switch to …" (without .md).', 'default_note', 'Voice Notes');
        this.textSetting(containerEl, 'Notes folder', 'Vault folder for the organised notes. Leave empty for the vault root.', 'notes_subfolder', '');
        this.textSetting(containerEl, 'Transcripts folder', 'Daily verbatim transcripts (YYYY-MM-DD_Transcript.md) with audio players.', 'transcripts_folder', 'Transcripts');
        this.textSetting(containerEl, 'Audio archive folder', 'Where the compressed voice clips (.ogg / Opus) are stored.', 'archive_dir', 'VoiceScribe_Archives');

        // ---------------------------------------------------------- Dashboard
        new obsidian.Setting(containerEl).setName('Dashboard').setHeading();
        const pluginToggle = (name, desc, key, after) => new obsidian.Setting(containerEl).setName(name).setDesc(desc).addToggle((t) => {
            t.setValue(!!this.plugin.settings[key]).onChange(async (v) => {
                this.plugin.settings[key] = v;
                await this.plugin.saveData(this.plugin.settings);
                if (after) after(v);
            });
        });
        new obsidian.Setting(containerEl).setName('When recording starts')
            .setDesc('Which note the session writes to. You can always switch by voice: “switch to …” / “create a new note named …”.')
            .addDropdown((dd) => {
                dd.addOption('ask', 'Ask me every time');
                dd.addOption('new', 'Always start a new note');
                dd.addOption('last', 'Continue the last note');
                dd.addOption('default', 'Use the default note');
                dd.setValue(this.plugin.settings.startNoteMode || 'ask').onChange(async (v) => {
                    this.plugin.settings.startNoteMode = v;
                    await this.plugin.saveData(this.plugin.settings);
                });
            });
        new obsidian.Setting(containerEl).setName('New note name')
            .setDesc('Template for new session notes. {date} and {time} are replaced.')
            .addText((t) => t.setPlaceholder(PLUGIN_DEFAULTS.newNoteTemplate).setValue(this.plugin.settings.newNoteTemplate || '')
                .onChange(async (v) => { this.plugin.settings.newNoteTemplate = v.trim() || PLUGIN_DEFAULTS.newNoteTemplate; await this.plugin.saveData(this.plugin.settings); }));
        pluginToggle('Compact source links', 'Show each bullet\'s sources as a small link icon that expands on hover (Live Preview and Reading view).', 'compactSources', () => this.plugin.refreshEditors());
        pluginToggle('Open dashboard when recording starts', 'Shows the session panel (status, audio timeline, console) in the right sidebar.', 'openDashboardOnStart');
        pluginToggle('Follow the active note', 'Open the note being written when recording starts and whenever you switch topic.', 'followActiveNote', () => this.plugin.updateViews('header'));
        pluginToggle('Save session timelines', 'Keep a small loudness timeline per session (VoiceScribe_Archives/sessions) so the last session can be reviewed.', 'saveSessionTimeline');
        pluginToggle('Show live notices', 'Also pop up what was heard after each thought (the dashboard console always shows it).', 'showSpeechNotices');

        // ---------------------------------------------------------- Microphone
        new obsidian.Setting(containerEl).setName('Microphone').setHeading();
        const micSetting = new obsidian.Setting(containerEl).setName('Input device').setDesc('Loading devices…');
        this.plugin.getInputDevices().then((mics) => {
            micSetting.setDesc(mics.length ? 'Microphone used for listening.' : 'Could not list devices (is the Python venv set up?).');
            micSetting.addDropdown((dd) => {
                dd.addOption('null', 'System default');
                mics.forEach((m) => dd.addOption(String(m.id), m.label));
                const cur = cfg.audio_device == null ? 'null' : String(cfg.audio_device);
                if (cur !== 'null' && !mics.some((m) => String(m.id) === cur)) dd.addOption(cur, `Device ${cur} (not found)`);
                dd.setValue(cur).onChange((v) => this.plugin.saveConfigFile({ audio_device: v === 'null' ? null : parseInt(v, 10) }));
            });
            micSetting.addExtraButton((b) => b.setIcon('refresh-cw').setTooltip('Rescan devices').onClick(() => this.display()));
        });

        const thrSetting = this.numberSetting(containerEl, 'Speech volume threshold',
            'Loudness (RMS) above which sound counts as speech. Too low: background noise is recorded. Too high: quiet speech is missed. Use Calibrate.',
            'energy_threshold', { min: 0.0001, max: 0.2, step: 0.0001 });
        thrSetting.addButton((b) => b.setButtonText('Calibrate').setTooltip('Measure background noise for 3 s').onClick(async () => {
            b.setDisabled(true);
            const res = await this.plugin.calibrateMic();
            b.setDisabled(false);
            if (res) {
                this.plugin.saveConfigFile({ energy_threshold: res.suggested_threshold });
                new obsidian.Notice(`Background noise ≈ ${res.noise_p90}. Threshold set to ${res.suggested_threshold}.`, 8000);
                this.display();
            }
        }));

        new obsidian.Setting(containerEl).setName('Pause before a thought ends')
            .setDesc('Seconds of silence that close one spoken thought.')
            .addSlider((s) => s.setLimits(0.8, 4.0, 0.1).setValue(cfg.pause_duration_sec).setDynamicTooltip()
                .onChange((v) => this.plugin.saveConfigFile({ pause_duration_sec: Math.round(v * 10) / 10 })));
        this.numberSetting(containerEl, 'Minimum thought length (s)', 'Shorter sounds (coughs, clicks) are ignored.', 'min_speech_duration_sec', { min: 0.2, max: 10, step: 0.1 });
        this.numberSetting(containerEl, 'Minimum real sound (s)', 'Clips with less sound above the threshold than this are treated as noise blips (stops Whisper inventing text from noise).', 'min_voiced_sec', { min: 0, max: 3, step: 0.05 });
        this.numberSetting(containerEl, 'Maximum thought length (s)', 'Long monologues are split at this length.', 'max_speech_duration_sec', { min: 5, max: 300, step: 1 });
        this.toggleSetting(containerEl, 'Normalise volume', 'Boost quiet microphones before transcription and archiving.', 'normalize_audio');

        // ---------------------------------------------------------- Transcription
        new obsidian.Setting(containerEl).setName('Transcription (Whisper, CPU)').setHeading();
        const known = WHISPER_MODELS.map((m) => m[0]);
        let customText;
        new obsidian.Setting(containerEl).setName('Whisper model').setDesc('Larger = more accurate but slower.')
            .addDropdown((dd) => {
                WHISPER_MODELS.forEach(([v, l]) => dd.addOption(v, l));
                dd.addOption('__custom__', 'Custom (model name or local folder)…');
                dd.setValue(known.includes(cfg.whisper_model) ? cfg.whisper_model : '__custom__').onChange((v) => {
                    if (customText) customText.inputEl.toggle(v === '__custom__');
                    if (v !== '__custom__') this.plugin.saveConfigFile({ whisper_model: v });
                });
            })
            .addText((t) => {
                customText = t;
                t.setPlaceholder('model name / path').setValue(known.includes(cfg.whisper_model) ? '' : cfg.whisper_model)
                    .onChange((v) => { if (v.trim()) this.plugin.saveConfigFile({ whisper_model: v.trim() }); });
                t.inputEl.toggle(!known.includes(cfg.whisper_model));
            });
        new obsidian.Setting(containerEl).setName('Run Whisper on')
            .setDesc('Auto uses your NVIDIA GPU when it works (much faster: about a second per thought) and falls back to the CPU otherwise. The GPU is tested once in a separate process, so a missing driver can never crash a session.')
            .addDropdown((dd) => {
                dd.addOption('auto', 'Auto (GPU if available)');
                dd.addOption('cuda', 'GPU (NVIDIA CUDA)');
                dd.addOption('cpu', 'CPU only');
                dd.setValue(cfg.whisper_device || 'auto').onChange((v) => this.plugin.saveConfigFile({ whisper_device: v }));
            });
        const gpuSetting = new obsidian.Setting(containerEl).setName('GPU support')
            .setDesc('The GPU needs NVIDIA’s CUDA libraries (cuBLAS 12 and cuDNN 9) in VoiceScribe’s Python environment. Installing them downloads about 1 GB once.');
        gpuSetting.addButton((b) => b.setButtonText('Check GPU').onClick(async () => {
            b.setDisabled(true);
            const st = await this.plugin.gpuStatus();
            b.setDisabled(false);
            if (!st) { new obsidian.Notice('VoiceScribe: could not query the GPU status.'); return; }
            const probes = Object.entries(st.probes || {}).map(([k, v]) => `${v.ok ? '✅' : '❌'} ${k.split('|')[0]}${v.ok ? '' : ': ' + (v.detail || '')}`);
            new obsidian.Notice([
                st.cuda_devices ? `✅ CUDA GPU detected (${st.cuda_devices})` : '❌ No usable CUDA GPU detected',
                st.nvidia_libs ? '✅ CUDA libraries installed' : '⚠️ CUDA libraries not installed (use “Install GPU support”)',
            ].concat(probes.length ? probes : ['Not tested yet: the test runs the next time you start recording.']).join('\n'), 15000);
        }));
        gpuSetting.addButton((b) => b.setButtonText('Install GPU support…').onClick(() => {
            new VoiceScribeGpuInstallModal(this.app, this.plugin, () => this.display()).open();
        }));
        new obsidian.Setting(containerEl).setName('GPU libraries folder')
            .setDesc(cfg.cuda_libs_dir ? 'Where VoiceScribe looks for the CUDA libraries.' : 'Empty = VoiceScribe’s Python environment. Use “Install GPU support…” to pick a folder.')
            .addText((t) => t.setPlaceholder('(Python environment)').setValue(cfg.cuda_libs_dir || '')
                .onChange((v) => { this.plugin.useCudaDir(v.trim()); }));
        new obsidian.Setting(containerEl).setName('GPU compute type').setDesc('int8_float16 uses the least video memory (leaves room for Ollama).')
            .addDropdown((dd) => {
                ['int8_float16', 'float16', 'int8'].forEach((v) => dd.addOption(v, v));
                dd.setValue(cfg.whisper_gpu_compute_type || 'int8_float16').onChange((v) => this.plugin.saveConfigFile({ whisper_gpu_compute_type: v }));
            });
        new obsidian.Setting(containerEl).setName('CPU compute type').setDesc('int8 is fastest on CPU with negligible accuracy loss.')
            .addDropdown((dd) => {
                ['int8', 'int8_float32', 'float32'].forEach((v) => dd.addOption(v, v));
                dd.setValue(cfg.whisper_compute_type).onChange((v) => this.plugin.saveConfigFile({ whisper_compute_type: v }));
            });
        this.numberSetting(containerEl, 'CPU threads', 'Threads used by Whisper.', 'whisper_threads', { min: 1, max: 64, int: true });
        this.numberSetting(containerEl, 'Release Whisper after (minutes idle)', 'Frees the GPU/RAM Whisper uses when you have not spoken for this long during a session (e.g. while paused). It reloads automatically on the next thought, which takes a few seconds. 0 = keep it loaded.', 'whisper_idle_unload_min', { min: 0, max: 600, step: 1 });
        this.numberSetting(containerEl, 'Beam size', '1 = fastest, 5 = most careful decoding.', 'whisper_beam_size', { min: 1, max: 10, int: true });
        this.textSetting(containerEl, 'Language', 'Language code (en, fr, sw, …) or "auto" to detect. ".en" models are English-only.', 'whisper_language', 'en');
        this.toggleSetting(containerEl, 'Context prompting', 'Prime Whisper with the current topic and recent sentences for better names/jargon.', 'context_prompt_enabled');
        this.toggleSetting(containerEl, 'Whisper silence filter', 'Skip non-speech inside each clip (recommended).', 'whisper_vad_filter');
        this.toggleSetting(containerEl, 'Allow model downloads',
            'VoiceScribe stays offline by default. Turn this on once to download a Whisper model you have not used before, then turn it off again.',
            'allow_model_downloads');

        // ---------------------------------------------------------- Organisation
        new obsidian.Setting(containerEl).setName('Note organisation (Ollama)').setHeading();
        this.toggleSetting(containerEl, 'Organise with local LLM', 'Group thoughts under theme headings and merge corrections. Off = append thoughts as-is.', 'llm_enabled');
        new obsidian.Setting(containerEl).setName('How to organise')
            .setDesc('Live: each thought is placed as soon as it is transcribed. In batches: thoughts appear at once in an “📥 Inbox” section and are organised together every few minutes and when you stop. The model then sees the whole batch, so it merges related thoughts and applies corrections better, and it can be a bigger model.')
            .addDropdown((dd) => {
                dd.addOption('live', 'Live (each thought)');
                dd.addOption('batch', 'In batches');
                dd.setValue(cfg.organise_mode || 'live').onChange((v) => { this.plugin.saveConfigFile({ organise_mode: v }); this.display(); });
            });
        if ((cfg.organise_mode || 'live') === 'batch') {
            this.numberSetting(containerEl, 'Batch: organise every (minutes)', '0 = only when recording stops (and “Organise now”).', 'batch_interval_min', { min: 0, max: 120, step: 0.5 });
            const bm = new obsidian.Setting(containerEl).setName('Batch: model')
                .setDesc('Leave empty to use the model above. A bigger model organises much better, and it also writes the session summary in batch mode. Because it only runs every few minutes, a model that runs partly from RAM is fine: e.g. qwen3:30b-a3b or gemma4:26b-a4b (best quality), granite4.2:8b or qwen3.5:9b (faster). Hidden “thinking” is switched off automatically, so reasoning models stay fast.');
            bm.addText((t) => t.setPlaceholder(`(same: ${cfg.ollama_model})`).setValue(cfg.batch_ollama_model || '')
                .onChange((v) => this.plugin.saveConfigFile({ batch_ollama_model: v.trim() })));
            this.plugin.getOllamaModels().then((models) => {
                if (!models.length) return;
                bm.addDropdown((dd) => {
                    dd.addOption('', 'Pick installed…');
                    models.forEach((m) => dd.addOption(m, m));
                    dd.onChange((v) => { if (v) { this.plugin.saveConfigFile({ batch_ollama_model: v }); this.display(); } });
                });
            });
            this.toggleSetting(containerEl, 'Batch: “On this page” list', 'Each batch pass refreshes a “🧭 On this page” section above the notes: links to every theme heading (with how many points it holds), the inbox and the session summaries. Appears once a note has two or more themes.', 'toc_enabled');
            this.numberSetting(containerEl, 'Batch: thoughts per pass', 'Larger inboxes are organised in parts of this size.', 'batch_max_thoughts', { min: 3, max: 100, int: true });
            this.numberSetting(containerEl, 'Batch: context window (tokens)', 'Room for the note plus the batch.', 'batch_num_ctx', { min: 2048, max: 131072, int: true });
            this.numberSetting(containerEl, 'Batch: max output tokens', '', 'batch_num_predict', { min: 256, max: 16384, int: true });
        }
        const ollamaSetting = new obsidian.Setting(containerEl).setName('Ollama model').setDesc('Scanning installed models…');
        this.plugin.getOllamaModels().then((models) => {
            ollamaSetting.setDesc(models.length ? 'Installed models on your Ollama server.' : 'Ollama not reachable — type a model name.');
            let txt;
            if (models.length) {
                ollamaSetting.addDropdown((dd) => {
                    models.forEach((m) => dd.addOption(m, m));
                    dd.addOption('__custom__', 'Other…');
                    dd.setValue(models.includes(cfg.ollama_model) ? cfg.ollama_model : '__custom__').onChange((v) => {
                        if (txt) txt.inputEl.toggle(v === '__custom__');
                        if (v !== '__custom__') this.plugin.saveConfigFile({ ollama_model: v });
                    });
                });
            }
            ollamaSetting.addText((t) => {
                txt = t;
                t.setPlaceholder('e.g. llama3.2:3b').setValue(cfg.ollama_model)
                    .onChange((v) => { if (v.trim()) this.plugin.saveConfigFile({ ollama_model: v.trim() }); });
                t.inputEl.toggle(!models.length || !models.includes(cfg.ollama_model));
            });
            ollamaSetting.addExtraButton((b) => b.setIcon('refresh-cw').setTooltip('Rescan').onClick(() => this.display()));
        });
        this.textSetting(containerEl, 'Ollama host', 'Local Ollama address.', 'ollama_host', 'http://127.0.0.1:11434');
        this.numberSetting(containerEl, 'Context window (tokens)', 'How much of the existing note the model can see.', 'ollama_num_ctx', { min: 1024, max: 131072, int: true });
        this.numberSetting(containerEl, 'Max output tokens', 'Upper limit for each organising answer.', 'ollama_num_predict', { min: 64, max: 4096, int: true });
        this.numberSetting(containerEl, 'Timeout (s)', 'Give up and append the thought as-is after this long.', 'ollama_timeout_sec', { min: 5, max: 900, int: true });
        this.numberSetting(containerEl, 'Unload model after (minutes idle)', 'Ollama frees the model’s GPU/RAM when it has not been used for this long. Before organising, VoiceScribe checks the model is loaded and loads it again if needed (the dashboard shows “loading …”). 0 = unload after every answer (slow), -1 = never.', 'ollama_keep_alive_min', { min: -1, max: 1440, step: 1 });
        this.toggleSetting(containerEl, 'Unload models when recording stops', 'Free the Ollama models’ memory as soon as a session (or import) finishes, instead of waiting for the idle time above.', 'unload_on_stop');
        const memSetting = new obsidian.Setting(containerEl).setName('Models in memory').setDesc('Checking Ollama…');
        const showMem = async () => {
            const loaded = await this.plugin.getLoadedModels();
            const gb = (b) => (b / 1e9).toFixed(1) + ' GB';
            if (loaded === null) memSetting.setDesc('Ollama is not running — nothing loaded.');
            else if (!loaded.length) memSetting.setDesc('No model loaded.');
            else memSetting.setDesc(loaded.map((m) => `${m.name}: ${gb(m.size)}${m.size ? ` (${Math.round(100 * m.vram / m.size)}% on GPU)` : ''}`).join(' · '));
        };
        memSetting.addExtraButton((b) => b.setIcon('refresh-cw').setTooltip('Refresh').onClick(showMem));
        memSetting.addButton((b) => b.setButtonText('Unload now').onClick(async () => { await this.plugin.unloadModelsNow(); showMem(); }));
        showMem();
        this.numberSetting(containerEl, 'Minimum words per thought', 'Shorter fragments are kept in the transcript but not added to notes.', 'min_substantive_words', { min: 1, max: 20, int: true });
        this.toggleSetting(containerEl, 'Session summary', 'When recording stops, add a “🧾 Session summary” section to each note: a short summary, action items (as checkboxes) and open questions, each linked to its sources.', 'summary_enabled');
        this.numberSetting(containerEl, 'Summary: minimum thoughts', 'Only summarise notes that got at least this many thoughts in the session.', 'summary_min_thoughts', { min: 1, max: 50, int: true });
        this.toggleSetting(containerEl, 'Link existing notes', 'Turn mentions of notes that already exist in the vault into [[links]] (multi-word names in any case; single words only when written like the note name).', 'autolink_enabled');
        this.toggleSetting(containerEl, 'Note properties', 'Keep created, updated, voice_sessions, voice_minutes and a tag in each note’s properties. Other properties are left untouched.', 'properties_enabled');
        this.textSetting(containerEl, 'Tag for voice notes', 'Added to the tags property (leave empty for none).', 'properties_tag', 'voicescribe');

        // ---------------------------------------------------------- Storage
        new obsidian.Setting(containerEl).setName('Audio storage').setHeading();
        new obsidian.Setting(containerEl).setName('Opus bitrate').setDesc('Compressed voice clips. 12k ≈ 5 MB per hour of speech.')
            .addDropdown((dd) => {
                ['8k', '10k', '12k', '16k', '24k', '32k'].forEach((v) => dd.addOption(v, v));
                if (!['8k', '10k', '12k', '16k', '24k', '32k'].includes(cfg.opus_bitrate)) dd.addOption(cfg.opus_bitrate, cfg.opus_bitrate);
                dd.setValue(cfg.opus_bitrate).onChange((v) => this.plugin.saveConfigFile({ opus_bitrate: v }));
            });

        // ---------------------------------------------------------- Backend
        new obsidian.Setting(containerEl).setName('Backend').setHeading();
        new obsidian.Setting(containerEl).setName('VoiceScribe folder')
            .setDesc(`Folder containing scribe_router.py. Currently: ${this.plugin.backendDir}`)
            .addText((t) => t.setPlaceholder('(default: next to this vault)').setValue(this.plugin.settings.backendDir)
                .onChange(async (v) => { this.plugin.settings.backendDir = v.trim(); await this.plugin.savePluginSettings(); }));
        new obsidian.Setting(containerEl).setName('Python executable')
            .setDesc(`Currently: ${this.plugin.pythonPath}`)
            .addText((t) => t.setPlaceholder('(default: venv in the VoiceScribe folder)').setValue(this.plugin.settings.pythonPath)
                .onChange(async (v) => { this.plugin.settings.pythonPath = v.trim(); await this.plugin.savePluginSettings(); }));

        new obsidian.Setting(containerEl).setName('Diagnostics').setDesc('Check Python, FFmpeg, Ollama and the selected model.')
            .addButton((b) => b.setButtonText('Run checks').setCta().onClick(async () => {
                b.setDisabled(true);
                await runDiagnostics(this.plugin);
                b.setDisabled(false);
            }));
    }
}

async function runDiagnostics(plugin) {
    plugin.resolvePaths();
    const cfg = plugin.loadConfigFile();
    const lines = [];
    lines.push((fs.existsSync(plugin.scriptPath) ? '✅' : '❌') + ' Backend: ' + plugin.backendDir);
    try {
        const v = await runFile(plugin.pythonPath, ['-c', 'import faster_whisper, sounddevice, ollama, numpy; print("ok")'], { cwd: plugin.backendDir, timeout: 30000 });
        lines.push(v.includes('ok') ? '✅ Python environment and packages' : '❌ Python packages missing');
    } catch (e) {
        lines.push('❌ Python/packages: ' + ((e.stderr || e.message || '').trim().split(/\r?\n/).pop()));
    }
    try {
        await runFile('ffmpeg', ['-hide_banner', '-version'], { timeout: 10000 });
        lines.push('✅ FFmpeg found');
    } catch (e) {
        lines.push('⚠️ FFmpeg not on PATH (built-in encoder will be used)');
    }
    if (cfg.whisper_device !== 'cpu') {
        const st = await plugin.gpuStatus();
        if (st && st.cuda_devices) lines.push(st.nvidia_libs ? '✅ NVIDIA GPU + CUDA libraries (GPU transcription available)' : '⚠️ NVIDIA GPU found, CUDA libraries missing: Settings → Transcription → Install GPU support');
        else lines.push('ℹ️ No CUDA GPU: Whisper runs on the CPU');
    }
    if (cfg.llm_enabled) {
        const models = await plugin.getOllamaModels();
        if (!models.length) lines.push(`⚠️ Ollama not reachable at ${cfg.ollama_host} (VoiceScribe will try to start it)`);
        else lines.push((models.includes(cfg.ollama_model) ? '✅' : '❌') + ` Ollama model '${cfg.ollama_model}'` + (models.includes(cfg.ollama_model) ? '' : ` not installed — run: ollama pull ${cfg.ollama_model}`));
    }
    new obsidian.Notice(lines.join('\n'), 15000);
}

// =============================================================================
// Quick options modal
// =============================================================================
class VoiceScribeOptionsModal extends obsidian.Modal {
    constructor(app, plugin) {
        super(app);
        this.plugin = plugin;
    }

    async onOpen() {
        const { contentEl } = this;
        contentEl.empty();
        const cfg = this.plugin.loadConfigFile();
        this.titleEl.setText('VoiceScribe quick options');

        const mics = await this.plugin.getInputDevices();
        new obsidian.Setting(contentEl).setName('Microphone').addDropdown((dd) => {
            dd.addOption('null', 'System default');
            mics.forEach((m) => dd.addOption(String(m.id), m.label));
            const cur = cfg.audio_device == null ? 'null' : String(cfg.audio_device);
            if (cur !== 'null' && !mics.some((m) => String(m.id) === cur)) dd.addOption(cur, `Device ${cur}`);
            dd.setValue(cur).onChange((v) => this.plugin.saveConfigFile({ audio_device: v === 'null' ? null : parseInt(v, 10) }));
        });

        new obsidian.Setting(contentEl).setName('Whisper model').addDropdown((dd) => {
            WHISPER_MODELS.forEach(([v, l]) => dd.addOption(v, l));
            if (!WHISPER_MODELS.some((m) => m[0] === cfg.whisper_model)) dd.addOption(cfg.whisper_model, cfg.whisper_model);
            dd.setValue(cfg.whisper_model).onChange((v) => this.plugin.saveConfigFile({ whisper_model: v }));
        });

        const models = await this.plugin.getOllamaModels();
        new obsidian.Setting(contentEl).setName('Ollama model').addDropdown((dd) => {
            const list = models.length ? models.slice() : [];
            if (!list.includes(cfg.ollama_model)) list.unshift(cfg.ollama_model);
            list.forEach((m) => dd.addOption(m, m));
            dd.setValue(cfg.ollama_model).onChange((v) => this.plugin.saveConfigFile({ ollama_model: v }));
        });

        new obsidian.Setting(contentEl).setName('How to organise').addDropdown((dd) => {
            dd.addOption('live', 'Live (each thought)');
            dd.addOption('batch', 'In batches');
            dd.setValue(cfg.organise_mode || 'live').onChange((v) => this.plugin.saveConfigFile({ organise_mode: v }));
        });

        new obsidian.Setting(contentEl).setName('Pause before a thought ends (s)').addSlider((s) =>
            s.setLimits(0.8, 4.0, 0.1).setValue(cfg.pause_duration_sec).setDynamicTooltip()
                .onChange((v) => this.plugin.saveConfigFile({ pause_duration_sec: Math.round(v * 10) / 10 })));

        new obsidian.Setting(contentEl)
            .addButton((b) => b.setButtonText('All settings…').onClick(() => {
                this.close();
                this.app.setting.open();
                this.app.setting.openTabById(this.plugin.manifest.id);
            }))
            .addButton((b) => b.setButtonText('Done').setCta().onClick(() => this.close()));
    }

    onClose() {
        this.contentEl.empty();
    }
}


// =============================================================================
// Guide: how to use VoiceScribe (opens as a tab; also from settings & dashboard)
// =============================================================================
const VIEW_TYPE_GUIDE = 'voicescribe-guide';

const GUIDE_COMMANDS = [
    ['toggle-voicescribe-recording', 'Start / stop recording', 'Alt+V'],
    ['focus-add', 'Focus here: add details to the selection', 'Alt+F'],
    ['focus-correct', 'Focus here: correct the selection', 'Alt+Shift+F'],
    ['clear-focus', 'Clear focus', ''],
    ['pause-resume', 'Pause / resume listening', 'Alt+P'],
    ['undo-last-change', 'Undo the last note change (scratch that)', 'Alt+Z'],
    ['summarize-now', 'Write the session summary now', ''],
    ['organise-now', 'Organise the inbox now (batch mode)', ''],
    ['toggle-organise-mode', 'Switch organising: live ⇄ in batches', ''],
    ['unload-models', 'Free memory: unload the Ollama models now', ''],
    ['import-recording', 'Import an audio recording…', ''],
    ['open-dashboard', 'Open session dashboard', ''],
    ['open-active-voice-notes', 'Open active voice note', ''],
    ['open-today-transcript', "Open today's transcript", ''],
    ['open-voicescribe-options-menu', 'Quick options (microphone & models)', 'Alt+Shift+V'],
    ['force-stop', 'Force stop (discard unfinished processing)', ''],
    ['open-guide', 'Open the guide', ''],
    ['open-voicescribe-settings-tab', 'Open settings', ''],
];

const VOICE_COMMANDS = [
    ['Create a new note named Grocery list', 'Creates a new note and switches to it'],
    ['New note called Ideas', 'Same as above'],
    ['Switch to a new note called Cloud Pro', 'Same as above'],
    ['Switch to Welcome.md', 'Opens an existing note anywhere in the vault (created if missing)'],
    ['Switch to the Welcome note', 'Same as above'],
    ['Go back to Voice Notes', 'Same as above'],
    ['Open the note called Mum', 'Same as above'],
    ['Topic Algorand', 'Short form of “switch to”'],
    ['Note French toast', 'Short form of “switch to”'],
    ['Scratch that', 'Undoes the last change to your notes'],
    ['Undo that · Remove the last point · Take that back', 'Same as above'],
];

class VoiceScribeGuideView extends obsidian.ItemView {
    constructor(leaf, plugin) {
        super(leaf);
        this.plugin = plugin;
    }

    getViewType() { return VIEW_TYPE_GUIDE; }
    getDisplayText() { return 'VoiceScribe guide'; }
    getIcon() { return 'book-open'; }

    async onOpen() { this.render(); }

    hotkeyFor(id, fallback) {
        // Show the user's own hotkey when they changed it (Settings → Hotkeys).
        try {
            const hm = this.plugin.app.hotkeyManager;
            const full = `${this.plugin.manifest.id}:${id}`;
            const keys = (hm && (hm.getHotkeys(full) || hm.getDefaultHotkeys(full))) || [];
            if (keys.length) {
                const k = keys[0];
                const mods = (k.modifiers || []).map((m) => (m === 'Mod' ? (process.platform === 'darwin' ? 'Cmd' : 'Ctrl') : m));
                return mods.concat([String(k.key).length === 1 ? k.key.toUpperCase() : k.key]).join('+');
            }
            if (hm && hm.getHotkeys(full)) return '';
        } catch (e) { /* private API changed: use the default */ }
        return fallback;
    }

    keys(el, combo) {
        if (!combo) { el.createSpan({ cls: 'vs-muted', text: '—' }); return; }
        combo.split('+').forEach((k, i) => {
            if (i) el.appendText('+');
            el.createEl('kbd', { text: k });
        });
    }

    action(el, label, fn, cta) {
        const b = el.createEl('button', { text: label, cls: cta ? 'mod-cta' : '' });
        b.addEventListener('click', fn);
        return b;
    }

    section(root, id, title) {
        const s = root.createEl('section', { cls: 'vs-guide-section', attr: { id: 'vs-guide-' + id } });
        s.createEl('h2', { text: title });
        this.toc.push([id, title]);
        return s;
    }

    table(el, head, rows, cellFn) {
        const t = el.createEl('table', { cls: 'vs-guide-table' });
        const tr = t.createEl('thead').createEl('tr');
        head.forEach((h) => tr.createEl('th', { text: h }));
        const body = t.createEl('tbody');
        rows.forEach((r) => {
            const row = body.createEl('tr');
            r.forEach((c, i) => {
                const td = row.createEl('td');
                if (cellFn && cellFn(td, c, i, r)) return;
                td.setText(c);
            });
        });
        return t;
    }

    render() {
        const p = this.plugin;
        const root = this.contentEl;
        root.empty();
        root.addClass('vs-guide');
        this.toc = [];

        const layout = root.createDiv({ cls: 'vs-guide-layout' });
        const nav = layout.createEl('nav', { cls: 'vs-guide-nav', attr: { 'aria-label': 'Guide sections' } });
        const main = layout.createDiv({ cls: 'vs-guide-main' });

        // ---------------------------------------------------------------- intro
        const hero = main.createDiv({ cls: 'vs-guide-hero' });
        hero.createEl('h1', { text: 'VoiceScribe guide' });
        hero.createEl('p', {
            text: 'Talk freely. VoiceScribe transcribes what you say, organises it into notes under theme headings, ' +
                'and links every point back to the transcript and the voice clip it came from. ' +
                'Everything runs on your computer: Whisper for transcription and Ollama for organising.',
        });
        const heroActions = hero.createDiv({ cls: 'vs-guide-actions' });
        this.action(heroActions, p.state === 'idle' ? 'Start recording' : 'Open dashboard',
            () => (p.state === 'idle' ? p.toggleRecording() : p.activateDashboard()), true);
        this.action(heroActions, 'Open settings', () => { this.plugin.app.setting.open(); this.plugin.app.setting.openTabById(p.manifest.id); });

        // ---------------------------------------------------------------- quick start
        let s = this.section(main, 'start', 'Quick start');
        const steps = s.createEl('ol', { cls: 'vs-guide-steps' });
        let li = steps.createEl('li');
        li.createEl('strong', { text: 'Check your setup. ' });
        li.appendText('Python environment, FFmpeg and your Ollama model. ');
        this.action(li, 'Run checks', () => runDiagnostics(p));
        li = steps.createEl('li');
        li.createEl('strong', { text: 'Calibrate the microphone. ' });
        li.appendText('Stay quiet for 3 seconds so VoiceScribe learns your background noise; otherwise noise can be recorded and turned into made-up text. ');
        this.action(li, 'Calibrate', async () => {
            const res = await p.calibrateMic();
            if (res) {
                p.saveConfigFile({ energy_threshold: res.suggested_threshold });
                new obsidian.Notice(`Threshold set to ${res.suggested_threshold} (background ≈ ${res.noise_p90}).`, 8000);
            }
        });
        li = steps.createEl('li');
        li.createEl('strong', { text: 'Start recording. ' });
        li.appendText('Press ');
        this.keys(li, this.hotkeyFor('toggle-voicescribe-recording', 'Alt+V'));
        li.appendText(' or the microphone icon, then choose where the session goes: a new note, a recent note or any note.');
        li = steps.createEl('li');
        li.createEl('strong', { text: 'Think out loud. ' });
        li.appendText('Pause naturally; each pause (about 1.7 s) ends one “thought”. The note and the dashboard update as you talk.');
        li = steps.createEl('li');
        li.createEl('strong', { text: 'Stop. ' });
        li.appendText('Press the same key again. VoiceScribe finishes what you already said, then stops (press again to force it).');

        // ---------------------------------------------------------------- notes
        s = this.section(main, 'notes', 'What gets written where');
        this.table(s, ['File', 'What it holds'], [
            ['Your note (e.g. “Voice note 2026-10-30 14.05”)', 'Organised bullet points under ### theme headings in a “💡 Notes” section. Your own text in the note is kept.'],
            [`${(p.config || CONFIG_DEFAULTS).transcripts_folder}/YYYY-MM-DD_Transcript.md`, 'Everything you said, word for word, with an audio player for each thought.'],
            [`${(p.config || CONFIG_DEFAULTS).archive_dir}/`, 'Compressed voice clips (.ogg, about 5 MB per hour of speech) and small session timelines.'],
        ]);
        s.createEl('p', { text: 'You can edit notes at any time, even while recording; VoiceScribe re-reads the note before each change.' });

        // ---------------------------------------------------------------- organise modes
        s = this.section(main, 'modes', 'Live or batch organising');
        s.createEl('p', { text: 'Choose how thoughts become notes, in the start dialog, in settings or with the command “Switch organising”. Both modes keep every source link, voice command, undo and the session summary.' });
        this.table(s, ['Mode', 'How it works', 'Good for'], [
            ['Live', 'Each thought is placed under a theme heading as soon as it is transcribed.', 'Watching the note take shape; short sessions; correcting with Focus as you go.'],
            ['In batches', 'Each thought appears at once in a “📥 Inbox (to organise)” section with its sources. Every few minutes and when you stop, the model organises the whole inbox together, so it can merge related thoughts, keep only the corrected version when you change your mind, and use a bigger model.', 'Long brain-dumps where structure matters more than seeing it live.'],
        ]);
        const bt = s.createDiv({ cls: 'vs-guide-tip' });
        bt.createEl('strong', { text: 'Batch details: ' });
        bt.appendText('“Organise now” on the dashboard runs it immediately. If the model is unavailable, thoughts simply stay in the inbox and nothing is lost. Thoughts the model leaves out are kept under “Other thoughts”. “Scratch that” removes the last inbox entry, or undoes the last organising pass. Focus always applies immediately, even in batch mode. Each pass also refreshes a “🧭 On this page” list above the notes, with links to every theme, the inbox and the summaries (setting “Batch: On this page list”).');

        // ---------------------------------------------------------------- voice commands
        s = this.section(main, 'voice', 'Voice commands');
        s.createEl('p', { text: 'Switch notes without touching the keyboard. Say the command as its own sentence; at the start or end of a longer thought also works.' });
        this.table(s, ['Say', 'What happens'], VOICE_COMMANDS, (td, c, i) => {
            if (i === 0) { td.createEl('code', { text: `“${c}”` }); return true; }
            return false;
        });
        const tip = s.createDiv({ cls: 'vs-guide-tip' });
        tip.createEl('strong', { text: 'Not commands: ' });
        tip.appendText('ordinary sentences such as “Note that the server is down” or “Switch to talking about elephants” are written into your note as normal. ' +
            '“Create a new note…” never reuses an existing note (it adds “ 2”); “Switch to…” opens the existing note if there is one. ' +
            'Every switch appears as a purple line on the dashboard timeline.');

        // ---------------------------------------------------------------- pause / undo / summary
        s = this.section(main, 'control', 'Pause, undo and summaries');
        this.table(s, ['', 'How', 'What it does'], [
            ['Pause', this.hotkeyFor('pause-resume', 'Alt+P'), 'Stops listening without ending the session (a phone call, a colleague). Anything you were saying is kept. Paused time is shaded on the timeline. Press again or Resume to continue.'],
            ['Undo', this.hotkeyFor('undo-last-change', 'Alt+Z'), 'Say “scratch that”, “undo that” or “remove the last point”, or press the key or the dashboard button. The last thought’s change is undone. If you edited the note since, only the points it added are removed and your edits stay.'],
            ['Summary', 'automatic', 'When you stop, each note that got enough thoughts gets a “🧾 Session summary” section: a short summary, action items as checkboxes and open questions, each linked to its sources. “Summary now” on the dashboard writes it mid-session; running it again replaces that session’s summary rather than adding a second.'],
        ], (td, c, i) => { if (i === 1 && c !== 'automatic') { this.keys(td, c); return true; } return false; });

        // ---------------------------------------------------------------- focus
        s = this.section(main, 'focus', 'Focus: guide where your words go');
        s.createEl('p', { text: 'Select part of a note — or just put the cursor on a bullet or a ### heading — then:' });
        this.table(s, ['Keys', 'Mode', 'What your next thoughts do'], [
            [this.hotkeyFor('focus-add', 'Alt+F'), 'Add details here', 'Added right after the focused bullet (or into the focused section). Thoughts clearly about something else still go elsewhere.'],
            [this.hotkeyFor('focus-correct', 'Alt+Shift+F'), 'Correct this', 'Rewrite only the focused bullet(s). Nothing else in the note changes; the old source links are kept and the new one is added.'],
        ], (td, c, i) => { if (i === 0) { this.keys(td, c); return true; } return false; });
        s.createEl('p', { text: 'Both are also in the right-click menu. The focused text is outlined in purple and a Focus card appears on the dashboard, where you can switch mode or clear it. Focus ends when you clear it, switch notes or stop recording. Focusing in another note makes that note the active one.' });
        const ex = s.createDiv({ cls: 'vs-guide-tip' });
        ex.createEl('strong', { text: 'Example: ' });
        ex.appendText('the note says “Soak each slice for a minute”. Put the cursor on it, press Alt+Shift+F and say “Actually thirty seconds, longer and it falls apart.” The bullet becomes “Soak each slice for about 30 seconds — longer and it falls apart”.');

        // ---------------------------------------------------------------- dashboard
        s = this.section(main, 'dashboard', 'The session dashboard');
        s.createEl('p', { text: 'Opens in the right sidebar when recording starts (or click the status bar item).' });
        const dl = s.createEl('ul');
        [
            ['Status & timer', 'listening, transcribing, organising; Start / Stop.'],
            ['Following the active note', 'the note being written to opens automatically and follows voice commands. “Pin this note” stops that for the session.'],
            ['Session audio', 'a live input meter (orange line = speech threshold) and a timeline of the whole session: teal = saved speech, blue = still processing, grey = silence, purple = note switch. Click a segment to play it; hover to see what was heard.'],
            ['Counters', 'thoughts captured, transcribed, note updates, waiting in the queue.'],
            ['Console', 'every status message. Filter by Speech, Notes, Issues or Raw backend output.'],
        ].forEach(([a, b]) => { const item = dl.createEl('li'); item.createEl('strong', { text: a + ': ' }); item.appendText(b); });
        this.action(s, 'Open dashboard', () => p.activateDashboard());

        // ---------------------------------------------------------------- sources
        s = this.section(main, 'sources', 'Sources and voice clips');
        s.createEl('p', { text: 'Every bullet ends with a small link icon (with a number when several thoughts support it). Hover it to see when you said it, the exact words from the transcript, and a Play button for the voice clip. Click the time to jump to that line of the transcript.' });
        s.createEl('p', { text: 'In Live Preview, put the cursor on the icon to see and edit the underlying links. Turn the compact icons off in Settings → Dashboard → Compact source links.' });

        // ---------------------------------------------------------------- import
        s = this.section(main, 'import', 'Importing recordings');
        s.createEl('p', { text: 'Thinking out loud often happens away from the desk. Record a voice memo on your phone, copy it to your computer or vault, then import it. VoiceScribe splits it into thoughts at pauses and runs them through the same steps as live recording: voice clips, transcript, organised note, voice commands (“create a new note named …” works inside recordings too), undo and a summary.' });
        s.createEl('p', { text: 'Supported: mp3, m4a, wav, ogg, opus, webm, flac, aac, mp4 and more. Times in the transcript come from the file’s date. The original file is not changed.' });
        this.action(s, 'Import a recording…', () => p.importRecording());

        // ---------------------------------------------------------------- links & properties
        s = this.section(main, 'links', 'Links and properties');
        s.createEl('p', { text: 'When a thought mentions a note that already exists (say “Cloud Pro”), the new bullet links to it as [[Cloud Pro]], so voice notes join your graph. Multi-word note names match in any case; one-word names only match when written exactly like the note (so a note called “Ideas” doesn’t link every “ideas”). Your own text is never changed.' });
        s.createEl('p', { text: 'Each voice note keeps these properties: created, updated, voice_sessions, voice_minutes and the tag “voicescribe”. They work with search, the Properties view and Dataview. Other properties you add are left alone. Both features can be turned off in settings.' });

        // ---------------------------------------------------------------- gpu
        s = this.section(main, 'memory', 'Freeing GPU and memory');
        s.createEl('p', { text: 'Models only use video memory and RAM while they are loaded. VoiceScribe frees them on its own and loads them again when needed:' });
        this.table(s, ['Model', 'When it is freed', 'Setting'], [
            ['Ollama (organising)', `After ${(p.config || CONFIG_DEFAULTS).ollama_keep_alive_min} min without use, and when recording stops. Before every organising step VoiceScribe checks the model is loaded and loads it first if not (“loading …” on the dashboard).`, 'Unload model after · Unload models when recording stops'],
            ['Whisper (transcription)', `After ${(p.config || CONFIG_DEFAULTS).whisper_idle_unload_min} min without speech in a session (e.g. paused), and always when the session ends. It reloads on your next thought.`, 'Release Whisper after'],
        ]);
        s.createEl('p', { text: 'To free memory right away, use the command “Free memory: unload the Ollama models now” or “Unload now” under Settings → Note organisation → Models in memory.' });

        s = this.section(main, 'gpu', 'Faster transcription on your GPU');
        s.createEl('p', { text: 'On the CPU, Whisper needs a few seconds per thought. With an NVIDIA GPU it takes about one second, and it still leaves room for Ollama. “Run Whisper on: Auto” uses the GPU whenever it works. The first time, VoiceScribe tests the GPU in a separate process and falls back to the CPU if anything is missing, so a session never crashes.' });
        const gpuSteps = s.createEl('ol', { cls: 'vs-guide-steps' });
        li = gpuSteps.createEl('li'); li.appendText('Settings → Transcription → '); li.createEl('strong', { text: 'Install GPU support…' }); li.appendText(' and choose where the libraries go (about 1 GB, once). The recommended folder is local and not synced. Your vault and OneDrive/MEGA folders would upload the whole download. If CUDA 12 is already installed, choose “Use CUDA already installed” and nothing is downloaded.');
        li = gpuSteps.createEl('li'); li.appendText('Start recording. The console shows “Whisper ready (…, GPU)”.');
        li = gpuSteps.createEl('li'); li.appendText('Use '); li.createEl('strong', { text: 'Check GPU' }); li.appendText(' in settings if it still says CPU.');

        // ---------------------------------------------------------------- commands
        s = this.section(main, 'commands', 'All commands and hotkeys');
        s.createEl('p', { text: 'Available from the command palette (Ctrl+P). Change hotkeys in Settings → Hotkeys (search “VoiceScribe”).' });
        this.table(s, ['Command', 'Hotkey', ''], GUIDE_COMMANDS.map(([id, name, def]) => [name, this.hotkeyFor(id, def), id]), (td, c, i) => {
            if (i === 1) { this.keys(td, c); return true; }
            if (i === 2) {
                this.action(td, 'Run', () => this.plugin.app.commands.executeCommandById(`${p.manifest.id}:${c}`));
                return true;
            }
            return false;
        });

        // ---------------------------------------------------------------- settings
        s = this.section(main, 'settings', 'Settings at a glance');
        this.table(s, ['Group', 'What you can change'], [
            ['Dashboard', 'which note a session starts in, new-note name template, compact sources, auto-open dashboard, follow the active note, notices'],
            ['Notes', 'default note, notes folder, transcripts folder, audio archive folder'],
            ['Microphone', 'device, speech threshold + Calibrate, pause length, minimum/maximum thought length, minimum real sound, volume normalisation'],
            ['Transcription', 'Whisper model, GPU / CPU, GPU support (install location / check), GPU libraries folder, compute types, CPU threads, release Whisper after idle, beam size, language, context prompting, silence filter, allow model downloads'],
            ['Organisation', 'use the local LLM, live or batch organising (interval, batch model, batch size, “On this page” list), Ollama model and host, context size, output limit, timeout, unload after idle / on stop, models in memory, minimum words, session summary, link existing notes, note properties and tag'],
            ['Storage / Backend', 'clip bitrate, VoiceScribe folder, Python path, diagnostics'],
        ]);

        // ---------------------------------------------------------------- troubleshooting
        s = this.section(main, 'help', 'Troubleshooting');
        [
            ['Nothing is transcribed', 'Open the dashboard console and look for errors. Run checks (Quick start). Make sure the right microphone is selected and the live meter moves when you speak.'],
            ['Random sentences appear that I never said', 'Background noise is being recorded and Whisper invents text from it (often “thanks for watching”). Calibrate the microphone, or raise “Speech volume threshold” / “Minimum real sound”.'],
            ['My first words get cut off / thoughts are split too often', 'Lower the threshold slightly, or increase “Pause before a thought ends”.'],
            ['Notes are not organised, thoughts are just appended', 'Ollama is not running or the model is missing. Install Ollama and run “ollama pull llama3.2:3b” (or pick an installed model in settings).'],
            ['“Model downloads are disabled”', 'VoiceScribe stays offline. Turn on “Allow model downloads” once to fetch a Whisper model you have not used before, then turn it off again.'],
            ['Transcription is slow', 'Use the GPU (see “Faster transcription on your GPU”). On the CPU, choose a smaller Whisper model (small.en or base.en), increase CPU threads, or set beam size to 1.'],
            ['It says it is using the CPU although I have a GPU', 'Install GPU support, then press Check GPU. The test result is remembered; installing GPU support clears it so the GPU is tested again on the next start.'],
            ['No summary appeared', 'Summaries need Ollama and at least “Summary: minimum thoughts” thoughts in that note during the session. A force-stop skips the summary.'],
            ['Stopping takes a while', 'VoiceScribe finishes transcribing everything you already said. Press stop again to quit immediately.'],
            ['A point was put in the wrong place', 'Move or edit it by hand — VoiceScribe keeps your edits. Or put the cursor on it and use Focus → Correct this.'],
        ].forEach(([q, a]) => {
            const d = s.createEl('details', { cls: 'vs-guide-faq' });
            d.createEl('summary', { text: q });
            d.createEl('p', { text: a });
        });
        s.createEl('p', { cls: 'vs-muted', text: 'Privacy: audio, transcripts and notes never leave your computer. The only network traffic is to your local Ollama server, plus Whisper model downloads if you allow them.' });

        // ---------------------------------------------------------------- nav
        nav.createDiv({ cls: 'vs-guide-nav-title', text: 'On this page' });
        this.toc.forEach(([id, title]) => {
            const a = nav.createEl('a', { text: title, href: '#' });
            a.addEventListener('click', (e) => {
                e.preventDefault();
                const target = main.querySelector('#vs-guide-' + id);
                if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            });
        });
    }
}

// =============================================================================
// Where to put the GPU (CUDA) libraries
// =============================================================================
class VoiceScribeGpuInstallModal extends obsidian.Modal {
    constructor(app, plugin, onDone) {
        super(app);
        this.plugin = plugin;
        this.onDone = onDone;
        const p = plugin;
        const cudaPath = process.env.CUDA_PATH || '';
        this.options = [
            { id: 'local', title: 'This computer only (recommended)', path: p.recommendedCudaDir(), install: true,
              desc: 'A local app-data folder that is not synced or backed up to the cloud.' },
            { id: 'venv', title: 'VoiceScribe’s Python environment', path: p.venvSitePackages(), install: true, store: '',
              desc: 'Next to the other Python packages (the original default).' },
            { id: 'custom', title: 'Another folder…', path: '', install: true, custom: true,
              desc: 'For example a second drive with more free space.' },
            { id: 'existing', title: 'Use CUDA already installed on this computer', path: cudaPath ? path.join(cudaPath, 'bin') : '', install: false, custom: true,
              desc: 'No download: point to a CUDA 12 Toolkit (…\\CUDA\\v12.x\\bin) or a folder where you installed these libraries before.' },
        ];
        const current = (p.config && p.config.cuda_libs_dir) || '';
        this.choice = current ? (p.findCudaDlls(current) && !/nvidia/i.test(current) ? 'existing' : 'custom') : 'local';
        if (current) this.options.find((o) => o.id === this.choice).path = current;
    }

    onOpen() {
        this.titleEl.setText('Where should the GPU libraries go?');
        this.render();
    }

    render() {
        const p = this.plugin;
        const { contentEl } = this;
        contentEl.empty();
        contentEl.addClass('vs-start');
        contentEl.createEl('p', { cls: 'vs-muted', text: 'Whisper needs NVIDIA’s cuBLAS 12 and cuDNN 9 libraries to run on your graphics card. Downloading them takes about 1 GB, once.' });
        const list = contentEl.createDiv({ cls: 'vs-gpu-options', attr: { role: 'radiogroup', 'aria-label': 'Install location' } });
        for (const o of this.options) {
            const on = this.choice === o.id;
            const row = list.createEl('label', { cls: 'vs-gpu-option' + (on ? ' is-on' : '') });
            const radio = row.createEl('input', { type: 'radio', attr: { name: 'vs-gpu-loc' } });
            radio.checked = on;
            radio.addEventListener('change', () => { this.choice = o.id; this.render(); });
            const body = row.createDiv({ cls: 'vs-gpu-option-body' });
            body.createDiv({ cls: 'vs-gpu-option-title', text: o.title });
            body.createDiv({ cls: 'vs-muted', text: o.desc });
            if (on && o.custom) {
                const r = body.createDiv({ cls: 'vs-start-row' });
                const input = r.createEl('input', { type: 'text', attr: { placeholder: o.id === 'existing' ? 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin' : 'D:\\VoiceScribe\\cuda', spellcheck: 'false' } });
                input.value = o.path;
                input.addEventListener('input', () => { o.path = input.value.trim(); this.renderStatus(); });
                const browse = r.createEl('button', { text: 'Browse…' });
                browse.addEventListener('click', async (e) => {
                    e.preventDefault();
                    const picked = await p.chooseFolder(o.path || undefined);
                    if (picked === undefined) { new obsidian.Notice('Type or paste the folder path.'); input.focus(); return; }
                    if (picked) { o.path = picked; input.value = picked; this.renderStatus(); }
                });
            } else if (on || !o.custom) {
                body.createDiv({ cls: 'vs-gpu-path', text: o.path || '—' });
            }
        }
        this.statusEl = contentEl.createDiv({ cls: 'vs-gpu-status' });
        const foot = contentEl.createDiv({ cls: 'vs-start-foot' });
        this.goBtn = foot.createEl('button', { cls: 'mod-cta' });
        this.goBtn.addEventListener('click', () => this.go());
        const cancel = foot.createEl('button', { text: 'Cancel' });
        cancel.addEventListener('click', () => this.close());
        this.renderStatus();
    }

    renderStatus() {
        const p = this.plugin;
        const o = this.options.find((x) => x.id === this.choice);
        const el = this.statusEl;
        el.empty();
        const lines = [];
        let ok = !!o.path;
        if (!o.path) lines.push(['warn', 'Choose a folder.']);
        if (o.path && o.install) {
            const gb = p.freeSpaceGB(o.path);
            if (gb !== null) lines.push([gb < 2.5 ? 'warn' : 'ok', `${gb.toFixed(1)} GB free on that drive (about 1 GB needed)`]);
            if (gb !== null && gb < 1.2) ok = false;
            if (p.isSyncedPath(o.path)) lines.push(['warn', 'This folder is synced (OneDrive / MEGA / Dropbox or your vault): about 1 GB of libraries would be uploaded. A local folder is better.']);
            if (p.findCudaDlls(o.path)) lines.push(['ok', 'Libraries are already there; installing again updates them.']);
        }
        if (o.path && !o.install) {
            const found = p.findCudaDlls(o.path);
            lines.push([found ? 'ok' : 'warn', found ? 'Found the CUDA libraries in this folder.' : 'No cublas64_12.dll found here. It needs a CUDA 12 bin folder.']);
            ok = ok && found;
        }
        lines.forEach(([kind, text]) => el.createDiv({ cls: 'vs-gpu-line is-' + kind, text: (kind === 'ok' ? '✓ ' : '⚠ ') + text }));
        this.goBtn.setText(o.install ? 'Download and install' : 'Use this folder');
        this.goBtn.disabled = !ok;
    }

    async go() {
        const p = this.plugin;
        const o = this.options.find((x) => x.id === this.choice);
        const dir = o.id === 'venv' ? '' : o.path;
        this.close();
        let done = true;
        if (o.install) done = await p.installGpuSupport(dir);
        else {
            p.useCudaDir(dir);
            new obsidian.Notice('✅ VoiceScribe will use the CUDA libraries in\n' + dir + '\nThe GPU is tested the next time you start recording.', 10000);
        }
        if (this.onDone) this.onDone(done);
    }

    onClose() { this.contentEl.empty(); }
}

// =============================================================================
// Start-of-session note picker
// =============================================================================
class VoiceScribeNotePicker extends obsidian.FuzzySuggestModal {
    constructor(app, onPick) {
        super(app);
        this.onPick = onPick;
        this.setPlaceholder('Type to find the note to record into…');
    }
    getItems() { return this.app.vault.getMarkdownFiles(); }
    getItemText(f) { return f.path.replace(/\.md$/, ''); }
    onChooseItem(f) { this.onPick(f.path); }
}

const AUDIO_IMPORT_EXT = ['mp3', 'm4a', 'wav', 'ogg', 'opus', 'webm', 'flac', 'aac', 'mp4', 'mov', 'wma', 'amr', '3gp'];

class VoiceScribeVaultAudioPicker extends obsidian.FuzzySuggestModal {
    constructor(app, onPick) {
        super(app);
        this.onPick = onPick;
        this.setPlaceholder('Pick an audio file from the vault…');
    }
    getItems() {
        return this.app.vault.getFiles().filter((f) => AUDIO_IMPORT_EXT.includes(f.extension.toLowerCase()))
            .sort((a, b) => b.stat.mtime - a.stat.mtime);
    }
    getItemText(f) { return f.path; }
    onChooseItem(f) { this.onPick(this.app.vault.adapter.getFullPath(f.path)); }
}

class VoiceScribeImportModal extends obsidian.Modal {
    constructor(app, plugin, onPick) {
        super(app);
        this.plugin = plugin;
        this.onPick = onPick;
    }

    pick(abs) {
        this.close();
        this.onPick(abs);
    }

    onOpen() {
        const { contentEl } = this;
        contentEl.empty();
        contentEl.addClass('vs-start');
        this.titleEl.setText('Import a recording');
        contentEl.createEl('p', {
            cls: 'vs-muted',
            text: 'Voice memos, meeting audio or any recording. VoiceScribe splits it into thoughts at pauses and runs them through the same steps as live recording: voice clips, transcript, organised note, voice commands and a summary. Nothing is uploaded.',
        });
        const row = contentEl.createDiv({ cls: 'vs-start-foot' });
        const browse = row.createEl('button', { cls: 'mod-cta', text: 'Choose a file on this computer…' });
        const input = contentEl.createEl('input', { type: 'file', attr: { accept: AUDIO_IMPORT_EXT.map((e) => '.' + e).join(','), style: 'display:none' } });
        browse.addEventListener('click', () => input.click());
        input.addEventListener('change', () => {
            const f = input.files && input.files[0];
            if (!f) return;
            let abs = '';
            try { abs = require('electron').webUtils.getPathForFile(f); } catch (e) { abs = ''; }
            if (!abs && f.path) abs = f.path;
            if (!abs) { new obsidian.Notice('VoiceScribe: could not read that file path. Put the file in your vault and pick it from the vault instead.'); return; }
            this.pick(abs);
        });
        const vaultBtn = row.createEl('button', { text: 'Pick from the vault…' });
        vaultBtn.addEventListener('click', () => {
            this.close();
            new VoiceScribeVaultAudioPicker(this.app, (abs) => this.onPick(abs)).open();
        });
        contentEl.createDiv({ cls: 'vs-muted vs-start-hint', text: 'Supported: ' + AUDIO_IMPORT_EXT.join(', ') + '. The original file is not changed.' });
    }

    onClose() { this.contentEl.empty(); }
}

class VoiceScribeStartModal extends obsidian.Modal {
    constructor(app, plugin, onStart, opts) {
        super(app);
        this.plugin = plugin;
        this.onStart = onStart;
        this.opts = opts || {};
    }

    start(rel) {
        this.close();
        this.onStart(rel);
    }

    onOpen() {
        const { contentEl } = this;
        const p = this.plugin;
        contentEl.empty();
        contentEl.addClass('vs-start');
        this.titleEl.setText(this.opts.title || 'Where should this recording go?');

        // New note
        const box = contentEl.createDiv({ cls: 'vs-start-new' });
        box.createEl('label', { text: 'New note', attr: { for: 'vs-new-note-name' } });
        const row = box.createDiv({ cls: 'vs-start-row' });
        const input = row.createEl('input', { type: 'text', attr: { id: 'vs-new-note-name', spellcheck: 'false' } });
        input.value = this.opts.newName || p.formatNoteName(p.settings.newNoteTemplate);
        const go = row.createEl('button', { cls: 'mod-cta', text: this.opts.startLabel || 'Start in new note' });
        const startNew = () => this.start(p.notePathFor(input.value));
        go.addEventListener('click', startNew);
        input.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); startNew(); } });
        const folder = (p.config && p.config.notes_subfolder) || '';
        box.createDiv({ cls: 'vs-muted', text: `Created in ${folder ? folder + '/' : 'the vault root'} · say “create a new note named …” or “switch to …” at any time to change notes.` });

        // Live or batch for this session (also the saved default)
        const modeRow = contentEl.createDiv({ cls: 'vs-start-mode', attr: { role: 'radiogroup', 'aria-label': 'How to organise' } });
        modeRow.createSpan({ cls: 'vs-start-heading', text: 'Organise' });
        const curMode = (p.config && p.config.organise_mode) || 'live';
        [['live', 'Live', 'each thought right away'], ['batch', 'In batches', `every ${(p.config && p.config.batch_interval_min) || 5} min + at the end`]].forEach(([id, label, hint]) => {
            const b = modeRow.createEl('button', { cls: 'vs-mode' + (curMode === id ? ' is-on' : ''), attr: { 'aria-pressed': String(curMode === id), title: hint } });
            b.createSpan({ text: label });
            b.createSpan({ cls: 'vs-muted', text: ' · ' + hint });
            b.addEventListener('click', () => {
                p.saveConfigFile({ organise_mode: id });
                modeRow.querySelectorAll('.vs-mode').forEach((x) => { x.removeClass('is-on'); x.setAttr('aria-pressed', 'false'); });
                b.addClass('is-on'); b.setAttr('aria-pressed', 'true');
            });
        });

        // Recent notes
        const recent = p.recentNoteFiles();
        if (recent.length) {
            contentEl.createDiv({ cls: 'vs-start-heading', text: 'Continue a recent note' });
            const list = contentEl.createDiv({ cls: 'vs-start-list' });
            recent.forEach((f) => {
                const b = list.createEl('button', { cls: 'vs-start-item' });
                b.createSpan({ cls: 'vs-start-item-name', text: f.basename });
                const dir = f.parent && f.parent.path !== '/' ? f.parent.path : '';
                if (dir) b.createSpan({ cls: 'vs-muted', text: dir });
                b.addEventListener('click', () => this.start(f.path));
            });
        }

        // Anything else
        const foot = contentEl.createDiv({ cls: 'vs-start-foot' });
        const pick = foot.createEl('button', { text: 'Choose another note…' });
        pick.addEventListener('click', () => {
            this.close();
            new VoiceScribeNotePicker(this.app, (rel) => this.onStart(rel)).open();
        });
        const active = this.app.workspace.getActiveFile();
        if (active && active.extension === 'md') {
            const cur = foot.createEl('button', { text: `Use “${active.basename}”` });
            cur.addEventListener('click', () => this.start(active.path));
        }
        foot.createDiv({ cls: 'vs-muted vs-start-hint', text: 'You can skip this question in Settings → VoiceScribe → “When recording starts”.' });
        setTimeout(() => { input.focus(); input.select(); }, 30);
    }

    onClose() { this.contentEl.empty(); }
}

module.exports = VoiceScribePlugin;
