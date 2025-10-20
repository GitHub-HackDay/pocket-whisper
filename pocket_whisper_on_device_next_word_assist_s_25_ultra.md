# Pocket Whisper — On‑Device Next‑Word Assist (S25 Ultra)

> **Goal**: Ship an Android app for Galaxy S25 Ultra that, when **turned ON**, listens locally and suggests a **single word or very short phrase** at the *right moment* when the speaker pauses or uses a filler ("uh", "like"). Suggestions can be **shown** (small on‑screen pill) and/or **whispered** via earpiece. A **Transcript Log** records heard text, suggestions shown, selection, and latency. **All processing is on‑device** using ExecuTorch and Qualcomm QNN.

---

## 0. TL;DR for Builders
- **Why**: Help users keep fluency in real conversations and dictation, privately and instantly.
- **What**: Streaming ASR → adaptive trigger → tiny LLM suggests 1–2 words; user taps or says "use 1".
- **Hard parts**: Latency (<350 ms p95), *when to jump in*, and non‑annoying UX.
- **Stack**: Kotlin + Compose app, Foreground Audio Service, Accessibility Service overlay, ExecuTorch `.pte` models (ASR + 1–2B LLM) with **QNN** delegate on Snapdragon NPU.
- **Ship bar**: End‑to‑end pause→suggestion ≤350 ms p95, false‑positive popups <5%, top‑3 acceptance ≥40% in live tests.

---

## 1. User Stories & Use Cases
- **Email/DM dictation**: User speaks; on pause, pill shows 2–3 choices (e.g., *clarify / confirm / pass*). Tap inserts.
- **Phone/Zoom call**: Earpiece whispers the top suggestion softly; user repeats or ignores. No auto‑send.
- **Public speaking**: During planned pauses, offers connective words ("first", "moreover", "in conclusion").
- **Second‑language assistance**: Non‑native speakers get context‑correct word options without cloud.
- **Voice notes**: While recording, prompts reduce disfluencies; final note is cleaner.

**Non‑Goals (v1)**
- Not a chatbot. Not medical/mental‑health advice. Not grammar correction beyond next‑word.
- No cloud processing; no background uploading.

---

## 2. UX Overview
### 2.1 Main Screens
1) **Home**
   - Big **Listening ON/OFF** toggle (primary control).  
   - Latency HUD (ASR ms, LLM ms, total).
   - Status: Microphone, Overlay, Accessibility permissions.
2) **Transcript Log**
   - Table: `[timestamp] [partial text] [suggestions shown] [chosen] [latency_ms]`.
   - Row detail: audio hash, trigger reason, thresholds.
3) **Settings**
   - Output mode: **Visual pill** / **Earpiece whisper** / Both.
   - Sensitivity (slider) → affects trigger thresholds.
   - Blocked words list; profanity filter ON by default.
   - Opt‑in: "**Learn my phrases**" (on‑device cache).

### 2.2 Overlay (Visual Mode)
- Small floating **pill** with up to **3 chips** (top suggestions).
- **Tap** to insert into focused text field (Accessibility Service).  
- **Long‑press** pill → snooze 5 minutes.

### 2.3 Audio Whisper Mode
- Route TTS to **earpiece (in‑call audio)** at low volume; fallback to speaker if no earpiece.
- Whisper only the **top‑1** suggestion.

### 2.4 Behavior Principles
- **Never auto‑insert** or auto‑send.  
- **Cancel** popup if speech resumes within ~200 ms (no interruption).
- Adapt timing to **user’s current speaking rate**.

---

## 3. Latency SLOs & Budgets
- **SLO**: Pause/Filler → first suggestion **≤ 350 ms p95** (target 250–330 ms typical).
- Budget (typical): VAD 10 ms → Filler TCN 5–10 ms → ASR partial 80–120 ms → LLM 100–180 ms → UI/TTS 20–30 ms.
- Warm‑start: Maintain **rolling ASR context** and **LLM KV‑cache** (when supported) for last ~12–20 tokens.

---

## 4. Trigger Policy (When to Jump In)
We compute a **dynamic jump threshold** and only surface suggestions when appropriate.

**Signals**
- `SR` (speaking rate): syllables/sec from ASR token times + energy peaks; keep EMA `SR_ema`.
- `EP` (expected pause): `EP = clamp(0.15 + 0.35*(1/SR_ema), 0.18, 0.65)` seconds.  
  Slower talker → larger natural pause.
- `F` (filler likelihood): 0–1 from a tiny 1D‑TCN over last ~500 ms (classes: filler vs fluent).
- `C` (completion confidence): 0–1 from LLM entropy on next token(s) + POS prior.
- `S` (ASR stability): last token stable flag (CTC/streaming heuristics).

**Fire condition**
```
(pause_duration ≥ EP * k1) AND
(F ≥ τ_filler  OR  pause_duration ≥ EP * k2) AND
(C ≥ τ_conf) AND
S == stable
```
Typical values: `k1=0.8`, `k2=1.15`, `τ_filler=0.6`, `τ_conf=0.55`.

**Back‑off**: If speech resumes ≤200 ms after popup, retract without action; raise suppression window briefly to avoid thrash.

---

## 5. System Architecture

```
+-------------------+     +----------------+     +----------------------+     +-----------------+
| Foreground Audio  | --> | VAD + Filler   | --> | Streaming ASR       | --> |  Suggest Engine |
| Service (16k PCM) |     | Detector (TCN) |     | (Whisper-tiny/Dist.)|     |  (LLM + Rerank) |
+-------------------+     +----------------+     +----------------------+     +-----------------+
         |                                                         |                 |
         |                                                         |                 v
         v                                                         v         +------------------+
  Transcript Logger  <----------------------------------------- Partial text  | Output Router   |
         |                                                                      | (Pill / TTS)  |
         +--------------------------------------------------------------------->+----------------+
```

### 5.1 Android Modules
- **App (Kotlin + Compose)**: UI screens, toggles, settings, logs.
- **Foreground Audio Service**: mic capture, buffering, VAD/TCN, ASR session, trigger policy.
- **Accessibility Service**: cross‑app text insertion, overlay draw.
- **ML Runtime**: ExecuTorch wrappers, model IO, pre/post‑processing.
- **Data Layer**: Room DB for transcript log; DataStore for settings.

### 5.2 Models
- **VAD**: classic RNNoise or tiny TCN (100–300k params), 16 kHz frames.
- **Filler Detector**: 1D‑TCN over mel or PCM, window ≈ 0.5–1.0 s.
- **ASR (Streaming)**: Whisper‑tiny/Distil or Qualcomm on‑device model — exported to `.pte`, quantized; partials every 200–300 ms.
- **LLM (Next‑word)**: ~1–1.5B params (Qwen‑1.5B/Phi‑2 distilled) → **int8**; QNN delegate on S25U NPU. Output capped at **1–2 tokens**.
- **Reranker**: POS validity + local bigram cache from accept history.

### 5.3 Execution Backends
- ExecuTorch runtime with **QNN delegate** for LLM; ASR can use CPU/XNNPACK or QNN if available.

---

## 6. Data Contracts & Storage
### 6.1 Transcript Log (Room)
```sql
Table transcript_event (
  id INTEGER PK AUTOINCREMENT,
  ts_epoch_ms INTEGER,
  text_before TEXT,            -- last N words
  suggestions TEXT,            -- JSON: ["pass","send","give"]
  suggestion_top TEXT,         -- top-1
  chosen TEXT NULL,
  latency_ms INTEGER,
  trigger_reason TEXT,         -- e.g., filler|long_pause|both
  sr_ema REAL,
  ep_ms INTEGER,
  asr_ms INTEGER,
  llm_ms INTEGER,
  mode TEXT,                   -- visual|ear
  audio_hash TEXT              -- rolling hash for debugging
)
```

### 6.2 Settings (DataStore)
```json
{
  "listening": true,
  "output_mode": "visual|ear|both",
  "sensitivity": 0.5,
  "blocked_words": ["..."],
  "learn_phrases": true
}
```

---

## 7. Model Export → `.pte` (ExecuTorch)
### 7.1 Python Env
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch executorch transformers torchaudio
```

### 7.2 Export Skeleton (Pseudo‑Py)
```python
import torch
from executorch import export

model = load_model(...)      # ASR or 1–1.5B LLM
model.eval()

sample_inputs = get_sample_inputs()
program = torch.export.export(model, sample_inputs)

pte = export.to_executorch(program, backend="qnn", quantization="int8")
with open("app/src/main/assets/nextword_llm_qnn_int8.pte", "wb") as f:
    f.write(pte)
```
- For ASR, start with CPU/XNNPACK backend; switch to QNN if supported.
- Keep prompts ≤ 20 tokens; generate ≤ 2 tokens.

### 7.3 Asset Layout
```
app/src/main/assets/
  asr_streaming_int8.pte
  nextword_llm_qnn_int8.pte
```

---

## 8. Android Integration Notes
### 8.1 Permissions (Manifest)
```xml
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_MICROPHONE"/>
<uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW"/>
```

### 8.2 Services (Manifest)
```xml
<service
  android:name=".audio.ListenForegroundService"
  android:foregroundServiceType="microphone"
  android:exported="false"/>

<service
  android:name=".accessibility.InsertTextAccessibilityService"
  android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE"
  android:exported="false">
  <intent-filter>
    <action android:name="android.accessibilityservice.AccessibilityService"/>
  </intent-filter>
</service>
```

### 8.3 ExecuTorch Runner (Pseudo‑Kotlin)
```kotlin
class EtSession(context: Context, assetName: String) {
  private val runtime = Runtime()
  private val module: Module by lazy {
    context.assets.open(assetName).use { runtime.load(it.readBytes()) }
  }
  fun run(io: Map<String, Tensor>): Map<String, Tensor> {
    val s = module.createSession()
    io.forEach { (k, v) -> s.setInput(k, v) }
    s.run()
    return s.getOutputs()
  }
}
```

---

## 9. Output Routing
- **Visual**: Compose overlay → chips for top‑3 → `AccessibilityService` inserts selected text at cursor.
- **Audio**: Top‑1 suggestion via on‑device TTS to **earpiece** (`AudioManager.MODE_IN_COMMUNICATION`), gain‑limited. Respect Do‑Not‑Disturb.

---

## 10. Privacy, Safety, and Controls
- **Local‑only**: No audio leaves device; airplane mode demo.
- **User control**: Big ON/OFF toggle; persistent foreground notification when ON; quick snooze.
- **Safety filter**: Block profanity by default; user‑editable blocked list.
- **Transparency**: Mic indicator; transcript log can be cleared; no background uploads.

---

## 11. Metrics & Telemetry (Local‑Only)
- Latency components: `asr_ms`, `llm_ms`, `total_ms`.
- Popup false‑positive rate (popup retracted without action).
- Top‑1 and Top‑3 acceptance rates.
- Session duration; battery impact estimate (coarse).

---

## 12. Bench & Validation Plan
- **Latency**: 200 scripted utterances (fast vs slow speech); report p50/p95.
- **Trigger quality**: % of popups that occur during true gaps.
- **A/B thresholds**: k1/k2/τ grid search; pick best acceptance/false‑positive trade‑off.
- **Battery burn**: 15‑min continuous session; record power draw with simple estimator.

---

## 13. Demo Script (3 Minutes)
1) Airplane mode → open app → **Listening ON**.
2) Gmail: "Could you please … (pause)…" → pill: *send / pass / give* → tap one.
3) Faster speech → earlier prompts; slower speech → later prompts.
4) Toggle **Audio whisper**; repeat; earpiece speaks *"pass"* softly.
5) Open **Transcript Log**: show line with ~280 ms latency.

---

## 14. Risks & Mitigations
- **Latency spikes** → keep LLM to 1–2 tokens; pre‑warm cache; QNN delegate.
- **Annoying popups** → adaptive EP + suppression window; quick snooze.
- **ASR errors** → rely on filler signal + entropy; only fire when ASR stable.
- **Permissions friction** → in‑app checklist with deep‑links to grant mic/overlay/accessibility.

---

## 15. Roadmap (Post‑MVP)
- Multilingual packs (ES/FR/HI) with per‑language ASR.
- Domain phrasebanks (work, school, sales) — on‑device only.
- Context personalization (opt‑in) from local docs/messages.
- Better prosody model to detect stress and slow down prompts (non‑medical).

---

## 16. File Tree (Proposed)
```
app/
  src/main/
    assets/
      asr_streaming_int8.pte
      nextword_llm_qnn_int8.pte
    java/
      com/pw/
        audio/ (Vad.kt, FillerTcn.kt, ListenForegroundService.kt)
        ml/    (EtSession.kt, AsrSession.kt, LlmSession.kt, Rerank.kt)
        ui/    (MainScreen.kt, TranscriptLog.kt, Overlay.kt, Settings.kt)
        acc/   (InsertTextAccessibilityService.kt)
        data/  (TranscriptDao.kt, Entities.kt, AppDb.kt, SettingsStore.kt)
    AndroidManifest.xml
  build.gradle
export/
  export_asr.py
  export_llm.py
README.md
LICENSE
```

---

## 17. Acceptance Criteria (Ship Bar)
- p95 pause→suggestion ≤ **350 ms**.
- Popup false‑positive rate **< 5%**.
- Top‑3 acceptance rate **≥ 40%** in live trials.
- ON/OFF toggle works; Transcript Log captures all events; all processing offline.

---

## 18. Glossary
- **ASR**: Automatic Speech Recognition.
- **VAD**: Voice Activity Detection.
- **LLM**: Small language model for next‑word prediction.
- **QNN**: Qualcomm’s NPU delegate.
- **ExecuTorch**: PyTorch runtime for on‑device inference (`.pte`).

---

*This spec is designed so a capable LLM or engineer can implement the MVP end‑to‑end on the Galaxy S25 Ultra without external dependencies beyond the Android SDK and ExecuTorch.*

