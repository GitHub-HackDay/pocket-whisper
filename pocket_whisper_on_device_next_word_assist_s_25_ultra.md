# Pocket Whisper — On‑Device Next‑Word Assist (S25 Ultra)

> **Goal**: Ship an Android app for Galaxy S25 Ultra that, when **turned ON**, listens locally and **automatically speaks and inserts** a **single word or very short phrase** at the *right moment* when the speaker pauses or uses a filler ("uh", "like"). Suggestions are **spoken** via phone speaker/earpiece and **auto-inserted** into active text fields. A **Transcript Log** records heard text, suggestions auto-inserted, user corrections, and latency. **All processing is on‑device** using ExecuTorch and Qualcomm QNN. **No visual UI for suggestions** — intelligence is in the trigger policy.

---

## 0. TL;DR for Builders
- **Why**: Help users keep fluency in real conversations and dictation, privately and instantly.
- **What**: Streaming ASR → adaptive trigger → tiny LLM suggests 1–2 words; **auto-speaks via audio + auto-inserts**.
- **Hard parts**: Latency (<350 ms p95), **extremely accurate trigger policy** (false inserts are costly), semantic coherence validation.
- **Stack**: Kotlin + Compose app, Foreground Audio Service, Accessibility Service (auto-insert), ExecuTorch `.pte` models (ASR + 1–2B LLM) with **QNN** delegate on Snapdragon NPU.
- **Ship bar**: End‑to‑end pause→audio output ≤350 ms p95, **unwanted auto-inserts <2%**, correction rate <10% in live tests.

---

## 1. User Stories & Use Cases
- **Email/DM dictation**: User speaks; on pause, app speaks suggestion and auto-inserts (e.g., *"confirm"*). User continues naturally or backspaces if wrong.
- **Phone/Zoom call**: Speaker outputs suggestion softly; user can repeat naturally or ignore.
- **Public speaking**: During planned pauses, speaks connective words ("first", "moreover", "in conclusion").
- **Second‑language assistance**: Non‑native speakers get context‑correct word auto-inserted without cloud.
- **Voice notes**: While recording, intelligently fills pauses; final note is cleaner.

**Non‑Goals (v1)**
- Not a chatbot. Not medical/mental‑health advice. Not grammar correction beyond next‑word.
- No cloud processing; no background uploading.
- **No visual suggestion UI** — audio output and auto-insert only.

---

## 2. UX Overview
### 2.1 Main Screens
1) **Home**
   - Big **Listening ON/OFF** toggle (primary control).  
   - Latency HUD (ASR ms, LLM ms, total ms).
   - Real-time stats: suggestions made, corrections detected, correction rate %.
   - Status: Microphone, Accessibility permissions.
2) **Transcript Log**
   - Table: `[timestamp] [text_before] [suggestion_inserted] [user_corrected] [latency_ms]`.
   - Row detail: audio hash, trigger reason, confidence scores, thresholds.
3) **Settings**
   - Output mode: **Speaker** / **Earpiece** / **Silent** (insert-only).
   - Sensitivity (slider) → affects trigger confidence thresholds.
   - Blocked words list; profanity filter ON by default.
   - Opt‑in: "**Learn my phrases**" (on‑device cache).
   - Emergency snooze: double-tap power button or shake phone.

### 2.2 Audio Output & Auto-Insert Behavior
- **Audio first**: Speak suggestion via phone speaker/earpiece at moderate volume.
- **Then insert**: Auto-insert into active text field via Accessibility Service (~100ms after audio starts).
- **Top-1 only**: No multiple choices — high confidence or nothing.
- **Correction detection**: Monitor next 3-5 seconds for backspace/deletion; log outcome.

### 2.3 Behavior Principles
- **Auto-insert with intelligence**: Only fire when confidence is very high (≥0.75).  
- **Cancel if speech resumes**: If user continues speaking within ~200 ms of pause, suppress trigger.
- **Semantic coherence check**: Verify suggestion fits grammatically and semantically.
- **Repetition suppression**: Don't suggest same word repeatedly or words just spoken.
- Adapt timing to **user's current speaking rate**.

---

## 3. Latency SLOs & Budgets
- **SLO**: Pause/Filler → first suggestion **≤ 350 ms p95** (target 250–330 ms typical).
- Budget (typical): VAD 10 ms → Filler TCN 5–10 ms → ASR partial 80–120 ms → LLM 100–180 ms → UI/TTS 20–30 ms.
- Warm‑start: Maintain **rolling ASR context** and **LLM KV‑cache** (when supported) for last ~12–20 tokens.

---

## 4. Trigger Policy (When to Jump In) — **CRITICAL**
We compute a **dynamic jump threshold** and only auto-insert suggestions when **very confident**. Since we're auto-inserting, false positives are costly.

**Signals**
- `SR` (speaking rate): syllables/sec from ASR token times + energy peaks; keep EMA `SR_ema`.
- `EP` (expected pause): `EP = clamp(0.15 + 0.35*(1/SR_ema), 0.18, 0.65)` seconds.  
  Slower talker → larger natural pause.
- `F` (filler likelihood): 0–1 from a tiny 1D‑TCN over last ~500 ms (classes: filler vs fluent).
- `C` (completion confidence): 0–1 from LLM entropy on next token(s) + POS prior.
- `S` (ASR stability): last token stable flag (CTC/streaming heuristics).
- **NEW** `SC` (semantic coherence): 0–1 from simple embedding similarity or grammar check; does suggestion fit context?
- **NEW** `RH` (recent history): has this word been suggested in last 10 seconds or spoken in last 5 words?

**Fire condition (STRICTER)**
```
(pause_duration ≥ EP * 1.0) AND                    // k1=1.0 (was 0.8)
(F ≥ 0.7  OR  pause_duration ≥ EP * 1.3) AND      // τ_filler=0.7 (was 0.6), k2=1.3 (was 1.15)
(C ≥ 0.75) AND                                      // τ_conf=0.75 (was 0.55) — MUCH HIGHER
(SC ≥ 0.8) AND                                      // NEW: semantic coherence check
(RH == false) AND                                   // NEW: not recently suggested/spoken
S == stable
```
Typical values: `k1=1.0`, `k2=1.3`, `τ_filler=0.7`, `τ_conf=0.75`, `τ_sc=0.8`.

**Back‑off**: If speech resumes ≤200 ms after trigger decision, abort audio/insert; raise suppression window for 2–3 seconds.

**Dynamic adjustment**: If correction rate >10% over last 20 insertions, increase `τ_conf` by 0.05. If correction rate <3%, decrease by 0.02 (min 0.70).

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
  Transcript Logger  <----------------------------------------- Partial text  | TTS + Auto-Insert|
         |                                                                      | (Audio + A11y) |
         +--------------------------------------------------------------------->+------------------+
                                                                                        |
                                                                                        v
                                                                              +-------------------+
                                                                              | Correction Detect |
                                                                              | (feedback loop)   |
                                                                              +-------------------+
```

### 5.1 Android Modules
- **App (Kotlin + Compose)**: UI screens (Home, Transcript Log, Settings), toggles, real-time stats.
- **Foreground Audio Service**: mic capture, buffering, VAD/TCN, ASR session, **enhanced trigger policy**.
- **Accessibility Service**: cross‑app text **auto-insertion**, correction detection (monitor backspace events).
- **ML Runtime**: ExecuTorch wrappers, model IO, pre/post‑processing.
- **Data Layer**: Room DB for transcript log (includes correction outcomes); DataStore for settings.

### 5.2 Models
- **VAD**: classic RNNoise or tiny TCN (100–300k params), 16 kHz frames.
- **Filler Detector**: 1D‑TCN over mel or PCM, window ≈ 0.5–1.0 s.
- **ASR (Streaming)**: Whisper‑tiny/Distil or Qualcomm on‑device model — exported to `.pte`, quantized; partials every 200–300 ms.
- **LLM (Next‑word)**: ~1–1.5B params (Qwen‑1.5B/Phi‑2 distilled) → **int8**; QNN delegate on S25U NPU. Output capped at **1 token** (single word).
- **Reranker & Coherence**: POS validity + local bigram cache from accept history + simple semantic coherence (embedding cosine sim or lightweight grammar check).

### 5.3 Execution Backends
- ExecuTorch runtime with **QNN delegate** for LLM; ASR can use CPU/XNNPACK or QNN if available.

---

## 6. Data Contracts & Storage
### 6.1 Transcript Log (Room)
```sql
Table transcript_event (
  id INTEGER PK AUTOINCREMENT,
  ts_epoch_ms INTEGER,
  text_before TEXT,            -- last N words before suggestion
  suggestion_inserted TEXT,    -- the word/phrase auto-inserted
  user_corrected BOOLEAN,      -- did user backspace/delete within 5s?
  correction_action TEXT,      -- "none"|"deleted"|"replaced"|"continued"
  latency_ms INTEGER,
  trigger_reason TEXT,         -- e.g., filler|long_pause|both
  confidence_scores TEXT,      -- JSON: {C: 0.82, SC: 0.85, F: 0.73}
  sr_ema REAL,
  ep_ms INTEGER,
  asr_ms INTEGER,
  llm_ms INTEGER,
  tts_ms INTEGER,
  mode TEXT,                   -- speaker|earpiece|silent
  audio_hash TEXT              -- rolling hash for debugging
)
```

### 6.2 Settings (DataStore)
```json
{
  "listening": true,
  "output_mode": "speaker|earpiece|silent",
  "sensitivity": 0.5,
  "confidence_threshold": 0.75,
  "blocked_words": ["..."],
  "learn_phrases": true,
  "emergency_snooze_enabled": true
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
<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS"/>
```

### 8.2 Services (Manifest)
```xml
<service
  android:name=".audio.ListenForegroundService"
  android:foregroundServiceType="microphone"
  android:exported="false"/>

<service
  android:name=".accessibility.AutoInsertAccessibilityService"
  android:permission="android.permission.BIND_ACCESSIBILITY_SERVICE"
  android:exported="false">
  <intent-filter>
    <action android:name="android.accessibilityservice.AccessibilityService"/>
  </intent-filter>
  <meta-data
    android:name="android.accessibilityservice"
    android:resource="@xml/accessibility_service_config"/>
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
- **Audio**: Top‑1 suggestion via on‑device TTS to **speaker** or **earpiece** (`AudioManager.MODE_IN_COMMUNICATION` for earpiece), moderate volume. Respect Do‑Not‑Disturb.
- **Auto-Insert**: Immediately after TTS starts (~100ms delay), `AccessibilityService` inserts text at cursor in active text field.
- **Correction Detection**: Monitor accessibility events for backspace/delete actions within 3–5 seconds post-insert; log outcome.

---

## 10. Privacy, Safety, and Controls
- **Local‑only**: No audio leaves device; airplane mode demo.
- **User control**: Big ON/OFF toggle; persistent foreground notification when ON; emergency snooze (double-tap power or shake).
- **Safety filter**: Block profanity by default; user‑editable blocked list.
- **Auto-insert safety**: High confidence thresholds (≥0.75); semantic coherence checks; dynamic adjustment based on correction rate.
- **Transparency**: Mic indicator; transcript log with correction tracking; log can be cleared; no background uploads.

---

## 11. Metrics & Telemetry (Local‑Only)
- Latency components: `asr_ms`, `llm_ms`, `tts_ms`, `total_ms`.
- Auto-insert false‑positive rate (user corrected/deleted within 5s).
- Correction rate: % of insertions that were deleted, replaced, or triggered backspace.
- Acceptance rate: % of insertions followed by natural continuation (inferred positive).
- Session duration; battery impact estimate (coarse).

---

## 12. Bench & Validation Plan
- **Latency**: 200 scripted utterances (fast vs slow speech); report p50/p95 total time from pause to audio output.
- **Trigger quality**: % of auto-inserts that are NOT corrected/deleted (target >90%).
- **A/B thresholds**: k1/k2/τ_conf/τ_sc grid search; optimize correction rate vs suggestion frequency trade‑off.
- **Correction detection accuracy**: % of corrections properly detected by accessibility service.
- **Battery burn**: 15‑min continuous session; record power draw with simple estimator.

---

## 13. Demo Script (3 Minutes)
1) Airplane mode → open app → **Listening ON**.
2) Gmail: "Could you please … (pause)…" → app speaks *"confirm"* via speaker + auto-inserts; user continues typing naturally.
3) Faster speech → suggestions adapt to shorter pauses; slower speech → waits longer.
4) Toggle to **Earpiece mode**; repeat; earpiece speaks suggestion softly + auto-inserts.
5) Intentionally say gibberish/pause awkwardly → no suggestion (confidence too low).
6) Open **Transcript Log**: show entries with ~280 ms latency, correction status, confidence scores.

---

## 14. Risks & Mitigations
- **Latency spikes** → keep LLM to 1 token; pre‑warm cache; QNN delegate.
- **Bad auto-inserts** → **very high confidence thresholds** (≥0.75); semantic coherence checks; dynamic adjustment from correction feedback.
- **ASR errors** → rely on filler signal + entropy + stability flag; only fire when ASR stable.
- **User annoyance from wrong suggestions** → correction detection + adaptive thresholds; emergency snooze (shake/double-tap power).
- **Permissions friction** → in‑app checklist with deep‑links to grant mic/accessibility.

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
        audio/   (Vad.kt, FillerTcn.kt, ListenForegroundService.kt, TtsOutput.kt)
        ml/      (EtSession.kt, AsrSession.kt, LlmSession.kt, Rerank.kt, CoherenceCheck.kt)
        ui/      (MainScreen.kt, TranscriptLog.kt, Settings.kt)
        acc/     (AutoInsertAccessibilityService.kt, CorrectionDetector.kt)
        data/    (TranscriptDao.kt, Entities.kt, AppDb.kt, SettingsStore.kt)
        trigger/ (TriggerPolicy.kt, SpeakingRateTracker.kt)
    res/
      xml/     (accessibility_service_config.xml)
    AndroidManifest.xml
  build.gradle
export/
  export_asr.py
  export_llm.py
  export_coherence_check.py
README.md
LICENSE
```

---

## 17. Acceptance Criteria (Ship Bar)
- p95 pause→audio output ≤ **350 ms**.
- Unwanted auto-insert rate **< 2%** (user corrects/deletes).
- Correction rate **< 10%** in live trials.
- Natural continuation rate **≥ 88%** (insertions not corrected within 5s).
- ON/OFF toggle works; Transcript Log captures all events with correction status; all processing offline.
- Emergency snooze functional (shake or double-tap power).

---

## 18. Glossary
- **ASR**: Automatic Speech Recognition.
- **VAD**: Voice Activity Detection.
- **LLM**: Small language model for next‑word prediction.
- **QNN**: Qualcomm’s NPU delegate.
- **ExecuTorch**: PyTorch runtime for on‑device inference (`.pte`).

---

*This spec is designed so a capable LLM or engineer can implement the MVP end‑to‑end on the Galaxy S25 Ultra without external dependencies beyond the Android SDK and ExecuTorch.*

