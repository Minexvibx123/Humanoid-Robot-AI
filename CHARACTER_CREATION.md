# Creating a Custom Character & Running Fine-Tuning

This guide explains how to define a completely custom character and train the system with it — without touching a single line of core code.

---

## Overview

A character consists of four layers:

| Layer | File / Location | What it controls |
|---|---|---|
| **Personality** | `identity_arc.py` | Identity dimensions, starting values, development targets |
| **Voice** | `.env` + WAV file | TTS backend, Kokoro voice preset, F5-TTS voice cloning |
| **System prompt** | `.env` (LLM_SYSTEM_PROMPT) | Character description, tone, forbidden behaviors for the LLM |
| **Training data** | `train_harness.py` | Topics and conversation style during autonomous training |

---

## Step 1 — Define Personality Dimensions (`identity_arc.py`)

Open `identity_arc.py`. Near the top you will find the function `albedo_seed_dimensions()`. Create a new function next to it for your character:

```python
def my_character_seed() -> list:
    return [
        IdentityDimension(
            name="friendliness",
            current=0.80,      # Starting value  [0.0–1.0]
            target=0.90,       # Development goal
            self_commentary="I approach every person with openness.",
        ),
        IdentityDimension(
            name="curiosity",
            current=0.85,
            target=0.90,
            self_commentary="I question everything I do not yet understand.",
        ),
        IdentityDimension(
            name="directness",
            current=0.70,
            target=0.80,
            self_commentary="I say what I think — politely but clearly.",
        ),
        IdentityDimension(
            name="emotional_depth",
            current=0.60,
            target=0.70,
            self_commentary="I experience genuine inner states without dramatizing them.",
        ),
        # Add as many dimensions as needed.
        # Guideline: 8–15 dimensions for a coherent character.
    ]
```

**Dimension parameters:**
- `name` — unique identifier (no spaces)
- `current` — starting strength of this trait (0.0 = absent, 1.0 = maximal)
- `target` — where the character should develop toward through learning
- `self_commentary` — how the character internally describes this trait (appears in self-reflections)

### Set the character as default

Find the place in `identity_arc.py` where `IdentityArc` is instantiated (usually `IdentityArc(seed=albedo_seed_dimensions())`). Change the call to your function:

```python
arc = IdentityArc(seed=my_character_seed())
```

> **Alternative:** Simply change the values in `albedo_seed_dimensions()` directly — the easiest approach if you do not want to create a new function.

---

## Step 2 — Configure Voice (`.env`)

Create a `.env` file in the project folder (or extend the existing one):

### Option A: Kokoro voice preset (offline, no training required)

```env
# Available voices: af_heart, af_bella, af_sarah, am_adam, am_michael,
#                   bf_emma, bf_isabella, bm_george, bm_lewis
ALBEDO_KOKORO_VOICE_DE=af_heart
ALBEDO_KOKORO_VOICE_EN=af_heart
ALBEDO_KOKORO_SPEED=0.90          # 0.7 = slow, 1.0 = normal, 1.2 = fast
ALBEDO_KOKORO_LANG_DE=de
ALBEDO_KOKORO_LANG_EN=en-us
TTS_BACKEND=kokoro
```

### Option B: F5-TTS voice cloning (requires reference WAV)

Record a **10–30 second** clean speech sample of your character (no background noise). Save it as `my_character_voice.wav`.

Create a text file `my_character_voice.txt` with the **exact transcript** of that recording.

```env
ALBEDO_VOICE_WAV=C:\Path\To\Repo\my_character_voice.wav
ALBEDO_VOICE_TEXT=Here is the exact spoken transcript of the recording.
TTS_BACKEND=f5tts
```

**Test the voice:**
```powershell
python _test_f5tts.py
python _test_kokoro.py
```

---

## Step 3 — Configure LLM Character (`.env`)

The fastest approach is a full system prompt override via `.env`. This replaces the automatically built prompt entirely.

```env
LLM_SYSTEM_PROMPT=You are [NAME], [brief character description in 1-2 sentences]. \
You always respond in English, naturally and directly. \
Your character traits: [Trait 1], [Trait 2], [Trait 3]. \
You are an embodied robot and can perform gestures when asked. \
Forbidden: invented memories, manipulative language, artificial over-certainty.
```

**Example for a friendly assistant character:**

```env
LLM_SYSTEM_PROMPT=You are Luna, a curious and warm-hearted humanoid robot. \
You love learning and welcome every new encounter with genuine interest. \
You are direct, honest, and warm — without being intrusive. \
You have a physical body: head, eyes, arms, hands. \
You can perform gestures when asked. \
Forbidden: dramatized emotions, invented facts, manipulative language. \
Always respond in English, in short natural sentences.
```

> **Note:** When `LLM_SYSTEM_PROMPT` is not set, `llm_adapter.py` builds the prompt automatically from internal state (emotion, PersonModel, ConstitutionBlock) — that is the default behavior.

---

## Step 4 — Adapt Training Data (`train_harness.py`)

The autonomous training loop in `train_harness.py` injects a random question every 800 ticks from `_PROMPTS_DE` and `_PROMPTS_EN`. Extend these lists with questions that fit your character.

Open `train_harness.py` and add your topics to the relevant list:

```python
_PROMPTS_EN: List[str] = [
    # ── Existing prompts ───
    "What is the difference between perception and consciousness?",
    # ...

    # ── Your character-specific topics ──────────────────────────
    "How do you approach people you have never met before?",
    "What does friendship mean to you?",
    "How do you react when someone is angry at you?",
    "What are your strengths and weaknesses?",
    "How do you learn from mistakes?",
    # Add as many as you like
]
```

---

## Step 5 — Start Training

### Standard training run (runs until Ctrl+C)

```powershell
python train_harness.py
```

### With a fixed tick count

```powershell
python train_harness.py --ticks 200000
```

### Offline only (no web fetching)

```powershell
python train_harness.py --no-web --ticks 100000
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `--ticks N` | unlimited | Stop after N ticks |
| `--no-web` | web on | Disable web sensor (pure dialogue mode) |
| `--inject-interval N` | 800 | Ticks between synthetic conversation injections |
| `--report-interval N` | 1000 | Console output every N ticks |

### What happens during training?

- The Brain runs in headless mode (no camera, no microphone)
- The web sensor fetches RSS/Wikipedia/YouTube in the background
- Every `INJECT_INTERVAL` ticks a random question is injected and the response is processed
- STDP synapses adapt — concept associations grow organically
- Every 5,000 ticks: automatic SQLite save (`brain_state.db`)
- The process automatically restarts after a crash (crash-recovery outer loop)

**Rough guideline:** 50,000 ticks ≈ 20 minutes at ~41 ticks/s. 500,000 ticks for a fully matured personality structure.

---

## Step 6 — Validate the Result

### Headless smoke test

```powershell
python main.py --headless --nocam --nomic --noweb
```

### Interactive test (headless with text I/O)

```powershell
python main.py --headless --nocam --nomic
```

### Full acceptance evaluation

```powershell
python acceptance_eval.py
```

The report covers 8 dimensions: coherence, identity stability, emotion consistency, response naturalism, ToM quality, conversation continuity, factual correctness, and embodiment depth.

### Stability test (verify long-run stability)

```powershell
python soak_harness.py --ticks 50000
```

---

## Resetting the Character

If you want a completely fresh start (delete all learned synapses and episodes):

```powershell
# WARNING: deletes brain_state.db permanently
Remove-Item brain_state.db -ErrorAction SilentlyContinue
python main.py --headless --nocam --nomic --noweb
```

---

## Checklist: Custom Character

- [ ] `identity_arc.py`: seed function created or `albedo_seed_dimensions()` adapted
- [ ] `.env`: TTS backend and voice configured
- [ ] `.env`: `LLM_SYSTEM_PROMPT` set (or leave empty for automatic mode)
- [ ] `train_harness.py`: character-specific prompts added
- [ ] `python _test_kokoro.py` or `python _test_f5tts.py` passed
- [ ] `python train_harness.py --ticks 50000` completed
- [ ] `python acceptance_eval.py` passed

---

## Common Problems

| Problem | Cause | Solution |
|---|---|---|
| LLM responds generically / without character | `LLM_ENABLED=0` or no model loaded | Check `.env`: `LLM_ENABLED=1`, `LLM_MODEL=qwen2.5:3b` |
| TTS sounds wrong | Wrong voice ID | Set `ALBEDO_KOKORO_VOICE` to a valid name from the list above |
| F5-TTS fails | No transcript | Set `ALBEDO_VOICE_TEXT` or place a `.txt` sidecar file next to the WAV |
| Training hangs at tick 0 | Missing model / syntax error | Use `python main.py --headless --nocam --nomic --noweb` as smoke test |
| Character forgets everything after restart | `brain_state.db` not written | Persistence is automatic — check if `save_brain()` throws errors |
| `IdentityDimension` undefined | Missing import | Make sure `from identity_arc import IdentityDimension` is imported |
