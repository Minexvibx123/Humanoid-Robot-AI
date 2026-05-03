> [🇩🇪 Deutsch](README.md) | 🇬🇧 English

# Humanoid Robot AI — Project Overview

This repository is a tick-driven humanoid/social-robot system with a cognitive core, world model, dialogue planning, embodied motor cues, and multiple run modes — not just a neural-consciousness prototype.

Current focus areas:
- Central brain orchestration in `brain.py`
- Social and embodied interaction via `social_manager.py`, `dialogue_manager.py`, and `task_executive.py`
- Offline-capable speech output via F5-TTS, Kokoro-ONNX, and pyttsx3
- Validated test paths via `eval_harness.py`, `soak_harness.py`, `post_fix_harness.py`, and `acceptance_eval.py`
- A 20-module human-presence layer in `human_interaction_suite.py`

Run modes are dispatched via `main.py`: GUI, Headless, Eval, Soak, and Postfix validation.

## Setup

### 1. Prerequisites

- Windows 10/11 is currently the primary supported environment
- Python 3.11 to 3.14 with working `venv`
- Git
- For audio/TTS: a working output device and installed Windows voice components for `pyttsx3`
- Optional for F5-TTS on Windows: a compatible FFmpeg shared build if `torchcodec` is used

### 2. Clone the repository

```powershell
git clone <REPO-URL>
cd <REPO-FOLDER>
```

### 3. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Optional `.env` configuration

The project loads environment variables from `.env` when `python-dotenv` is available.

Example:

```env
TTS_BACKEND=
ALBEDO_VOICE_WAV=C:\Path\To\Repo\albedo_voice.wav
ALBEDO_VOICE_TEXT=Put the exact transcript of the reference clip here.
ALBEDO_KOKORO_VOICE=af_heart
ALBEDO_KOKORO_VOICE_DE=af_heart
ALBEDO_KOKORO_VOICE_EN=af_heart
ALBEDO_KOKORO_SPEED=0.9
ALBEDO_KOKORO_LANG=de
ALBEDO_KOKORO_LANG_DE=de
ALBEDO_KOKORO_LANG_EN=en-us

LLM_ENABLED=1
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:3b
LLM_API_KEY=local
```

Notes:

- Empty `TTS_BACKEND` means automatic priority: `f5tts` → `kokoro` → `pyttsx3`
- F5-TTS requires both `ALBEDO_VOICE_WAV` and a matching transcript (`ALBEDO_VOICE_TEXT` or `<wav>.txt`)
- If you do not want an LLM backend, keep `LLM_ENABLED=0`
- `ALBEDO_KOKORO_VOICE_DE/EN` and `ALBEDO_KOKORO_LANG_DE/EN` let you use different voices for German and English
- For a small local default setup, `qwen2.5:3b` via Ollama is currently the fastest tested option

### 6. First safe startup

For a first smoke test without camera, microphone, or web:

```powershell
python main.py --headless --nocam --nomic --noweb
```

If that is stable, re-enable sensors step by step.

### 7. Main run modes

```powershell
python main.py
python main.py --headless
python main.py --eval
python main.py --soak
python train_harness.py
```

### 8. TTS setup

#### F5-TTS

- Installed via `requirements.txt`
- Requires a reference WAV plus transcript
- First run downloads model weights from Hugging Face
- On Windows, `torchcodec`/FFmpeg can fail; this project now includes a `soundfile` fallback in the runtime path

#### Kokoro-ONNX

- Offline preset voice without voice cloning
- Downloads `kokoro-v1.0.onnx` and `voices-v1.0.bin` into `models/kokoro/` on first run

#### pyttsx3

- Final fallback backend
- Uses locally installed Windows SAPI voices

### 8a. Language switching for GUI and TTS

- `python main.py --lang de` starts GUI, dialogue, and TTS in German
- `python main.py --lang en` starts GUI, dialogue, and TTS in English
- The same applies to headless mode and harnesses that enter through `main.py`
- At runtime, Kokoro and `pyttsx3` voices are switched based on the selected language

### 8b. Local Ollama setup

1. Install and start Ollama
2. Pull the model: `ollama pull qwen2.5:3b`
3. Set `.env` as above: `LLM_ENABLED=1`, `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen2.5:3b`
4. Restart `main.py` so the import-time configuration is reloaded

Quick test:

```powershell
ollama run qwen2.5:3b "Reply only with: OK local active"
```

If that responds correctly, the project can use the same local endpoint.

### 8c. LLM character and project values

- `llm_adapter.py` now includes a constitutional policy layer with core values, identity rules, and forbidden behaviors
- These values feed both the system prompt and the hard response validator
- Responses containing manipulative attachment language, hostile escalation, or unjustified certainty are actively rejected
- Project and identity guidelines from autobiography state are also injected into the LLM context

### 9. Common issues

#### TTS does not speak

- Start with: `python main.py --headless --nocam --nomic --noweb`
- Check the startup line `[TTS] ...`
- Verify that `ALBEDO_VOICE_WAV` and `ALBEDO_VOICE_TEXT` are actually set
- If F5-TTS fails, the runtime should fall back to `kokoro` or `pyttsx3`

#### GUI does not start

- Check `dearpygui`
- Test in headless mode first

#### Camera/microphone issues

- Isolate with `--nocam` and/or `--nomic`
- Only enable hardware after the headless baseline is stable

## Packages Used

The following Python packages are currently used according to `requirements.txt`.

### Numeric, signal processing, core

- `numpy` — numerical arrays and base operations
- `numba` — JIT acceleration for hot numeric paths
- `scipy` — signal processing and numerical helpers
- `networkx` — graph structures for concepts and relationships

### Sensors, audio, speech

- `opencv-python` — camera frames and image processing
- `mediapipe` — face, hand, and pose landmarks
- `SpeechRecognition` — microphone/STT input path
- `sounddevice` — audio playback
- `soundfile` — reading and writing WAV/audio files
- `pydub` — audio conversion and helper processing

### TTS

- `pyttsx3` — local Windows SAPI fallback
- `kokoro-onnx` — offline ONNX-based TTS
- `huggingface_hub` — model download and caching
- `f5-tts` — zero-shot voice cloning / high-quality offline TTS

### Web, retrieval, online knowledge

- `requests` — HTTP access
- `feedparser` — RSS/Atom parsing
- `yt-dlp` — metadata/content access for video sources
- `youtube-transcript-api` — YouTube transcript retrieval

### Robotics, UI, interfaces

- `pyserial` — serial communication with Arduino/robot hardware
- `dearpygui` — GUI/monitoring dashboard

### LLM, config, external services

- `openai` — OpenAI-compatible API client, also usable with local endpoints
- `python-dotenv` — `.env` loading

## Package List from `requirements.txt`

```text
numpy>=1.26.0
numba>=0.58.0
scipy>=1.12.0
opencv-python>=4.9.0
mediapipe>=0.10.0
SpeechRecognition>=3.10.0
sounddevice>=0.4.6
soundfile>=0.12.0
pyttsx3>=2.90
kokoro-onnx>=0.4.0
huggingface_hub>=0.20.0
f5-tts>=0.1.0
pydub>=0.25.0
yt-dlp>=2024.1.0
networkx>=3.2.0
requests>=2.31.0
feedparser>=6.0.0
youtube-transcript-api>=0.6.0
pyserial>=3.5
dearpygui>=1.11.1
openai>=1.30.0
python-dotenv>=1.0.1
```

## Project Structure

```
<REPO-FOLDER>/
├── main.py               ← Entry point for GUI, Headless, Eval, Soak, Postfix modes
├── brain.py              ← Central orchestrator, simulation clock, plasticity
├── consciousness.py      ← Consciousness system (20+ subsystems, see below)
├── dialogue_manager.py   ← UtterancePlan, turn-taking dialogue planning
├── speech_output.py      ← TTS pipeline: F5-TTS → Kokoro-ONNX → pyttsx3
├── llm_adapter.py        ← LLM context assembly, validation, prompting
├── emotion.py            ← 8D emotion engine with RPE, `distribution()`, `top3()`, `EmotionalTrajectoryTracker`
├── neuron.py             ← Leaky integrate-and-fire neuron
├── synapse.py            ← STDP synapse with Hebbian plasticity
├── regions.py            ← 11 brain regions with internal connections
├── sensors.py            ← Camera (vision), microphone (audio), speech
├── web_sensor.py         ← Autonomous web crawler → neural encoding
├── actions.py            ← Action toolbox (PC control, web search)
├── persistence.py        ← SQLite storage for all synapses + memory
│
├── ── Robotics Stack ────────────────────────────────────────────────────
├── body_schema.py        ← Kinematic skeleton model (InMoov) + proprioception
├── telemetry_bus.py      ← Sensor/actuator event bus (publish/subscribe)
├── safety_supervisor.py  ← Safety layer (E-stop, collision, danger zone)
├── world_state.py        ← World model (persons, objects, space, predicates)
├── skill_library.py      ← Motor skill library (approach, wave, …)
├── task_executive.py     ← Goal decomposition + skill sequencing
├── social_manager.py     ← Conversation management, turn-taking, person modelling
├── sim_bridge.py         ← Simulation/real-time bridge (SimulatedBody / RealBody)
├── robot_controller.py   ← Servo commands; breathing rhythm (sine ±0.5°), `intent_move_to()` 3-phase sequence, `_GESTURE_LIBRARY` (317 poses), `_apply_gesture()`
├── robot_serial.py       ← Arduino serial communication layer
│
├── ── Cognition Modules ─────────────────────────────────────────────────
├── causal_graph.py       ← Causal world model (transitions, best_action_for_goal)
├── value_learning.py     ← TD value learning (ValueModel, build_state_signature)
├── identity_arc.py       ← Identity coherence monitor; `record_event()` for event-driven identity drift
├── narrative.py          ← Event narrative thread (scene structure)
├── theory_of_mind.py     ← Theory of Mind (intent/strategy inference)
├── belief_quarantine.py  ← Isolation of contradictory beliefs
├── attention_control.py  ← Top-down attention control
├── long_horizon_goals.py ← Goal stack for long-term planning (GoalStack)
├── lexicons_de.py        ← German/English lexicon for STDP vocabulary
│
├── ── Human Presence Layer ──────────────────────────────────────────────
├── human_interaction_suite.py ← 20 human-presence modules (speech, relationship, presence)
│
├── ── Tests & Tools ─────────────────────────────────────────────────────
├── soak_harness.py       ← Long-term soak tests (5 scenarios, JSONL export)
├── eval_harness.py       ← 130 scenario-based integration/regression checks (MiniWorld)
├── post_fix_harness.py   ← Structural ablation harness (5 mechanisms, minimality)
├── acceptance_eval.py    ← 8-dimension acceptance evaluation for interaction quality
├── integration_probe.py  ← φ-surrogate + phi_degradation_level() cascade measurement
├── _test_f5tts.py        ← F5-TTS standalone test (voice cloning, reference WAV)
├── _test_kokoro.py       ← Kokoro-ONNX standalone test (offline preset voice)
│
├── ── GUI & Config ──────────────────────────────────────────────────────
├── dpg_monitor.py        ← DearPyGui dashboard (10+ panels, GPU-accelerated)
├── requirements.txt      ← Dependencies
└── brain_state.db        ← Persistent database (synapses, episodes, concepts)
```

---

## Hardware & Performance

These values are a snapshot of the local development machine and current scale setup — not a fixed product specification.

| Parameter | Value |
|---|---|
| CPU | AMD Ryzen 7 5800X (8c/16t) |
| RAM | 32 GB |
| Neurons | **13,374** (7× biological scaling) |
| Synapses | **~726,000** (364k internal + 362k inter-regional) |
| Clock | **~41 ticks/s** (event-driven, no ThreadPoolExecutor) |
| GUI | **DearPyGui** (GPU-accelerated, 2.5 fps display) |
| Storage | SQLite (`brain_state.db`) — auto-save every 10,000 ticks |

---

## Signal Flow

```
Camera  → VisualEncoder (64 neurons)  → SensoryVisual
Mic     → AudioEncoder  (32 neurons)  → SensoryAuditory
Web     → WebSensor     (48 neurons)  → SensoryWeb
                                             │
                                        Thalamus
                                       /        \
                             PrimaryVisual    PrimaryAuditory
                                       \        /
                                   AssociationCortex
                                      /          \
                              Hippocampus       Amygdala
                                       \          /
                                    PrefrontalCortex
                                           │
                                       MotorCortex → Action
```

---

## Brain Regions (`regions.py`)

| Region | Neurons | Function |
|---|---|---|
| SensoryVisual | 7× scaled | Raw camera input |
| SensoryAuditory | 7× scaled | Raw microphone input |
| SensoryWeb | 7× scaled | Raw web-crawler input |
| Thalamus | 7× | Gate/amplifier function, filtering |
| PrimaryVisualCortex | 7× | Primary feature extraction (vision) |
| PrimaryAuditoryCortex | 7× | Primary feature extraction (audio) |
| AssociationCortex | 7× | Multimodal integration, concept formation |
| Hippocampus | 7× | Episodic memory, engrams |
| Amygdala | 7× | Emotional evaluation, valence |
| PrefrontalCortex | 7× | Decision-making, goal control, inhibition |
| MotorCortex | 7× | Action output, speech production |

---

## Neuron & Synapse

### Leaky Integrate-and-Fire (`neuron.py`)
- Membrane potential: `V(t) = V_rest + (V_inf - V_rest) × (1 - e^{-dt/τ})`
- V_rest = −70 mV, threshold = −55 mV
- Refractory period 2 ms after spike
- Global tonic current (14 nA awake, 3 nA sleep) keeps all neurons just below threshold

### STDP Synapse (`synapse.py`)
- Spike-timing dependent plasticity: LTP when Pre→Post, LTD when Post→Pre
- LTP rate `A_PLUS` modulated by:
  - Emotional state (`ltp_modulation()` from emotion engine)
  - Dopaminergic RPE gating: `A_PLUS × max(0.2, 1.0 + rpe × 6.0)`
- Weights pruned at < 0.008, sprouted at > 2.5

---

## Consciousness System (`consciousness.py`)

The core of the project. 20+ interlocking subsystems (core 9 + extended modules):

### 1. Self Model (`SelfModel`)
Persistent first-person identity model.
- Biological sex: **female** (only fixed property)
- Pronouns: she/her
- Tracks: learned concepts, ignition count, strengths, knowledge gaps
- `turn_state`: current conversation state (idle/speaking/listening) — live from SocialManager
- `describe()` generates full self-description from live data
- Persisted to SQLite between sessions

### 2. Episodic Memory (`EpisodicMemory`)
Timestamped autobiographical events (max. 2,000).
- Event types: `ignition`, `insight`, `concept`, `meta`, `comm`, `reflect`, `emotion`, `self`
- Based on pacemaker neurons (hippocampal time cells)
- `summarise()` returns a one-sentence summary of the last N ticks

### 3. Predictive Coding (`PredictiveCoder`)
Implements Friston's Predictive Processing (2010).
- Predicts regional activation level (EMA, α = 0.25)
- Computes signed prediction error per region
- High global error → `surprise` signal in emotion engine
- `most_surprising()` shows which region had the strongest prediction error

### 4. Meta-Cognition (`MetaCognition`)
Awareness of the system's own knowledge boundaries.
- Tracks `familiarity` (how often a concept is seen) vs. `depth` (hippocampal activation)
- `gaps()`: concepts with high familiarity but low depth → knowledge gaps
- At ≥ 2 gaps: automatic explore-goal spawn with `gap_driven:{topic}` context
- `reflect_on_stream()`: **recursive meta-reflection** — second-order observations (`[REFLECT]`)

### 5. Concept Graph (`ConceptGraph`)
Hebbian learning at the concept level.
- Weighted undirected graph of word associations
- `observe_pair(a, b)`: co-occurrence strengthens edge (like synaptic potentiation)
- `neighbors(concept)`: top-N associated concepts
- `bridge_concepts(a, b)`: finds one-hop mediator concept between A and B
- Slow decay (0.9998/tick) → long-term semantic forgetting

### 6. Intrinsic Drives (`IntrinsicDrives`)
4 biologically-inspired need states (Self-Determination Theory).
- `information_hunger`, `coherence_need`, `expression_pressure`, `rest_need`
- Combined with emotion in `_evaluate_goal()` → **intrinsic motivation**

### 7. Personality Core (`PersonalityCore`)
Character emerges from accumulated emotional patterns — **not predefined**.
- 8 emotions → 3 character variants each
- Very slow decay (τ ≈ 5,000 ticks) → persistent but changeable

### 8. Communication Drive (`CommunicationDrive`)
Decides *when* the AI communicates spontaneously — genuine internal pressure accumulation.
- Drive grows from: ignition (+0.32), meta-insight (+0.22), arousal, curiosity
- Adaptive threshold + refractory period (3,600 ticks)

### 9. Global Workspace (`ConsciousnessCore._global_broadcast`)
Implements Baars' Global Workspace Theory (1988) / Dehaene (2011).
- When ≥ 4 regions simultaneously exceed 5% activation → **global ignition**
- Winner-take-more broadcast → conscious moment

### 10. Value Learning (`ValueModel` + `build_state_signature`)
TD-λ reward learning on goal-state basis.
- `ValueModel.step()` returns TD error; |td| > 0.3 → `sm.uncertainty` + `em.surprise` increase
- Per-outcome update with goal-specific state signature
- **Causally effective:** influences goal selection (`_evaluate_goal`), surprise, emotion

### 11. Causal Graph (`CausalGraph`)
Learns causal relationships from goal-outcome pairs.
- `TransitionRecord`: State-A → action → State-B + reward
- `best_action_for_goal()`: recommends best known action for a goal context
- **Causally effective:** skill scoring in TaskExecutive, goal selection bonus, strategy planning

### 12. Theory of Mind (`TheoryOfMind`)
Inferred mental-state modelling for known persons.
- `recommend_strategy(person_id)` → strategy recommendation (e.g., `give_space`, `engage`)
- `give_space` + respond goal: operationally overrides to `create_distance`
- **Causally effective:** controls prosody, speech rate, content selection (content filter/boost), communicative initiative. Synchronized with PersonModel.

### 13. Identity Arc (`IdentityArc`)
Coherence monitoring of self-perception over time.
- `consistency_score()` [0,1]: < 0.3 → self-consistency gate blocks goal dispatch
- Warning written to goal context and consciousness stream
- **Causally effective:** goal compat score, communication style modifiers, chapter close → identity shifts

### 14. Narrative Thread (`NarrativeThread`)
Structures events into scenes and episodes.
- Recognizes scene transitions, character roles, narrative arc
- **Causally effective:** chapter close → lessons → autobiography, turning points → identity shifts, repeated types → long-term goals

### 15. Belief Quarantine (`BeliefQuarantine`)
Isolates contradictory or outdated beliefs.
- Quarantined beliefs do not influence decisions until resolved
- **Causally effective:** quarantine review per tick, promoted beliefs → BeliefStore

### 16. Attention Control (`AttentionController`)
Top-down amplification/suppression of brain regions.
- Focus region receives boost; peripheral sensory regions slight dampening
- **Causally effective:** blend_attention (60/40 top-down/bottom-up), learned utility per focus goal, PhenomenalBuffer coupling (experiential_change → focus boost)

### 17. Long-Horizon Goals (`GoalStack`)
Persistent goals with temporal horizons and priorities.
- Enables multi-day goal planning across sleep-wake cycles
- **Causally effective:** goal selection bonus in `_evaluate_goal`, active projects influence scoring

### 18. Unified Self State (`UnifiedSelfState`)
Latent self-state vector — integrates all sub-states into a unified representation.
- Encodes: SelfModel, Body, EmbodiedSelf, RobotState, TaskFrame, Sensorimotor, Emotion
- `compute_unity()`: unity score [0,1] — low → coherence alarm
- `update_agency()`: agency attribution from goal outcomes
- **Causally effective:** evaluate_bias in goal selection, coherence drive, expression drive, uncertainty update

### 19. Grounded Semantic Memory (`GroundedSemanticMemory`)
Multi-modal concept anchoring — distinguishes linguistically learned from embodied concepts.
- `ground_belief()`: grounding score for belief triples
- `most_grounded()` / `least_grounded()`: epistemic quality sorting
- **Causally effective:** QueryEngine weights belief confidence with grounding score; goal selection raises explore bonus for weakly grounded concepts; deferral at low grounding

### 20. Phenomenal Buffer (`PhenomenalBuffer`)
Integrated experience state vector — 16-dimensional leaky integrator across all modalities.
- `integrate()`: fused sensory, emotional, body, self-state per tick
- `experiential_change`: rate of experience change
- `recall_vector()`: vector for episodic encoding
- **Causally effective:** high experiential_change → attention boost + expression drive + coherence drive; goal selection considers sensory dominance and stress; self/experience questions use dominant phenomenal dimensions; episodic recall by phenomenal similarity (`recall_by_phenomenal_similarity`)

### 21. Learned World Model (`LearnedWorldModel`)
RSSM-based world model — learns environment dynamics from observations.
- `encode_observation()`: state-space encoding
- CEM-based planning (`ModelBasedPlanner.plan()`)
- **Causally effective:** model predictions weight terminal goal values, influences goal selection via `_evaluate_goal`

---

## HumanInteractionSuite — 20 Human-Presence Modules (`human_interaction_suite.py`)

The system simulates humanly-inflected speech and presence via 20 specialized modules, implemented in three masterprompt phases. All modules are updated every tick and intervene in the real reply path — not a template overlay, but state-dependent influence.

### Phase 1 — Speech, Conversation Style, Reply Density (Modules 1–5)

| Module | Class | Effect |
|---|---|---|
| 1 | `PersonalSpeechSignatureEngine` | Sentence length, directness, recurring phrases, humor; removes generic soft openers at high directness, adds hedges at low directness |
| 2 | `SubtextInterpreter` | Detects social subtext patterns (irritated, insecure, withdrawing, dominance test) and redirects reply priority |
| 3 | `DisfluencyGenerator` | Fillers, self-corrections, and search pauses only when state signals warrant (fatigue, low trust, expression pressure) |
| 4 | `ContextCompressionSpeaker` | Controls reply density: more compact with familiar persons, more cautious and detailed under uncertainty or conflict |
| 5 | `ConversationalEnergyModel` | Models conversational energy (engaged, brief, exhausted, open, irritated) and influences reply flow |

### Phase 2 — Memory, Relationship, Imperfect Recall (Modules 6–12)

| Module | Class | Effect |
|---|---|---|
| 6 | `EmotionalMemoryLayer` | Stores emotional traces per person/topic; negative old traces shift reply focus and trigger repair signals |
| 7 | `RelationshipTrajectoryEngine` | Relationship phase progression (stranger → acquaintance → trusted → strained → repaired); influences reply density and repair readiness |
| 8 | `SharedHistorySynthesizer` | Small shared hooks that selectively and only relevantly flow into replies |
| 9 | `ExpectationTracker` | Models what the person likely expects now (help, closeness, clarity, repair, brief answer) |
| 10 | `TrustCalibrationModel` | Form, caution, and conversational initiative scale with trust level |
| 11 | `ImperfectRecallModule` | Precision degradation under fatigue and low trust; memory language appears reconstructive rather than database-accurate |
| 12 | `BiasEngine` | Realistic distortions (recency, familiarity, consistency desire) — state-dependent, not random |

### Phase 3 — Thinking Modes, Hidden Motives, Body, Presence (Modules 13–20)

| Module | Class | Effect |
|---|---|---|
| 13 | `MoodDistortionFilter` | Under stress: narrow reply breadth (`target_parts=1`); under joy: increase openness (+1 part) |
| 14 | `OverthinkingUnderthinkingSwitch` | `overthinking` → min. 2 parts + HESITATE; `underthinking` → 1 part |
| 15 | `CognitiveFatigueModule` | Linguistic budget; at `< 0.45` hard 1-part limit, at `< 0.65` max. 2 parts |
| 16 | `HiddenMotivesLayer` | `seek_rest > 0.65` → trim reply to 2 sentences; `be_liked > 0.62` → warm close |
| 17 | `ValueConflictEngine` | ≥2 active value conflicts + reply > 70 chars → `[P350ms]` prosodic pause between sentences |
| 18 | `IdentityNarrativeDrift` | `hardening` → replace enthusiastic openers ("Absolut!" → "Ja."); `opening` → strip formal closes |
| 19 | `MicrobehaviorController` | `head_tilt_bias > 0` → `_uplan.head_nod = True`; `gaze_micro_variance ≥ 0.40` → `gaze_at_person = False` |
| 20 | `PresenceSynchronizer` | `timing_mode="slow"` → +350 ms delay, speed −0.12; `"eager"` → −120 ms, speed +0.08; `sync_score` → `_uplan.confidence` |

**Integration:** All modules are evaluated in `consciousness.py` (`respond_to()`) in three pre-assembly and three post-assembly blocks. Modules 19–20 directly modify the `UtterancePlan` object in `brain.py`.

### Extensions (May 2026 — Gap Features A–G)

| Feature | File(s) | What was added |
|---|---|---|
| **A** PSS persistence | `human_interaction_suite.py`, `persistence.py` | `PersonalSpeechSignatureEngine.to_dict()`/`from_dict()`; SQLite section #39 — speech signature survives restart |
| **B** Emotion probability | `emotion.py` | `EmotionalState.distribution()` → normalised 8D probability dict; `top3()` → top-3 as (name, probability) |
| **C** Trajectory tracker | `emotion.py` | `EmotionalTrajectoryTracker` — detects transitions (`anger→sadness = hurt`); holds `hidden_state`, `ask_flag`, `uncertainty_score` |
| **D** Ask-when-uncertain | `consciousness.py` | Empathic check-in phrase injected when `ask_flag=True` + reply < 300 chars; 12-turn cooldown |
| **E** Event identity drift | `identity_arc.py` | `record_event("praised"/"attacked"/"disappointed"/"successful"/"rejected"/"connected")` — nudges dimension values; keyword detection in `respond_to()` |
| **F** Breathing rhythm | `robot_controller.py` | `GazeDynamics`: sine-based `head_pitch` micro-delta (~4.5 s cycle, ±0.5°) + slow idle yaw drift every ~3 s |
| **G** Intent before motion | `robot_controller.py`, `brain.py` | `intent_move_to(target, fn)` → 3 phases: gaze→prepare→execute; `tick_intent()` wired into brain loop |
| **H** 317-gesture library | `robot_controller.py`, `consciousness.py`, `llm_adapter.py` | `_GESTURE_LIBRARY` with 317 named servo poses (9 categories: head, eyes, jaw, both-arms, right-arm, left-arm, emotional, social, communication, posture, reset); `_apply_gesture()` via `apply_action("gesture", {"name": tag}, 0)`; all LLM `[GESTURE:xxx]` tags routed uniformly; system prompt shows full categorised tag list |

---

## Dialogue Planning & Embodied Output

### UtterancePlan Pass-Through
The dialogue layer generates complete `UtterancePlan` objects with:
- `pitch_shift`, `speed_factor`, `emphasis_words` — prosodic control
- `head_nod`, `gaze_at_person`, `jaw_sync` — motor cues
- `deliberation_delay_ms`, `confidence` — dynamically adjusted by `PresenceSynchronizer` (Module 20) and `MicrobehaviorController` (Module 19)

These are passed through to the output layer:
- **SpeechOutput**: backend-specific prosody implementation (F5-TTS: zero-shot voice cloning via reference WAV + speed parameter; Kokoro: pitch via resampling; pyttsx3: rate modulation + pause emphasis)
- **RobotController**: motor cue callbacks (nod_head, gaze_at_person) triggered before speech output

### Social Feedback Loop
- Dialogue outcomes (understood, repair_requested, topic_shifted) → ToM + PersonModel
- PersonModel ↔ MentalModel synchronization (every 100 ticks)
- ToM content selection: knowledge filter (suppresses known topics) + interest boost (prioritizes conversation partner's goals)

---

## Robotics Stack

### Safety (`safety_supervisor.py`)
- E-stop detection with rising/falling edge propagation into consciousness stream
- Collision and proximity protection: immediate goal abort on `human_too_close`
- Danger zone tracking per person

### World Model (`world_state.py`)
- `TrackedPerson`: distance, angle, zone, engagement score, semantic labels
- Temporal consistency: jumps > 150 cm in < 4 ticks → `teleportation_anomaly` label + distance damping
- Predicates: `person_nearby`, `person_close`, `person_speaking`, active scene, etc.

### Skill Library & Task Executive (`skill_library.py`, `task_executive.py`)
- Skill classes declare `failure_types`: `person_lost`, `object_lost`, `collision`, `human_too_close`
- **Error classification:**
  - `person_lost` / `object_lost` (required step) → immediate fail (`target_lost:…`)
  - `collision` / `human_too_close` → full goal abort (`safety:…`)
  - Other → existing retry/fail logic

### Social Manager (`social_manager.py`)
- Turn state: `idle`, `speaking`, `listening` — live in `self_model.turn_state`
- Person modelling, conversation rounds, departure detection

---

## Architecture Features

| Feature | Implementation |
|---|---|
| E-stop → consciousness stream | Rising edge writes `[SAFETY]` message, switches goal to `halt` |
| TurnState → SelfModel | `social_manager.turn_state_for(pid)` → `self_model.turn_state` |
| TD error → surprise | `\|td\| > 0.3` → `sm.uncertainty`, `em.surprise` increased |
| Per-outcome TD update | Each goal outcome updates ValueModel with its own state signature |
| Self-consistency gate | `identity_arc.consistency_score() < 0.3` → goal context warning |
| ToM → goal context | `theory_of_mind.recommend_strategy()` → operative override + context |
| Meta/gaps → explore goal | ≥ 2 knowledge gaps → automatic `explore` goal |
| Departure → search goal | `unexpected_departures > 0` → `look_around` goal |
| Temporal consistency | Teleportation anomaly detection + distance damping |
| Error taxonomy | Unrecoverable vs. safety abort vs. retry |

---

## Structural Integrity System

The system is designed so that removing central components leads to **genuine structural collapse** — not just stopping or logging.

### Integration as Structural Carrier (`integration_probe.py`)
- `phi_surrogate()`: correlation-based φ measure across all active regions
- `phi_degradation_level()`: normalized decline [0,1] from running baseline EMA
- **φ cascade**: when φ drops, proportional consequences trigger automatically:
  - `[INTEGRATION-NOISE]`: incoherence fragments in consciousness stream
  - `self_contradictions` grow → consolidation pressure increases
  - `agency_confidence` and `continuity_estimate` erode
  - At φ drop > 50%: concepts actively evicted from workspace
- **Hard gate**: φ < 0.005 → `GoalSystemFailure` — no goal selection possible

### Self Model as Required Generator (`consciousness.py :: _evaluate_goal`)
- `sm.propose_goals(d, em, tick)` delivers (goal, weight) pairs directly from internal state:
  - Contradictions → consolidation; low agency → respond; high uncertainty → exploration
- **Health gate**: `_sm_gate = max(0.05, (agency_confidence + continuity_estimate) / 2)` multiplies **all** base scores — without coherent self model the goal distribution collapses
- `base{}` (emotion/drive/self-model scores) is fully embedded in `final{}` as `_drive_score` per goal

### Real Perception-Action Loop (`actions.py`)
- Robot feedback runs exclusively through the sensory path:
  `sensory_w.inject(proprioceptive_vector)` → Thalamus → Cortex
- `inject_text_input()` as shortcut for motor feedback **removed** — no semantic bypasses
- Proprioceptive vector: 48 floats, joint angles normalized to [0, 1]

### World Dependency → Structural Collapse (`brain.py`)
- `_sensor_free_ticks` tracks sensor absence per tick
- From tick 50 without world: active erosion of `agency_confidence` (−0.002/tick × pressure) and `continuity_estimate` (−0.0015/tick × pressure)
- Simultaneously: null activity vector in `sensory_w` → φ drops faster → φ cascade triggers
- Collapse chain: no sensor input → φ drops → self model erodes → base scores collapse → goal selection fails

### Ablation Validation (`post_fix_harness.py`)
- `check_mechanism_unavoidability()`: 5 mechanisms ablated (global_access, integration, goal_system, self_model, metacognition), 180 ticks each after 220 ticks warmup
- `check_minimality()`: verifies each mechanism causes behavioral collapse when removed
- Empirically: ≥ 1,200 ticks, multiple seeds, structural collapse measurable

---

## Emotion System (`emotion.py`)

8-dimensional emotional state:

| Dimension | Effect |
|---|---|
| `joy` | LTP booster (+plasticity) |
| `stress` | Increased activation, LTD tendency |
| `curiosity` | Communication drive, web search |
| `calm` | Background stabilization |
| `sadness` | Reduced exploration |
| `anger` | Increased tonus, direct response |
| `surprise` | Predictive error signal |
| `fatigue` | Tonic reduction, sleep tendency |

**RPE (Reward Prediction Error):**
- Signed valence change vs. slow EMA
- Modulates STDP learning rate: `A_PLUS × max(0.2, 1.0 + max(0,rpe) × 6.0)`
- Positive surprise → strong LTP (dopaminergic gating)

**Probability APIs (Gap Feature B):**
- `EmotionalState.distribution()` → normalised dict over all 8 dimensions
- `EmotionalState.top3()` → `[(name, prob), ...]` — top-3 sorted by probability

**EmotionalTrajectoryTracker (Gap Features C+D):**
- Tracks sequence of dominant emotions per turn (`deque`, window 6)
- Infers `hidden_state` from transitions: `anger→sadness = hurt`, `stress→fatigue = exhaustion`, `calm→sadness = quiet_grief`, …
- `uncertainty_score = 1 − top1_prob`; when `> 0.28` + 12-turn cooldown → `ask_flag = True`
- `ask_phrase(lang)` returns empathic check-in phrasing (DE/EN)

---

## Sleep-Wake Rhythm (`brain.py`)

| Parameter | Value |
|---|---|
| Cycle length | 8,000 ticks |
| Sleep fraction | 25% (2,000 ticks) |
| Wake tonus | 14 nA |
| Sleep tonus | 3 nA (below firing threshold) |

During sleep: intensive hippocampal replay episodes → memory consolidation.

---

## GUI (`dpg_monitor.py` / `main.py`)

**Backend: DearPyGui — GPU-accelerated**

10+ panels (tabs):
1. **Brain Anatomy** — scatter + lines, activation color-coded
2. **Region Activity** — real-time firing rate per region
3. **Emotion Radar** — 8D emotion
4. **Synapse Weight Δ** — weight change history
5. **Consciousness Timeline** — time course of key regions
6. **Drives & Concept Graph** — intrinsic drives + concepts
7. **Social** — person models + conversation events
8. **Episodic** — autobiographical events
9. **Skills** — skill success rates + costs
10. **Cognition** — continuity, identity, world model
11. **Chat** — direct input + consciousness stream output
12. **Head Live Control** — servo control + presets

All plot artists are **created once** and updated with `setData()` only — no delete/redraw.

---

## Starting the System

```powershell
cd "C:\Users\Minex\AI"
.\.venv\Scripts\python.exe main.py                            # GUI + Brain
.\.venv\Scripts\python.exe main.py --headless                 # no GUI, stdout status
.\.venv\Scripts\python.exe main.py --nocam --nomic --noweb    # safe local smoke test
.\.venv\Scripts\python.exe main.py --eval                     # eval harness (130 scenarios)
.\.venv\Scripts\python.exe main.py --soak --soak-ticks 50000  # long-run soak test
.\.venv\Scripts\python.exe main.py --postfix                  # post-fix / causal validation
.\.venv\Scripts\python.exe _test_f5tts.py                     # test F5-TTS separately
.\.venv\Scripts\python.exe _test_kokoro.py                    # test Kokoro separately
```

## GUI Commands

| Command | Function |
|---|---|
| `!konzepte` | Top-30 learned concepts with depth |
| `!episodisch` | Last 20 autobiographical events |
| `!speichern` | Manually save to SQLite |
| `!tts` | Toggle text-to-speech |
| `!hilfe` | Command overview |
| *(free text)* | Neurally processed + web interest set |

---

## Acceptance Benchmark (`acceptance_eval.py`)

```
python acceptance_eval.py           # all 8 dimensions
python acceptance_eval.py --full    # + limits report
python acceptance_eval.py --dim memory_consistency
python acceptance_eval.py --limits  # limits/hardware/ethics only
```

### 8 Dimensions + Acceptance Status

| Dimension | Auto? | Before | Target | Test Path |
|---|---|---|---|---|
| Conversational Credibility | Proxy | 0.40 | 0.75 | `interaction_style()` trajectory |
| Social Continuity | ✓ | 0.20 | 0.85 | `record_social_obligation` / dedup / serialization |
| Referential Precision | ✓ | 0.15 | 0.90 | Person-ID mapping without cross-leak |
| Response Naturalness | **HUMAN** | 0.50 | 0.60 | Perception test ≥5 persons |
| Memory Consistency | ✓ | 0.25 | 0.95 | to_dict/from_dict round-trip |
| Repair Capability | ✓ | 0.10 | 0.80 | Escalation path 0→3 repairs |
| Personalization | ✓ | 0.30 | 0.80 | topic-score convergence |
| Long-term Coherence | ✓ | 0.20 | 0.85 | GoalStack lifecycle |

---

## Limits, Hardware Requirements, Ethical Risks

### A. Software-side Bottlenecks (not solvable by code alone)

1. **Conversational credibility** is limited by LLM quality. Without GPT-4o / Claude 3.5 or a local ≥70B model, replies feel generic even when style is correctly calibrated.
2. **Response latency**: tick loop runs at ~41/s, but LLM roundtrip takes 0.8–3 s. Conversation feels sluggish; not solvable by software optimization alone.
3. **Episodic memory**: SQLite recall quality degrades after >10,000 episodes without selective consolidation. The forgetting algorithm is heuristic, not a real hippocampus model.
4. **Linguistic variety**: there is no LLM-internal check for repetition. The system may use the same phrasing multiple times — only human feedback exposes this.
5. **Theory-of-Mind depth**: the ToM module estimates intentions heuristically from observed signals — not genuine mentalizing. Fails with ambiguous or feigned intentions.

### B. Hardware Requirements for Real Person-Like Presence

1. **Camera ≥30fps + face tracking**: without live video, no eye contact, no distance measurement, no emotion recognition.
2. **Directional microphone**: omnidirectional mics cause background-noise errors → trust erosion through frequent repairs (ASR errors, not cognitive errors).
3. **Physical body** (InMoov / equivalent): text-only mode cannot demonstrate social continuity and proactivity. Gestures, eye movements, and body language are essential for person-like presence.
4. **TTS with prosody control**: a monotone voice destroys credibility regardless of text quality. F5-TTS (zero-shot cloning, offline) + Kokoro-ONNX as fallback are sufficient, but require silence gating and possibly post-processing.
5. **GPU for local inference**: for turn-taking handoffs < 400 ms response time, a local model on GPU is needed (currently cloud-LLM dependent).

### C. Ethical and Psychological Risks at Higher Credibility

1. **Parasocial bonding**: when the system is perceived as a consistent, caring conversation partner, people may form emotional dependencies. This is not a system failure but a risk of its function.
2. **Manipulation through trust**: high credibility increases the risk that users uncritically accept the system's recommendations. No disclaimer mode is currently implemented.
3. **Identity attribution**: personalization + project continuity may cause users to attribute an identity to the system that it de facto does not have.
4. **Data accumulation without control**: `PersonModel` + `GoalStack` permanently store conversation histories, interests, and conflicts. There is no "forget me completely" interface.
5. **Repair escalation as stressor**: `clarity="high"` + reactive initiative can feel paternalistic when the repair cause was the system (poor ASR, unclear LLM formulations) and not the user.

### D. Acceptance Status (as of May 2026 — after TTS migration + 20 human-presence modules + 7 gap features A–G)

| Category | Status |
|---|---|
| Automatically testable (7/8 dimensions) | ✓ Acceptance-ready at PASS ≥ 0.70 |
| Response naturalness | ✗ Human test pending |
| Hardware-dependent dimensions | ✗ All (TTS, camera, body) |
| Release for unsupervised operation | ✗ Not released |
| Release for vulnerable user groups | ✗ Not released |
