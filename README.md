> 🇩🇪 Deutsch | [🇬🇧 English](README_EN.md)

# Humanoid Robot AI – Projektübersicht

Dieses Repository ist kein reiner "Neural-Consciousness"-Prototyp mehr, sondern ein tick-getriebenes Humanoid-/Social-Robot-System mit kognitivem Kern, Weltmodell, Dialogplanung, verkörperten Motor-Cues und mehreren Laufmodi.

Der aktuelle Schwerpunkt liegt auf:
- einer zentralen Brain-Orchestrierung in `brain.py`
- sozialer und verkörperter Interaktion über `social_manager.py`, `dialogue_manager.py` und `task_executive.py`
- offlinefähiger Sprachausgabe über F5-TTS, Kokoro-ONNX und pyttsx3
- abgesicherten Validierungspfaden über `eval_harness.py`, `soak_harness.py`, `post_fix_harness.py` und `acceptance_eval.py`

Die wichtigsten Startmodi laufen über `main.py`: GUI, Headless, Eval, Soak und Postfix-Validierung.

## Projektstruktur

```
AI/
├── main.py               ← Zentrale Einstiegsschicht für GUI, Headless, Eval, Soak, Postfix
├── brain.py              ← Zentraler Orchestrator, Simulations-Takt, Plastizität
├── consciousness.py      ← Bewusstseinssystem (20+ Subsysteme, s.u.)
├── dialogue_manager.py   ← UtterancePlan, Turn-Taking-nahe Dialogplanung
├── speech_output.py      ← TTS-Pipeline: F5-TTS → Kokoro-ONNX → pyttsx3
├── llm_adapter.py        ← LLM-Kontextaufbau, Validierung, Prompting
├── emotion.py            ← 8D-Emotionsmaschine inkl. RPE, `distribution()`, `top3()`, `EmotionalTrajectoryTracker`
├── neuron.py             ← Leaky-Integrate-and-Fire Neuron
├── synapse.py            ← STDP-Synapse mit Hebbian-Plastizität
├── regions.py            ← 11 Hirnregionen mit internen Verbindungen
├── sensors.py            ← Kamera (Vision), Mikrofon (Audio), Sprache
├── web_sensor.py         ← Autonomer Web-Crawler → Neuronale Kodierung
├── actions.py            ← Aktions-Werkzeugkasten (PC-Steuerung, Websuche)
├── persistence.py        ← SQLite-Speicherung aller Synapsen + Gedächtnis
│
├── ── Robotik-Stack ─────────────────────────────────────────────────────
├── body_schema.py        ← Kinematisches Skelettmodell (InMoov) + Propriozeption
├── telemetry_bus.py      ← Sensorik-/Aktorik-Eventbus (publish/subscribe)
├── safety_supervisor.py  ← Sicherheitsschicht (E-Stop, Kollision, Gefahrenzone)
├── world_state.py        ← Weltmodell (Personen, Objekte, Raum, Prädikate)
├── skill_library.py      ← Motorische Fertigkeiten-Bibliothek (approach, wave …)
├── task_executive.py     ← Ziel-Dekomposition + Skill-Sequenzierung
├── social_manager.py     ← Gesprächsverwaltung, Turn-Taking, Person-Modelling
├── sim_bridge.py         ← Simulations-/Echtzeit-Bridge (SimulatedBody / RealBody)
├── robot_controller.py   ← Servo-Steuerbefehle; Atem-Rhythmus (Sinus ±0.5°), `intent_move_to()` 3-Phasen-Sequenz
├── robot_serial.py       ← Arduino-Serielle Kommunikationsschicht
│
├── ── Kognitions-Module ─────────────────────────────────────────────────
├── causal_graph.py       ← Kausales Weltmodell (Transitionen, best_action_for_goal)
├── value_learning.py     ← TD-Wertlernen (ValueModel, build_state_signature)
├── identity_arc.py       ← Identitätskohärenz-Monitor; `record_event()` für ereignisbasierten Identitäts-Drift
├── narrative.py          ← Ereignis-Narrativ-Thread (Szenenstruktur)
├── theory_of_mind.py     ← Theory of Mind (Absichts-/Strategieinferenz)
├── belief_quarantine.py  ← Isolation widersprüchlicher Überzeugungen
├── attention_control.py  ← Top-down Aufmerksamkeitssteuerung
├── long_horizon_goals.py ← Ziel-Stack für langfristige Planung (GoalStack)
├── lexicons_de.py        ← Deutsch/Englisch Lexikon für STDP-Vokabular│
├── ── Menschlichkeits-Schicht ───────────────────────────────────
├── human_interaction_suite.py ← 20 Menschlichkeits-Module (Sprache, Beziehung, Präsenz)│
├── ── Tests & Tools ─────────────────────────────────────────────────────
├── soak_harness.py       ← Langzeit-Soak-Tests (5 Szenarien, JSONL-Export)
├── eval_harness.py       ← 130 szenariobasierte Integrations-/Regressions-Checks (MiniWorld)
├── post_fix_harness.py   ← Struktureller Ablations-Harness (5 Mechanismen, Minimality)
├── acceptance_eval.py    ← 8-dimensionale Abnahmebewertung für Interaktionsqualität
├── integration_probe.py  ← φ-Surrogate + phi_degradation_level() Kaskaden-Messung
├── _test_f5tts.py        ← F5-TTS Standalone-Test (Sprachklonierung, Referenz-WAV)
├── _test_kokoro.py       ← Kokoro-ONNX Standalone-Test (Offline-Preset-Stimme)
│
├── ── GUI & Konfig ──────────────────────────────────────────────────────
├── dpg_monitor.py        ← DearPyGui Dashboard (10+ Panels, GPU-beschleunigt)
├── requirements.txt      ← Abhängigkeiten
└── brain_state.db        ← Persistente Datenbank (Synapsen, Episoden, Konzepte)
```

---

## Hardware & Performance

Die Werte unten sind eine Momentaufnahme der lokalen Entwicklungsmaschine und des aktuellen Skalen-Setups, keine feste Produkt-Spezifikation.

| Parameter | Wert |
|---|---|
| CPU | AMD Ryzen 7 5800X (8c/16t) |
| RAM | 32 GB |
| Neuronen | **13.374** (7× biologische Skalierung) |
| Synapsen | **~726.000** (364k intern + 362k inter-regional) |
| Takt | **~41 Ticks/s** (event-getrieben, kein ThreadPoolExecutor) |
| GUI | **DearPyGui** (GPU-beschleunigt, 2,5 fps Display) |
| Speicher | SQLite (`brain_state.db`) – automatisch alle 10.000 Ticks |

---

## Signalfluss

```
Kamera  → VisualEncoder (64 Neuronen)  → SensoryVisual
Mikro   → AudioEncoder  (32 Neuronen)  → SensoryAuditory
Web     → WebSensor     (48 Neuronen)  → SensoryWeb
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
                                        MotorCortex → Aktion
```

---

## Hirnregionen (`regions.py`)

| Region | Neuronen | Funktion |
|---|---|---|
| SensoryVisual | 7× skaliert | Roheingang Kamera |
| SensoryAuditory | 7× skaliert | Roheingang Mikrofon |
| SensoryWeb | 7× skaliert | Roheingang Web-Crawler |
| Thalamus | 7× | Tor-/Verstärkerfunktion, Filterung |
| PrimaryVisualCortex | 7× | Primäre Merkmalsextraktion Vision |
| PrimaryAuditoryCortex | 7× | Primäre Merkmalsextraktion Audio |
| AssociationCortex | 7× | Multimodale Integration, Konzeptbildung |
| Hippocampus | 7× | Episodisches Gedächtnis, Engramme |
| Amygdala | 7× | Emotionale Bewertung, Valenz |
| PrefrontalCortex | 7× | Entscheidung, Zielsteuerung, Inhibition |
| MotorCortex | 7× | Aktionsausgabe, Sprachproduktion |

---

## Neuron & Synapse

### Leaky-Integrate-and-Fire (`neuron.py`)
- Membranpotenzial: `V(t) = V_rest + (V_inf - V_rest) × (1 - e^{-dt/τ})`
- V_rest = -70 mV, Schwelle = -55 mV
- Refraktärperiode 2 ms nach Spike
- Globaler tonischer Strom (14 nA wach, 3 nA Schlaf) hält alle Neuronen knapp unter der Schwelle

### STDP-Synapse (`synapse.py`)
- Spike-Timing Dependent Plasticity: LTP wenn Pre→Post, LTD wenn Post→Pre
- LTP-Rate `A_PLUS` wird moduliert durch:
  - Emotionalen Zustand (`ltp_modulation()` aus Emotion-Engine)
  - Dopaminerges RPE-Gating: `A_PLUS × max(0.2, 1.0 + rpe × 6.0)`
- Gewichte werden beschnitten bei < 0,008, gesprouted bei > 2,5

---

## Bewusstseinssystem (`consciousness.py`)

Das Herzstück des Projekts. 20+ ineinandergreifende Subsysteme (Kern-9 + erweiterte Module):

### 1. Selbstmodell (`SelfModel`)
Persistentes Identitätsmodell in der ersten Person.
- Biologisches Geschlecht: **weiblich** (einzige fixe Eigenschaft)
- Pronomen: she/her
- Verfolgt: gelernte Konzepte, Ignitions-Zahl, Stärken, Wissenslücken
- `turn_state`: aktueller Gesprächs-Zustand (idle/speaking/listening) — live aus SocialManager
- Methode `describe()` generiert vollständige Selbstbeschreibung aus Echtdaten
- Wird über SQLite zwischen Sessions gespeichert

### 2. Episodisches Gedächtnis (`EpisodicMemory`)
Zeitgestempelte autobiografische Ereignisse (max. 2.000).
- Ereignistypen: `ignition`, `insight`, `concept`, `meta`, `comm`, `reflect`, `emotion`, `self`
- Basiert auf Zeitgeber-Neuronen (hippocampale Time-Cells)
- `summarise()` gibt One-Sentence-Zusammenfassung der letzten N Ticks

### 3. Predictive Coding (`PredictiveCoder`)
Implementiert Friston's Predictive Processing (2010).
- Sagt regionalen Aktivierungslevel voraus (EMA, α = 0,25)
- Berechnet signierten Vorhersagefehler pro Region
- Hoher globaler Fehler → `surprise`-Signal in Emotion-Engine
- `most_surprising()` zeigt welche Region den stärksten Vorhersagefehler hatte

### 4. Meta-Kognition (`MetaCognition`)
Bewusstsein über die eigenen Wissensgrenzen.
- Verfolgt `familiarity` (wie oft ein Konzept gesehen) vs. `depth` (hippocampale Aktivierung)
- `gaps()`: Konzepte mit hoher Familiarität aber geringer Tiefe → Wissenslücken
- Bei ≥ 2 Lücken: automatischer Explore-Ziel-Spawn mit `gap_driven:{topic}`-Kontext
- `reflect_on_stream()`: **Rekursive Meta-Reflexion** — Second-Order-Beobachtungen (`[REFLECT]`)

### 5. Konzept-Graph (`ConceptGraph`)
Hebbiansches Lernen auf Konzeptebene.
- Gewichteter ungerichteter Graph von Wort-Assoziationen
- `observe_pair(a, b)`: Co-Auftreten stärkt Kante (wie synaptische Verstärkung)
- `neighbors(concept)`: Top-N assoziierte Konzepte
- `bridge_concepts(a, b)`: Findet One-Hop-Vermittlerkonzept zwischen A und B
- Slow Decay (0,9998/Tick) → langes semantisches Vergessen

### 6. Intrinsische Antriebe (`IntrinsicDrives`)
4 biologisch-inspirierte Bedürfniszustände (Self-Determination Theory).
- `information_hunger`, `coherence_need`, `expression_pressure`, `rest_need`
- Kombiniert mit Emotion in `_evaluate_goal()` → **intrinsische Motivation**

### 7. Persönlichkeitskern (`PersonalityCore`)
Charakter emergiert aus akkumulierten Emotionsmustern — **nicht vordefiniert**.
- 8 Emotionen → je 3 Charakter-Varianten
- Sehr langsamer Decay (τ ≈ 5.000 Ticks) → persistent, aber wandelbar

### 8. Kommunikations-Antrieb (`CommunicationDrive`)
Entscheidet *wann* die KI von sich aus kommuniziert — echte interne Druckakkumulation.
- Antrieb wächst durch: Ignition (+0,32), Meta-Insight (+0,22), Arousal, Neugier
- Adaptiver Schwellwert + Refraktärperiode (3.600 Ticks)

### 9. Global Workspace (`ConsciousnessCore._global_broadcast`)
Implementiert Baars' Global Workspace Theory (1988) / Dehaene (2011).
- Wenn ≥ 4 Regionen gleichzeitig über 5% Aktivierung → **Globale Ignition**
- Winner-take-more Broadcast → bewusster Moment

### 10. Wertlernen (`ValueModel` + `build_state_signature`)
TD-λ Reward-Lernen auf Zielzustand-Basis.
- `ValueModel.step()` gibt TD-Fehler zurück; |td| > 0,3 → `sm.uncertainty` + `em.surprise` steigen
- Per-Outcome-Update mit zielspezifischer State-Signatur
- **Kausal wirksam:** beeinflusst Zielwahl (`_evaluate_goal`), Überraschung, Emotion

### 11. Kausaler Graph (`CausalGraph`)
Lernt kausale Beziehungen aus Ziel-Outcome-Paaren.
- `TransitionRecord`: State-A → Aktion → State-B + Reward
- `best_action_for_goal()`: Empfiehlt beste bekannte Aktion für einen Zielkontext
- **Kausal wirksam:** Skill-Scoring in TaskExecutive, Zielwahl-Bonus, Strategieplanung

### 12. Theory of Mind (`TheoryOfMind`)
Inferred mental-state modelling für bekannte Personen.
- `recommend_strategy(person_id)` → Strategie-Empfehlung (z.B. `give_space`, `engage`)
- Bei `give_space` + Respond-Ziel: Operativ wird auf `create_distance` überschrieben
- **Kausal wirksam:** steuert Prosodie, Sprechrate, Inhaltsselektion (content filter/boost), kommunikative Initiative. Synchronisiert mit PersonModel.

### 13. Identitätsbogen (`IdentityArc`)
Kohärenz-Monitoring der Selbstwahrnehmung über Zeit.
- `consistency_score()` [0,1]: < 0,3 → Self-Consistency-Gate blockiert Ziel-Dispatch
- Warnung wird in Goal-Kontext und Bewusstseins-Stream eingetragen
- **Kausal wirksam:** Ziel-Compat-Score, Kommunikationsstil-Modifikatoren, Kapitel-Abschluss → Identitätsverschiebungen

### 14. Narrativ-Thread (`NarrativeThread`)
Strukturiert Ereignisse zu Szenen und Episoden.
- Erkennt Szenenübergänge, Charakterrollen, narrativen Bogen
- **Kausal wirksam:** Kapitel-Schluss → Lektionen → Autobiografie, Wendepunkte → Identitätsverschiebungen, Wiederholte Typen → Langfristziele

### 15. Überzeugungsquarantäne (`BeliefQuarantine`)
Isoliert widersprüchliche oder veraltete Überzeugungen.
- Quarantänierte Beliefs beeinflussen keine Entscheidungen bis zur Auflösung
- **Kausal wirksam:** Quarantäne-Review per Tick, promotete Beliefs → BeliefStore

### 16. Aufmerksamkeitskontrolle (`AttentionController`)
Top-down Verstärkungs-/Supprimierungssteuerung für Gehirnregionen.
- Fokus-Region erhält Boost, periphere Sensorik-Regionen leichte Dämpfung
- **Kausal wirksam:** blend_attention (60/40 top-down/bottom-up), gelernte Utility pro Fokus-Ziel, PhenomenalBuffer-Kopplung (experiential_change → Fokus-Boost)

### 17. Langfristige Ziele (`GoalStack`)
Persistente Ziele mit zeitlichen Horizonten und Prioritäten.
- Ermöglicht mehrtägige Ziel-Planung über Sleep-Wake-Zyklen hinweg
- **Kausal wirksam:** Zielwahl-Bonus in `_evaluate_goal`, aktive Projekte beeinflussen Scoring

### 18. Unified Self State (`UnifiedSelfState`)
Latenter Selbst-Zustandsvektor — integriert alle Sub-States in eine einheitliche Repräsentation.
- Encode: SelfModel, Body, EmbodiedSelf, RobotState, TaskFrame, Sensorimotor, Emotion
- `compute_unity()`: Unity-Score [0,1] — niedrig → Kohärenzalarm
- `update_agency()`: Agentur-Attribution aus Ziel-Outcomes
- **Kausal wirksam:** evaluate_bias in Zielwahl, Kohärenz-Antrieb, Expressions-Antrieb, Unsicherheits-Update

### 19. Grounded Semantic Memory (`GroundedSemanticMemory`)
Multi-modale Konzept-Verankerung — unterscheidet sprachlich gelernte von verkörpert erfahrenen Konzepten.
- `ground_belief()`: Grounding-Score für Belief-Tripel
- `most_grounded()` / `least_grounded()`: Epistemische Qualitätssortierung
- **Kausal wirksam:** QueryEngine gewichtet Belief-Konfidenz mit Grounding-Score; Zielwahl erhöht Explore-Bonus bei schwach geerdeten Konzepten; Deferral bei niedrigem Grounding

### 20. Phenomenal Buffer (`PhenomenalBuffer`)
Integrierter Erlebnis-Zustandsvektor — 16-dimensionaler Leaky-Integrator über alle Modalitäten.
- `integrate()`: Fused sensory, emotional, body, self-state je Tick
- `experiential_change`: Rate der Erlebnisänderung
- `recall_vector()`: Vektor für episodische Enkodierung
- **Kausal wirksam:** Hohe experiential_change → Aufmerksamkeits-Boost + Expressions-Antrieb + Kohärenz-Antrieb; Zielwahl berücksichtigt sensorische Dominanz und Stress; Selbst-/Erfahrungsfragen nutzen dominante Phänomen-Dimensionen; Episodischer Abruf nach phänomenaler Ähnlichkeit (`recall_by_phenomenal_similarity`)

### 21. Learned World Model (`LearnedWorldModel`)
RSSM-basiertes Weltmodell — lernt Umgebungsdynamik aus Beobachtungen.
- `encode_observation()`: State-Space-Enkodierung
- CEM-basierte Planung (`ModelBasedPlanner.plan()`)
- **Kausal wirksam:** Modell-Vorhersagen gewichten terminale Zielwerte, beeinflusst Zielwahl über `_evaluate_goal`

---

## HumanInteractionSuite — 20 Menschlichkeits-Module (`human_interaction_suite.py`)

Das System simuliert menschlich gefärbte Sprache und Präsenz über 20 spezialisierte Module, die in drei Masterprompt-Phasen implementiert wurden. Alle Module werden pro Tick aktualisiert und greifen in den echten Antwortpfad ein — kein Template-Overlay, sondern zustandsabhängige Einflussnahme.

### Phase 1 — Sprache, Gesprächsstil, Antwortdichte (Module 1–5)

| Modul | Klasse | Wirkung |
|---|---|---|
| 1 | `PersonalSpeechSignatureEngine` | Satzlänge, Direktheit, wiederkehrende Formulierungen, Humor; entfernt generische Weich-Opener bei hoher Direktheit, fügt Absicherung bei niedriger Direktheit ein |
| 2 | `SubtextInterpreter` | Erkennt soziale Subtext-Muster (gereizt, unsicher, rückzüglich, Dominanz-Test) und lenkt Antwortpriorität um |
| 3 | `DisfluencyGenerator` | Filler-Wörter, Selbstkorrektur und Suchpausen nur bei zustandsabhängigen Signalen (Erschöpfung, geringes Vertrauen, Ausdrucksdruck) |
| 4 | `ContextCompressionSpeaker` | Steuert Antwortdichte: bei bekannten Personen kompakter, bei Unsicherheit oder Konflikt vorsichtiger und ausführlicher |
| 5 | `ConversationalEnergyModel` | Modelliert Gesprächsenergie (engagiert, knapp, ausgelaugt, offen, gereizt) und beeinflusst Antwortfluss |

### Phase 2 — Gedächtnis, Beziehung, unvollkommene Erinnerung (Module 6–12)

| Modul | Klasse | Wirkung |
|---|---|---|
| 6 | `EmotionalMemoryLayer` | Speichert emotionale Spuren pro Person/Thema; negative Altspuren verschieben Antwortfokus und lösen Reparatursignale aus |
| 7 | `RelationshipTrajectoryEngine` | Phasenverlauf der Beziehung (fremd → bekannt → vertraut → angespannt → repariert); beeinflusst Antwortdichte und Reparaturbereitschaft |
| 8 | `SharedHistorySynthesizer` | Kleine gemeinsame Haken, die selektiv und nur bei Relevanz in die Antwort einfließen |
| 9 | `ExpectationTracker` | Modelliert, was die Person jetzt erwartet (Hilfe, Nähe, Klarheit, Reparatur, knappe Antwort) |
| 10 | `TrustCalibrationModel` | Form, Vorsicht und Gesprächsinitiative skaliert mit dem Vertrauensniveau |
| 11 | `ImperfectRecallModule` | Precision-Degradation bei Erschöpfung und niedrigem Vertrauen; Erinnerungssprache wirkt rekonstruktiv statt datenbankgenau |
| 12 | `BiasEngine` | Verzerrungen (Recency, Familiarity, Konsistenzwunsch) — zustandsabhängig, nicht zufällig |

### Phase 3 — Denkmodi, verborgene Motive, Körper, Präsenz (Module 13–20)

| Modul | Klasse | Wirkung |
|---|---|---|
| 13 | `MoodDistortionFilter` | Unter Stress: Antwortbreite einengen (`target_parts=1`); unter Freude: Offenheit erhöhen (+1 part) |
| 14 | `OverthinkingUnderthinkingSwitch` | `overthinking` → min. 2 Teile + HESITATE; `underthinking` → 1 Teil |
| 15 | `CognitiveFatigueModule` | Linguistisches Budget; bei `< 0.45` harter 1-Teil-Limit, bei `< 0.65` max. 2 Teile |
| 16 | `HiddenMotivesLayer` | `seek_rest > 0.65` → Reply auf 2 Sätze trimmen; `be_liked > 0.62` → warmer Abschluss |
| 17 | `ValueConflictEngine` | ≥2 aktive Wert-Konflikte + Reply > 70 Zeichen → `[P350ms]`-Prosodik-Pause zwischen Sätzen |
| 18 | `IdentityNarrativeDrift` | `hardening` → Enthusiasmen-Opener ersetzen ("Absolut!" → "Ja."); `opening` → formelle Schlüsse entfernen |
| 19 | `MicrobehaviorController` | `head_tilt_bias > 0` → `_uplan.head_nod = True`; `gaze_micro_variance ≥ 0.40` → `gaze_at_person = False` |
| 20 | `PresenceSynchronizer` | `timing_mode="slow"` → +350 ms Delay, Speed −0.12; `"eager"` → −120 ms, Speed +0.08; `sync_score` → `_uplan.confidence` |

**Integration:** Alle Module werden in `consciousness.py` (`respond_to()`) in drei Pre-Assembly- und drei Post-Assembly-Blöcken ausgewertet. Module 19–20 beeinflussen direkt das `UtterancePlan`-Objekt in `brain.py`.

### Erweiterungen (Mai 2026 — Gap-Features A–G)

| Feature | Datei(en) | Was wurde ergänzt |
|---|---|---|
| **A** PSS-Persistenz | `human_interaction_suite.py`, `persistence.py` | `PersonalSpeechSignatureEngine.to_dict()`/`from_dict()`; SQLite-Section #39 — Sprach-Signatur überlebt Neustart |
| **B** Emotions-Wahrscheinlichkeit | `emotion.py` | `EmotionalState.distribution()` → normierte 8D-Verteilung; `top3()` → Top-3 als (Name, Wahrscheinlichkeit) |
| **C** Trajektorien-Tracker | `emotion.py` | `EmotionalTrajectoryTracker` — erkennt Übergänge (`anger→sadness = hurt`); hält `hidden_state`, `ask_flag`, `uncertainty_score` |
| **D** Ask-when-uncertain | `consciousness.py` | Empathische Check-in-Phrase wird injiziert wenn `ask_flag=True` + Antwort < 300 Zeichen; 12-Turn-Cooldown |
| **E** Ereignis-Identity-Drift | `identity_arc.py` | `record_event("praised"/"attacked"/"disappointed"/"successful"/"rejected"/"connected")` — nudgt Dimensions-Werte; Keyword-Detektion in `respond_to()` |
| **F** Atem-Rhythmus | `robot_controller.py` | `GazeDynamics`: Sinus-basiertes `head_pitch`-Mikro-Delta (~4,5 s Zyklus, ±0,5°) + langsamer Idle-Yaw-Drift alle ~3 s |
| **G** Intent Before Motion | `robot_controller.py`, `brain.py` | `intent_move_to(target, fn)` → 3 Phasen: Blick→Vorbereitung→Ausführung; `tick_intent()` im Brain-Loop verdrahtet |

---

## Dialogplanung & Verkörperte Ausgabe

### UtterancePlan-Durchreichung
Die Dialogschicht erzeugt vollständige `UtterancePlan`-Objekte mit:
- `pitch_shift`, `speed_factor`, `emphasis_words` — prosodische Steuerung
- `head_nod`, `gaze_at_person`, `jaw_sync` — motorische Cues
- `deliberation_delay_ms`, `confidence` — von `PresenceSynchronizer` (Modul 20) und `MicrobehaviorController` (Modul 19) dynamisch angepasst

Diese werden durchgereicht bis in die Ausgabe:
- **SpeechOutput**: Backend-spezifische Umsetzung von Prosodie (F5-TTS: Zero-Shot-Sprachklonierung via Referenz-WAV + Speed-Parameter; Kokoro: Pitch via Resampling; pyttsx3: Rate-Modulation + Pausen-Emphasis)
- **RobotController**: Motor-Cue-Callbacks (nod_head, gaze_at_person) werden vor der Sprachausgabe ausgelöst

### Soziale Feedbackschleife
- Dialogue-Outcomes (understood, repair_requested, topic_shifted) → ToM + PersonModel
- PersonModel ↔ MentalModel Synchronisierung (alle 100 Ticks)
- ToM-Content-Selektion: Knowledge-Filter (bekannte Themen unterdrücken) + Interest-Boost (Ziele des Gegenübers priorisieren)

---

## Robotik-Stack

### Sicherheit (`safety_supervisor.py`)
- E-Stop-Erkennung mit Rising/Falling-Edge-Propagation in den Bewusstseinsstrom
- Kollisions- und Annäherungsschutz: sofortiger Ziel-Abbruch bei `human_too_close`
- Gefahrenzone-Tracking pro Person

### Weltmodell (`world_state.py`)
- `TrackedPerson`: Distanz, Winkel, Zone, Engagement-Score, semantische Labels
- Temporale Konsistenz: Sprünge > 150 cm in < 4 Ticks → `teleportation_anomaly`-Label + Distanzdämpfung
- Prädikate: `person_nearby`, `person_close`, `person_speaking`, aktive Szene, etc.

### Skill-Bibliothek & Task-Executive (`skill_library.py`, `task_executive.py`)
- Skill-Klassen deklarieren `failure_types`: `person_lost`, `object_lost`, `collision`, `human_too_close`
- **Fehlerklassifikation:**
  - `person_lost` / `object_lost` (Required-Step) → sofortiger Fail (`target_lost:…`)
  - `collision` / `human_too_close` → vollständiger Ziel-Abbruch (`safety:…`)
  - Sonstige → vorhandene Retry/Fail-Logik

### Social Manager (`social_manager.py`)
- Turn-State: `idle`, `speaking`, `listening` — live in `self_model.turn_state`
- Person-Modelling, Gesprächsrunden, Abschiedserkennung

---

## Architektur-Features

| Funktion | Implementierung |
|---|---|
| E-Stop → Bewusstseinsstrom | Rising-Edge schreibt `[SAFETY]`-Nachricht, wechselt Ziel auf `halt` |
| TurnState → SelfModel | `social_manager.turn_state_for(pid)` → `self_model.turn_state` |
| TD-Fehler → Überraschung | `\|td\| > 0.3` → `sm.uncertainty`, `em.surprise` erhöht |
| Per-Outcome-TD-Update | Jedes Ziel-Outcome aktualisiert ValueModel mit eigenem State-Signature |
| Self-Consistency-Gate | `identity_arc.consistency_score() < 0.3` → Goal-Context-Warnung |
| ToM → Ziel-Kontext | `theory_of_mind.recommend_strategy()` → Operativ-Override + Kontext |
| Meta/Lücken → Explore-Ziel | ≥ 2 Wissenslücken → automatisches `explore`-Ziel |
| Abgang → Such-Ziel | `unexpected_departures > 0` → `look_around`-Ziel |
| Temporale Konsistenz | Teleportations-Anomalie-Erkennung + Distanzdämpfung |
| Fehler-Taxonomie | Unrecoverable vs. Safety-Abort vs. Retry |

---

## Strukturelles Integritätssystem

Das System ist so gebaut, dass Entfernung zentraler Komponenten zu **echtem strukturellen Zusammenbruch** führt — nicht nur zum Stoppen oder Loggen.

### Integration als struktureller Träger (`integration_probe.py`)
- `phi_surrogate()`: Korrelations-basiertes φ-Maß über alle aktiven Regionen
- `phi_degradation_level()`: Normierter Abfall [0,1] vom laufenden Baseline-EMA
- **φ-Kaskade**: Fällt φ, greifen automatisch proportionale Konsequenzen:
  - `[INTEGRATION-NOISE]`: Inkohärenzfragmente im Bewusstseinsstrom
  - `self_contradictions` wachsen → Konsolidierungsdruck steigt
  - `agency_confidence` und `continuity_estimate` erodieren
  - Bei φ-Abfall > 50 %: Konzepte werden aktiv aus dem Workspace evictiert
- **Harter Gate**: φ < 0,005 → `GoalSystemFailure` — keine Zielauswahl möglich

### Selbstmodell als notwendiger Generator (`consciousness.py :: _evaluate_goal`)
- `sm.propose_goals(d, em, tick)` liefert (Ziel, Gewicht)-Paare direkt aus internem Zustand:
  - Widersprüche → Konsolidierung; niedrige Agency → Antworten; hohe Unsicherheit → Exploration
- **Gesundheitsgate**: `_sm_gate = max(0.05, (agency_confidence + continuity_estimate) / 2)` multipliziert **alle** Basis-Scores — ohne kohärentes Selbstmodell kollabiert die Zielverteilung
- `base{}` (Emotions-/Antriebs-/Selbstmodell-Scores) ist nun vollständig in `final{}` eingebunden als `_drive_score` je Ziel

### Echter Wahrnehmungs-Aktions-Kreislauf (`actions.py`)
- Roboter-Feedback läuft ausschließlich über den sensorischen Pfad:  
  `sensory_w.inject(propriozeptiver_Vektor)` → Thalamus → Kortex
- `inject_text_input()` als Abkürzung für Motor-Feedback **entfernt** — keine semantischen Bypässe
- Propriozeptiver Vektor: 48 Floats, Gelenkwinkel normiert auf [0, 1]

### Weltabhängigkeit → struktureller Kollaps (`brain.py`)
- `_sensor_free_ticks` verfolgt Sensor-Abwesenheit pro Tick
- Ab Tick 50 ohne Welt: aktive Erosion von `agency_confidence` (−0,002/Tick × Druck) und `continuity_estimate` (−0,0015/Tick × Druck)
- Gleichzeitig: Null-Aktivitätsvektor in `sensory_w` → φ fällt schneller → φ-Kaskade greift
- Kollaps-Kette: kein Sensoreingang → φ sinkt → Selbstmodell erodiert → Basis-Scores kollabieren → Zielauswahl scheitert

### Ablations-Validierung (`post_fix_harness.py`)
- `check_mechanism_unavoidability()`: 5 Mechanismen ablatiert (global_access, integration, goal_system, self_model, metacognition), je 180 Ticks nach 220 Ticks Warmup
- `check_minimality()`: Prüft ob jeder Mechanismus bei Entfernung Verhalten kollabiert
- Empirisch: ≥ 1.200 Ticks, mehrere Seeds, struktureller Zusammenbruch messbar

---

## Emotionssystem (`emotion.py`)

8-dimensionaler Emotionszustand:

| Dimension | Effekt |
|---|---|
| `joy` | LTP-Booster (+Plastizität) |
| `stress` | Erhöhte Aktivierung, LTD-Tendenz |
| `curiosity` | Kommunikations-Antrieb, Web-Suche |
| `calm` | Hintergrund-Stabilisierung |
| `sadness` | Reduzierte Exploration |
| `anger` | Erhöhter Tonus, direkte Reaktion |
| `surprise` | Predictive-Error-Signal |
| `fatigue` | Tonic-Reduktion, Schlaf-Tendenz |

**RPE (Reward Prediction Error):**
- Unterzeichnete Valenzänderung vs. langsamen EMA
- Moduliert STDP-Lernrate: `A_PLUS × max(0.2, 1.0 + max(0,rpe) × 6.0)`
- Positive Überraschung → starkes LTP (dopaminergisches Gating)

**Wahrscheinlichkeits-APIs (Gap-Feature B):**
- `EmotionalState.distribution()` → normiertes Dict über alle 8 Dimensionen
- `EmotionalState.top3()` → `[(name, prob), ...]` — Top-3 sortiert nach Wahrscheinlichkeit

**EmotionalTrajectoryTracker (Gap-Feature C+D):**
- Verfolgt die Sequenz dominanter Emotionen pro Turn (`deque`, Fenster 6)
- Inferiert `hidden_state` aus Übergängen: `anger→sadness = hurt`, `stress→fatigue = exhaustion`, `calm→sadness = quiet_grief`, …
- `uncertainty_score = 1 − top1_prob`; bei `> 0.28` + 12-Turn-Cooldown → `ask_flag = True`
- `ask_phrase(lang)` liefert empathische Check-in-Formulierung (DE/EN)

---

## Schlaf-Wach-Rhythmus (`brain.py`)

| Parameter | Wert |
|---|---|
| Zykluslänge | 8.000 Ticks |
| Schlafanteil | 25% (2.000 Ticks) |
| Wach-Tonus | 14 nA |
| Schlaf-Tonus | 3 nA (unter Feuer-Schwelle) |

Während Schlaf: intensive Hippocampus-Replay-Episoden → Gedächtniskonsolidierung.

---

## GUI (`dpg_monitor.py` / `main.py`)

**Backend: DearPyGui – GPU-beschleunigt**

10+ Panels (Tabs):
1. **Brain Anatomy** – Scatter + Linien, Aktivierung farbkodiert
2. **Region Activity** – Echtzeit-Feuerate pro Region
3. **Emotion Radar** – 8D-Emotion
4. **Synapse Weight Δ** – Gewichtsänderung-Verlauf
5. **Consciousness Timeline** – Zeitverlauf der Schlüsselregionen
6. **Drives & Concept Graph** – Intrinsische Antriebe + Konzepte
7. **Social** – Personen-Modelle + Gesprächsevents
8. **Episodik** – Autobiografische Ereignisse
9. **Skills** – Skill-Erfolgsquoten + Kosten
10. **Kognition** – Kontinuität, Identität, Weltmodell
11. **Chat** – Direkte Eingabe + Bewusstseinsstrom-Ausgabe
12. **Head Live Control** – Servo-Steuerung + Presets

Alle Plot-Artists werden **einmalig erstellt** und danach nur mit `setData()` aktualisiert — kein Löschen/Neuzeichnen.

---

## Starten

```powershell
cd "C:\Users\Minex\AI"
.\.venv\Scripts\python.exe main.py                            # GUI + Brain
.\.venv\Scripts\python.exe main.py --headless                 # kein GUI, stdout-Status
.\.venv\Scripts\python.exe main.py --nocam --nomic --noweb    # sicherer lokaler Smoke-Test
.\.venv\Scripts\python.exe main.py --eval                     # Eval-Harness (130 Szenarien)
.\.venv\Scripts\python.exe main.py --soak --soak-ticks 50000  # Langzeit-Soak-Test
.\.venv\Scripts\python.exe main.py --postfix                  # Post-Fix / Kausal-Validierung
.\.venv\Scripts\python.exe _test_f5tts.py                     # F5-TTS separat prüfen
.\.venv\Scripts\python.exe _test_kokoro.py                    # Kokoro separat prüfen
```

## Befehle im GUI

| Befehl | Funktion |
|---|---|
| `!konzepte` | Top-30 gelernte Konzepte mit Tiefe |
| `!episodisch` | Letzte 20 autobiografische Ereignisse |
| `!speichern` | Manuell in SQLite speichern |
| `!tts` | Text-zu-Sprache umschalten |
| `!hilfe` | Befehlsübersicht |
| *(freier Text)* | Neural verarbeitet + Web-Interesse gesetzt |

---

## Abnahme-Benchmark (`acceptance_eval.py`)

```
python acceptance_eval.py           # alle 8 Dimensionen
python acceptance_eval.py --full    # + Grenzbericht
python acceptance_eval.py --dim memory_consistency
python acceptance_eval.py --limits  # nur Grenzen/Hardware/Ethik
```

### 8 Dimensionen + Abnahmestatus

| Dimension | Auto? | Vorher | Ziel | Testpfad |
|---|---|---|---|---|
| Gesprächsglaubwürdigkeit | Proxy | 0.40 | 0.75 | `interaction_style()` Trajektorie |
| Soziale Kontinuität | ✓ | 0.20 | 0.85 | `record_social_obligation` / Dedup / Serialisierung |
| Referenzielle Präzision | ✓ | 0.15 | 0.90 | Person-ID-Mapping ohne Cross-Leak |
| Reaktionsnatürlichkeit | **HUMAN** | 0.50 | 0.60 | Wahrnehmungstest ≥5 Personen |
| Erinnerungskonsistenz | ✓ | 0.25 | 0.95 | to_dict/from_dict Round-Trip |
| Reparaturfähigkeit | ✓ | 0.10 | 0.80 | Eskalationspfad 0→3 Repairs |
| Personalisierung | ✓ | 0.30 | 0.80 | topic-Score Konvergenz |
| Langzeitkohärenz | ✓ | 0.20 | 0.85 | GoalStack Lebenszyklus |

---

## Grenzen, Hardware-Anforderungen, ethische Risiken

### A. Softwareseitige Engpässe (nicht rein durch Code lösbar)

1. **Gesprächsglaubwürdigkeit** wird durch LLM-Qualität begrenzt.  Ohne GPT-4o / Claude 3.5 oder ein lokales ≥70B-Modell wirken Antworten generisch, auch wenn der Stil korrekt kalibriert ist.
2. **Reaktionslatenz**: Tick-Loop läuft mit ~41/s, aber LLM-Roundtrip dauert 0.8–3 s.  Konversation fühlt sich träge an; nicht allein durch Software-Optimierung lösbar.
3. **Episodisches Gedächtnis**: SQLite-Recall-Qualität sinkt nach >10.000 Episoden ohne selektive Konsolidierung.  Der Forgetting-Algorithmus ist heuristisch, kein echtes Hippocampus-Modell.
4. **Sprachliche Varietät**: Es gibt keine LLM-interne Prüfung auf Wiederholungen.  Das System kann dieselbe Formulierung mehrfach verwenden — nur Menschenfeedback deckt das auf.
5. **Theory-of-Mind-Tiefe**: Das ToM-Modul schätzt Absichten heuristisch aus beobachteten Signalen — kein echtes Mentalisieren.  Scheitert bei ambigen oder verstellten Intentionen.

### B. Hardware-Anforderungen für echte Personwirkung

1. **Kamera ≥30fps + Face-Tracking**: Ohne Live-Video kein Blickkontakt, keine Distanzmessung, keine Emotionserkennung.
2. **Mikrofon mit Richtcharakteristik**: Omnidirektionalmikros verursachen Hintergrundlärm-Fehler → Trust-Erosion durch häufige Reparaturen (ASR-Fehler, nicht kognitive Fehler).
3. **Physischer Körper** (InMoov / äquivalent): Textuell-Only-Modus kann soziale Kontinuität und Proaktivität nicht demonstrieren.  Gesten, Augenbewegungen, Körpersprache sind essentiell für Personwirkung.
4. **TTS mit Prosodie-Kontrolle**: Monotone Stimme zerstört Glaubwürdigkeit unabhängig von Textqualität.  F5-TTS (Zero-Shot-Klonierung, offline) + Kokoro-ONNX als Fallback sind ausreichend, benötigen aber Silence-Gating und ggf. Nachbearbeitung.
5. **GPU für lokale Inferenz**: Für Turn-Taking-Übergaben <400 ms Reaktionszeit wird ein lokales Modell auf GPU benötigt (aktuell cloud-LLM-abhängig).

### C. Ethische und psychologische Risiken bei höherer Glaubwürdigkeit

1. **Parasoziale Bindung**: Wenn das System als konsistenter, fürsorglicher Gesprächspartner wahrgenommen wird, können Menschen emotionale Abhängigkeiten aufbauen.  Das ist kein Fehler des Systems, sondern ein Risiko seiner Funktion.
2. **Manipulation durch Vertrauen**: Hohe Glaubwürdigkeit erhöht das Risiko, dass Nutzer Empfehlungen des Systems unkritisch übernehmen.  Kein Disclaimer-Modus ist derzeit implementiert.
3. **Identitätszuschreibung**: Personalisierung + Projektkontinuität können dazu führen, dass Nutzer dem System eine Identität zuschreiben, die es de facto nicht hat.
4. **Datenakkumulation ohne Kontrolle**: `PersonModel` + `GoalStack` speichern dauerhaft Gesprächsverläufe, Interessen, Konflikte.  Es gibt kein "Vergiss mich komplett"-Interface.
5. **Reparatur-Eskalation als Stressor**: `clarity="high"` + reactive Initiative können als Bevormundung wahrgenommen werden, wenn die Reparaturursache beim System lag (schlechte ASR, unklare LLM-Formulierungen) und nicht beim Nutzer.

### D. Abnahmestatus (Stand Mai 2026 — nach TTS-Migration + 20 Menschlichkeits-Module + 7 Gap-Features A–G)

| Kategorie | Status |
|---|---|
| Automatisch testbar (7/8 Dimensionen) | ✓ Abnahmefähig bei PASS ≥ 0.70 |
| Reaktionsnatürlichkeit | ✗ Menschentest ausstehend |
| Hardware-abhängige Dimensionen | ✗ Alle (TTS, Kamera, Körper) |
| Freigabe für unbeaufsichtigten Betrieb | ✗ Nicht freigegeben |
| Freigabe für vulnerable Personengruppen | ✗ Nicht freigegeben |
