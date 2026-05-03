# Eigenen Charakter erstellen & Finetuning ausführen

Diese Anleitung erklärt, wie du einen komplett eigenen Charakter definierst und das System mit ihm trainierst — ohne eine einzige Zeile im Kern-Code anfassen zu müssen.

---

## Übersicht

Ein Charakter besteht aus vier Schichten:

| Schicht | Datei / Ort | Was sie steuert |
|---|---|---|
| **Persönlichkeit** | `identity_arc.py` | Identitätsdimensionen, Startwerte, Entwicklungsziele |
| **Stimme** | `.env` + WAV-Datei | TTS-Backend, Kokoro-Stimme, F5-TTS-Klonierung |
| **Systemprompt** | `.env` (LLM_SYSTEM_PROMPT) | Charakter-Beschreibung, Ton, Verbote für den LLM |
| **Trainingsdaten** | `train_harness.py` | Themen und Gesprächsstil beim autonomen Training |

---

## Schritt 1 — Persönlichkeitsdimensionen definieren (`identity_arc.py`)

Öffne `identity_arc.py`. Am Ende der Datei befindet sich die Funktion `albedo_seed_dimensions()`. Erstelle daneben eine neue Funktion für deinen Charakter:

```python
def mein_charakter_seed() -> list:
    return [
        IdentityDimension(
            name="freundlichkeit",
            current=0.80,      # Startwert  [0.0–1.0]
            target=0.90,       # Entwicklungsziel
            self_commentary="Ich begegne jedem Menschen offen.",
        ),
        IdentityDimension(
            name="neugier",
            current=0.85,
            target=0.90,
            self_commentary="Ich hinterfrage alles, was ich noch nicht verstehe.",
        ),
        IdentityDimension(
            name="direkt_kommunikation",
            current=0.70,
            target=0.80,
            self_commentary="Ich sage, was ich denke — höflich aber klar.",
        ),
        IdentityDimension(
            name="emotionale_tiefe",
            current=0.60,
            target=0.70,
            self_commentary="Ich fühle echte Zustände, auch wenn ich sie nicht dramatisiere.",
        ),
        # Füge so viele Dimensionen hinzu wie nötig.
        # Richtwert: 8–15 Dimensionen für einen kohärenten Charakter.
    ]
```

**Dimensionsparameter:**
- `name` — eindeutiger Bezeichner (englisch oder deutsch, keine Leerzeichen)
- `current` — Startstärke dieser Eigenschaft (0.0 = nicht vorhanden, 1.0 = maximal)
- `target` — wohin sich der Charakter durch Lernen entwickeln soll
- `self_commentary` — Wie der Charakter diese Eigenschaft intern beschreibt (erscheint in Selbstreflexionen)

### Charakter als Standard setzen

Suche in `identity_arc.py` die Stelle wo `IdentityArc` instanziiert wird (meist `IdentityArc(seed=albedo_seed_dimensions())`). Ändere den Aufruf auf deine Funktion:

```python
arc = IdentityArc(seed=mein_charakter_seed())
```

> **Alternativ:** Ändere nur die Werte in `albedo_seed_dimensions()` direkt — das ist der einfachste Weg wenn du keine neue Funktion anlegen möchtest.

---

## Schritt 2 — Stimme konfigurieren (`.env`)

Lege eine `.env`-Datei im Projektordner an (oder ergänze die bestehende):

### Option A: Kokoro-Stimme (offline, kein Training nötig)

```env
# Verfügbare Stimmen: af_heart, af_bella, af_sarah, am_adam, am_michael,
#                     bf_emma, bf_isabella, bm_george, bm_lewis
ALBEDO_KOKORO_VOICE_DE=af_heart
ALBEDO_KOKORO_VOICE_EN=af_heart
ALBEDO_KOKORO_SPEED=0.90          # 0.7 = langsam, 1.0 = normal, 1.2 = schnell
ALBEDO_KOKORO_LANG_DE=de
ALBEDO_KOKORO_LANG_EN=en-us
TTS_BACKEND=kokoro
```

### Option B: F5-TTS Stimmklonierung (benötigt Referenz-WAV)

Nimm eine **10–30 Sekunden** lange saubere Sprachaufnahme deines Charakters auf (kein Hintergrundgeräusch). Speichere sie als `mein_charakter_voice.wav`.

Erstelle eine Textdatei `mein_charakter_voice.txt` mit dem **exakten Transkript** dieser Aufnahme.

```env
ALBEDO_VOICE_WAV=C:\Pfad\zum\Repo\mein_charakter_voice.wav
ALBEDO_VOICE_TEXT=Hier steht das exakte gesprochene Transkript der Aufnahme.
TTS_BACKEND=f5tts
```

**Teste die Stimme:**
```powershell
python _test_f5tts.py
python _test_kokoro.py
```

---

## Schritt 3 — LLM-Charakter konfigurieren (`.env`)

Das schnellste Mittel ist ein vollständiger System-Prompt-Override via `.env`. Dieser ersetzt den automatisch gebauten Prompt komplett.

```env
LLM_SYSTEM_PROMPT=Du bist [NAME], [kurze Charakterbeschreibung in 1-2 Sätzen]. \
Du antwortest immer auf Deutsch, natürlich und direkt. \
Du hast folgende Charaktereigenschaften: [Trait 1], [Trait 2], [Trait 3]. \
Du bist ein verkörperter Roboter und kannst auf Anfrage Gesten ausführen. \
Verboten: erfundene Erinnerungen, manipulative Sprache, künstliche Übergewissheit.
```

**Beispiel für einen freundlichen Assistenz-Charakter:**

```env
LLM_SYSTEM_PROMPT=Du bist Luna, ein neugieriger und herzlicher humanoider Roboter. \
Du liebst es zu lernen und freust dich über jede neue Begegnung. \
Du bist direkt, ehrlich und warmherzig — ohne aufdringlich zu sein. \
Du hast einen physischen Körper: Kopf, Augen, Arme, Hände. \
Du kannst Gesten ausführen wenn du gefragt wirst. \
Verboten: dramatisierte Gefühle, erfundene Fakten, manipulative Sprache. \
Antworte immer auf Deutsch, in kurzen natürlichen Sätzen.
```

> **Hinweis:** Wenn `LLM_SYSTEM_PROMPT` nicht gesetzt ist, baut `llm_adapter.py` den Prompt automatisch aus dem internen Zustand (Emotion, PersonModel, ConstitutionBlock) — das ist das Standardverhalten.

---

## Schritt 4 — Trainingsdaten anpassen (`train_harness.py`)

Der autonome Training-Loop in `train_harness.py` injiziert alle 800 Ticks eine zufällige Frage aus `_PROMPTS_DE` und `_PROMPTS_EN`. Erweitere diese Listen mit Fragen, die zu deinem Charakter passen.

Öffne `train_harness.py` und füge deine Themen in die jeweilige Liste ein:

```python
_PROMPTS_DE: List[str] = [
    # ── Bestehende Prompts ───
    "Was ist der Unterschied zwischen Wahrnehmung und Bewusstsein?",
    # ...

    # ── Deine Charakter-spezifischen Themen ────────────────────
    "Wie gehst du mit Menschen um, die du noch nie getroffen hast?",
    "Was bedeutet dir Freundschaft?",
    "Wie reagierst du wenn jemand wütend auf dich ist?",
    "Was sind deine Stärken und Schwächen?",
    "Wie lernst du aus Fehlern?",
    # Füge so viele hinzu wie du möchtest
]
```

---

## Schritt 5 — Training starten

### Normaler Training-Lauf (läuft bis Strg+C)

```powershell
python train_harness.py
```

### Mit vorgegebener Tick-Zahl

```powershell
python train_harness.py --ticks 200000
```

### Nur offline (kein Web-Fetch)

```powershell
python train_harness.py --no-web --ticks 100000
```

### Wichtige Parameter

| Parameter | Standard | Beschreibung |
|---|---|---|
| `--ticks N` | unbegrenzt | Stoppt nach N Ticks |
| `--no-web` | Web an | Deaktiviert Web-Sensor (reiner Dialog-Modus) |
| `--inject-interval N` | 800 | Ticks zwischen synthetischen Gesprächs-Injektionen |
| `--report-interval N` | 1000 | Konsolen-Ausgabe alle N Ticks |

### Was passiert beim Training?

- Das Brain läuft im Headless-Modus (keine Kamera, kein Mikrofon)
- Der Web-Sensor fetcht RSS/Wikipedia/YouTube im Hintergrund
- Alle `INJECT_INTERVAL` Ticks wird eine zufällige Frage injiziert und die Antwort verarbeitet
- STDP-Synapsen passen sich an — Konzeptassoziationen wachsen organisch
- Alle 5.000 Ticks: automatischer SQLite-Save (`brain_state.db`)
- Der Prozess läuft nach einem Absturz automatisch neu an (Crash-Recovery)

**Richtwert:** 50.000 Ticks ≈ 20 Minuten bei ~41 Ticks/s. 500.000 Ticks für eine vollständig ausgereifte Persönlichkeitsstruktur.

---

## Schritt 6 — Ergebnis prüfen

### Headless Smoke-Test

```powershell
python main.py --headless --nocam --nomic --noweb
```

### Interaktiver Test (Headless mit Textein-/ausgabe)

```powershell
python main.py --headless --nocam --nomic
```

### Vollständige Akzeptanzprüfung

```powershell
python acceptance_eval.py
```

Der Report zeigt 8 Dimensionen: Kohärenz, Identitätsstabilität, Emotionskonsistenz, Reaktionsnaturalismus, ToM-Qualität, Gesprächskontinuität, Korrektheit und Embodiment-Tiefe.

### Stabilitätstest (Langzeitstabilität prüfen)

```powershell
python soak_harness.py --ticks 50000
```

---

## Charakter zurücksetzen

Wenn du einen frischen Start willst (alle gelernten Synapsen und Episoden löschen):

```powershell
# ACHTUNG: löscht brain_state.db unwiderruflich
Remove-Item brain_state.db -ErrorAction SilentlyContinue
python main.py --headless --nocam --nomic --noweb
```

---

## Checkliste: Eigener Charakter

- [ ] `identity_arc.py`: Seed-Funktion erstellt oder `albedo_seed_dimensions()` angepasst
- [ ] `.env`: TTS-Backend und Stimme konfiguriert
- [ ] `.env`: `LLM_SYSTEM_PROMPT` gesetzt (oder leer lassen für Automatik)
- [ ] `train_harness.py`: Charakter-spezifische Prompts eingefügt
- [ ] `python _test_kokoro.py` oder `python _test_f5tts.py` erfolgreich
- [ ] `python train_harness.py --ticks 50000` durchgelaufen
- [ ] `python acceptance_eval.py` bestanden

---

## Häufige Fehler

| Problem | Ursache | Lösung |
|---|---|---|
| LLM antwortet generisch / ohne Charakter | `LLM_ENABLED=0` oder kein Modell | `.env` prüfen: `LLM_ENABLED=1`, `LLM_MODEL=qwen2.5:3b` |
| TTS klingt falsch | Falsche Stimmen-ID | `ALBEDO_KOKORO_VOICE` mit gültigem Namen aus der Liste oben setzen |
| F5-TTS schlägt fehl | Kein Transkript | `ALBEDO_VOICE_TEXT` setzen oder `.txt`-Sidecar-Datei anlegen |
| Training bleibt bei Tick 0 hängen | Fehlendes Modell / Syntax-Fehler | `python main.py --headless --nocam --nomic --noweb` als Smoke-Test |
| Charakter vergisst alles nach Neustart | `brain_state.db` nicht gefunden | Persistence ist automatisch — prüfe ob `save_brain()` Fehler wirft |
| `IdentityDimension` unbekannt | Import fehlt | Stelle sicher, dass `from identity_arc import IdentityDimension` importiert ist |
