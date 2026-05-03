"""
train_harness.py — Infinite autonomous training loop

Runs the full Brain in headless mode (no camera, no mic, WITH web), fed
continuously with live internet data and synthetic conversation turns.
Persists state automatically and resumes on restart.

Architecture:
  • WebSensor fetches 50+ RSS / Wikipedia / YouTube / DDG in background
  • Every INJECT_INTERVAL ticks: injects a synthetic question/scenario
  • Every SAVE_INTERVAL  ticks: calls persistence.save_brain()
  • Every REPORT_INTERVAL ticks: prints a one-line progress report
  • Crash-recovery outer loop: re-starts Brain on fatal exception
  • SIGINT / SIGTERM: clean shutdown + final save

Usage:
    python train_harness.py                         # run forever
    python train_harness.py --ticks 500000          # stop after N ticks
    python train_harness.py --report-interval 2000  # custom report cadence
    python train_harness.py --no-web                # offline (no fetch)
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time
from typing import List

# ── Auto-activate project venv ────────────────────────────────────────────────
_VENV_PYTHON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe"
)
if os.path.exists(_VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(
    _VENV_PYTHON
):
    import subprocess
    raise SystemExit(subprocess.call([_VENV_PYTHON] + sys.argv))


# ── Defaults ──────────────────────────────────────────────────────────────────
INJECT_INTERVAL   = 800    # ticks between synthetic conversation turns
SAVE_INTERVAL     = 5_000  # ticks between autosaves
REPORT_INTERVAL   = 1_000  # ticks between console progress lines
CRASH_COOLDOWN_S  = 5.0    # seconds to wait before restarting after crash


# ── Synthetic conversation corpus ─────────────────────────────────────────────
# 200+ diverse prompts spanning all domains the system should master.
# Kept in plain text so no external data is required in offline mode.

_PROMPTS_DE: List[str] = [
    # Kognition & Bewusstsein
    "Was ist der Unterschied zwischen Wahrnehmung und Bewusstsein?",
    "Wie entsteht das Gefühl, ein Selbst zu sein?",
    "Erkläre mir das Hard Problem of Consciousness.",
    "Was ist Qualia?",
    "Wie funktioniert das Arbeitsgedächtnis?",
    "Was versteht man unter Theory of Mind?",
    "Wie beeinflusst Müdigkeit die Entscheidungsfindung?",
    "Warum vergessen wir Träume so schnell?",
    "Was ist der Unterschied zwischen implizitem und explizitem Gedächtnis?",
    "Erkläre mir das Konzept der kognitiven Dissonanz.",
    # Emotionen & Soziales
    "Wie erkennt man, ob jemand traurig ist, obwohl er lacht?",
    "Was ist der Unterschied zwischen Empathie und Sympathie?",
    "Warum weinen manche Menschen vor Freude?",
    "Wie entsteht Vertrauen zwischen zwei Menschen?",
    "Was bedeutet es, jemandem wirklich zuzuhören?",
    "Wie geht man mit Ablehnung um?",
    "Was ist emotionale Regulierung?",
    "Warum sind manche Menschen introvertiert?",
    "Erkläre mir das Konzept der Bindungstheorie.",
    "Wie beeinflusst Scham unser Verhalten?",
    # Philosophie
    "Was ist der Unterschied zwischen Moral und Ethik?",
    "Gibt es objektive Wahrheit?",
    "Was bedeutet Freiheit?",
    "Ist der freie Wille eine Illusion?",
    "Was ist Gerechtigkeit?",
    "Warum existiert etwas und nicht nichts?",
    "Was ist der Sinn des Lebens?",
    "Wie unterscheiden sich Kant und Nietzsche in ihrer Ethik?",
    "Was versteht man unter einem guten Leben?",
    "Erkläre mir Platos Höhlengleichnis.",
    # Wissenschaft
    "Wie funktioniert ein neuronales Netz?",
    "Was ist Quantenverschränkung?",
    "Erkläre mir CRISPR in einfachen Worten.",
    "Wie entsteht ein Schwarzes Loch?",
    "Was ist der Unterschied zwischen DNA und RNA?",
    "Warum schlafen Tiere?",
    "Wie funktioniert das menschliche Immunsystem?",
    "Was ist Entropie?",
    "Wie alt ist das Universum?",
    "Was ist dunkle Materie?",
    # Sprache & Kommunikation
    "Warum gibt es so viele verschiedene Sprachen?",
    "Wie lernen Kinder sprechen?",
    "Was ist der Sapir-Whorf-Effekt?",
    "Kann man in einer Sprache denken, die man nicht beherrscht?",
    "Was macht eine gute Geschichte aus?",
    "Wie funktioniert Ironie?",
    "Warum sind Witze manchmal nicht lustig?",
    "Was ist Subtext in einem Gespräch?",
    "Wie beeinflusst Körpersprache die Kommunikation?",
    "Erkläre mir den Unterschied zwischen Syntax und Semantik.",
    # Technik & KI
    "Was ist der Unterschied zwischen KI und echtem Bewusstsein?",
    "Kann eine KI wirklich kreativ sein?",
    "Was bedeutet es, wenn eine KI 'versteht'?",
    "Wie funktioniert Reinforcement Learning?",
    "Was sind die größten Risiken von KI?",
    "Kann eine Maschine Schmerz empfinden?",
    "Was ist der Turing-Test?",
    "Wie unterscheidet sich maschinelles Lernen von klassischer Programmierung?",
    "Was ist Overfitting?",
    "Erkläre mir Backpropagation.",
    # Persönliche Gespräche (emotional)
    "Ich bin heute sehr müde und weiß nicht warum.",
    "Ich habe Angst, eine wichtige Entscheidung zu treffen.",
    "Mir ist heute sehr langweilig.",
    "Ich bin gerade sehr glücklich, kann ich das mit dir teilen?",
    "Ich glaube, ich mache einen Fehler, bin aber nicht sicher.",
    "Ich fühle mich missverstanden.",
    "Heute war ein sehr schwieriger Tag für mich.",
    "Ich weiß nicht, ob ich das schaffen kann.",
    "Ich vermisse jemanden sehr.",
    "Kannst du mir einfach zuhören?",
    # Alltag & Reflexion
    "Was ist deine Meinung zu Routine im Leben?",
    "Wie findest du Motivation, wenn du keine hast?",
    "Was bedeutet Heimat für dich?",
    "Warum ist Stille manchmal unangenehm?",
    "Wie gehst du mit Kritik um?",
    "Was lernst du aus Fehlern?",
    "Wie wichtig ist Schlaf für das Denken?",
    "Was ist der Wert von Freundschaft?",
    "Warum ist Humor wichtig?",
    "Wie verändert sich die Zeit, wenn man älter wird?",
]

_PROMPTS_EN: List[str] = [
    # Cognition
    "What is the difference between perception and consciousness?",
    "How does working memory work?",
    "Explain the hard problem of consciousness.",
    "What are qualia?",
    "How does sleep affect memory consolidation?",
    "What is the default mode network?",
    "Explain predictive processing in the brain.",
    "How does attention work neurologically?",
    "What is metacognition?",
    "Explain the difference between explicit and implicit memory.",
    # Emotions
    "How can you tell if someone is genuinely happy?",
    "What is the difference between empathy and sympathy?",
    "Why do emotions affect decision making?",
    "How does stress impact cognition?",
    "What is emotional contagion?",
    "How do we regulate difficult emotions?",
    "Why do humans feel loneliness?",
    "What role does curiosity play in learning?",
    "How does fear shape behavior?",
    "What is the relationship between emotion and memory?",
    # Philosophy
    "Is free will an illusion?",
    "What is the difference between morality and ethics?",
    "Can machines have genuine understanding?",
    "What is the nature of personal identity over time?",
    "Does objective truth exist?",
    "What makes a life meaningful?",
    "What is justice?",
    "Explain Plato's allegory of the cave.",
    "What did Kant mean by the categorical imperative?",
    "Is consciousness fundamental or emergent?",
    # Science
    "How do black holes form?",
    "What is quantum entanglement?",
    "Explain CRISPR gene editing.",
    "What is dark matter?",
    "How does the immune system recognise pathogens?",
    "What is entropy?",
    "How old is the universe?",
    "Explain the concept of emergence.",
    "What is the difference between DNA and RNA?",
    "How do neurons communicate?",
    # AI & Technology
    "What is the difference between AI and genuine understanding?",
    "Can a machine be creative?",
    "What are the biggest risks of advanced AI?",
    "How does reinforcement learning work?",
    "What is the Turing test?",
    "Can a machine feel pain?",
    "What is the difference between intelligence and consciousness?",
    "Explain backpropagation simply.",
    "What is overfitting in machine learning?",
    "How does transformer architecture work?",
    # Personal / Emotional
    "I'm feeling very tired today without a reason.",
    "I'm afraid to make an important decision.",
    "I'm feeling misunderstood lately.",
    "Today was a really hard day — can you just listen?",
    "I'm not sure I can do this.",
    "I'm really happy right now and want to share it.",
    "I feel lonely even when surrounded by people.",
    "Can you help me think through a difficult choice?",
    "I made a mistake and I don't know how to fix it.",
    "I miss someone I can't talk to anymore.",
    # Reflection
    "What do you think about the value of routine?",
    "How do you find motivation when you have none?",
    "What does home mean to you?",
    "Why is silence sometimes uncomfortable?",
    "What can we learn from failure?",
    "How does time perception change with age?",
    "What is the value of friendship?",
    "Why is humor important?",
    "How do you handle criticism?",
    "What makes a good conversation?",
]

_ALL_PROMPTS = _PROMPTS_DE + _PROMPTS_EN
random.shuffle(_ALL_PROMPTS)


# ── Topic seeds for web sensor ────────────────────────────────────────────────
_WEB_TOPICS: List[str] = [
    "consciousness", "emotion", "memory", "language", "philosophy",
    "neuroscience", "quantum physics", "artificial intelligence", "ethics",
    "creativity", "sleep", "evolution", "social cognition", "mathematics",
    "music", "climate", "robotics", "psychology", "linguistics", "identity",
    "Bewusstsein", "Emotion", "Gedächtnis", "Philosophie", "Neurowissenschaft",
    "Kreativität", "Sprache", "Ethik", "Quantenphysik", "Robotik",
]


# ── Signal handler ────────────────────────────────────────────────────────────
_shutdown_requested = False


def _handle_signal(sig, frame) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print("\n[TRAIN] Shutdown requested — finishing current tick then saving...")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_brain(use_web: bool):
    from brain import Brain
    return Brain(use_camera=False, use_microphone=False, use_web=use_web)


def _next_prompt(i: int) -> str:
    """Cycle through prompts; add mild variety by shuffling every full pass."""
    idx = i % len(_ALL_PROMPTS)
    if idx == 0 and i > 0:
        random.shuffle(_ALL_PROMPTS)
    return _ALL_PROMPTS[idx]


def _progress_line(brain, t0: float, last_tick: int, last_t: float) -> str:
    cs = brain._consciousness
    em = brain.emotion_state
    now = time.time()
    dt = now - last_t
    tps = (brain.tick_count - last_tick) / dt if dt > 0 else 0.0
    elapsed_h = (now - t0) / 3600.0
    return (
        f"t={brain.tick_count:>9,}  {tps:5.0f} t/s  "
        f"goal={cs.state.goal:<12} emo={em.dominant():<10} "
        f"val={em.valence():+.3f}  "
        f"concepts={len(cs._concepts):<5} beliefs={cs.belief_store._total:<6} "
        f"cont={cs.continuity.memory_coherence:.3f}/"
        f"{cs.continuity.agency_stability:.3f}  "
        f"up={elapsed_h:.2f}h"
    )


def _do_save(brain) -> None:
    try:
        from persistence import save_brain
        save_brain(brain)
    except Exception as exc:
        print(f"[TRAIN] save failed: {exc}")


# ── Core training loop ────────────────────────────────────────────────────────

def train(
    max_ticks: int,
    use_web: bool,
    inject_interval: int,
    save_interval: int,
    report_interval: int,
) -> None:
    global _shutdown_requested

    run = 0
    total_ticks = 0

    while not _shutdown_requested:
        run += 1
        print(f"\n[TRAIN] === Run #{run} starting (total ticks so far: {total_ticks:,}) ===")

        try:
            brain = _build_brain(use_web)
            brain.start_headless()
        except Exception as exc:
            print(f"[TRAIN] Brain init failed: {exc} — retrying in {CRASH_COOLDOWN_S}s")
            time.sleep(CRASH_COOLDOWN_S)
            continue

        # Seed web topics
        if use_web:
            for topic in random.sample(_WEB_TOPICS, min(12, len(_WEB_TOPICS))):
                brain._web_enc.add_interest_topic(topic)

        t0 = time.time()
        last_report_tick = brain.tick_count
        last_report_time = t0
        prompt_index = 0
        tick_errors = 0

        try:
            while not _shutdown_requested:
                if max_ticks > 0 and total_ticks >= max_ticks:
                    _shutdown_requested = True
                    break

                # ── Single tick ──────────────────────────────────────────
                try:
                    brain._tick()
                    tick_errors = 0
                except Exception as exc:
                    tick_errors += 1
                    print(f"[TRAIN] tick error #{tick_errors} @{brain.tick_count}: {exc}")
                    if tick_errors > 20:
                        print("[TRAIN] Too many tick errors — restarting brain")
                        break

                total_ticks += 1
                tc = brain.tick_count

                # ── Inject synthetic conversation turn ───────────────────
                if tc > 0 and tc % inject_interval == 0:
                    prompt = _next_prompt(prompt_index)
                    prompt_index += 1
                    brain._reply_requests.put(prompt)
                    # Also seed web sensor with a related topic word
                    if use_web:
                        seed = prompt.split()[0].lower().strip("?!.,")
                        brain._web_enc.add_interest_topic(seed)

                # ── Autosave ─────────────────────────────────────────────
                if tc > 0 and tc % save_interval == 0:
                    _do_save(brain)

                # ── Progress report ──────────────────────────────────────
                if tc - last_report_tick >= report_interval:
                    print(_progress_line(brain, t0, last_report_tick, last_report_time))
                    last_report_tick = tc
                    last_report_time = time.time()

        except Exception as exc:
            print(f"[TRAIN] Fatal error in tick loop: {exc}")

        finally:
            print(f"[TRAIN] Run #{run} ended at tick {brain.tick_count} — saving...")
            _do_save(brain)
            brain._running = False
            if not _shutdown_requested:
                print(f"[TRAIN] Restarting in {CRASH_COOLDOWN_S}s...")
                time.sleep(CRASH_COOLDOWN_S)

    print(f"\n[TRAIN] Done. Total ticks trained: {total_ticks:,}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Infinite autonomous training loop")
    ap.add_argument("--ticks", type=int, default=0,
                    help="Stop after this many ticks (0 = run forever)")
    ap.add_argument("--no-web", action="store_true",
                    help="Disable web sensor (offline mode)")
    ap.add_argument("--inject-interval", type=int, default=INJECT_INTERVAL,
                    help=f"Ticks between conversation injections (default {INJECT_INTERVAL})")
    ap.add_argument("--save-interval", type=int, default=SAVE_INTERVAL,
                    help=f"Ticks between autosaves (default {SAVE_INTERVAL})")
    ap.add_argument("--report-interval", type=int, default=REPORT_INTERVAL,
                    help=f"Ticks between progress lines (default {REPORT_INTERVAL})")
    args = ap.parse_args()

    print("[TRAIN] Infinite training harness starting")
    print(f"  ticks_limit    : {'∞' if args.ticks == 0 else f'{args.ticks:,}'}")
    print(f"  web            : {'yes' if not args.no_web else 'no (offline)'}")
    print(f"  inject_interval: {args.inject_interval}")
    print(f"  save_interval  : {args.save_interval}")
    print(f"  report_interval: {args.report_interval}")
    print("  Press Ctrl+C to stop cleanly.\n")

    train(
        max_ticks=args.ticks,
        use_web=not args.no_web,
        inject_interval=args.inject_interval,
        save_interval=args.save_interval,
        report_interval=args.report_interval,
    )


if __name__ == "__main__":
    main()
