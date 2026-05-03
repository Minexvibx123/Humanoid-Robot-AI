"""main.py — AI Consciousness System launcher.

Modes
-----
  (default)   Launch the DearPyGui live monitor (GUI).
  --headless  Run brain without any GUI; print status to stdout.
  --eval      Run eval_harness.py test suite and exit.
  --soak      Run soak_harness.py long-run stress test and exit.
    --postfix   Run post_fix_harness.py causal validation and exit.

Sensor flags
------------
  --nocam   Disable camera
  --nomic   Disable microphone
  --noweb   Disable web sensor
  --web-interval SEC  Web fetch interval (default 15 s)
  --camera IDX        Camera device index (default 0)

Soak options
------------
  --soak-scenario NAME   Scenario to run (default: stability)
  --soak-ticks N         Ticks to run (default: 10000)

Post-fix options
----------------
    --postfix-ticks N      Pressure-run ticks for post-fix validation (default: 1200)
    --postfix-seed N       Seed for reproducible post-fix runs (default: 1337)
    --postfix-export DIR   Artifact directory for JSONL/summaries
    --postfix-markdown     Also export a Markdown summary
    --postfix-allow-short  Allow sub-1200-tick debug runs

Other
-----
  --lang {de,en}          AI output language (default de)
  --status-interval SEC   [headless] Print interval (default 5 s)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# ── Auto-activate project venv ───────────────────────────────────────────────
_VENV_PYTHON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".venv",
    "Scripts",
    "python.exe",
)
if os.path.exists(_VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(
    _VENV_PYTHON
):
    import subprocess

    raise SystemExit(subprocess.call([_VENV_PYTHON] + sys.argv))

# ── Load .env (TTS config, voice WAV path, etc.) ─────────────────────────────
try:
    import importlib

    _setup_voice = importlib.import_module("_setup_voice")
    _load_env = getattr(_setup_voice, "load_env", None)
    if callable(_load_env):
        _load_env()
    else:
        raise ImportError("_setup_voice.load_env missing")
except Exception:
    try:
        from dotenv import load_dotenv as _load_dotenv

        _load_dotenv()
    except Exception:
        pass


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="AI Consciousness System — Neural Brain + DearPyGui Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--headless",
        action="store_true",
        help="Run brain without GUI; print periodic status to stdout",
    )
    mode.add_argument(
        "--eval",
        action="store_true",
        help="Run eval_harness.py test suite and exit",
    )
    mode.add_argument(
        "--soak",
        action="store_true",
        help="Run soak_harness.py long-run tests and exit",
    )
    mode.add_argument(
        "--postfix",
        action="store_true",
        help="Run post_fix_harness.py causal validation and exit",
    )
    p.add_argument(
        "--soak-scenario",
        default="stability",
        metavar="NAME",
        help="Soak scenario to run (default: stability)",
    )
    p.add_argument(
        "--soak-ticks",
        type=int,
        default=10000,
        dest="soak_ticks",
        metavar="N",
        help="Ticks to run in soak test (default: 10000)",
    )
    p.add_argument(
        "--postfix-ticks",
        type=int,
        default=1200,
        dest="postfix_ticks",
        metavar="N",
        help="Ticks to run in post-fix pressure validation (default: 1200)",
    )
    p.add_argument(
        "--postfix-seed",
        type=int,
        default=1337,
        dest="postfix_seed",
        metavar="N",
        help="Seed for reproducible post-fix validation (default: 1337)",
    )
    p.add_argument(
        "--postfix-export",
        default="postfix_runs",
        dest="postfix_export",
        metavar="DIR",
        help="Artifact directory for post-fix JSONL and summaries (default: postfix_runs)",
    )
    p.add_argument(
        "--postfix-markdown",
        action="store_true",
        dest="postfix_markdown",
        help="Also export a Markdown summary for post-fix validation",
    )
    p.add_argument(
        "--postfix-allow-short",
        action="store_true",
        dest="postfix_allow_short",
        help="Allow sub-1200-tick debug runs for the post-fix harness",
    )
    p.add_argument("--nocam", action="store_true", help="Disable camera sensor")
    p.add_argument("--nomic", action="store_true", help="Disable microphone sensor")
    p.add_argument("--noweb", action="store_true", help="Disable web sensor")
    p.add_argument(
        "--web-interval",
        type=float,
        default=15.0,
        dest="web_interval",
        metavar="SEC",
        help="Web fetch interval in seconds (default: 15)",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="IDX",
        help="Camera device index (default: 0)",
    )
    p.add_argument(
        "--lang",
        choices=["de", "en"],
        default="de",
        help="Language for AI output (default: de)",
    )
    p.add_argument(
        "--status-interval",
        type=float,
        default=5.0,
        dest="status_interval",
        metavar="SEC",
        help="[headless] Status print interval in seconds (default: 5)",
    )
    return p


def _run_eval() -> int:
    """Run evaluation harness in-process and return exit code."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "eval_harness",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_harness.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["eval_harness"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    # Temporarily replace argv so eval_harness.main() doesn't pick up --eval
    _saved_argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        mod.main()
    finally:
        sys.argv = _saved_argv
    return 0


def _run_soak(args: argparse.Namespace) -> int:
    """Run soak_harness scenarios and return exit code."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "soak_harness",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "soak_harness.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["soak_harness"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _saved_argv, sys.argv = sys.argv, [
        sys.argv[0],
        "--scenario",
        args.soak_scenario,
        "--ticks",
        str(args.soak_ticks),
    ]
    try:
        mod.main()
    finally:
        sys.argv = _saved_argv
    return 0


def _run_postfix(args: argparse.Namespace) -> int:
    """Run post-fix causal validation harness and return exit code."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "post_fix_harness",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "post_fix_harness.py"),
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["post_fix_harness"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    _saved_argv, sys.argv = sys.argv, [
        sys.argv[0],
        "--ticks",
        str(args.postfix_ticks),
        "--seed",
        str(args.postfix_seed),
        "--export-dir",
        str(args.postfix_export),
    ]
    if args.postfix_markdown:
        sys.argv.append("--markdown")
    if args.postfix_allow_short:
        sys.argv.append("--allow-short")
    try:
        mod.main()
    finally:
        sys.argv = _saved_argv
    return 0


def _run_headless(args: argparse.Namespace) -> int:
    """Run brain without GUI, printing periodic status to stdout."""
    import signal
    import time

    from brain import Brain

    print("=== AI Consciousness — Headless Mode ===")
    print(
        f"  cam={not args.nocam}  mic={not args.nomic}  "
        f"web={not args.noweb}  lang={args.lang}"
    )

    brain = Brain(
        camera_index=args.camera,
        use_camera=not args.nocam,
        use_microphone=not args.nomic,
        use_web=not args.noweb,
        web_fetch_interval=args.web_interval,
    )
    brain._speech.language = "en-US" if args.lang == "en" else "de-DE"
    brain._consciousness.lang._lang = args.lang
    brain._speech_output.set_language(args.lang)
    brain.start()

    _stop = [False]

    def _sig(*_) -> None:
        _stop[0] = True

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    last_print = 0.0
    try:
        while not _stop[0]:
            time.sleep(0.05)
            now = time.time()
            if now - last_print >= args.status_interval:
                last_print = now
                cs = brain._consciousness
                cont = cs.continuity
                em = brain.emotion_state
                st = brain.consciousness_state
                auto = cs.autobiography
                print(
                    f"[t={brain.tick_count:,}]  goal={st.goal}  emo={em.describe()}\n"
                    f"  continuity : {cont.describe()}\n"
                    f"  identity   : consistency={auto.identity_consistency:.3f}"
                    f"  guidelines={len(auto.guidelines)}\n"
                    f"  beliefs    : active={cs.belief_store.size}"
                    f"  quarantined={len(cs.belief_store.quarantined())}\n"
                    f"  meta gaps  : {cs.meta.gaps(4)}\n"
                )
    finally:
        print("Shutting down …")
        try:
            from persistence import save_brain

            n = save_brain(brain)
            print(f"Saved {n:,} synapses.")
        except Exception as exc:
            print(f"Save error: {exc}")
        brain.stop()
    return 0


def _run_gui(args: argparse.Namespace) -> int:
    from dpg_monitor import main as run_monitor

    return run_monitor(args)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = _build_parser()
    args = parser.parse_args()

    if args.eval:
        raise SystemExit(_run_eval())
    elif args.soak:
        raise SystemExit(_run_soak(args))
    elif args.postfix:
        raise SystemExit(_run_postfix(args))
    elif args.headless:
        raise SystemExit(_run_headless(args))
    else:
        raise SystemExit(_run_gui(args))
