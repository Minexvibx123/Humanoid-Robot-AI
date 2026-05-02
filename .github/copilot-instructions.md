# Project Guidelines

## Code Style
- Follow the existing Python style in the repo: type hints where already used, dataclasses for structured state, and concise module docstrings that explain the subsystem boundary.
- Keep changes localized. This codebase is highly coupled, so avoid broad refactors unless the task explicitly requires them.
- Preserve the existing naming style, including mixed English/German domain terms when working in established modules.
- Prefer extending existing tick-driven components over adding parallel control flows or background workers.

## Architecture
- Treat [main.py](../main.py) as the entry point for run modes only. It dispatches to GUI, headless, eval, or soak paths and will re-exec into `.venv\Scripts\python.exe` automatically when that interpreter exists.
- Treat [brain.py](../brain.py) as the central orchestrator. It wires sensors, regions, emotion, consciousness, robotics, and persistence into a single tick loop.
- Treat [consciousness.py](../consciousness.py) as a tightly integrated subsystem. Changes there can affect goal selection, belief handling, narrative state, social behavior, and output generation.
- Keep the test boundary clear:
  - [eval_harness.py](../eval_harness.py) is a lightweight scenario harness built around `MiniWorld` and does not instantiate the full `Brain`.
  - [soak_harness.py](../soak_harness.py) runs the real brain for long stability and persistence checks.

## Build And Test
- Install dependencies with `pip install -r requirements.txt`.
- Default launch: `python main.py`
- Safe headless smoke test: `python main.py --headless --nocam --nomic --noweb`
- Scenario evals: `python main.py --eval` or `python eval_harness.py`
- Long-run stability: `python main.py --soak` or `python soak_harness.py --ticks 50000`
- Use the smallest meaningful validation for the area you changed. Prefer `eval_harness.py` for social/task logic and `soak_harness.py` for tick-loop, persistence, or stability changes.

## Conventions
- This project is tick-driven. New behavior should usually integrate into an existing `tick()` path instead of introducing ad hoc polling or blocking loops.
- Sensor and hardware dependencies are optional in development. Prefer disabled-sensor modes for automated checks unless the task is specifically about camera, microphone, web, or robot hardware.
- Treat persistence and telemetry as first-class behavior, not debug extras. Changes that affect long-running state should avoid breaking SQLite save/load or JSONL metric export.
- Keep voice/TTS code resilient to missing credentials or models. `.env` and cached voice/model files are optional and are ignored by git.
- Avoid changing global tuning constants in [brain.py](../brain.py) or [consciousness.py](../consciousness.py) without running an end-to-end validation path.

## Environment Notes
- Secrets live in `.env`; do not hardcode API keys or voice IDs.
- The repo expects large optional assets under `models/` and `models/kokoro/`. Do not commit generated caches.
- For hardware-facing changes, keep graceful fallback behavior when serial devices, camera, microphone, or TTS backends are unavailable.

## References
- See [README.md](../README.md) for the system overview and subsystem map.