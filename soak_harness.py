"""
soak_harness.py — Full-Brain Long-Run Soak Tests

Unlike eval_harness.py (which uses MiniWorld without Brain/Consciousness),
this module instantiates the REAL Brain with all subsystems and runs
extended sessions (50k–200k ticks) to verify:

  • Tick stability      — no crashes, no tick-thread death
  • Queue health        — reply queue doesn't grow unbounded
  • Continuity          — identity metrics stay above critical thresholds
  • Save/Load fidelity  — mid-run restart doesn't lose operative state
  • Sleep cycles        — correct tonic switching, no stuck sleep/wake
  • Metric export       — JSONL per-session output for benchmarking

Usage:
    python soak_harness.py                      # default 50k ticks
    python soak_harness.py --ticks 200000       # long run
    python soak_harness.py --scenario restart   # mid-run restart test
    python soak_harness.py --all                # all soak scenarios
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

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


# ─────────────────────────────────────────────────────────────
# Soak Metrics — collected every sample_interval ticks
# ─────────────────────────────────────────────────────────────


@dataclass
class SoakSample:
    """Snapshot taken at regular intervals during a soak run."""

    tick: int = 0
    wall_time_s: float = 0.0
    ticks_per_second: float = 0.0
    sleeping: bool = False
    ignition_count: int = 0
    concept_count: int = 0
    belief_count: int = 0
    quarantine_count: int = 0
    episode_count: int = 0
    guideline_count: int = 0
    identity_consistency: float = 0.0
    continuity_memory: float = 0.0
    continuity_agency: float = 0.0
    continuity_value: float = 0.0
    energy_reserve: float = 0.0
    integrity: float = 0.0
    goal: str = ""
    emotion_dominant: str = ""
    valence: float = 0.0
    reply_queue_size: int = 0
    sensorimotor_agency: float = 0.0
    sensorimotor_surprise: float = 0.0
    stream_size: int = 0
    veto_count: int = 0
    save_latency_ms: float = 0.0


@dataclass
class SoakResult:
    """Final outcome of a soak run."""

    scenario: str = ""
    target_ticks: int = 0
    actual_ticks: int = 0
    passed: bool = False
    wall_time_s: float = 0.0
    avg_ticks_per_second: float = 0.0
    min_continuity_memory: float = 1.0
    min_continuity_agency: float = 1.0
    min_continuity_value: float = 1.0
    max_reply_queue: int = 0
    sleep_cycles: int = 0
    total_ignitions: int = 0
    save_count: int = 0
    errors: List[str] = field(default_factory=list)
    samples: List[SoakSample] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{self.scenario}] {status}",
            f"  ticks: {self.actual_ticks:,} / {self.target_ticks:,} "
            f"({self.wall_time_s:.1f}s, {self.avg_ticks_per_second:.1f} t/s)",
            f"  continuity: mem={self.min_continuity_memory:.3f} "
            f"agency={self.min_continuity_agency:.3f} "
            f"value={self.min_continuity_value:.3f}",
            f"  ignitions={self.total_ignitions} sleep_cycles={self.sleep_cycles} "
            f"saves={self.save_count}",
            f"  max_reply_queue={self.max_reply_queue}",
        ]
        if self.errors:
            lines.append(f"  errors: {self.errors[:5]}")
        return "\n".join(lines)

    def export_jsonl(self, path: str) -> None:
        """Write per-sample JSONL for benchmarking."""
        with open(path, "w", encoding="utf-8") as f:
            # Header record
            header = {
                "type": "soak_header",
                "scenario": self.scenario,
                "target_ticks": self.target_ticks,
                "actual_ticks": self.actual_ticks,
                "passed": self.passed,
                "wall_time_s": round(self.wall_time_s, 2),
                "avg_tps": round(self.avg_ticks_per_second, 1),
            }
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            # Sample records
            for s in self.samples:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
# Brain runner — headless tick loop with metric collection
# ─────────────────────────────────────────────────────────────


def _create_brain():
    """Create a headless Brain with all sensors disabled."""
    from brain import Brain

    return Brain(
        use_camera=False,
        use_microphone=False,
        use_web=False,
    )


def _sample(brain, t0: float, last_tick: int, last_time: float) -> SoakSample:
    """Take a metric snapshot from a live Brain."""
    now = time.time()
    dt = now - last_time
    tps = (brain.tick_count - last_tick) / dt if dt > 0 else 0.0
    cs = brain._consciousness
    em = brain.emotion_state
    # Estimate last save latency from autosave thread timing if available
    _save_lat = 0.0
    try:
        from persistence import last_save_report

        _rep = last_save_report()
        if _rep is not None:
            _save_lat = round((now - _rep.timestamp) * 1000.0, 1)
    except Exception:
        pass
    return SoakSample(
        tick=brain.tick_count,
        wall_time_s=round(now - t0, 2),
        ticks_per_second=round(tps, 1),
        sleeping=brain.sleeping,
        ignition_count=cs._ignition_count,
        concept_count=len(cs._concepts),
        belief_count=cs.belief_store._total,
        quarantine_count=len(cs.belief_store._quarantine),
        episode_count=len(cs.episodic._events),
        guideline_count=len(cs.autobiography._guidelines),
        identity_consistency=round(cs.autobiography._identity_consistency, 4),
        continuity_memory=round(cs.continuity.memory_coherence, 4),
        continuity_agency=round(cs.continuity.agency_stability, 4),
        continuity_value=round(cs.continuity.value_stability, 4),
        energy_reserve=round(cs.body.energy_reserve, 4),
        integrity=round(cs.body.integrity, 4),
        goal=cs.state.goal,
        emotion_dominant=em.dominant(),
        valence=round(em.valence(), 4),
        reply_queue_size=brain._reply_requests.qsize(),
        sensorimotor_agency=round(cs.sensorimotor.agency, 4),
        sensorimotor_surprise=round(cs.sensorimotor.surprise, 4),
        stream_size=len(cs.stream),
        veto_count=getattr(cs.goal_system, "_veto_count", 0),
        save_latency_ms=_save_lat,
    )


def _run_ticks(
    brain,
    n_ticks: int,
    result: SoakResult,
    sample_interval: int = 2000,
    inject_replies: bool = False,
) -> None:
    """Run n_ticks on the brain tick thread and collect samples."""
    t0 = time.time()
    last_sample_tick = brain.tick_count
    last_sample_time = t0
    prev_sleeping = False
    tick_errors = 0

    for i in range(n_ticks):
        # Inject reply requests periodically to stress test the queue
        if inject_replies and i > 0 and i % 5000 == 0:
            brain._reply_requests.put(
                f"Soak-Test Tick {brain.tick_count}: was denkst du?"
            )

        try:
            brain._tick()
        except Exception as exc:
            tick_errors += 1
            result.errors.append(
                f"tick_error@{brain.tick_count}: {type(exc).__name__}: {exc}"
            )
            if tick_errors > 10:
                result.errors.append("ABORT: too many tick errors")
                break

        # Detect sleep cycle transitions
        if brain.sleeping and not prev_sleeping:
            result.sleep_cycles += 1
        prev_sleeping = brain.sleeping

        # Sample metrics at intervals
        if brain.tick_count - last_sample_tick >= sample_interval:
            s = _sample(brain, t0, last_sample_tick, last_sample_time)
            result.samples.append(s)
            result.max_reply_queue = max(result.max_reply_queue, s.reply_queue_size)
            result.min_continuity_memory = min(
                result.min_continuity_memory, s.continuity_memory
            )
            result.min_continuity_agency = min(
                result.min_continuity_agency, s.continuity_agency
            )
            result.min_continuity_value = min(
                result.min_continuity_value, s.continuity_value
            )
            last_sample_tick = brain.tick_count
            last_sample_time = time.time()
            # Print progress
            print(
                f"  [{brain.tick_count:>8,}/{result.target_ticks:,}] "
                f"tps={s.ticks_per_second:.0f} goal={s.goal} "
                f"emo={s.emotion_dominant} sleep={s.sleeping} "
                f"cont={s.continuity_memory:.3f}/{s.continuity_agency:.3f}/{s.continuity_value:.3f}"
            )

    result.wall_time_s = time.time() - t0
    result.actual_ticks = brain.tick_count
    result.total_ignitions = brain._consciousness._ignition_count
    result.avg_ticks_per_second = (
        n_ticks / result.wall_time_s if result.wall_time_s > 0 else 0.0
    )


# ─────────────────────────────────────────────────────────────
# Soak Scenarios
# ─────────────────────────────────────────────────────────────


def scenario_stability(n_ticks: int = 50000) -> SoakResult:
    """
    Basic stability soak: run brain for n_ticks, check for crashes,
    queue growth, and continuity collapse.
    """
    result = SoakResult(scenario="stability", target_ticks=n_ticks)
    print(f"\n=== Soak: stability ({n_ticks:,} ticks) ===")

    brain = _create_brain()
    brain.start_headless()

    try:
        _run_ticks(brain, n_ticks, result, inject_replies=True)
    except Exception as exc:
        result.errors.append(f"fatal: {type(exc).__name__}: {exc}")
    finally:
        brain._running = False

    # Evaluate pass criteria
    checks = [
        result.actual_ticks >= n_ticks * 0.99,  # completed ≥99% of ticks
        result.min_continuity_memory > 0.2,  # no memory collapse
        result.min_continuity_agency > 0.1,  # no agency collapse
        result.max_reply_queue < 50,  # queue bounded
        len(result.errors) == 0,  # no tick errors
    ]
    result.passed = all(checks)
    if not result.passed and not result.errors:
        if result.actual_ticks < n_ticks * 0.99:
            result.errors.append(f"incomplete: {result.actual_ticks}/{n_ticks}")
        if result.min_continuity_memory <= 0.2:
            result.errors.append(
                f"continuity_memory_collapse: {result.min_continuity_memory:.3f}"
            )
        if result.min_continuity_agency <= 0.1:
            result.errors.append(
                f"continuity_agency_collapse: {result.min_continuity_agency:.3f}"
            )
        if result.max_reply_queue >= 50:
            result.errors.append(f"reply_queue_overflow: {result.max_reply_queue}")

    print(result.summary())
    return result


def scenario_restart(n_ticks: int = 20000) -> SoakResult:
    """
    Mid-run restart test: run half the ticks, save, create fresh brain,
    load, run remaining ticks. Verify operative state survives.
    """
    result = SoakResult(scenario="restart", target_ticks=n_ticks)
    half = n_ticks // 2
    print(f"\n=== Soak: restart ({n_ticks:,} ticks, restart at {half:,}) ===")

    db_path = os.path.join(tempfile.gettempdir(), "_soak_restart.db")

    # Phase 1: Run first half
    brain1 = _create_brain()
    brain1.start_headless()
    try:
        _run_ticks(brain1, half, result, sample_interval=2000)
    except Exception as exc:
        result.errors.append(f"phase1_fatal: {type(exc).__name__}: {exc}")
        brain1._running = False
        return result

    # Capture pre-save state
    cs1 = brain1._consciousness
    pre_concepts = len(cs1._concepts)
    pre_beliefs = cs1.belief_store._total
    pre_episodes = len(cs1.episodic._events)
    pre_identity = cs1.autobiography._identity_consistency
    pre_guidelines = len(cs1.autobiography._guidelines)
    pre_ignitions = cs1._ignition_count
    pre_tick = brain1.tick_count

    # Save
    from persistence import load_brain, save_brain

    t_save_start = time.time()
    try:
        os.remove(db_path)
    except FileNotFoundError:
        pass
    n_syn = save_brain(brain1, db_path=db_path)
    save_lat = (time.time() - t_save_start) * 1000
    result.save_count += 1
    print(f"  Saved {n_syn:,} synapses in {save_lat:.0f}ms")

    brain1._running = False

    # Phase 2: Load into fresh brain
    brain2 = _create_brain()
    brain2.start_headless()
    t_load_start = time.time()
    n_restored = load_brain(brain2, db_path=db_path)
    load_lat = (time.time() - t_load_start) * 1000
    print(f"  Loaded {n_restored:,} synapses in {load_lat:.0f}ms")

    # Verify state survived
    cs2 = brain2._consciousness
    restore_checks = []
    restore_checks.append(("tick_count", brain2.tick_count >= pre_tick * 0.9))
    restore_checks.append(("concepts", len(cs2._concepts) >= pre_concepts * 0.8))
    restore_checks.append(("beliefs", cs2.belief_store._total >= pre_beliefs * 0.8))
    restore_checks.append(("episodes", len(cs2.episodic._events) >= pre_episodes * 0.8))
    restore_checks.append(
        ("identity", abs(cs2.autobiography._identity_consistency - pre_identity) < 0.3)
    )
    restore_checks.append(("ignitions", cs2._ignition_count >= pre_ignitions * 0.5))

    for name, ok in restore_checks:
        if not ok:
            result.errors.append(f"restore_fail:{name}")

    # Phase 2: run remaining ticks
    remaining = n_ticks - half
    try:
        _run_ticks(brain2, remaining, result, sample_interval=2000, inject_replies=True)
    except Exception as exc:
        result.errors.append(f"phase2_fatal: {type(exc).__name__}: {exc}")
    finally:
        brain2._running = False

    # Cleanup
    try:
        os.remove(db_path)
    except Exception:
        pass

    # Evaluate pass criteria
    checks = [
        result.actual_ticks >= n_ticks * 0.9,
        all(ok for _, ok in restore_checks),
        result.min_continuity_memory > 0.15,
        len([e for e in result.errors if "fatal" in e]) == 0,
    ]
    result.passed = all(checks)
    print(result.summary())
    return result


def scenario_sleep_stress(n_ticks: int = 30000) -> SoakResult:
    """
    Sleep stress test: verify sleep/wake cycles occur correctly
    and the system doesn't get stuck in one mode.
    """
    result = SoakResult(scenario="sleep_stress", target_ticks=n_ticks)
    print(f"\n=== Soak: sleep_stress ({n_ticks:,} ticks) ===")

    brain = _create_brain()
    brain.start_headless()

    try:
        _run_ticks(brain, n_ticks, result, sample_interval=2000, inject_replies=True)
    except Exception as exc:
        result.errors.append(f"fatal: {type(exc).__name__}: {exc}")
    finally:
        brain._running = False

    # Check sleep cycle count — with 8000 tick cycles, 30k ticks → ~3-4 cycles
    expected_min_cycles = max(1, n_ticks // 10000)  # at least 1 per 10k ticks
    if result.sleep_cycles < expected_min_cycles:
        result.errors.append(
            f"too_few_sleep_cycles: {result.sleep_cycles} "
            f"(expected >= {expected_min_cycles})"
        )

    # Check we had wake samples (sleeping=False) AND sleep samples
    wake_samples = [s for s in result.samples if not s.sleeping]
    sleep_samples = [s for s in result.samples if s.sleeping]
    if not wake_samples:
        result.errors.append("no_wake_samples")
    if not sleep_samples:
        result.errors.append("no_sleep_samples")

    checks = [
        result.sleep_cycles >= expected_min_cycles,
        len(wake_samples) > 0,
        len(sleep_samples) > 0,
        len([e for e in result.errors if "fatal" in e]) == 0,
    ]
    result.passed = all(checks)
    print(result.summary())
    return result


def scenario_reply_stress(n_ticks: int = 15000) -> SoakResult:
    """
    Reply queue stress: inject many reply requests and verify
    they're processed without queue overflow or tick blocking.
    """
    result = SoakResult(scenario="reply_stress", target_ticks=n_ticks)
    print(f"\n=== Soak: reply_stress ({n_ticks:,} ticks) ===")

    brain = _create_brain()
    brain.start_headless()

    # Pre-inject several reply requests
    for i in range(5):
        brain._reply_requests.put(
            f"Stress Anfrage {i}: erzähl mir etwas über Bewusstsein"
        )

    try:
        _run_ticks(brain, n_ticks, result, sample_interval=1000)
    except Exception as exc:
        result.errors.append(f"fatal: {type(exc).__name__}: {exc}")
    finally:
        brain._running = False

    # Check all replies were processed
    final_queue = brain._reply_requests.qsize()
    n_replies = len(brain._reply_results)
    print(f"  Replies generated: {n_replies}, queue remaining: {final_queue}")

    if final_queue > 2:
        result.errors.append(f"unprocessed_replies: {final_queue}")
    if n_replies < 3:
        result.errors.append(f"too_few_replies: {n_replies}")

    checks = [
        final_queue <= 2,
        n_replies >= 3,
        result.max_reply_queue < 20,
        len([e for e in result.errors if "fatal" in e]) == 0,
    ]
    result.passed = all(checks)
    print(result.summary())
    return result


def scenario_autosave_verify(n_ticks: int = 10000) -> SoakResult:
    """
    Autosave verification: run past multiple autosave intervals,
    measure save latency, verify file grows.
    """
    result = SoakResult(scenario="autosave_verify", target_ticks=n_ticks)
    print(f"\n=== Soak: autosave_verify ({n_ticks:,} ticks) ===")

    db_path = os.path.join(tempfile.gettempdir(), "_soak_autosave.db")
    # Temporarily override DB_PATH for this test
    import persistence

    orig_path = persistence.DB_PATH
    persistence.DB_PATH = db_path

    try:
        os.remove(db_path)
    except FileNotFoundError:
        pass

    brain = _create_brain()
    brain.start_headless()

    try:
        _run_ticks(brain, n_ticks, result, sample_interval=2000)
    except Exception as exc:
        result.errors.append(f"fatal: {type(exc).__name__}: {exc}")
    finally:
        brain._running = False
        persistence.DB_PATH = orig_path

    # Check DB was created and has content
    from persistence import db_stats

    stats = db_stats(db_path)
    if not stats.get("exists"):
        result.errors.append("no_db_file")
    else:
        if stats.get("synapses", 0) == 0:
            result.errors.append("no_synapses_in_db")
        result.save_count = max(1, n_ticks // 2000)  # approximate
        print(f"  DB stats: {stats}")

    try:
        os.remove(db_path)
    except Exception:
        pass

    checks = [
        stats.get("exists", False),
        stats.get("synapses", 0) > 0,
        len([e for e in result.errors if "fatal" in e]) == 0,
    ]
    result.passed = all(checks)
    print(result.summary())
    return result


ALL_SOAK_SCENARIOS = {
    "stability": scenario_stability,
    "restart": scenario_restart,
    "sleep_stress": scenario_sleep_stress,
    "reply_stress": scenario_reply_stress,
    "autosave_verify": scenario_autosave_verify,
}


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="soak_harness.py",
        description="Full-Brain Soak Tests — long-run stability verification",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=50000,
        help="Number of ticks per scenario (default: 50000)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=list(ALL_SOAK_SCENARIOS.keys()),
        help="Run a single scenario (default: all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all soak scenarios",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export JSONL results to this directory",
    )
    args = parser.parse_args()

    if args.scenario:
        scenarios = {args.scenario: ALL_SOAK_SCENARIOS[args.scenario]}
    else:
        scenarios = ALL_SOAK_SCENARIOS

    results: Dict[str, SoakResult] = {}
    for name, func in scenarios.items():
        result = func(n_ticks=args.ticks)
        results[name] = result
        if args.export:
            os.makedirs(args.export, exist_ok=True)
            export_path = os.path.join(args.export, f"soak_{name}.jsonl")
            result.export_jsonl(export_path)
            print(f"  Exported: {export_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SOAK TEST RESULTS")
    print("=" * 60)
    n_pass = sum(1 for r in results.values() if r.passed)
    n_fail = sum(1 for r in results.values() if not r.passed)
    for name, r in results.items():
        print(r.summary())
        print()
    print(f"Total: {n_pass} PASS / {n_fail} FAIL / {len(results)} scenarios")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
