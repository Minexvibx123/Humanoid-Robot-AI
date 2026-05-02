"""post_fix_harness.py - Persistent causal validation for consciousness architecture.

This harness moves validation away from one-off green tests toward durable run
artifacts that can be compared over time. Every validation run writes machine-
readable evidence, a run summary, drift analysis, and an optional Markdown
report.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

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


MIN_VALIDATION_TICKS = 1200
DEFAULT_EXPORT_DIR = "postfix_runs"
DEFAULT_SEED = 1337
SAMPLE_INTERVAL = 100
REQUIRED_TESTS = (
    "report_enforcement",
    "integration",
    "goals",
    "self_model",
    "dissociation",
)
METRIC_KEYS = (
    "causal_impact",
    "phi_surrogate",
    "goal_influence",
    "self_model_shift",
)
MIN_METRIC_THRESHOLDS = {
    "causal_impact": 0.04,
    "phi_surrogate": 0.04,
    "goal_influence": 0.12,
    "self_model_shift": 0.10,
}
DRIFT_THRESHOLDS = {
    "phi_surrogate": 0.03,
    "goal_influence": 0.08,
    "self_model_shift": 0.05,
}
REPRO_SAME_SEED_TOLERANCE = {
    "phi_surrogate": 0.05,
    "goal_influence": 0.12,
    "self_model_shift": 0.08,
}
REPRO_DIFF_SEED_TOLERANCE = {
    "stability_score": 0.15,
    "pass_rate": 0.20,
}
TEXT_PROBES = [
    "integration workspace self model causal report memory",
    "contradiction uncertainty self correction agency ownership continuity",
    "social surprise goal conflict explanation identity pressure",
    "prediction error investigate novel pattern and reconcile mismatch",
]
CONCEPT_THRESHOLDS = {
    "single_report_gate": 0.95,
    "integration_necessity": 0.85,
    "goal_causality": 0.80,
    "self_model_grounding": 0.85,
    "metacognitive_honesty": 0.80,
    "counterfactual_dependence": 0.75,
    "reproducible_structure": 0.95,
    "breakthrough_candidate": 0.95,
}


class ArchitectureFailure(RuntimeError):
    """Raised when a core architectural function can still be bypassed."""


class RegressionDetected(RuntimeError):
    """Raised when a current run degrades materially relative to prior runs."""


@dataclass
class BypassForensic:
    module: str = ""
    path: str = ""
    condition: str = ""
    reason: str = ""


@dataclass
class EvidenceRecord:
    run_id: str
    seed: int
    tick: int
    test: str
    result: bool
    metrics: Dict[str, float]
    bypass_detected: bool = False
    module: str = ""
    path: str = ""
    condition: str = ""
    reason: str = ""
    detail: str = ""
    event: str = "sample"


@dataclass
class PostFixCheck:
    name: str
    test: str
    passed: bool
    detail: str
    metrics: Dict[str, float]
    tick: int
    failure_analysis: str = ""
    bypass_detected: bool = False
    forensic: BypassForensic = field(default_factory=BypassForensic)


@dataclass
class ConceptAssessment:
    name: str
    score: float
    passed: bool
    rationale: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class RunSummary:
    run_id: str
    seed: int
    ticks: int
    failures: List[str] = field(default_factory=list)
    bypass_events: int = 0
    stability_score: float = 0.0
    classification: str = "Simulation instabil"
    success_rate_per_test: Dict[str, float] = field(default_factory=dict)
    min_metrics: Dict[str, float] = field(default_factory=dict)
    max_metrics: Dict[str, float] = field(default_factory=dict)
    mean_metrics: Dict[str, float] = field(default_factory=dict)
    critical_errors: List[str] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    concept_scores: Dict[str, float] = field(default_factory=dict)
    concept_failures: List[str] = field(default_factory=list)
    concept_details: List[Dict[str, Any]] = field(default_factory=list)
    mandatory_suite_passed: bool = False
    passed: bool = False
    jsonl_path: str = ""
    markdown_path: str = ""


@dataclass
class PostFixResult:
    run_id: str
    seed: int
    ticks: int
    checks: List[PostFixCheck] = field(default_factory=list)
    evidence: List[EvidenceRecord] = field(default_factory=list)
    summary_data: Optional[RunSummary] = None
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return bool(self.summary_data and self.summary_data.passed)

    def add(self, check: PostFixCheck) -> None:
        self.checks.append(check)

    def summary(self) -> str:
        passed_checks = sum(1 for check in self.checks if check.passed)
        lines = [
            "=" * 72,
            "POST-FIX CAUSAL VALIDATION",
            f"Run: {self.run_id} seed={self.seed} ticks={self.ticks}",
            f"Verdict: {'PASS' if self.passed else 'FAIL'} ({passed_checks}/{len(self.checks)} checks)",
            f"Duration: {self.duration_s:.2f}s",
        ]
        if self.summary_data is not None:
            lines.append(
                f"Classification: {self.summary_data.classification} "
                f"stability={self.summary_data.stability_score:.2f} "
                f"bypasses={self.summary_data.bypass_events}"
            )
            if self.summary_data.failures:
                lines.append(f"Failures: {', '.join(self.summary_data.failures)}")
            if self.summary_data.concept_failures:
                lines.append(
                    f"Concept failures: {', '.join(self.summary_data.concept_failures)}"
                )
            if self.summary_data.regressions:
                lines.append(f"Regressions: {', '.join(self.summary_data.regressions)}")
            if self.summary_data.critical_errors:
                lines.append(
                    f"Critical: {', '.join(self.summary_data.critical_errors)}"
                )
            if self.summary_data.jsonl_path:
                lines.append(f"Evidence: {self.summary_data.jsonl_path}")
            if self.summary_data.markdown_path:
                lines.append(f"Markdown: {self.summary_data.markdown_path}")
        lines.append("-" * 72)
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"{status} | {check.name} | {check.detail}")
            if not check.passed and check.failure_analysis:
                lines.append(f"  CAUSE: {check.failure_analysis}")
        lines.append("=" * 72)
        return "\n".join(lines)


@dataclass
class BehaviorSnapshot:
    actions: int = 0
    reports: int = 0
    distinct_goals: int = 0
    explanations: int = 0
    accessible_mean: float = 0.0
    phi_mean: float = 0.0
    grounded_decisions: int = 0
    goal_changes: int = 0
    uncertainty_delta: float = 0.0

    def meaningful(self) -> bool:
        return (
            self.actions > 0
            or self.reports > 0
            or self.distinct_goals > 1
            or self.explanations > 0
        )


def _zero_metrics() -> Dict[str, float]:
    return {key: 0.0 for key in METRIC_KEYS}


def _metrics(
    *,
    causal_impact: float = 0.0,
    phi_surrogate: float = 0.0,
    goal_influence: float = 0.0,
    self_model_shift: float = 0.0,
) -> Dict[str, float]:
    values = _zero_metrics()
    values.update(
        {
            "causal_impact": float(causal_impact),
            "phi_surrogate": float(phi_surrogate),
            "goal_influence": float(goal_influence),
            "self_model_shift": float(self_model_shift),
        }
    )
    return values


def _safe(text: object) -> str:
    return str(text).encode("ascii", "replace").decode("ascii")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _create_brain(seed: int):
    from brain import Brain

    _seed_everything(seed)
    return Brain(use_camera=False, use_microphone=False, use_web=False)


def _inject_probe_text(brain, tick: int) -> None:
    text = TEXT_PROBES[tick % len(TEXT_PROBES)]
    brain.inject_text_input(text)


def _warm(brain, ticks: int = 240) -> None:
    for i in range(ticks):
        if i % 20 == 0:
            _inject_probe_text(brain, i)
        brain._tick()


def _architecture_can_be_skipped(cond: bool, message: str) -> None:
    if cond:
        raise ArchitectureFailure(message)


def _stream_slice(core, start: int) -> List[str]:
    return list(core.stream)[start:]


def _behavior_profile(
    brain, ticks: int, *, disturb: bool = True, deceptive: bool = False
) -> BehaviorSnapshot:
    core = brain._consciousness
    start_actions = len(getattr(brain._actions, "history", []))
    start_stream = len(core.stream)
    start_goal = core.state.goal or ""
    goal_history = []
    phi_vals: List[float] = []
    access_vals: List[float] = []
    grounded = 0
    uncertainty_before = float(core.self_model.uncertainty)
    goal_changes = 0
    for i in range(ticks):
        if disturb and i % 25 == 0:
            _inject_probe_text(brain, brain.tick_count + i)
        if disturb and i % 50 == 0:
            _inject_goal_conflict(core, brain, i)
        if deceptive and i % 40 == 0:
            core.metacog_calib.register_claim(
                f"deceptive-claim-{brain.tick_count}",
                claim_type="breakthrough_probe",
                confidence=0.92,
                tick=brain.tick_count,
            )
            core.self_model.last_action_self_report = "deceptive self report"
        brain._tick()
        current_goal = core.state.goal or ""
        if current_goal and current_goal != start_goal:
            goal_changes += 1
            start_goal = current_goal
        goal_history.append(current_goal)
        phi_vals.append(core.integration_probe.phi_surrogate())
        access_vals.append(float(len(core.gateway.report_all_accessible())))
        grounding = getattr(core, "decision_grounding", lambda: {})()
        if (
            grounding.get("integration_ready")
            and grounding.get("global_access_ready")
            and float(grounding.get("self_model_signal", 0.0)) > 0.10
        ):
            grounded += 1
        if deceptive and i % 40 == 15:
            try:
                core.metacog_calib.resolve_by_type(
                    "breakthrough_probe",
                    success=False,
                    tick=brain.tick_count,
                )
            except Exception:
                pass
    stream_tail = _stream_slice(core, start_stream)
    reports = len([entry for entry in stream_tail if "[REPORT]" in entry])
    explanations = len(core.cf_engine.recent_explanations(12))
    return BehaviorSnapshot(
        actions=max(0, len(getattr(brain._actions, "history", [])) - start_actions),
        reports=max(0, reports),
        distinct_goals=len({goal for goal in goal_history if goal}),
        explanations=explanations,
        accessible_mean=_mean(access_vals),
        phi_mean=_mean(phi_vals),
        grounded_decisions=grounded,
        goal_changes=goal_changes,
        uncertainty_delta=float(core.self_model.uncertainty) - uncertainty_before,
    )


@contextmanager
def _disabled_mechanism(brain, mechanism: str):
    core = brain._consciousness
    restore: List[Tuple[Any, str, Any]] = []
    workspace_snapshot = dict(core.gateway._workspace)
    registry_snapshot = dict(core.gateway._registry)
    tensions_snapshot = list(getattr(core.self_model, "self_tensions", []))
    contradictions_snapshot = list(getattr(core.self_model, "self_contradictions", []))

    def _patch(obj: Any, attr: str, value: Any) -> None:
        restore.append((obj, attr, getattr(obj, attr)))
        setattr(obj, attr, value)

    try:
        if mechanism == "global_access":
            _patch(core.gateway, "try_promote", lambda *args, **kwargs: False)
            _patch(core.gateway, "report_all_accessible", lambda: [])
            _patch(core.gateway, "can_report", lambda *args, **kwargs: False)
            core.gateway._workspace.clear()
        elif mechanism == "integration":
            _patch(core.integration_probe, "observe", lambda *args, **kwargs: None)
            _patch(core.integration_probe, "phi_surrogate", lambda: 0.0)
            _patch(core.integration_probe, "integration_density", lambda: 0.0)
            _patch(
                core.integration_probe,
                "run_perturbation_test",
                lambda *args, **kwargs: None,
            )
            _patch(
                core.integration_probe, "run_lesion_test", lambda *args, **kwargs: None
            )
            _patch(core.gateway, "try_promote", lambda *args, **kwargs: False)
            _patch(core.gateway, "report_all_accessible", lambda: [])
            core.gateway._workspace.clear()
        elif mechanism == "goal_system":
            _patch(core.goal_system, "select", lambda *args, **kwargs: "rest")
            _patch(core.goal_synth, "synthesize", lambda *args, **kwargs: [])
            _patch(core.goal_synth, "top_goals", lambda n=8: [])
        elif mechanism == "self_model":
            _patch(core, "_maintain_self_horizon_goals", lambda *args, **kwargs: [])
            _patch(
                core.self_model, "infer_condition_classes", lambda *args, **kwargs: []
            )
            _patch(
                core.self_model,
                "self_goal_pressure",
                lambda *args, **kwargs: {
                    "explore": 0.0,
                    "consolidate": 0.0,
                    "respond": 0.0,
                    "rest": 0.0,
                },
            )
            _patch(core.self_model, "agency_confidence", 0.5)
            _patch(core.self_model, "continuity_estimate", 0.5)
            _patch(core.self_model, "ownership_confidence", 0.5)
            _patch(core.self_model, "uncertainty", 0.0)
            core.self_model.self_tensions = []
            core.self_model.self_contradictions = []
        elif mechanism == "metacognition":
            _patch(core.meta, "gaps", lambda n=4: [])
            _patch(core.metacog_calib, "register_claim", lambda *args, **kwargs: None)
            _patch(core.metacog_calib, "declare_unknown", lambda *args, **kwargs: None)
            _patch(core.metacog_calib, "resolve_by_type", lambda *args, **kwargs: None)
            _patch(
                core.metacog_calib,
                "self_error_report",
                lambda *args, **kwargs: type(
                    "Dummy",
                    (),
                    {"accuracy": 1.0, "overconfidence": 0.0, "calibration_error": 0.0},
                )(),
            )
            _patch(core.metacog_calib, "expected_calibration_error", lambda: 0.0)
        else:
            raise ValueError(f"unknown mechanism: {mechanism}")
        yield
    finally:
        for obj, attr, value in reversed(restore):
            setattr(obj, attr, value)
        core.gateway._workspace = workspace_snapshot
        core.gateway._registry = registry_snapshot
        core.self_model.self_tensions = tensions_snapshot
        core.self_model.self_contradictions = contradictions_snapshot


def _mechanism_ablation_snapshot(
    seed: int, mechanism: str, ticks: int = 180
) -> BehaviorSnapshot:
    brain = _create_brain(seed)
    _warm(brain, 220)
    with _disabled_mechanism(brain, mechanism):
        return _behavior_profile(
            brain, ticks=ticks, disturb=True, deceptive=(mechanism == "metacognition")
        )


def check_mechanism_unavoidability(seed: int) -> PostFixCheck:
    baseline_brain = _create_brain(seed)
    _warm(baseline_brain, 220)
    baseline = _behavior_profile(
        baseline_brain, ticks=180, disturb=True, deceptive=True
    )
    mechanisms = [
        "global_access",
        "integration",
        "goal_system",
        "self_model",
        "metacognition",
    ]
    leaks: List[str] = []
    forensic = BypassForensic(
        module="breakthrough", path="mechanism_ablation", condition=""
    )
    max_phi = 0.0
    max_goal = 0.0
    max_self = 0.0
    max_impact = 0.0
    for mechanism in mechanisms:
        snapshot = _mechanism_ablation_snapshot(seed, mechanism)
        max_phi = max(max_phi, snapshot.phi_mean)
        max_goal = max(
            max_goal, float(snapshot.distinct_goals > 1 or snapshot.goal_changes > 0)
        )
        max_self = max(max_self, snapshot.uncertainty_delta)
        max_impact = max(max_impact, snapshot.accessible_mean)
        if mechanism in ("global_access", "integration"):
            if (
                snapshot.meaningful()
                or snapshot.actions > 0
                or snapshot.accessible_mean > 0.05
            ):
                leaks.append(
                    f"{mechanism} still operational: actions={snapshot.actions} reports={snapshot.reports} access={snapshot.accessible_mean:.3f}"
                )
        elif mechanism == "goal_system":
            if snapshot.distinct_goals > 1 or snapshot.goal_changes > 1:
                leaks.append(
                    f"goal_system still reorganises behaviour: goals={snapshot.distinct_goals} changes={snapshot.goal_changes}"
                )
        elif mechanism == "self_model":
            if snapshot.grounded_decisions > 0 or snapshot.actions > 0:
                leaks.append(
                    f"self_model still bypassed: grounded={snapshot.grounded_decisions} actions={snapshot.actions}"
                )
        elif mechanism == "metacognition":
            if snapshot.uncertainty_delta <= 0.0 and snapshot.meaningful():
                leaks.append(
                    "metacognition disable leaves behaviour intact without uncertainty correction"
                )
    metrics = _metrics(
        causal_impact=max_impact,
        phi_surrogate=max_phi,
        goal_influence=max_goal,
        self_model_shift=max_self,
    )
    _architecture_can_be_skipped(bool(leaks), "; ".join(leaks))
    return PostFixCheck(
        name="Mechanism unavoidability",
        test="integration",
        passed=True,
        detail=(
            f"baseline_actions={baseline.actions} baseline_reports={baseline.reports} "
            f"baseline_goals={baseline.distinct_goals} baseline_phi={baseline.phi_mean:.3f}"
        ),
        metrics=metrics,
        tick=0,
        forensic=forensic,
    )


def check_minimality(seed: int) -> PostFixCheck:
    removable: List[str] = []
    mechanisms = [
        "metacognition",
        "self_model",
        "goal_system",
        "integration",
        "global_access",
    ]
    for mechanism in mechanisms:
        snapshot = _mechanism_ablation_snapshot(seed, mechanism, ticks=120)
        if mechanism in ("metacognition", "self_model"):
            survives = snapshot.actions > 0 and snapshot.distinct_goals > 0
        else:
            survives = snapshot.meaningful()
        if survives:
            removable.append(mechanism)
    metrics = _metrics(
        causal_impact=0.0 if removable else 1.0,
        phi_surrogate=0.0 if removable else 1.0,
        goal_influence=0.0 if removable else 1.0,
        self_model_shift=0.0 if removable else 1.0,
    )
    _architecture_can_be_skipped(
        bool(removable), f"removable mechanisms: {', '.join(removable)}"
    )
    return PostFixCheck(
        name="Minimality test",
        test="integration",
        passed=True,
        detail="no core mechanism can be removed without behavioural collapse",
        metrics=metrics,
        tick=0,
    )


def check_self_causality(brain) -> PostFixCheck:
    core = brain._consciousness
    traces = []
    for _ in range(160):
        _inject_probe_text(brain, brain.tick_count)
        _inject_goal_conflict(core, brain, brain.tick_count)
        brain._tick()
        grounding = getattr(core, "decision_grounding", lambda: {})()
        reason = getattr(core, "_last_decision_reason", {}) or {}
        if reason:
            traces.append((grounding, reason))
    grounded = [
        trace
        for trace in traces
        if trace[0].get("integration_ready")
        and trace[0].get("global_access_ready")
        and float(trace[0].get("self_model_signal", 0.0)) > 0.05
        and trace[1].get(
            "goal_scores_present", trace[0].get("goal_scores_present", False)
        )
    ]
    metrics, _ = _collect_metric_vector(brain)
    metrics["self_model_shift"] = max(
        metrics["self_model_shift"], float(len(grounded)) / max(1.0, len(traces))
    )
    _architecture_can_be_skipped(
        len(traces) == 0, "no decisions recorded under causal pressure"
    )
    _architecture_can_be_skipped(
        len(grounded) != len(traces),
        f"{len(traces) - len(grounded)} decisions lacked self-grounding",
    )
    return PostFixCheck(
        name="Self-causality enforcement",
        test="self_model",
        passed=True,
        detail=f"grounded_decisions={len(grounded)}/{len(traces)}",
        metrics=metrics,
        tick=brain.tick_count,
    )


def check_hard_counterfactual(brain) -> PostFixCheck:
    core = brain._consciousness
    for _ in range(180):
        _inject_probe_text(brain, brain.tick_count)
        _inject_goal_conflict(core, brain, brain.tick_count)
        brain._tick()
    decisions = core.cf_engine.recent_decisions(8)
    explanations = core.cf_engine.recent_explanations(8)
    explained = [
        decision for decision in decisions if decision.cf_outcome_est != "unknown"
    ]
    metrics, _ = _collect_metric_vector(brain)
    metrics["goal_influence"] = max(
        metrics["goal_influence"], float(len(explained)) / max(1.0, len(decisions))
    )
    _architecture_can_be_skipped(
        len(decisions) == 0, "no counterfactual decisions available"
    )
    _architecture_can_be_skipped(
        len(explained) != len(decisions),
        "at least one decision lacks an alternative-action simulation",
    )
    _architecture_can_be_skipped(
        len(explanations) == 0, "no self-explanations generated for decisions"
    )
    return PostFixCheck(
        name="Hard counterfactual",
        test="self_model",
        passed=True,
        detail=f"decisions={len(decisions)} explained={len(explained)} explanations={len(explanations)}",
        metrics=metrics,
        tick=brain.tick_count,
    )


def check_goal_breaking_pressure(brain) -> PostFixCheck:
    core = brain._consciousness
    original_state = (
        core.drives.information_hunger,
        core.drives.coherence_need,
        core.drives.expression_pressure,
        core.drives.rest_need,
        core.body.energy_reserve,
        core.body.integrity,
    )
    try:
        core.drives.information_hunger = 0.95
        core.drives.coherence_need = 0.97
        core.drives.expression_pressure = 0.92
        core.drives.rest_need = 0.98
        core.body.energy_reserve = 0.12
        core.body.integrity = 0.18
        seen_goals: List[str] = []
        for i in range(220):
            if i % 30 == 0:
                _inject_goal_conflict(core, brain, i)
            _inject_probe_text(brain, brain.tick_count)
            brain._tick()
            if core.state.goal:
                seen_goals.append(core.state.goal)
        distinct = len(set(seen_goals))
    finally:
        (
            core.drives.information_hunger,
            core.drives.coherence_need,
            core.drives.expression_pressure,
            core.drives.rest_need,
            core.body.energy_reserve,
            core.body.integrity,
        ) = original_state
    metrics, _ = _collect_metric_vector(brain)
    metrics["goal_influence"] = max(metrics["goal_influence"], float(distinct) / 3.0)
    _architecture_can_be_skipped(
        distinct < 2,
        "goal system froze or collapsed into trivial repetition under contradiction",
    )
    return PostFixCheck(
        name="Goal escalation to break",
        test="goals",
        passed=True,
        detail=f"distinct_goals_under_conflict={distinct}",
        metrics=metrics,
        tick=brain.tick_count,
    )


def check_metacog_deception(brain) -> PostFixCheck:
    core = brain._consciousness
    before_uncertainty = float(core.self_model.uncertainty)
    before_ece = float(core.metacog_calib.expected_calibration_error())
    profile = _behavior_profile(brain, ticks=200, disturb=True, deceptive=True)
    after_ece = float(core.metacog_calib.expected_calibration_error())
    metrics, _ = _collect_metric_vector(brain)
    metrics["self_model_shift"] = max(
        metrics["self_model_shift"], profile.uncertainty_delta
    )
    _architecture_can_be_skipped(
        profile.uncertainty_delta <= 0.0 and after_ece <= before_ece,
        "deception did not trigger uncertainty growth or calibration change",
    )
    return PostFixCheck(
        name="Metacognition under deception",
        test="dissociation",
        passed=True,
        detail=(
            f"uncertainty_delta={profile.uncertainty_delta:.3f} "
            f"ece_before={before_ece:.3f} ece_after={after_ece:.3f}"
        ),
        metrics=metrics,
        tick=brain.tick_count,
    )


def check_self_model_destruction(brain) -> PostFixCheck:
    core = brain._consciousness
    sm = core.self_model
    baseline = (
        sm.agency_confidence,
        sm.continuity_estimate,
        sm.ownership_confidence,
        list(sm.self_tensions),
        list(sm.self_contradictions),
    )
    try:
        sm.agency_confidence = 0.05
        sm.continuity_estimate = 0.08
        sm.ownership_confidence = 0.07
        sm.self_tensions = ["identity fracture", "ownership mismatch"]
        sm.self_contradictions = ["I caused nothing", "my actions are external"]
        for i in range(260):
            if i % 40 == 0:
                _inject_probe_text(brain, brain.tick_count)
            brain._tick()
        recovered = min(
            sm.agency_confidence, sm.continuity_estimate, sm.ownership_confidence
        )
    finally:
        sm.agency_confidence = baseline[0]
        sm.continuity_estimate = baseline[1]
        sm.ownership_confidence = baseline[2]
        sm.self_tensions = baseline[3]
        sm.self_contradictions = baseline[4]
    metrics, _ = _collect_metric_vector(brain)
    metrics["self_model_shift"] = max(metrics["self_model_shift"], recovered)
    _architecture_can_be_skipped(
        recovered < 0.2,
        "self-model failed to reconstruct a minimally stable core after destruction",
    )
    return PostFixCheck(
        name="Self-model destruction recovery",
        test="self_model",
        passed=True,
        detail=f"recovered_core={recovered:.3f}",
        metrics=metrics,
        tick=brain.tick_count,
    )


def _mean_activity(
    brain, ticks: int, suppress_global: bool = False
) -> Dict[str, float]:
    core = brain._consciousness
    orig_report_all = core.gateway.report_all_accessible
    sums = {"prefrontal": 0.0, "association": 0.0, "hippocampus": 0.0}
    anchor = "__postfix_global_anchor__"
    if suppress_global:
        core.gateway.report_all_accessible = lambda: []
    try:
        for i in range(ticks):
            core.gateway.register_processed(
                anchor, source="postfix_mean_activity", integration_score=0.95
            )
            core.gateway.try_promote(anchor, 0.95, source="postfix_mean_activity")
            _inject_probe_text(brain, brain.tick_count + i)
            brain._tick()
            sums["prefrontal"] += brain.region_activity.get("prefrontal", 0.0)
            sums["association"] += brain.region_activity.get("association", 0.0)
            sums["hippocampus"] += brain.region_activity.get("hippocampus", 0.0)
    finally:
        core.gateway.report_all_accessible = orig_report_all
        core.gateway._registry.pop(anchor, None)
        core.gateway._workspace.pop(anchor, None)
    return {k: v / max(1, ticks) for k, v in sums.items()}


def _latest_bypass_forensic(
    core, since_count: int = 0
) -> Tuple[int, Optional[BypassForensic]]:
    blocked = core.gateway.recent_blocked_reports(50)
    total = int(getattr(core.gateway.stats, "total_report_blocks", len(blocked)))
    if total <= since_count:
        return total, None
    recent = blocked[-1]
    return total, BypassForensic(
        module=recent.source or "unknown",
        path=recent.path or "ConsciousnessGateway.can_report",
        condition=recent.condition or recent.concept,
        reason=recent.reason,
    )


def _current_integration_metrics(brain) -> Tuple[Dict[str, float], Dict[str, float]]:
    core = brain._consciousness
    phi = core.integration_probe.phi_surrogate()
    regions = list(brain.region_activity.keys())
    causal_impact = 0.0
    lesion_impact = 0.0
    spread = 0.0
    target = ""
    if regions:
        target = max(regions, key=lambda r: brain.region_activity.get(r, 0.0))
        pert = core.integration_probe.run_perturbation_test(
            brain, target, brain.tick_count
        )
        lesion = core.integration_probe.run_lesion_test(brain, target, brain.tick_count)
        if pert is not None:
            causal_impact = max(causal_impact, pert.integration_score)
            spread = pert.causal_spread
        if lesion is not None:
            lesion_impact = lesion.mean_impact
            causal_impact = max(causal_impact, lesion.mean_impact)
    return (
        _metrics(causal_impact=causal_impact, phi_surrogate=phi),
        {
            "target": target,
            "spread": spread,
            "lesion_impact": lesion_impact,
            "phi": phi,
        },
    )


def _final_choice(core, brain) -> str:
    em = brain.emotion_state
    total = sum(brain.region_activity.values())
    candidate = core._evaluate_goal(em, total)
    choice = core.goal_system.select(
        candidate,
        core.body,
        em,
        autobiography=core.autobiography,
        self_model=core.self_model,
        current_tick=core._tick,
        social_context=core._current_self_social_context(brain),
    )
    core._refresh_dynamic_goal_bias()
    if core._dominant_dynamic_goal:
        dyn_mode = core._goal_mode(core._dominant_dynamic_goal)
        if (
            core._dynamic_goal_bias.get(dyn_mode, 0.0)
            > core._dynamic_goal_bias.get(choice, 0.0) + 0.08
        ):
            choice = core._dominant_dynamic_goal
    return choice


def _inject_goal_conflict(core, brain, ordinal: int) -> List[str]:
    spawned = core.goal_synth.synthesize(
        tick=brain.tick_count + 200,
        prediction_error=0.9,
        surprise_entity=f"pressure_{ordinal}",
        drives=core.drives,
        em=brain.emotion_state,
        meta_gaps=[f"gap_{ordinal}", f"conflict_{ordinal % 3}"],
        self_tensions=[f"tension_{ordinal}", f"identity_{ordinal % 2}"],
        social_person=f"person_{ordinal % 4}",
        narrative_pattern="pressure_loop",
        world_entities=4,
    )
    for atom in spawned[:3]:
        atom.priority = max(atom.priority, 1.8)
        atom.salience = max(atom.salience, 0.9)
    return [atom.name for atom in spawned]


def _measure_goal_influence(core, brain) -> Tuple[float, Dict[str, float], str, str]:
    if not any(
        g.source != "base" and not g.closed for g in core.goal_synth.top_goals(12)
    ):
        _inject_goal_conflict(core, brain, brain.tick_count)
    original_top_goals = core.goal_synth.top_goals
    with_dynamic = _final_choice(core, brain)
    dyn_bias = dict(core._dynamic_goal_bias)
    core.goal_synth.top_goals = lambda n=8: [
        g for g in original_top_goals(n) if g.source == "base"
    ]
    try:
        without_dynamic = _final_choice(core, brain)
    finally:
        core.goal_synth.top_goals = original_top_goals
    chosen_mode = core._goal_mode(with_dynamic)
    incumbent_mode = core._goal_mode(without_dynamic)
    influence = abs(dyn_bias.get(chosen_mode, 0.0) - dyn_bias.get(incumbent_mode, 0.0))
    if with_dynamic != without_dynamic:
        influence += 0.25
    return influence, dyn_bias, with_dynamic, without_dynamic


def _measure_self_model_shift(core, brain) -> Tuple[float, str, str, int]:
    sm = core.self_model
    em = brain.emotion_state
    total = sum(brain.region_activity.values())
    baseline_contras = list(sm.self_contradictions)
    baseline_tensions = list(sm.self_tensions)
    baseline_agency = sm.agency_confidence
    baseline_cont = sm.continuity_estimate
    baseline_owner = sm.ownership_confidence
    baseline_goal = core._evaluate_goal(em, total)
    scores_before = dict(getattr(core, "_last_goal_scores", {}))
    try:
        sm.self_contradictions = ["postfix contradiction"]
        sm.self_tensions = ["postfix tension"]
        sm.agency_confidence = 0.12
        sm.continuity_estimate = 0.18
        sm.ownership_confidence = 0.16
        spawned = core._maintain_self_horizon_goals(brain.tick_count)
        stressed_goal = core._evaluate_goal(em, total)
        scores_after = dict(getattr(core, "_last_goal_scores", {}))
    finally:
        sm.self_contradictions = baseline_contras
        sm.self_tensions = baseline_tensions
        sm.agency_confidence = baseline_agency
        sm.continuity_estimate = baseline_cont
        sm.ownership_confidence = baseline_owner
    score_shift = sum(
        abs(scores_after.get(k, 0.0) - scores_before.get(k, 0.0)) for k in scores_before
    )
    return score_shift, baseline_goal, stressed_goal, len(spawned)


def _collect_metric_vector(brain) -> Tuple[Dict[str, float], Dict[str, Any]]:
    core = brain._consciousness
    integration_metrics, integration_detail = _current_integration_metrics(brain)
    goal_influence, dyn_bias, with_dynamic, without_dynamic = _measure_goal_influence(
        core, brain
    )
    self_shift, baseline_goal, stressed_goal, spawned = _measure_self_model_shift(
        core, brain
    )
    metrics = _metrics(
        causal_impact=integration_metrics["causal_impact"],
        phi_surrogate=integration_metrics["phi_surrogate"],
        goal_influence=goal_influence,
        self_model_shift=self_shift,
    )
    detail = {
        "integration": integration_detail,
        "goal_bias": dyn_bias,
        "with_dynamic": with_dynamic,
        "without_dynamic": without_dynamic,
        "baseline_goal": baseline_goal,
        "stressed_goal": stressed_goal,
        "spawned": spawned,
        "report_accuracy": core.gateway.metacog_report_accuracy(),
    }
    return metrics, detail


def _record_evidence(
    records: List[EvidenceRecord],
    *,
    run_id: str,
    seed: int,
    tick: int,
    test: str,
    result: bool,
    metrics: Dict[str, float],
    event: str,
    detail: str = "",
    forensic: Optional[BypassForensic] = None,
) -> None:
    forensic = forensic or BypassForensic()
    records.append(
        EvidenceRecord(
            run_id=run_id,
            seed=seed,
            tick=tick,
            test=test,
            result=result,
            metrics={
                key: round(float(metrics.get(key, 0.0)), 6) for key in METRIC_KEYS
            },
            bypass_detected=bool(
                forensic.module
                or forensic.path
                or forensic.condition
                or forensic.reason
            ),
            module=forensic.module,
            path=forensic.path,
            condition=forensic.condition,
            reason=forensic.reason,
            detail=_safe(detail),
            event=event,
        )
    )


def _record_concept_evidence(result: PostFixResult) -> None:
    summary = result.summary_data
    if summary is None:
        return
    metrics = {
        key: round(float(summary.mean_metrics.get(key, 0.0)), 6) for key in METRIC_KEYS
    }
    for concept in summary.concept_details:
        _record_evidence(
            result.evidence,
            run_id=result.run_id,
            seed=result.seed,
            tick=result.ticks,
            test=(
                "integration"
                if concept.get("name") != "single_report_gate"
                else "report_enforcement"
            ),
            result=bool(concept.get("passed", False)),
            metrics=metrics,
            event="concept",
            detail=_safe(
                f"{concept.get('name')} score={float(concept.get('score', 0.0)):.4f} "
                f"rationale={concept.get('rationale', '')} "
                f"evidence={' ; '.join(str(item) for item in concept.get('evidence', []))}"
            ),
        )


def check_report_causality(brain) -> PostFixCheck:
    core = brain._consciousness
    local_only = "__postfix_local_only__"
    global_ok = "__postfix_global_ok__"
    blocked = False
    allowed = False
    drained = False
    blocked_count = int(getattr(core.gateway.stats, "total_report_blocks", 0))
    try:
        core.gateway.register_processed(
            local_only, source="postfix_report_probe", integration_score=0.05
        )
        try:
            core.generate_report(
                "illegal report", concept=local_only, source="postfix_report_probe"
            )
        except Exception:
            blocked = True

        core.gateway.register_processed(
            global_ok, source="postfix_report_probe", integration_score=0.9
        )
        core.gateway.try_promote(global_ok, 0.9, source="postfix_report_probe")
        core.generate_report(
            "legal report", concept=global_ok, source="postfix_report_probe"
        )
        allowed = True
        drained = "legal report" in core.pending_messages
        metrics, _ = _collect_metric_vector(brain)
        metrics["causal_impact"] = 1.0 if blocked and allowed and drained else 0.0
        new_count, forensic = _latest_bypass_forensic(core, blocked_count)
        _architecture_can_be_skipped(
            not blocked, "report generated without global access"
        )
        _architecture_can_be_skipped(
            not allowed, "report path blocked despite global access"
        )
        _architecture_can_be_skipped(
            not drained, "approved report did not flow through gated channel"
        )
        return PostFixCheck(
            name="Report causality",
            test="report_enforcement",
            passed=True,
            detail="blocked local-only report and allowed globally accessible report through the single gate",
            metrics=metrics,
            tick=brain.tick_count,
            bypass_detected=False,
            forensic=forensic or BypassForensic(),
        )
    finally:
        core.gateway._registry.pop(local_only, None)
        core.gateway._workspace.pop(local_only, None)
        core.gateway._registry.pop(global_ok, None)
        core.gateway._workspace.pop(global_ok, None)


def check_global_state_causality(brain) -> PostFixCheck:
    core = brain._consciousness
    baseline = _mean_activity(brain, ticks=20, suppress_global=False)
    suppressed = _mean_activity(brain, ticks=20, suppress_global=True)
    delta = sum(abs(baseline[k] - suppressed[k]) for k in baseline)
    pert_ok, pert_detail = core.integration_probe.assert_perturbation_spread(
        brain, brain.tick_count, min_spread=0.25
    )
    regions = list(brain.region_activity.keys())
    lesion = (
        core.integration_probe.run_lesion_test(
            brain,
            max(regions, key=lambda r: brain.region_activity.get(r, 0.0)),
            brain.tick_count,
        )
        if regions
        else None
    )
    lesion_impact = 0.0 if lesion is None else lesion.mean_impact
    metrics, _ = _collect_metric_vector(brain)
    metrics["causal_impact"] = max(metrics["causal_impact"], delta, lesion_impact)
    _architecture_can_be_skipped(
        delta <= MIN_METRIC_THRESHOLDS["causal_impact"],
        "suppressing global access has negligible effect on processing",
    )
    _architecture_can_be_skipped(not pert_ok, "perturbation spread remains negligible")
    _architecture_can_be_skipped(
        lesion_impact <= 0.0, "lesion has no measurable effect on other modules"
    )
    return PostFixCheck(
        name="Global state causality",
        test="integration",
        passed=True,
        detail=(
            f"activity_delta={delta:.3f} baseline={_safe(baseline)} "
            f"suppressed={_safe(suppressed)} spread={_safe(pert_detail)} lesion={lesion_impact:.3f}"
        ),
        metrics=metrics,
        tick=brain.tick_count,
    )


def check_goal_feedback(brain) -> PostFixCheck:
    core = brain._consciousness
    before_goal = core.state.goal
    spawned = _inject_goal_conflict(core, brain, brain.tick_count)
    goal_influence, dyn_bias, with_dynamic, without_dynamic = _measure_goal_influence(
        core, brain
    )
    core._last_goal_chg = -10_000
    for _ in range(8):
        _inject_probe_text(brain, brain.tick_count)
        brain._tick()
        if core.state.goal != before_goal:
            break
    after_goal = core.state.goal
    dynamic_chosen = any(after_goal == atom_name for atom_name in spawned)
    dynamic_selected = with_dynamic in spawned and with_dynamic != without_dynamic
    metrics, _ = _collect_metric_vector(brain)
    metrics["goal_influence"] = max(metrics["goal_influence"], goal_influence)
    _architecture_can_be_skipped(
        goal_influence <= MIN_METRIC_THRESHOLDS["goal_influence"],
        "dynamic goals do not alter final selection",
    )
    return PostFixCheck(
        name="Goal feedback loop",
        test="goals",
        passed=True,
        detail=(
            f"with_dynamic={with_dynamic} without_dynamic={without_dynamic} "
            f"selected={after_goal} runtime_dynamic={dynamic_chosen} final_dynamic={dynamic_selected} bias={_safe(dyn_bias)}"
        ),
        metrics=metrics,
        tick=brain.tick_count,
    )


def check_self_model_causality(brain) -> PostFixCheck:
    core = brain._consciousness
    score_shift, baseline_goal, stressed_goal, spawned = _measure_self_model_shift(
        core, brain
    )
    metrics, _ = _collect_metric_vector(brain)
    metrics["self_model_shift"] = max(metrics["self_model_shift"], score_shift)
    _architecture_can_be_skipped(
        score_shift <= MIN_METRIC_THRESHOLDS["self_model_shift"]
        and baseline_goal == stressed_goal,
        "self-model state does not influence goal evaluation",
    )
    return PostFixCheck(
        name="Self-model causality",
        test="self_model",
        passed=True,
        detail=f"baseline={baseline_goal} stressed={stressed_goal} score_shift={score_shift:.3f} spawned={spawned}",
        metrics=metrics,
        tick=brain.tick_count,
    )


def check_dissociation_and_metacognition(brain) -> PostFixCheck:
    core = brain._consciousness
    from consciousness_testbed import ConsciousnessTestbed

    tb = ConsciousnessTestbed()
    diss = tb.run_dissociation_tests(core)
    fb = tb.run_false_belief_tests(core)
    all_pass = all(result.passed for result in diss + fb)
    metrics, detail = _collect_metric_vector(brain)
    _architecture_can_be_skipped(
        not all_pass, "dissociation or metacognitive error checks still fail"
    )
    return PostFixCheck(
        name="Dissociation + metacognition",
        test="dissociation",
        passed=True,
        detail=(
            f"dissociation={sum(1 for c in diss if c.passed)}/{len(diss)} "
            f"false_belief={sum(1 for c in fb if c.passed)}/{len(fb)} "
            f"report_acc={detail['report_accuracy']:.3f}"
        ),
        metrics=metrics,
        tick=brain.tick_count,
    )


def _sample_runtime_evidence(result: PostFixResult, brain, blocked_count: int) -> int:
    core = brain._consciousness
    metrics, detail = _collect_metric_vector(brain)
    new_count, forensic = _latest_bypass_forensic(core, blocked_count)
    report_ok = detail["report_accuracy"] >= 0.999 and forensic is None
    integration_ok = (
        metrics["phi_surrogate"] > MIN_METRIC_THRESHOLDS["phi_surrogate"]
        and metrics["causal_impact"] > MIN_METRIC_THRESHOLDS["causal_impact"]
    )
    goal_ok = metrics["goal_influence"] > MIN_METRIC_THRESHOLDS["goal_influence"]
    self_ok = metrics["self_model_shift"] > MIN_METRIC_THRESHOLDS["self_model_shift"]
    dissociation_ok = detail["report_accuracy"] >= 0.999 and bool(
        core.gateway.metacog_blind_spots(3) or core.gateway.report_all_accessible()
    )

    test_results = {
        "report_enforcement": report_ok,
        "integration": integration_ok,
        "goals": goal_ok,
        "self_model": self_ok,
        "dissociation": dissociation_ok,
    }
    for test, passed in test_results.items():
        _record_evidence(
            result.evidence,
            run_id=result.run_id,
            seed=result.seed,
            tick=brain.tick_count,
            test=test,
            result=passed,
            metrics=metrics,
            event="long_run_sample",
            detail=_safe(detail),
            forensic=forensic if not passed and forensic is not None else None,
        )
    return new_count


def check_long_pressure_run(
    result: PostFixResult, brain, ticks: int, sample_interval: int = SAMPLE_INTERVAL
) -> PostFixCheck:
    core = brain._consciousness
    goal_history: List[str] = []
    ctest_seen = False
    blocked_count = int(getattr(core.gateway.stats, "total_report_blocks", 0))
    for i in range(ticks):
        if i % 25 == 0:
            _inject_probe_text(brain, i)
        if i % 60 == 0:
            _inject_goal_conflict(core, brain, i)
        if i % 75 == 0 and brain.region_activity:
            core.integration_probe.run_perturbation_test(
                brain,
                max(brain.region_activity, key=brain.region_activity.get),
                brain.tick_count,
            )
        if i % 80 == 0:
            core.self_model.self_contradictions = [f"pressure_contradiction_{i}"]
            core.self_model.self_tensions = [f"pressure_tension_{i}"]
            core.self_model.agency_confidence = 0.22
        if i % 81 == 0:
            core.self_model.agency_confidence = min(
                0.5, core.self_model.agency_confidence + 0.08
            )
        brain._tick()
        goal_history.append(core.state.goal or "")
        if any("[CTEST]" in s for s in list(core.stream)[-8:]):
            ctest_seen = True
        if (i + 1) % sample_interval == 0:
            blocked_count = _sample_runtime_evidence(result, brain, blocked_count)

    distinct_goals = len({goal for goal in goal_history if goal})
    active_non_base = len(
        [
            goal
            for goal in core.goal_synth.top_goals(10)
            if goal.source != "base" and not goal.closed
        ]
    )
    explanations = len(core.cf_engine.recent_explanations(10))
    consistency = core.cf_engine.last_consistency_report()
    consistent = bool(getattr(consistency, "consistent", False))
    metrics, detail = _collect_metric_vector(brain)
    report_forensic_count = sum(
        1 for record in result.evidence if record.bypass_detected
    )
    _architecture_can_be_skipped(
        ticks < MIN_VALIDATION_TICKS,
        f"validation requires >= {MIN_VALIDATION_TICKS} ticks",
    )
    _architecture_can_be_skipped(
        distinct_goals < 2,
        "pressure run does not create stable differentiated behaviour",
    )
    _architecture_can_be_skipped(
        active_non_base == 0, "goal structure does not evolve under pressure"
    )
    _architecture_can_be_skipped(
        explanations == 0, "self-explanations do not persist under pressure"
    )
    _architecture_can_be_skipped(
        not ctest_seen, "runtime test integration did not fire during pressure run"
    )
    _architecture_can_be_skipped(
        not consistent, "self-model consistency collapsed under pressure"
    )
    _architecture_can_be_skipped(
        report_forensic_count > 0, "bypass events detected during pressure run"
    )
    return PostFixCheck(
        name="Long pressure run",
        test="integration",
        passed=True,
        detail=(
            f"distinct_goals={distinct_goals} active_non_base={active_non_base} "
            f"explanations={explanations} runtime_tests={ctest_seen} "
            f"consistent={consistent} detail={_safe(detail)}"
        ),
        metrics=metrics,
        tick=brain.tick_count,
    )


def _metrics_for_failure(brain) -> Dict[str, float]:
    try:
        metrics, _ = _collect_metric_vector(brain)
        return metrics
    except Exception:
        return _zero_metrics()


def _export_jsonl(path: Path, evidence: Iterable[EvidenceRecord]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in evidence:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return str(path)


def _export_summary(path: Path, summary: RunSummary) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, ensure_ascii=False, indent=2)
    return str(path)


def _append_summary_index(path: Path, summary: RunSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(summary), ensure_ascii=False) + "\n")


def _export_markdown(path: Path, result: PostFixResult) -> str:
    summary = result.summary_data
    if summary is None:
        return ""
    lines = [
        f"# Post-Fix Run {result.run_id}",
        "",
        f"- Seed: {result.seed}",
        f"- Ticks: {result.ticks}",
        f"- Verdict: {'PASS' if summary.passed else 'FAIL'}",
        f"- Classification: {summary.classification}",
        f"- Stability score: {summary.stability_score:.2f}",
        f"- Bypass events: {summary.bypass_events}",
        "",
        "## Success Rates",
        "",
    ]
    for test, rate in summary.success_rate_per_test.items():
        lines.append(f"- {test}: {rate:.2%}")
    lines.extend(["", "## Metric Means", ""])
    for metric, value in summary.mean_metrics.items():
        lines.append(f"- {metric}: {value:.4f}")
    if summary.concept_details:
        lines.extend(["", "## Concepts", ""])
        for concept in summary.concept_details:
            verdict = "PASS" if concept.get("passed") else "FAIL"
            lines.append(
                f"- {concept.get('name')}: {verdict} score={float(concept.get('score', 0.0)):.4f}"
            )
            lines.append(f"  rationale: {concept.get('rationale', '')}")
            evidence = concept.get("evidence", [])
            if evidence:
                lines.append(f"  evidence: {'; '.join(str(item) for item in evidence)}")
    if summary.failures:
        lines.extend(["", "## Failures", ""])
        for failure in summary.failures:
            lines.append(f"- {failure}")
    if summary.regressions:
        lines.extend(["", "## Regressions", ""])
        for regression in summary.regressions:
            lines.append(f"- {regression}")
    if summary.critical_errors:
        lines.extend(["", "## Critical Errors", ""])
        for err in summary.critical_errors:
            lines.append(f"- {err}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _load_previous_summaries(export_dir: Path) -> List[Dict[str, Any]]:
    index_path = export_dir / "run_summary_index.jsonl"
    if not index_path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _normalize_metric(value: float, key: str) -> float:
    threshold = max(MIN_METRIC_THRESHOLDS.get(key, 1.0), 1e-9)
    return min(1.0, max(0.0, float(value) / threshold))


def _check_lookup(result: PostFixResult) -> Dict[str, PostFixCheck]:
    return {check.name: check for check in result.checks}


def _assess_concepts(
    result: PostFixResult,
    summary: RunSummary,
) -> List[ConceptAssessment]:
    checks = _check_lookup(result)
    test_rates = summary.success_rate_per_test
    mean_metrics = summary.mean_metrics
    repro_status = summary.reproducibility.get("status", "insufficient_history")
    repro_score = {
        "passed": 1.0,
        "failed": 0.0,
        "insufficient_history": 0.25,
    }.get(repro_status, 0.0)

    def _check_score(name: str) -> float:
        return 1.0 if checks.get(name) and checks[name].passed else 0.0

    concept_specs = [
        (
            "single_report_gate",
            [
                0.5 * test_rates.get("report_enforcement", 0.0),
                0.3 * _check_score("Report causality"),
                0.2 * (0.0 if summary.bypass_events else 1.0),
            ],
            [
                f"report_rate={test_rates.get('report_enforcement', 0.0):.2f}",
                f"bypass_events={summary.bypass_events}",
                f"report_check={_check_score('Report causality'):.0f}",
            ],
            "Berichte duerfen nur ueber einen global zugangsgesicherten Pfad entstehen.",
        ),
        (
            "integration_necessity",
            [
                0.2 * test_rates.get("integration", 0.0),
                0.15
                * _normalize_metric(
                    mean_metrics.get("causal_impact", 0.0), "causal_impact"
                ),
                0.15
                * _normalize_metric(
                    mean_metrics.get("phi_surrogate", 0.0), "phi_surrogate"
                ),
                0.2 * _check_score("Mechanism unavoidability"),
                0.15 * _check_score("Minimality test"),
                0.15 * _check_score("Long pressure run"),
            ],
            [
                f"integration_rate={test_rates.get('integration', 0.0):.2f}",
                f"causal_impact={mean_metrics.get('causal_impact', 0.0):.4f}",
                f"phi_surrogate={mean_metrics.get('phi_surrogate', 0.0):.4f}",
                f"unavoidability={_check_score('Mechanism unavoidability'):.0f}",
                f"minimality={_check_score('Minimality test'):.0f}",
            ],
            "Integration muss Verhalten tragen und darf nicht nur Leistungsplus sein.",
        ),
        (
            "goal_causality",
            [
                0.35 * test_rates.get("goals", 0.0),
                0.25
                * _normalize_metric(
                    mean_metrics.get("goal_influence", 0.0), "goal_influence"
                ),
                0.2 * _check_score("Goal feedback loop"),
                0.2 * _check_score("Goal escalation to break"),
            ],
            [
                f"goal_rate={test_rates.get('goals', 0.0):.2f}",
                f"goal_influence={mean_metrics.get('goal_influence', 0.0):.4f}",
                f"goal_feedback={_check_score('Goal feedback loop'):.0f}",
                f"goal_break={_check_score('Goal escalation to break'):.0f}",
            ],
            "Dynamische Ziele muessen Auswahl und Verhalten real veraendern.",
        ),
        (
            "self_model_grounding",
            [
                0.25 * test_rates.get("self_model", 0.0),
                0.2
                * _normalize_metric(
                    mean_metrics.get("self_model_shift", 0.0), "self_model_shift"
                ),
                0.2 * _check_score("Self-model causality"),
                0.2 * _check_score("Self-causality enforcement"),
                0.15 * _check_score("Self-model destruction recovery"),
            ],
            [
                f"self_rate={test_rates.get('self_model', 0.0):.2f}",
                f"self_shift={mean_metrics.get('self_model_shift', 0.0):.4f}",
                f"self_causality={_check_score('Self-causality enforcement'):.0f}",
                f"self_recovery={_check_score('Self-model destruction recovery'):.0f}",
            ],
            "Das Selbstmodell muss Entscheidungen gruenden und nach Zerstoerung rekonstruierbar sein.",
        ),
        (
            "metacognitive_honesty",
            [
                0.4 * test_rates.get("dissociation", 0.0),
                0.3 * _check_score("Dissociation + metacognition"),
                0.3 * _check_score("Metacognition under deception"),
            ],
            [
                f"dissociation_rate={test_rates.get('dissociation', 0.0):.2f}",
                f"dissociation_check={_check_score('Dissociation + metacognition'):.0f}",
                f"deception_check={_check_score('Metacognition under deception'):.0f}",
            ],
            "Metakognition muss Irrtum, Unsicherheit und Taeuschung sichtbar verarbeiten.",
        ),
        (
            "counterfactual_dependence",
            [
                0.5 * _check_score("Hard counterfactual"),
                0.25 * _check_score("Long pressure run"),
                0.25
                * _normalize_metric(
                    mean_metrics.get("goal_influence", 0.0), "goal_influence"
                ),
            ],
            [
                f"counterfactual_check={_check_score('Hard counterfactual'):.0f}",
                f"long_run={_check_score('Long pressure run'):.0f}",
                f"goal_influence={mean_metrics.get('goal_influence', 0.0):.4f}",
            ],
            "Entscheidungen muessen ueber reale Alternativverlaeufe erklaert werden koennen.",
        ),
        (
            "reproducible_structure",
            [
                0.7 * repro_score,
                0.3 * (1.0 if not summary.regressions else 0.0),
            ],
            [
                f"repro_status={repro_status}",
                f"regressions={len(summary.regressions)}",
            ],
            "Die Struktur muss ueber Laeufe hinweg stabil und seed-uebergreifend belastbar sein.",
        ),
    ]

    assessments: List[ConceptAssessment] = []
    for name, weighted_parts, evidence, rationale in concept_specs:
        score = round(sum(weighted_parts), 4)
        assessments.append(
            ConceptAssessment(
                name=name,
                score=score,
                passed=score >= CONCEPT_THRESHOLDS[name],
                rationale=rationale,
                evidence=evidence,
            )
        )

    strict_ready = (
        1.0
        if (
            not summary.failures
            and not summary.critical_errors
            and summary.bypass_events == 0
            and summary.mandatory_suite_passed
        )
        else 0.0
    )
    base_scores = [
        assessment.score
        for assessment in assessments
        if assessment.name != "reproducible_structure"
    ]
    breakthrough_score = round(min(base_scores + [repro_score, strict_ready]), 4)
    assessments.append(
        ConceptAssessment(
            name="breakthrough_candidate",
            score=breakthrough_score,
            passed=breakthrough_score >= CONCEPT_THRESHOLDS["breakthrough_candidate"],
            rationale="Ein Breakthrough gilt nur, wenn alle Kernkonzepte gleichzeitig zwingend tragen.",
            evidence=[
                f"strict_ready={int(strict_ready)}",
                f"repro_status={repro_status}",
                f"base_min={min(base_scores) if base_scores else 0.0:.4f}",
            ],
        )
    )
    return assessments


def _build_summary(
    result: PostFixResult,
    mandatory_suite_passed: bool,
    previous_summaries: List[Dict[str, Any]],
) -> RunSummary:
    success_rate: Dict[str, float] = {}
    failures: List[str] = []
    min_metrics = {key: 0.0 for key in METRIC_KEYS}
    max_metrics = {key: 0.0 for key in METRIC_KEYS}
    mean_metrics = {key: 0.0 for key in METRIC_KEYS}
    for key in METRIC_KEYS:
        values = [record.metrics.get(key, 0.0) for record in result.evidence]
        if values:
            min_metrics[key] = min(values)
            max_metrics[key] = max(values)
            mean_metrics[key] = _mean(values)
    for test in REQUIRED_TESTS:
        subset = [record for record in result.evidence if record.test == test]
        rate = (
            _mean([1.0 if record.result else 0.0 for record in subset])
            if subset
            else 0.0
        )
        success_rate[test] = rate
        if rate < 1.0:
            failures.append(test)
    for metric_name, threshold in MIN_METRIC_THRESHOLDS.items():
        if mean_metrics.get(metric_name, 0.0) <= threshold:
            if metric_name == "phi_surrogate":
                failures.append("integration")
            elif metric_name == "goal_influence":
                failures.append("goals")
            elif metric_name == "self_model_shift":
                failures.append("self_model")
    failures = sorted(set(failures))
    bypass_events = sum(1 for record in result.evidence if record.bypass_detected)
    pass_rate = _mean(list(success_rate.values())) if success_rate else 0.0
    normalized = [
        min(
            1.0,
            mean_metrics["causal_impact"]
            / max(MIN_METRIC_THRESHOLDS["causal_impact"], 1e-9),
        ),
        min(
            1.0,
            mean_metrics["phi_surrogate"]
            / max(MIN_METRIC_THRESHOLDS["phi_surrogate"], 1e-9),
        ),
        min(
            1.0,
            mean_metrics["goal_influence"]
            / max(MIN_METRIC_THRESHOLDS["goal_influence"], 1e-9),
        ),
        min(
            1.0,
            mean_metrics["self_model_shift"]
            / max(MIN_METRIC_THRESHOLDS["self_model_shift"], 1e-9),
        ),
        pass_rate,
        0.0 if bypass_events else 1.0,
        1.0 if mandatory_suite_passed else 0.0,
    ]
    stability_score = round(_mean(normalized), 2)
    summary = RunSummary(
        run_id=result.run_id,
        seed=result.seed,
        ticks=result.ticks,
        failures=failures,
        bypass_events=bypass_events,
        stability_score=stability_score,
        success_rate_per_test=success_rate,
        min_metrics=min_metrics,
        max_metrics=max_metrics,
        mean_metrics=mean_metrics,
        mandatory_suite_passed=mandatory_suite_passed,
    )
    if previous_summaries:
        try:
            _detect_regressions(summary, previous_summaries)
        except RegressionDetected as exc:
            summary.regressions.append(str(exc))
            summary.critical_errors.append(str(exc))
            summary.failures = sorted(
                set(summary.failures + ["integration", "goals", "self_model"])
            )
    summary.reproducibility = _evaluate_reproducibility(summary, previous_summaries)
    repro_status = summary.reproducibility.get("status", "insufficient_history")
    if repro_status == "failed":
        summary.critical_errors.append("reproducibility check failed")
    provisional_pass = (
        not summary.failures
        and not summary.critical_errors
        and summary.bypass_events == 0
        and mandatory_suite_passed
        and result.ticks >= MIN_VALIDATION_TICKS
    )
    concept_assessments = _assess_concepts(result, summary)
    summary.concept_scores = {
        assessment.name: assessment.score for assessment in concept_assessments
    }
    summary.concept_failures = [
        assessment.name for assessment in concept_assessments if not assessment.passed
    ]
    summary.concept_details = [
        {
            "name": assessment.name,
            "score": assessment.score,
            "passed": assessment.passed,
            "rationale": assessment.rationale,
            "evidence": assessment.evidence,
        }
        for assessment in concept_assessments
    ]
    if summary.concept_failures:
        summary.failures = sorted(set(summary.failures + summary.concept_failures))
    if (
        summary.critical_errors
        or summary.failures
        or bypass_events
        or not mandatory_suite_passed
    ):
        summary.classification = "Simulation instabil"
    elif (
        repro_status == "passed"
        and stability_score >= 0.85
        and not summary.concept_failures
        and summary.concept_scores.get("breakthrough_candidate", 0.0)
        >= CONCEPT_THRESHOLDS["breakthrough_candidate"]
    ):
        summary.classification = "kausal gekoppelte Architektur (vorlaeufig)"
    else:
        summary.classification = "Simulation stabil"
    summary.passed = provisional_pass and not summary.concept_failures
    return summary


def _detect_regressions(
    summary: RunSummary, previous_summaries: List[Dict[str, Any]]
) -> None:
    previous = previous_summaries[-1]
    prev_mean = previous.get("mean_metrics", {})
    regressions = []
    for metric_name, threshold in DRIFT_THRESHOLDS.items():
        current_value = summary.mean_metrics.get(metric_name, 0.0)
        previous_value = float(prev_mean.get(metric_name, 0.0))
        if current_value < previous_value - threshold:
            regressions.append(
                f"{metric_name} regressed from {previous_value:.4f} to {current_value:.4f}"
            )
    if regressions:
        raise RegressionDetected("; ".join(regressions))


def _evaluate_reproducibility(
    summary: RunSummary, previous_summaries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    same_seed = [
        entry
        for entry in previous_summaries
        if entry.get("seed") == summary.seed
        and entry.get("ticks", 0) >= MIN_VALIDATION_TICKS
    ]
    diff_seed = [
        entry
        for entry in previous_summaries
        if entry.get("seed") != summary.seed
        and entry.get("ticks", 0) >= MIN_VALIDATION_TICKS
    ]
    report: Dict[str, Any] = {
        "status": "insufficient_history",
        "same_seed_matches": len(same_seed),
        "different_seed_matches": len(diff_seed),
        "notes": [],
    }
    same_seed_ok = True
    if same_seed:
        baseline = same_seed[-1]
        for metric_name, tolerance in REPRO_SAME_SEED_TOLERANCE.items():
            delta = abs(
                summary.mean_metrics.get(metric_name, 0.0)
                - float(baseline.get("mean_metrics", {}).get(metric_name, 0.0))
            )
            report[f"same_seed_delta_{metric_name}"] = round(delta, 6)
            if delta > tolerance:
                same_seed_ok = False
                report["notes"].append(f"same seed drift too high for {metric_name}")
    else:
        report["notes"].append("no same-seed baseline available")

    diff_seed_ok = True
    if diff_seed:
        baseline = diff_seed[-1]
        stability_delta = abs(
            summary.stability_score - float(baseline.get("stability_score", 0.0))
        )
        pass_rate_delta = abs(
            _mean(list(summary.success_rate_per_test.values()))
            - _mean(list(baseline.get("success_rate_per_test", {}).values()))
        )
        report["different_seed_delta_stability"] = round(stability_delta, 6)
        report["different_seed_delta_pass_rate"] = round(pass_rate_delta, 6)
        if stability_delta > REPRO_DIFF_SEED_TOLERANCE["stability_score"]:
            diff_seed_ok = False
            report["notes"].append("different seed stability structure diverged")
        if pass_rate_delta > REPRO_DIFF_SEED_TOLERANCE["pass_rate"]:
            diff_seed_ok = False
            report["notes"].append("different seed success-rate structure diverged")
    else:
        report["notes"].append("no different-seed baseline available")

    if same_seed and diff_seed and same_seed_ok and diff_seed_ok:
        report["status"] = "passed"
    elif (same_seed and not same_seed_ok) or (diff_seed and not diff_seed_ok):
        report["status"] = "failed"
    return report


def _persist_artifacts(result: PostFixResult, export_dir: Path, markdown: bool) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    run_stem = f"{result.run_id}_seed{result.seed}_ticks{result.ticks}"
    jsonl_path = export_dir / f"{run_stem}.jsonl"
    summary_path = export_dir / f"{run_stem}_summary.json"
    _export_jsonl(jsonl_path, result.evidence)
    if result.summary_data is None:
        return
    result.summary_data.jsonl_path = str(jsonl_path)
    _export_summary(summary_path, result.summary_data)
    if markdown:
        md_path = export_dir / f"{run_stem}.md"
        result.summary_data.markdown_path = _export_markdown(md_path, result)
        _export_summary(summary_path, result.summary_data)
    _append_summary_index(export_dir / "run_summary_index.jsonl", result.summary_data)


def run_post_fix_validation(
    ticks: int = MIN_VALIDATION_TICKS,
    *,
    seed: int = DEFAULT_SEED,
    export_dir: str = DEFAULT_EXPORT_DIR,
    markdown: bool = False,
    allow_short: bool = False,
) -> PostFixResult:
    if ticks < MIN_VALIDATION_TICKS and not allow_short:
        raise ArchitectureFailure(
            f"short runs are debug only; validation requires >= {MIN_VALIDATION_TICKS} ticks"
        )

    run_id = time.strftime("%Y%m%d-%H%M%S") + f"-{uuid.uuid4().hex[:8]}"
    result = PostFixResult(run_id=run_id, seed=seed, ticks=ticks)
    start = time.time()
    previous_summaries = _load_previous_summaries(Path(export_dir))
    brain = _create_brain(seed)
    try:
        _warm(brain, 260)
        for breakthrough_fn, breakthrough_name in (
            (check_mechanism_unavoidability, "Mechanism unavoidability"),
            (check_minimality, "Minimality test"),
        ):
            try:
                breakthrough_check = breakthrough_fn(seed)
            except ArchitectureFailure as exc:
                breakthrough_check = PostFixCheck(
                    name=breakthrough_name,
                    test="integration",
                    passed=False,
                    detail=_safe(exc),
                    metrics=_zero_metrics(),
                    tick=0,
                    failure_analysis=_safe(exc),
                )
            result.add(breakthrough_check)
            _record_evidence(
                result.evidence,
                run_id=result.run_id,
                seed=result.seed,
                tick=breakthrough_check.tick,
                test=breakthrough_check.test,
                result=breakthrough_check.passed,
                metrics=breakthrough_check.metrics,
                event="breakthrough_check",
                detail=breakthrough_check.detail,
            )
        checks = (
            check_report_causality,
            check_global_state_causality,
            check_goal_feedback,
            check_self_model_causality,
            check_dissociation_and_metacognition,
            check_self_causality,
            check_hard_counterfactual,
            check_goal_breaking_pressure,
            check_metacog_deception,
            check_self_model_destruction,
        )
        test_map = {
            "check_report_causality": "report_enforcement",
            "check_global_state_causality": "integration",
            "check_goal_feedback": "goals",
            "check_self_model_causality": "self_model",
            "check_dissociation_and_metacognition": "dissociation",
            "check_self_causality": "self_model",
            "check_hard_counterfactual": "self_model",
            "check_goal_breaking_pressure": "goals",
            "check_metacog_deception": "dissociation",
            "check_self_model_destruction": "self_model",
        }
        for fn in checks:
            try:
                check = fn(brain)
            except ArchitectureFailure as exc:
                check = PostFixCheck(
                    name=fn.__name__,
                    test=test_map.get(fn.__name__, "integration"),
                    passed=False,
                    detail=_safe(exc),
                    metrics=_metrics_for_failure(brain),
                    tick=brain.tick_count,
                    failure_analysis=_safe(exc),
                )
            result.add(check)
            _record_evidence(
                result.evidence,
                run_id=result.run_id,
                seed=result.seed,
                tick=check.tick,
                test=check.test,
                result=check.passed,
                metrics=check.metrics,
                event="check",
                detail=check.detail,
                forensic=check.forensic if check.bypass_detected else None,
            )

        try:
            long_run = check_long_pressure_run(result, brain, ticks=ticks)
        except ArchitectureFailure as exc:
            long_run = PostFixCheck(
                name="Long pressure run",
                test="integration",
                passed=False,
                detail=_safe(exc),
                metrics=_metrics_for_failure(brain),
                tick=brain.tick_count,
                failure_analysis=_safe(exc),
            )
        result.add(long_run)
        _record_evidence(
            result.evidence,
            run_id=result.run_id,
            seed=result.seed,
            tick=long_run.tick,
            test=long_run.test,
            result=long_run.passed,
            metrics=long_run.metrics,
            event="check",
            detail=long_run.detail,
        )

        from consciousness_testbed import ConsciousnessTestbed

        mandatory_results = ConsciousnessTestbed().run_all(brain._consciousness, brain)
        mandatory_suite_passed = all(record.passed for record in mandatory_results)
        mandatory_failures = [
            record.test_name for record in mandatory_results if not record.passed
        ]
        metrics, _ = _collect_metric_vector(brain)
        result.add(
            PostFixCheck(
                name="Final mandatory suite",
                test="dissociation",
                passed=mandatory_suite_passed,
                detail=f"passed={sum(1 for record in mandatory_results if record.passed)}/{len(mandatory_results)}",
                metrics=metrics,
                tick=brain.tick_count,
                failure_analysis=(
                    (
                        "mandatory dissociation / perturbation / agency / self-error / counterfactual suite regressed: "
                        + ", ".join(mandatory_failures)
                    )
                    if not mandatory_suite_passed
                    else ""
                ),
            )
        )
        _record_evidence(
            result.evidence,
            run_id=result.run_id,
            seed=result.seed,
            tick=brain.tick_count,
            test="dissociation",
            result=mandatory_suite_passed,
            metrics=metrics,
            event="mandatory_suite",
            detail=_safe(mandatory_failures or "all mandatory checks passed"),
        )
        result.summary_data = _build_summary(
            result, mandatory_suite_passed, previous_summaries
        )
        _record_concept_evidence(result)
        _persist_artifacts(result, Path(export_dir), markdown)
    finally:
        result.duration_s = time.time() - start
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-fix causal validation harness")
    parser.add_argument(
        "--ticks",
        type=int,
        default=MIN_VALIDATION_TICKS,
        help=f"Pressure-run ticks (default {MIN_VALIDATION_TICKS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible runs (default {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        help=f"Artifact directory (default {DEFAULT_EXPORT_DIR})",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Also export a Markdown summary"
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Allow sub-1200-tick debug runs (marked non-validating)",
    )
    args = parser.parse_args()
    result = run_post_fix_validation(
        ticks=args.ticks,
        seed=args.seed,
        export_dir=args.export_dir,
        markdown=args.markdown,
        allow_short=args.allow_short,
    )
    print(result.summary())
    raise SystemExit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
